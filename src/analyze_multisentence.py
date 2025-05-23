# %%
import torch
import numpy as np
import pandas as pd
# import plotly.express as px # Keep if you want visualization later
# from phate import PHATE # Keep if you want visualization later
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize # Import minimize

# --- SONAR Model Loading ---

if not hasattr(locals(), "SONAR_AVAILABLE"):
    from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
    print("Attempting to load SONAR models...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the Sonar text-to-embedding model
    text2vec = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder", # Although not explicitly used for predict, good practice
        device=device
    )
    print("SONAR text-to-embedding model loaded successfully.")
    SONAR_AVAILABLE = True

def get_embeddings(texts, batch_size=32):
    """Generates embeddings for a list of texts with batching."""
    if not SONAR_AVAILABLE:
        raise RuntimeError("SONAR models are not available.")
    all_embeddings_np = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings_tensor = text2vec.predict(batch_texts, "eng_Latn")
            batch_embeddings_np = [emb.cpu().numpy() for emb in batch_embeddings_tensor]
            all_embeddings_np.extend(batch_embeddings_np)
            # Simple progress indicator
            print(f"\r  Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}", end="")
    print("\nEmbedding generation complete.") # Newline after loop
    return np.vstack(all_embeddings_np)

# %%
# Objective function for scipy.optimize.minimize
def objective_scalar(params, v1, v2, v12, embedding_dim):
    """
    Calculates the sum of squared errors for the model:
    v12[i] ≈ C + a1*v1[i] + a2*v2[i]

    Args:
        params (np.ndarray): 1D array containing [a1, a2, C_0, C_1, ..., C_{D-1}].
        v1 (np.ndarray): Embeddings for sentence 1, shape (N, D).
        v2 (np.ndarray): Embeddings for sentence 2, shape (N, D).
        v12 (np.ndarray): Embeddings for concatenated sentence, shape (N, D).
        embedding_dim (int): The dimensionality of the embeddings (D).

    Returns:
        float: The sum of squared errors over all samples and dimensions.
    """
    if len(params) != 2 + embedding_dim:
        raise ValueError(f"Expected {2 + embedding_dim} params, got {len(params)}")

    a1 = params[0]
    a2 = params[1]
    C = params[2:] # Shape (D,)

    # Calculate predicted v12 using broadcasting
    # C needs to be reshaped to (1, D) to broadcast across N samples
    v12_pred = C.reshape(1, -1) + a1 * v1 + a2 * v2 # Shape (N, D)

    # Calculate total sum of squared errors
    error = np.sum((v12 - v12_pred)**2)
    return error

# NEW: Gradient (Jacobian) of the objective function
def objective_scalar_grad(params, v1, v2, v12, embedding_dim):
    """
    Calculates the gradient of the objective_scalar function w.r.t. params.

    Args:
        params (np.ndarray): 1D array containing [a1, a2, C_0, ..., C_{D-1}].
        v1 (np.ndarray): Embeddings for sentence 1, shape (N, D).
        v2 (np.ndarray): Embeddings for sentence 2, shape (N, D).
        v12 (np.ndarray): Embeddings for concatenated sentence, shape (N, D).
        embedding_dim (int): The dimensionality of the embeddings (D).

    Returns:
        np.ndarray: The gradient vector, shape (2 + D,).
    """
    N = v1.shape[0]
    if len(params) != 2 + embedding_dim:
        raise ValueError(f"Expected {2 + embedding_dim} params, got {len(params)}")

    a1 = params[0]
    a2 = params[1]
    C = params[2:] # Shape (D,)

    # Calculate the prediction error matrix (Delta)
    v12_pred = C.reshape(1, -1) + a1 * v1 + a2 * v2 # Shape (N, D)
    delta = v12_pred - v12 # Shape (N, D), error term (predicted - actual)

    # Gradient w.r.t. a1: dE/da1 = sum_{i,j} 2 * delta_ij * v1_ij
    grad_a1 = 2 * np.sum(delta * v1)

    # Gradient w.r.t. a2: dE/da2 = sum_{i,j} 2 * delta_ij * v2_ij
    grad_a2 = 2 * np.sum(delta * v2)

    # Gradient w.r.t. C_k: dE/dC_k = sum_i 2 * delta_ik
    # Vectorized calculation for all C_k: sum delta over the samples axis (axis=0)
    grad_C = 2 * np.sum(delta, axis=0) # Shape (D,)

    # Combine gradients into a single flat array matching params structure
    gradient = np.concatenate(([grad_a1, grad_a2], grad_C))
    return gradient

# %%
if __name__ == "__main__" and SONAR_AVAILABLE:

    print("\n--- Multi-Sentence Scalar Relationship Analysis (v12 ≈ C + a1*v1 + a2*v2) ---")

    # 1. Define Sentence Pairs
    sentence_pairs = [
        ("The cat sat on the mat", "The dog chased the ball"),
        ("Apples are red", "Bananas are yellow"),
        ("The sun rises in the east", "The moon orbits the Earth"),
        ("Water is essential for life", "Plants need sunlight to grow"),
        ("He reads books every day", "She enjoys listening to music"),
        ("The train arrived late", "The passengers were annoyed"),
        ("Coding can be challenging", "It is also very rewarding"),
        ("Birds fly in the sky", "Fish swim in the sea"),
        ("Paris is the capital of France", "Berlin is the capital of Germany"),
        ("Machine learning requires data", "Deep learning uses neural networks"),
        # Add more diverse pairs for better results (10-20+ recommended)
    ]
    num_pairs = len(sentence_pairs)
    print(f"Using {num_pairs} sentence pairs for analysis.")

    # 2. Create Lists of S1, S2, and S1S2
    s1_list = [pair[0] for pair in sentence_pairs]
    s2_list = [pair[1] for pair in sentence_pairs]
    s1s2_list = [f"{pair[0]}. {pair[1]}" for pair in sentence_pairs] # Consistent format

    # Combine all unique texts to embed efficiently
    all_texts = list(set(s1_list + s2_list + s1s2_list))
    print(f"\nGenerating embeddings for {len(all_texts)} unique texts...")

    # 3. Generate Embeddings
    try:
        all_embeddings = get_embeddings(all_texts)
        text_to_embedding = {text: emb for text, emb in zip(all_texts, all_embeddings)}

        # Retrieve embeddings for our specific lists
        v1_embeds = np.array([text_to_embedding[text] for text in s1_list])
        v2_embeds = np.array([text_to_embedding[text] for text in s2_list])
        v12_embeds = np.array([text_to_embedding[text] for text in s1s2_list])

        print(f"Shape of v1 embeddings: {v1_embeds.shape}")
        print(f"Shape of v2 embeddings: {v2_embeds.shape}")
        print(f"Shape of v12 embeddings: {v12_embeds.shape}")

        # === Determine Embedding Dimension ===
        if v1_embeds.ndim != 2 or v1_embeds.shape[0] != num_pairs:
             raise ValueError("v1_embeds has unexpected shape!")
        EMBEDDING_DIM = v1_embeds.shape[1]
        print(f"Determined embedding dimension: {EMBEDDING_DIM}")
        # ====================================

        # 4. Perform Optimization to find C, a1, a2
        print("\nOptimizing parameters (C, a1, a2) using scipy.optimize.minimize...")

        # Initial guess for parameters
        initial_a1 = 1.0
        initial_a2 = 1.0
        # Calculate initial C based on the assumption a1=1, a2=1
        initial_C = np.mean(v12_embeds - initial_a1 * v1_embeds - initial_a2 * v2_embeds, axis=0)
        # Assemble initial parameters array [a1, a2, C_0, ..., C_{D-1}]
        initial_params = np.concatenate(([initial_a1, initial_a2], initial_C))

        print(f"  Initial guess: a1={initial_a1:.2f}, a2={initial_a2:.2f}, C norm={np.linalg.norm(initial_C):.4f}")

        # Run the optimization - NOW WITH jac=objective_scalar_grad
        print("  Running optimization with analytical gradients...")
        result = minimize(
            objective_scalar,
            initial_params,
            args=(v1_embeds, v2_embeds, v12_embeds, EMBEDDING_DIM),
            method='L-BFGS-B',
            jac=objective_scalar_grad, # Use the analytical gradient function
            options={'disp': True, 'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-7} # Set gtol, display output
        )

        # 5. Extract and Evaluate Results
        if result.success:
            fitted_params = result.x
            final_a1 = fitted_params[0]
            final_a2 = fitted_params[1]
            final_C = fitted_params[2:]
            min_error = result.fun # Minimized objective function value (Sum of Squared Errors)

            print("\nOptimization Successful!")
            print(f"  Fitted a1: {final_a1:.4f}")
            print(f"  Fitted a2: {final_a2:.4f}")
            print(f"  Fitted C vector norm: {np.linalg.norm(final_C):.4f}")
            print(f"  Final Sum of Squared Errors: {min_error:.4f}")

            # Evaluate the fit
            print("\nEvaluating the scalar model fit...")
            v12_pred_scalar = final_C.reshape(1, -1) + final_a1 * v1_embeds + final_a2 * v2_embeds

            # MSE (average squared error per sample)
            mse_scalar = min_error / num_pairs
            # MSE per dimension (average squared error per dimension value)
            if EMBEDDING_DIM is None or EMBEDDING_DIM == 0:
                print("  Warning: Cannot calculate MSE per dimension (EMBEDDING_DIM is invalid).")
                mse_per_dim = np.nan # Or handle as appropriate
            else:
                mse_per_dim = min_error / (num_pairs * EMBEDDING_DIM)

            # Calculate R-squared manually
            # Total sum of squares (variance in the original v12 data)
            mean_v12 = np.mean(v12_embeds, axis=0)
            ss_total = np.sum((v12_embeds - mean_v12)**2)
            # Residual sum of squares is the minimized error
            ss_residual = min_error
            # R-squared
            r2_scalar = 1 - (ss_residual / ss_total) if ss_total > 1e-9 else 0

            print(f"  Mean Squared Error (MSE) per sample: {mse_scalar:.6f}")
            print(f"  Mean Squared Error (MSE) per dimension: {mse_per_dim:.8f}")
            print(f"  R-squared (R2) Score: {r2_scalar:.4f}")

            # Optional: Check residual norms
            residuals_scalar = v12_embeds - v12_pred_scalar
            avg_residual_norm_scalar = np.mean(np.linalg.norm(residuals_scalar, axis=1))
            avg_target_norm = np.mean(np.linalg.norm(v12_embeds, axis=1))
            print(f"  Average L2 norm of residuals: {avg_residual_norm_scalar:.4f}")
            print(f"  Average L2 norm of target vectors (v12): {avg_target_norm:.4f}")
            if avg_target_norm > 1e-6:
                 print(f"  Relative residual norm: {avg_residual_norm_scalar / avg_target_norm:.4f}")

        else:
            print("\nOptimization Failed!")
            print(f"  Message: {result.message}")

        print("\nAnalysis complete.")
        print("Interpretation:")
        print("- This model assumes v12 is a simple scaled sum of v1 and v2 plus a constant offset.")
        print("- The R2 score indicates how much variance this simple model explains.")
        print("- Compare this R2 to the R2 from the previous linear model (with U1, U2 matrices) if you ran it.")
        print("  - If R2_scalar is nearly as high as R2_linear, the simpler model might be sufficient.")
        print("  - If R2_scalar is much lower, it suggests more complex transformations (like rotations/scaling captured by U1, U2) are needed.")

        # --- BEGIN ADDED DECODING BLOCK ---

        # %% Decode Original vs. Reconstructed Embeddings
        print("\n--- Decoding Original vs. Reconstructed Scalar Embeddings ---")

        # Check if text decoder needs loading
        if 'text_decoder' not in locals():
            try:
                print("Loading SONAR text decoder model...")
                from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline as DecoderPipeline # Avoid name clash

                # Load the SONAR decoder model
                text_decoder = DecoderPipeline(
                    decoder="text_sonar_basic_decoder",
                    tokenizer="text_sonar_basic_decoder", # Or appropriate decoder tokenizer
                    device=device
                )
                print("SONAR text decoder model loaded successfully.")
            except ImportError:
                print("Error: Could not import SONAR decoder pipeline. Is SONAR installed?")
                text_decoder = None
            except Exception as e:
                print(f"Error loading SONAR text decoder model: {e}")
                text_decoder = None

        if 'text_decoder' in locals() and text_decoder is not None:
            # Convert numpy arrays to torch tensors on the correct device
            v12_embeds_tensor = torch.tensor(v12_embeds, dtype=torch.float32, device=device)
            v12_pred_scalar_tensor = torch.tensor(v12_pred_scalar, dtype=torch.float32, device=device)

            num_samples_to_decode = min(len(s1s2_list), 10) # Decode first 10 or fewer
            print(f"Decoding first {num_samples_to_decode} samples...")

            with torch.no_grad():
                # Decode original combined embeddings
                decoded_original_texts = text_decoder.predict(
                    v12_embeds_tensor[:num_samples_to_decode],
                    "eng_Latn" # Assuming English
                )
                # Decode reconstructed combined embeddings
                decoded_reconstructed_texts = text_decoder.predict(
                    v12_pred_scalar_tensor[:num_samples_to_decode],
                    "eng_Latn" # Assuming English
                )

            # Print comparison table
            print("\nComparison of Original vs. Reconstructed Decodings:")
            print("-" * 150)
            print(f"{'Original S1':<40} | {'Original S2':<40} | {'Decoded Original v12':<40} | {'Decoded Reconstructed v12':<40}")
            print("-" * 150)
            for i in range(num_samples_to_decode):
                s1 = s1_list[i]
                s2 = s2_list[i]
                # s1s2 = s1s2_list[i] # Original text for reference if needed
                decoded_orig = decoded_original_texts[i]
                decoded_recon = decoded_reconstructed_texts[i]
                print(f"{s1:<40} | {s2:<40} | {decoded_orig:<40} | {decoded_recon:<40}")
            print("-" * 150)
            print("\nNote: Decoding quality depends on the decoder model and the accuracy of the scalar reconstruction.")

        else:
             print("\nSkipping decoding comparison as the SONAR text decoder is not available.")


# %%
