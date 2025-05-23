# %% New Cell for Comparison
import os
import json
import torch
from tqdm import tqdm
import numpy as np
import random
import time
import math

# Assuming utils_sonar.py and the SAE definition are accessible
from utils_sonar import SonarDecoderCELoss # Import the loss calculator
from utils_load_data import load_embeds, load_split_paragraphs
import torch.nn as nn
torch.set_grad_enabled(False)

# Import the SAE class definition (copy it here or import from the training script)
# --- SAE Model Definition (Copied from previous script) ---
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, l1_coeff): # l1_coeff not needed for inference
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # self.l1_coeff = l1_coeff # Not needed for inference

        self.encoder = nn.Linear(input_dim, hidden_dim, bias=False)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)
        self.relu = nn.ReLU()

        # No need for initialization here, weights will be loaded

    def forward(self, x):
        # Simplified forward for inference (no loss calculation here)
        encoded = self.encoder(x)
        activations = self.relu(encoded)
        reconstructed = self.decoder(activations)
        return reconstructed, activations # Return activations too if needed

    def encode(self, x):
        encoded = self.encoder(x)
        activations = self.relu(encoded)
        return activations

    def decode(self, activations):
        return self.decoder(activations)
# --- End SAE Model Definition ---


# --- Configuration for Comparison ---
NUM_SAMPLES = 100
MAX_FILES_TO_TRY = 20 # Try loading from a few files to get samples
SAE_MODEL_DIR = "./sae_models" # Directory where models are saved
# *** IMPORTANT: Specify the exact model and config file you want to test ***
import os
import json
import matplotlib.pyplot as plt
from utils_sonar import TextToEmbeddingModelPipeline, EmbeddingToTextModelPipeline

# Update model filenames based on the instructions
# SAE_MODEL_FILENAME = "sae_final_H8192_L1_5.0e-06.pt"
# SAE_CONFIG_FILENAME = "sae_config_H8192_L1_5.0e-06.json"
SAE_MODEL_FILENAME = "sae_final_H8192_L1_8.0e-06.pt"
SAE_CONFIG_FILENAME = "sae_config_H8192_L1_8.0e-06.json"

SAE_MODEL_PATH = os.path.join(SAE_MODEL_DIR, SAE_MODEL_FILENAME)
SAE_CONFIG_PATH = os.path.join(SAE_MODEL_DIR, SAE_CONFIG_FILENAME)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_STATE = 42 # Use the same seed if needed for consistency

# --- Set Seed ---
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

# --- Helper Functions ---
def load_sae_model(model_path, config_path, device):
    """Loads a trained SAE model."""
    print(f"Loading SAE config from: {config_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"Loading SAE model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = SparseAutoencoder(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        l1_coeff=config.get('l1_coeff', 0) # Provide default if missing
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set to evaluation mode
    print("SAE model loaded successfully.")
    return model

def load_comparison_data(num_samples, max_files_to_try=10):
    """Loads a specific number of embed/text pairs."""
    original_embeddings = []
    texts = []
    files_loaded = 0
    print(f"Attempting to load {num_samples} samples...")

    for i in range(max_files_to_try):
        if len(original_embeddings) >= num_samples:
            break
        try:
            embeds_batch = load_embeds(i)
            texts_batch = load_split_paragraphs(i)
            files_loaded += 1
            for embed, text in zip(embeds_batch, texts_batch):
                if len(text) > 0: # Basic check for non-empty text
                    original_embeddings.append(embed.cpu()) # Ensure on CPU for consistency
                    texts.append(text)
                    if len(original_embeddings) >= num_samples:
                        break
        except FileNotFoundError:
            print(f"Warning: File {i} not found, skipping.")
            continue
        except Exception as e:
            print(f"Warning: Error loading file {i}: {e}")
            continue

    if len(original_embeddings) < num_samples:
        print(f"Warning: Could only load {len(original_embeddings)} samples out of {num_samples} requested.")
    else:
        print(f"Successfully loaded {len(original_embeddings)} samples from {files_loaded} files.")

    return original_embeddings, texts

# --- Main Comparison Logic ---
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # 1. Load the trained SAE model
    sae_model = load_sae_model(SAE_MODEL_PATH, SAE_CONFIG_PATH, DEVICE)

    # 2. Load the SONAR Decoder Loss Calculator
    # Use default repo, adjust if needed
    print("Initializing SONAR Decoder Loss Calculator...")
    sonar_decoder_loss_calc = SonarDecoderCELoss(device=DEVICE)
    # Set the decoder model to eval mode as well
    sonar_decoder_loss_calc.model.eval()
    print("SONAR Decoder Loss Calculator initialized.")

    # 3. Load Comparison Data
    original_embeddings, texts = load_comparison_data(NUM_SAMPLES, MAX_FILES_TO_TRY)

    if not original_embeddings:
        print("No data loaded for comparison. Exiting.")
        exit()

    # 4. Perform Comparison
    original_losses = []
    reconstructed_losses = []

    print(f"\n--- Comparing Decoder Loss on {len(original_embeddings)} Samples ---")
    with torch.no_grad(): # Ensure no gradients are calculated
        for i in tqdm(range(len(original_embeddings)), desc="Processing Samples"):
            original_embed = original_embeddings[i].to(DEVICE)
            text = texts[i]

            # Ensure correct shape (add batch dim if needed)
            if original_embed.dim() == 1:
                original_embed = original_embed.unsqueeze(0) # Shape: [1, input_dim]

            try:
                # Get SAE reconstruction
                reconstructed_embed, _ = sae_model(original_embed) # Forward pass for reconstruction

                # Calculate loss for original embedding
                # Need to pass text as a list for tokenize_text
                loss_orig = sonar_decoder_loss_calc(original_embed, [text])
                original_losses.append(loss_orig.item())

                # Calculate loss for reconstructed embedding
                loss_recon = sonar_decoder_loss_calc(reconstructed_embed, [text])
                reconstructed_losses.append(loss_recon.item())

            except Exception as e:
                print(f"\nWarning: Error processing sample {i}: {e}")
                print(f"  Text: '{text[:100]}...'")
                # Append NaN or skip? Let's append NaN for now to keep lists aligned
                original_losses.append(float('nan'))
                reconstructed_losses.append(float('nan'))

    # 5. Analyze and Print Results
    # Filter out potential NaNs before calculating averages
    valid_indices = [k for k, (lo, lr) in enumerate(zip(original_losses, reconstructed_losses)) if not (math.isnan(lo) or math.isnan(lr))]
    if not valid_indices:
        print("\nNo valid results obtained after processing samples.")
    else:
        valid_original_losses = np.array([original_losses[k] for k in valid_indices])
        valid_reconstructed_losses = np.array([reconstructed_losses[k] for k in valid_indices])

        avg_loss_orig = np.mean(valid_original_losses)
        avg_loss_recon = np.mean(valid_reconstructed_losses)
        loss_diff = avg_loss_recon - avg_loss_orig
        perc_increase = (loss_diff / avg_loss_orig) * 100 if avg_loss_orig != 0 else float('inf')

        print("\n--- Comparison Results ---")
        print(f"Number of valid samples compared: {len(valid_indices)}")
        print(f"Average Original Embedding Loss:      {avg_loss_orig:.6f}")
        print(f"Average Reconstructed Embedding Loss: {avg_loss_recon:.6f}")
        print(f"Average Loss Difference (Recon - Orig): {loss_diff:.6f}")
        print(f"Average Percentage Loss Increase:     {perc_increase:.2f}%")

        # Print first few results for inspection
        print("\nFirst 10 Sample Losses (Original vs Reconstructed):")
        for k in range(min(10, len(valid_indices))):
            idx = valid_indices[k]
            print(f"  Sample {idx+1}: Orig={original_losses[idx]:.4f}, Recon={reconstructed_losses[idx]:.4f}")
        # 6. Visualize the original vs reconstructed text decoding

# %%
if __name__ == "__main__":
    print("\n--- Text Decoding Comparison ---")
    print("Comparing original embedding vs reconstructed embedding decoding:")

    # Take a few samples for visualization
    num_vis_samples = min(5, len(valid_indices))
    for k in range(num_vis_samples):
        idx = valid_indices[k]
        sample_idx = idx  # Use the valid sample index

        # Get the original embedding and text
        original_embed = original_embeddings[sample_idx].to(DEVICE)
        if original_embed.dim() == 1:
            original_embed = original_embed.unsqueeze(0)

        # Get the reconstructed embedding
        reconstructed_embed, _ = sae_model(original_embed)

        # Get the original text
        original_text = texts[sample_idx]

        # Decode both embeddings using SONAR
        try:
            # This assumes sonar_decoder_loss_calc has a method to get the decoded text
            # If not available, you might need to implement this functionality
            decoded_original = sonar_decoder_loss_calc.decode(original_embed)
            decoded_reconstructed = sonar_decoder_loss_calc.decode(reconstructed_embed)

            print(f"\nSample {idx+1}:")
            print(f"Original text: '{original_text[:100]}...'")
            print(f"Decoded from original embedding: '{decoded_original[:100]}...'")
            print(f"Decoded from reconstructed embedding: '{decoded_reconstructed[:100]}...'")

        except Exception as e:
            print(f"\nWarning: Error decoding sample {idx+1}: {e}")

# %%
