# %%
import os
import json
import torch
from tqdm import tqdm # Keep tqdm for data loading etc.
import numpy as np
import random
import time
import warnings # To optionally ignore warnings

# --- Configuration ---
MAX_FILES_TO_LOAD = 100 # Adjust as needed
PROBE_SUBSAMPLE_SIZE = 20000 # Subsample data points for probe training (adjust based on memory/time)
USE_PROBE_SUBSAMPLING = True
RANDOM_STATE = 42
# Limit classification report output (set to 0 or None to disable)
MAX_CLASSES_IN_REPORT = 20
MIN_TOKEN_FREQUENCY = 10 # Minimum times a token must appear in train set to be probed

# --- Optional: Ignore sklearn warnings ---
from sklearn.exceptions import UndefinedMetricWarning
# Ignore the specific UserWarning from multiclass about labels not present
# This warning might reappear if a token is present in ALL samples after filtering,
# but the filtering logic should prevent fitting on such columns.
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.multiclass')
# Ignore UndefinedMetricWarning if classification_report causes them for some tokens
warnings.filterwarnings('ignore', category=UndefinedMetricWarning, module='sklearn.metrics')


# Assuming these utils are in the same directory or accessible
from utils_load_data import load_embeds, load_split_paragraphs, load_split_paragraphs_tokenized
# Use the specific SONAR tokenizer loader
try:
    from sonar.models.sonar_text import load_sonar_tokenizer
    # Updated load_tokenizer function provided by user
    def load_tokenizer(repo="text_sonar_basic_encoder"):
        print(f"Loading SONAR tokenizer from repo: {repo}")
        orig_tokenizer = load_sonar_tokenizer(repo)
        tokenizer = orig_tokenizer.create_encoder()
        vocab_size = orig_tokenizer.vocab_info.size

        class TokenizerWrapper:
            def __init__(self, tokenizer, vocab_size):
                self.tokenizer = tokenizer
                self.vocab_size = vocab_size
                self._orig_tokenizer = orig_tokenizer # Store for decoder access

            def __getattr__(self, name):
                return getattr(self.tokenizer, name)

            def __call__(self, *args, **kwargs):
                return self.tokenizer(*args, **kwargs)

            def encode(self, text):
                return self.tokenizer(text)

            def encode_as_tokens(self, text):
                if hasattr(self.tokenizer, 'encode_as_tokens'):
                    return self.tokenizer.encode_as_tokens(text)
                else:
                    print("Warning: encode_as_tokens might not be directly available on the wrapped encoder.")
                    return None

            def get_decoder(self):
                 if hasattr(self._orig_tokenizer, 'create_decoder'):
                     return self._orig_tokenizer.create_decoder()
                 else:
                     return None

        wrapped_tokenizer = TokenizerWrapper(tokenizer, vocab_size)
        print(f"Tokenizer loaded. Vocab size: {wrapped_tokenizer.vocab_size}")
        return wrapped_tokenizer

except ImportError:
    print("ERROR: SONAR library not found or load_sonar_tokenizer failed.")
    class DummyTokenizer:
        vocab_size = 10
        def __init__(self): self._vocab_size=10
        def encode(self, text): return torch.tensor([random.randint(0,9) for _ in text.split()])
        def get_decoder(self): return None
    tokenizer = DummyTokenizer()
except Exception as e:
    print(f"An unexpected error occurred during tokenizer loading: {e}")
    raise


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier # Use this again for parallelization
from sklearn.metrics import f1_score, hamming_loss, accuracy_score # Subset accuracy
from sklearn.metrics import classification_report # Can be very verbose
from scipy.sparse import csr_matrix # For efficient sparse matrix


# --- Load Tokenizer ---
tokenizer = load_tokenizer()
VOCAB_SIZE = tokenizer.vocab_size

# --- Data Loading ---
def load_data_for_probes(max_files=MAX_FILES_TO_LOAD):
    """Load vectors and texts for probing."""
    all_vecs = []
    all_texts = []
    all_tokens = []
    print(f"Loading data from {max_files} files...")
    num_skipped = 0
    for i in tqdm(range(max_files), desc="Loading files"):
        try:
            vecs = load_embeds(i)
            texts = load_split_paragraphs(i)
            tokens = load_split_paragraphs_tokenized(i)
        except FileNotFoundError:
            print(f"Warning: File {i} not found, skipping.")
            continue
        for v, t, tk in zip(vecs, texts, tokens):
            if len(t) <= 0:
                num_skipped += 1
                continue
            all_vecs.append(v.cpu())
            all_texts.append(t)
            all_tokens.append(tk)
    print(f"Loaded {len(all_vecs)} vectors/texts. Skipped {num_skipped} empty texts.")
    if not all_vecs:
        raise ValueError("No vectors were loaded.")
    vec_array = torch.stack(all_vecs).numpy()
    return vec_array, all_texts, all_tokens

# --- Target Matrix Creation ---
def create_token_presence_target_matrix(tokens_list):
    """Creates a sparse matrix Y where Y[i,j]=1 if token j is in text i."""
    print(f"Creating sparse target matrix for {len(tokens_list)} texts and vocab size {VOCAB_SIZE}...")
    rows, cols = [], []
    for i, _tokens in enumerate(tqdm(tokens_list, desc="Tokenizing texts")):
        try:
            unique_ids = set(_tokens)
            for token_id in unique_ids:
                if 0 <= token_id and token_id < VOCAB_SIZE:
                    rows.append(i)
                    cols.append(token_id)
        except Exception as e:
            print(f"Warning: Error processing text sample {i}: {e}")
            raise e
    y_sparse = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(tokens_list), VOCAB_SIZE))
    print(f"Sparse target matrix created. Sparsity: {y_sparse.nnz / (y_sparse.shape[0] * y_sparse.shape[1]):.6f}")
    return y_sparse

# --- Probing Function (REVISED with Filtering before Parallel Training) ---
def run_token_presence_probes(X, Y, test_size=0.2):
    """Train and evaluate token presence linear probes using parallel processing."""
    print(f"\nRunning token presence probes on {X.shape[0]} samples...")

    # 1. Train/Test Split
    print(f"Splitting data (test_size={test_size})...")
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=RANDOM_STATE
    )
    n_train_samples = X_train.shape[0]
    print(f"Train shape: X={X_train.shape}, Y={Y_train.shape}")
    print(f"Test shape:  X={X_test.shape}, Y={Y_test.shape}")

    # 2. Filter Tokens by Frequency and Variability in Training Set
    print(f"Filtering tokens: Keeping tokens present >= {MIN_TOKEN_FREQUENCY} times "
          f"and present in < {n_train_samples} samples in the training set...")
    token_frequencies = np.array(Y_train.sum(axis=0)).flatten()

    # Indices of tokens meeting min frequency AND not present in all samples
    variable_token_indices = np.where(
        (token_frequencies >= MIN_TOKEN_FREQUENCY) &
        (token_frequencies < n_train_samples)
    )[0]
    num_variable_tokens = len(variable_token_indices)

    if num_variable_tokens == 0:
        print("ERROR: No tokens meet the frequency and variability requirements. Aborting probe.")
        return None

    print(f"Found {num_variable_tokens} tokens (out of {VOCAB_SIZE}) to probe.")

    # 3. Create Filtered Target Matrices
    print("Creating filtered target matrices...")
    # Select only the columns corresponding to variable tokens
    Y_train_filtered = Y_train[:, variable_token_indices]
    Y_test_filtered = Y_test[:, variable_token_indices]
    print(f"Filtered Train Y shape: {Y_train_filtered.shape}")
    print(f"Filtered Test Y shape:  {Y_test_filtered.shape}")

    # 4. Define and Train Parallel Model
    print("Initializing OneVsRestClassifier with LogisticRegression (using n_jobs=-1)...")
    # Note: n_jobs=-1 uses all available CPU cores
    base_classifier = LogisticRegression(
        solver='liblinear',
        C=1.0,
        random_state=RANDOM_STATE,
        class_weight='balanced',
        max_iter=1000
    )
    # Train on the X_train and the *filtered* Y_train_filtered
    model = OneVsRestClassifier(base_classifier, n_jobs=-1)

    print("Training the probe model in parallel (this might take a while)...")
    start_time = time.time()
    # Fit the model on the filtered data
    model.fit(X_train, Y_train_filtered)
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")

    # 5. Evaluate Model on Filtered Test Data
    print("Evaluating the probe model on the test set (only filtered tokens)...")
    # Predict using the trained model
    Y_pred_filtered = model.predict(X_test)

    # Ensure predictions are in the same format (sparse or dense) as Y_test_filtered
    # OneVsRestClassifier usually returns dense numpy array, Y_test_filtered is sparse
    # Convert Y_test_filtered to dense for metric calculation if needed
    if isinstance(Y_test_filtered, csr_matrix):
        Y_test_filtered_dense = Y_test_filtered.toarray()
    else:
        Y_test_filtered_dense = Y_test_filtered # Assume already dense if not sparse

    # Calculate Metrics (using dense versions)
    hamming = hamming_loss(Y_test_filtered_dense, Y_pred_filtered)
    subset_acc = accuracy_score(Y_test_filtered_dense, Y_pred_filtered)
    f1_micro = f1_score(Y_test_filtered_dense, Y_pred_filtered, average='micro', zero_division=0)
    f1_macro = f1_score(Y_test_filtered_dense, Y_pred_filtered, average='macro', zero_division=0)
    f1_samples = f1_score(Y_test_filtered_dense, Y_pred_filtered, average='samples', zero_division=0)

    print("\n--- Probe Evaluation Results (Filtered Tokens Only) ---")
    print(f"Number of Tokens Probed: {num_variable_tokens}")
    print(f"Hamming Loss:         {hamming:.4f}")
    print(f"Subset Accuracy:      {subset_acc:.4f}")
    print(f"F1 Score (Micro):     {f1_micro:.4f}")
    print(f"F1 Score (Macro):     {f1_macro:.4f}")
    print(f"F1 Score (Samples):   {f1_samples:.4f}")

    # Optional: Detailed report for the *probed* tokens
    if MAX_CLASSES_IN_REPORT is not None and MAX_CLASSES_IN_REPORT > 0 and num_variable_tokens > 0:
        print(f"\nClassification Report (Top {min(MAX_CLASSES_IN_REPORT, num_variable_tokens)} probed classes by original support):")
        try:
            # Get the original indices of the probed tokens
            probed_original_indices = variable_token_indices

            # Get the original frequencies for *only* the probed tokens
            original_support_for_probed = token_frequencies[probed_original_indices]
            # Sort the *indices within the probed set* based on original frequency
            sorted_indices_within_probed = np.argsort(original_support_for_probed)[::-1]

            # Select top N indices *within the probed set* for the report
            report_indices_in_filtered_array = sorted_indices_within_probed[:min(MAX_CLASSES_IN_REPORT, num_variable_tokens)]
            # Get the original token IDs corresponding to these top report indices
            report_original_token_ids = probed_original_indices[report_indices_in_filtered_array]

            # Create labels for the report using original token IDs
            target_names_for_report = [f"TokenID_{i}" for i in report_original_token_ids]

            report = classification_report(
                Y_test_filtered_dense, # Use dense filtered true labels
                Y_pred_filtered,       # Use filtered predictions
                # labels should be the indices *within the filtered arrays* that we want to report on
                labels=report_indices_in_filtered_array,
                target_names=target_names_for_report, # Names corresponding to these indices
                zero_division=0,
                digits=3
            )
            print(report)

        except Exception as e:
            print(f"Could not generate detailed classification report: {e}")

    print("----------------------------------------------------")
    # Return the trained OneVsRest model and the indices it was trained on
    return model, variable_token_indices

# %%

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load Data
    vec_array, texts_list, tokens_list = load_data_for_probes(max_files=MAX_FILES_TO_LOAD)

    # 2. Create Target Matrix
    Y_sparse = create_token_presence_target_matrix(tokens_list)

    # 3. Optional Subsampling for Probing
    X_probe_data = vec_array
    Y_probe_data = Y_sparse

    if USE_PROBE_SUBSAMPLING and len(vec_array) > PROBE_SUBSAMPLE_SIZE:
        print(f"\nSubsampling data from {len(vec_array)} to {PROBE_SUBSAMPLE_SIZE} points for probe training...")
        random.seed(RANDOM_STATE)
        indices = random.sample(range(len(vec_array)), PROBE_SUBSAMPLE_SIZE)
        X_probe_data = vec_array[indices]
        Y_probe_data = Y_sparse[indices] # Subsample sparse matrix
        print("Subsampling complete.")
    else:
        print("\nUsing full dataset for probe training (or dataset smaller than subsample size).")

    # 4. Run Probes (using revised function with filtering and parallel training)
    results = run_token_presence_probes(X_probe_data, Y_probe_data)
    if results:
        trained_model, probed_indices = results
        print(f"\nToken presence probe analysis complete. Model trained for {len(probed_indices)} tokens.")
    else:
        print("\nToken presence probe analysis aborted or failed.")


# %%
# Save the trained model and probed indices
import pickle
import os
import datetime

# Create a directory for saving models if it doesn't exist
os.makedirs('../data/tae_models', exist_ok=True)

# Save the trained model
model_path = '../data/tae_models/token_presence_scipy_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(trained_model, f)
print(f"Trained model saved to {model_path}")

# Save the probed indices
indices_path = '../data/tae_models/token_presence_scipy_probed_indices.pkl'
with open(indices_path, 'wb') as f:
    pickle.dump(probed_indices, f)
print(f"Probed indices saved to {indices_path}")

# Optional: Save a metadata file with information about the model
metadata = {
    'model_type': 'token_presence_classifier',
    'num_tokens': len(probed_indices),
    'training_date': str(datetime.datetime.now()),
    'random_state': RANDOM_STATE,
    'subsample_size': PROBE_SUBSAMPLE_SIZE if USE_PROBE_SUBSAMPLING else None
}

metadata_path = '../data/tae_models/token_presence_scipy_metadata.pkl'
with open(metadata_path, 'wb') as f:
    pickle.dump(metadata, f)
print(f"Model metadata saved to {metadata_path}")

# %% Load the model and probed indices
import pickle

# Load the trained model
model_path = '../data/tae_models/token_presence_scipy_model.pkl'
with open(model_path, 'rb') as f:
    trained_model = pickle.load(f)

# Load the probed indices
indices_path = '../data/tae_models/token_presence_scipy_probed_indices.pkl'
with open(indices_path, 'rb') as f:
    probed_indices = pickle.load(f)

# Load the metadata
metadata_path = '../data/tae_models/token_presence_scipy_metadata.pkl'
with open(metadata_path, 'rb') as f:
    metadata = pickle.load(f)

print(f"Loaded model trained on {metadata['num_tokens']} tokens.")

# %%


tokenizer_decoder = tokenizer.get_decoder()
token_map = {}
for token_id in probed_indices:
    token_map[token_id] = tokenizer_decoder(torch.tensor([token_id]))
print(token_map)
# %%

# Analyze and visualize the token presence classifier results
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import seaborn as sns

if __name__ == "__main__":
    print("\nAnalyzing token presence classifier results...")

    # vec_array, texts_list, tokens_list = load_data_for_probes(max_files=MAX_FILES_TO_LOAD)
    # Y_sparse = create_token_presence_target_matrix(tokens_list)

    X_probe_data = vec_array[:1000]
    Y_probe_data = Y_sparse[:1000]


    # Calculate token-wise metrics from the classification results
    token_metrics = {}

    # We need to use the data that's already available, not X_probe_eval which isn't defined
    # Get predictions using the trained model on the available data
    if hasattr(trained_model, 'predict'):
        # For scikit-learn models
        y_pred = trained_model.predict(X_probe_data)
        y_true = Y_probe_data.toarray() if hasattr(Y_probe_data, 'toarray') else Y_probe_data
    else:
        # For PyTorch models (if that's what we're using)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = trained_model.to(device)
        model.eval()

        X_tensor = torch.tensor(X_probe_data, dtype=torch.float32)
        batch_size = 128
        all_preds = []

        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_tensor[i:i+batch_size].to(device)
                logits = model(batch_X)
                preds = torch.sigmoid(logits) > 0.5
                all_preds.append(preds.cpu())

        y_pred = torch.cat(all_preds, dim=0).numpy()
        y_true = Y_probe_data.toarray() if hasattr(Y_probe_data, 'toarray') else Y_probe_data

    # Calculate metrics for each token
    print("Calculating metrics for each token...")
    for i, token_idx in enumerate(probed_indices):
        y_true_i = y_true[:, i]
        if isinstance(y_true_i, csr_matrix):
            y_true_i = y_true_i.toarray()
        y_pred_i = y_pred[:, i]
        if isinstance(y_pred_i, csr_matrix):
            y_pred_i = y_pred_i.toarray()

        true_positives = np.sum((y_pred_i == 1) & (y_true_i == 1))
        false_positives = np.sum((y_pred_i == 1) & (y_true_i == 0))
        false_negatives = np.sum((y_pred_i == 0) & (y_true_i == 1))
        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        support = np.sum(y_true_i == 1)

        token_metrics[token_idx] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'token': token_map[token_idx]
        }

    # Sort tokens by F1 score
    sorted_tokens = sorted(token_metrics.items(), key=lambda x: x[1]['f1'], reverse=True)

    # Print top 20 tokens by F1 score
    print("\nTop 20 tokens by F1 score:")
    print(f"{'Token':<15} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'Support':<10}")
    print("-" * 60)

    for token_idx, metrics in sorted_tokens[:20]:
        token_name = metrics['token']
        print(f"{token_name:<15} {metrics['precision']:.3f}      {metrics['recall']:.3f}      {metrics['f1']:.3f}       {int(metrics['support'])}")

    # Visualize F1 score distribution
    plt.figure(figsize=(12, 6))
    f1_scores = [metrics['f1'] for _, metrics in sorted_tokens]
    plt.subplot(1, 2, 1)
    plt.hist(f1_scores, bins=20, alpha=0.7)
    plt.title('Distribution of F1 Scores')
    plt.xlabel('F1 Score')
    plt.ylabel('Number of Tokens')

    # Plot F1 scores in descending order (log-log scale)
    plt.subplot(1, 2, 2)
    plt.loglog(range(1, len(f1_scores) + 1), f1_scores)
    plt.title('F1 Scores (Log-Log Scale)')
    plt.xlabel('Token Rank')
    plt.ylabel('F1 Score')
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig('../data/tae_models/token_f1_distribution.png')
    plt.show()

    # Analyze relationship between token frequency and F1 score
    plt.figure(figsize=(10, 6))
    supports = [metrics['support'] for _, metrics in sorted_tokens]
    f1_scores = [metrics['f1'] for _, metrics in sorted_tokens]

    plt.scatter(supports, f1_scores, alpha=0.5)
    plt.xscale('log')
    plt.title('Token Frequency vs F1 Score - SciPy Binary')
    plt.xlabel('Token Frequency (log scale)')
    plt.ylabel('F1 Score')
    plt.grid(True, which="both", ls="--")
    plt.savefig('../data/tae_models/token_frequency_vs_f1.png')
    plt.loglog()
    plt.show()

    # Create a heatmap of precision, recall, and F1 for top tokens
    top_n = 15  # Number of top tokens to display
    plt.figure(figsize=(12, 8))

    top_tokens = sorted_tokens[:top_n]
    token_names = [metrics['token'] for _, metrics in top_tokens]
    metrics_data = np.array([
        [metrics['precision'], metrics['recall'], metrics['f1']]
            for _, metrics in top_tokens
    ])

    sns.heatmap(metrics_data, annot=True, fmt=".3f", cmap="YlGnBu",
                xticklabels=['Precision', 'Recall', 'F1'],
                yticklabels=token_names)
    plt.title(f'Performance Metrics for Top {top_n} Tokens')
    plt.tight_layout()
    plt.savefig('../data/tae_models/top_tokens_metrics_heatmap.png')
    plt.show()

    print("\nAnalysis complete. Visualization images saved to '../data/tae_models/'")


# %%
