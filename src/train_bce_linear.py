# %%
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm # Import tqdm for progress bars
import numpy as np
import random
import time
import warnings # To optionally ignore warnings

# --- Configuration ---
MAX_FILES_TO_LOAD = 100 # Adjust as needed
PROBE_SUBSAMPLE_SIZE = 100_000 # Subsample data points for probe training (adjust based on memory/time)
USE_PROBE_SUBSAMPLING = True
RANDOM_STATE = 42
# Limit classification report output (set to 0 or None to disable)
MAX_CLASSES_IN_REPORT = 20
MIN_TOKEN_FREQUENCY = 10 # Minimum times a token must appear in train set to be probed

# --- PyTorch Training Configuration ---
LEARNING_RATE = 1e-3
BATCH_SIZE = 1024 # Adjust based on GPU memory
NUM_EPOCHS = 10 # Adjust based on convergence

# --- Optional: Ignore sklearn warnings ---
from sklearn.exceptions import UndefinedMetricWarning
# Ignore UndefinedMetricWarning if classification_report causes them for some tokens
warnings.filterwarnings('ignore', category=UndefinedMetricWarning, module='sklearn.metrics')


# Assuming these utils are in the same directory or accessible
from utils_load_data import load_embeds, load_split_paragraphs
torch.set_grad_enabled(True)
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
                # Delegate attributes/methods to the wrapped encoder
                return getattr(self.tokenizer, name)

            def __call__(self, *args, **kwargs):
                # Delegate the __call__ method (likely encode)
                return self.tokenizer(*args, **kwargs)

            def encode(self, text):
                # Assuming the encoder itself is callable for encoding
                return self.tokenizer(text)

            def encode_as_tokens(self, text):
                 # Check if the underlying encoder has this method
                if hasattr(self.tokenizer, 'encode_as_tokens'):
                    return self.tokenizer.encode_as_tokens(text)
                else:
                    print("Warning: encode_as_tokens might not be directly available on the wrapped encoder.")
                    return None

            # Add a method to potentially access the decoder later
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
# from sklearn.linear_model import LogisticRegression # No longer needed
# from sklearn.multiclass import OneVsRestClassifier # No longer needed
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
    print(f"Loading data from {max_files} files...")
    num_skipped = 0
    for i in tqdm(range(max_files), desc="Loading files"):
        try:
            vecs = load_embeds(i)
            texts = load_split_paragraphs(i)
        except FileNotFoundError:
            print(f"Warning: File {i} not found, skipping.")
            continue
        for v, t in zip(vecs, texts):
            if len(t) <= 0:
                num_skipped += 1
                continue
            all_vecs.append(v.cpu())
            all_texts.append(t)
    print(f"Loaded {len(all_vecs)} vectors/texts. Skipped {num_skipped} empty texts.")
    if not all_vecs:
        raise ValueError("No vectors were loaded.")
    vec_array = torch.stack(all_vecs).numpy()
    return vec_array, all_texts

# --- Target Matrix Creation ---
def create_token_presence_target_matrix(texts, tokenizer):
    """Creates a sparse matrix Y where Y[i,j]=1 if token j is in text i."""
    print(f"Creating sparse target matrix for {len(texts)} texts and vocab size {VOCAB_SIZE}...")
    rows, cols = [], []
    for i, text in enumerate(tqdm(texts, desc="Tokenizing texts")):
        try:
            token_ids_tensor = tokenizer.encode(text)
            unique_ids = set(token_ids_tensor.cpu().numpy())
            for token_id in unique_ids:
                if 0 <= token_id < VOCAB_SIZE:
                    rows.append(i)
                    cols.append(token_id)
        except Exception as e:
            print(f"Warning: Error processing text sample {i}: {e}")
            continue
    y_sparse = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(texts), VOCAB_SIZE))
    print(f"Sparse target matrix created. Sparsity: {y_sparse.nnz / (y_sparse.shape[0] * y_sparse.shape[1]):.6f}")
    return y_sparse

# --- PyTorch Dataset ---
class ProbeDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        # Ensure labels are dense and float32 for BCEWithLogitsLoss
        if isinstance(labels, csr_matrix):
            # Convert sparse matrix to dense tensor
            self.labels = torch.tensor(labels.toarray(), dtype=torch.float32)
        elif isinstance(labels, np.ndarray):
             self.labels = torch.tensor(labels, dtype=torch.float32)
        else:
             # Assuming labels might already be a tensor
             self.labels = labels.clone().detach().to(dtype=torch.float32)


    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# --- PyTorch Model ---
class LinearProbe(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Return raw logits
        return self.linear(x)

# --- PyTorch Probing Function ---
def run_pytorch_token_presence_probes(X, Y, test_size=0.1):
    """Train and evaluate token presence linear probes using PyTorch."""
    print(f"\nRunning PyTorch token presence probes on {X.shape[0]} samples...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Train/Test Split (on original indices/data before filtering Y)
    print(f"Splitting data indices (test_size={test_size})...")
    indices = np.arange(X.shape[0])
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=RANDOM_STATE
    )
    X_train, X_test = X[train_indices], X[test_indices]
    # Keep Y split sparse for now for efficient frequency calculation
    Y_train_sparse, Y_test_sparse = Y[train_indices], Y[test_indices]

    n_train_samples = X_train.shape[0]
    print(f"Train shape: X={X_train.shape}, Y={Y_train_sparse.shape}")
    print(f"Test shape:  X={X_test.shape}, Y={Y_test_sparse.shape}")

    # 2. Filter Tokens by Frequency and Variability in Training Set
    print(f"Filtering tokens: Keeping tokens present >= {MIN_TOKEN_FREQUENCY} times "
          f"and present in < {n_train_samples} samples in the training set...")
    token_frequencies = np.array(Y_train_sparse.sum(axis=0)).flatten()

    variable_token_indices = np.where(
        (token_frequencies >= MIN_TOKEN_FREQUENCY) &
        (token_frequencies < n_train_samples)
    )[0]
    num_variable_tokens = len(variable_token_indices)

    if num_variable_tokens == 0:
        print("ERROR: No tokens meet the frequency and variability requirements. Aborting probe.")
        return None

    print(f"Found {num_variable_tokens} tokens (out of {VOCAB_SIZE}) to probe.")

    # 3. Create Filtered Target Matrices (Sparse initially)
    print("Creating filtered target matrices (sparse)...")
    Y_train_filtered_sparse = Y_train_sparse[:, variable_token_indices]
    Y_test_filtered_sparse = Y_test_sparse[:, variable_token_indices]
    print(f"Filtered Train Y shape: {Y_train_filtered_sparse.shape}")
    print(f"Filtered Test Y shape:  {Y_test_filtered_sparse.shape}")

    # Calculate positive weights for loss balancing
    print("Calculating positive weights for loss balancing...")
    # Ensure Y_train_filtered_sparse is used
    num_train_samples = Y_train_filtered_sparse.shape[0]
    # Sum occurrences of each token (positive class count)
    pos_counts = np.array(Y_train_filtered_sparse.sum(axis=0)).flatten()
    # Avoid division by zero for tokens that might somehow have 0 count despite filtering
    # Though the filtering should prevent counts < MIN_TOKEN_FREQUENCY
    pos_counts = np.maximum(pos_counts, 1e-8)  # Add small epsilon
    neg_counts = num_train_samples - pos_counts
    pos_weight_values = neg_counts / pos_counts

    # Create a tensor for the weights, move to device
    pos_weight_tensor = torch.tensor(pos_weight_values, dtype=torch.float32).to(device)
    print(f"Calculated pos_weights. Min: {pos_weight_tensor.min():.2f}, Max: {pos_weight_tensor.max():.2f}, Mean: {pos_weight_tensor.mean():.2f}")

    # 4. Create Datasets and DataLoaders (using filtered sparse Y, converted in Dataset)
    print("Creating PyTorch Datasets and DataLoaders...")
    train_dataset = ProbeDataset(X_train, Y_train_filtered_sparse)
    test_dataset = ProbeDataset(X_test, Y_test_filtered_sparse)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 5. Initialize Model, Loss, Optimizer
    input_dim = X_train.shape[1] # Should be 1024
    output_dim = num_variable_tokens # Number of tokens being probed

    model = LinearProbe(input_dim, output_dim).to(device)
    # Use weighted loss to address class imbalance
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor) # Handles sigmoid + BCE with class weights
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 6. Training Loop
    print("Starting Training...")
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
        for embeddings, labels in pbar:
            embeddings, labels = embeddings.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(embeddings)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} Average Training Loss: {avg_loss:.4f}")
        # Optional: Add validation loss calculation within the epoch loop for monitoring

    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")

    # 7. Evaluation
    print("Evaluating...")
    model.eval()
    all_preds_list = []
    all_labels_list = []
    with torch.no_grad():
        for embeddings, labels in tqdm(test_loader, desc="Evaluating"):
            embeddings = embeddings.to(device)
            logits = model(embeddings)
            # Apply sigmoid and threshold to get binary predictions
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).cpu() # Keep as tensor for now

            all_preds_list.append(preds)
            # Labels are already float tensors from dataset
            all_labels_list.append(labels.cpu())

    # Concatenate results from all batches
    all_preds = torch.cat(all_preds_list, dim=0).numpy()
    all_labels = torch.cat(all_labels_list, dim=0).numpy()

    # 8. Calculate Metrics using sklearn
    print("Calculating evaluation metrics...")
    hamming = hamming_loss(all_labels, all_preds)
    subset_acc = accuracy_score(all_labels, all_preds)
    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_samples = f1_score(all_labels, all_preds, average='samples', zero_division=0)

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
                all_labels, # Use collected true labels
                all_preds,  # Use collected predictions
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
    # Return the trained PyTorch model and the indices it was trained on
    return model, variable_token_indices


# --- Main Execution ---
if __name__ == "__main__":
    # Set seed for reproducibility
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)

    # 1. Load Data
    vec_array, texts_list = load_data_for_probes(max_files=MAX_FILES_TO_LOAD)

    # 2. Create Target Matrix
    Y_sparse = create_token_presence_target_matrix(texts_list, tokenizer)

    # 3. Optional Subsampling for Probing
    X_probe_data = vec_array
    Y_probe_data = Y_sparse

    if USE_PROBE_SUBSAMPLING and len(vec_array) > PROBE_SUBSAMPLE_SIZE:
        print(f"\nSubsampling data from {len(vec_array)} to {PROBE_SUBSAMPLE_SIZE} points for probe training...")
        # Subsample indices first, then select data
        indices_full = np.arange(len(vec_array))
        indices_subsampled = random.sample(list(indices_full), PROBE_SUBSAMPLE_SIZE)
        X_probe_data = vec_array[indices_subsampled]
        Y_probe_data = Y_sparse[indices_subsampled] # Subsample sparse matrix
        print("Subsampling complete.")
    else:
        print("\nUsing full dataset for probe training (or dataset smaller than subsample size).")

    # 4. Run Probes (using PyTorch function)
    results = run_pytorch_token_presence_probes(X_probe_data, Y_probe_data)
    if results:
        trained_pytorch_model, probed_indices = results
        print(f"\nPyTorch token presence probe analysis complete. Model trained for {len(probed_indices)} tokens.")
        # You can access weights via: trained_pytorch_model.linear.weight.data
    else:
        print("\nToken presence probe analysis aborted or failed.")


# %%

tokenizer_decoder = tokenizer.get_decoder()
token_map = {}
for token_id in probed_indices:
    token_map[token_id] = tokenizer_decoder(torch.tensor([token_id]))
print(token_map)

if __name__ == "__main__":
    # Print top tokens by F1 score from existing results
    print("\nShowing top tokens by F1 score...")

    # We need to extract token-wise metrics from the classification report
    # that was already generated in the run_pytorch_token_presence_probes function

    # Since we already have the trained model and evaluation results,
    # we can just sort and display the top tokens by F1 score

    # Create a dictionary to store token metrics from the existing evaluation
    token_metrics = {}

    # Set up evaluation dataset to get predictions if needed
    X_eval = torch.tensor(X_probe_data, dtype=torch.float32)
    Y_eval = Y_probe_data.toarray() if isinstance(Y_probe_data, csr_matrix) else Y_probe_data
    Y_eval = torch.tensor(Y_eval, dtype=torch.float32)

    # Get predictions for all tokens
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = trained_pytorch_model.to(device)
    model.eval()

    batch_size = 128
    all_preds = []

    print("Generating predictions for selected tokens...")
    with torch.no_grad():
        for i in tqdm(range(0, len(X_eval), batch_size), desc="Evaluating"):
            batch_X = X_eval[i:i+batch_size].to(device)
            logits = model(batch_X)
            preds = torch.sigmoid(logits) > 0.5
            all_preds.append(preds.cpu())

    Y_pred = torch.cat(all_preds, dim=0).numpy()
    Y_true = Y_eval.numpy()

    # Calculate F1 score for each token in our probed_indices (tokens that met frequency threshold)
    print(f"Calculating F1 scores for {len(probed_indices)} selected tokens...")
    for i, token_idx in enumerate(tqdm(probed_indices)):
        # Calculate metrics for this token
        true_positives = np.sum((Y_pred[:, i] == 1) & (Y_true[:, token_idx] == 1))
        false_positives = np.sum((Y_pred[:, i] == 1) & (Y_true[:, token_idx] == 0))
        false_negatives = np.sum((Y_pred[:, i] == 0) & (Y_true[:, token_idx] == 1))

        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        support = np.sum(Y_true[:, token_idx] == 1)

        token_metrics[token_idx] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        }

    # Sort tokens by F1 score
    sorted_tokens = sorted(token_metrics.items(), key=lambda x: x[1]['f1'], reverse=True)

    # Print top 20 tokens by F1 score
    print("\nClassification Report (Top 20 probed tokens by F1 score):")
    print("                precision    recall  f1-score   support")
    print()

    for token_idx, metrics in sorted_tokens[:20]:
        token_name =  "'" + token_map[token_idx] + "'"
        print(f"{token_name:>15}      {metrics['precision']:.3f}     {metrics['recall']:.3f}     {metrics['f1']:.3f}      {int(metrics['support'])}")

    print("\n----------------------------------------------------")


# %%
import matplotlib.pyplot as plt
plt.loglog([v['f1'] for (k, v) in sorted_tokens])
plt.title("F1 Score vs Token")
plt.show()
# %%
# Plot token frequency vs metric score
def plot_token_frequency_vs_metric(token_metrics, metric='f1', title=None, color='steelblue'):
    plt.figure(figsize=(10, 6))
    token_frequencies = [token_metrics[token_idx]['support'] for token_idx in token_metrics.keys()]
    metric_scores = [token_metrics[token_idx][metric] for token_idx in token_metrics.keys()]

    plt.scatter(token_frequencies, metric_scores, alpha=0.6, s=20, color=color)
    plt.xscale('log')
    plt.yscale('linear')
    plt.xlabel('Token Frequency (log scale)')
    plt.ylabel(f'{metric.capitalize()} Score')

    if title is None:
        title = f'Token Frequency vs {metric.capitalize()} Score - PyTorch Binary'
    plt.title(title)

    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.show()

# Example usage
plot_token_frequency_vs_metric(token_metrics, metric='f1')
plot_token_frequency_vs_metric(token_metrics, metric='precision')
plot_token_frequency_vs_metric(token_metrics, metric='recall')

# definitions:
# precision = true positives / (true positives + false positives)
# recall = true positives / (true positives + false negatives)
# f1 = 2 * precision * recall / (precision + recall)
# support = number of true positives

# Optional: Save the plot
# plt.savefig('../figures/token_frequency_vs_f1.png', dpi=300, bbox_inches='tight')
# plt.close()

# %%
