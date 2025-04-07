# %%
import os
import json
import torch
from tqdm import tqdm
import numpy as np
import random
import time

# Assuming these utils are in the same directory or accessible
from utils_load_data import load_embeds, load_split_paragraphs
# Use the specific SONAR tokenizer loader

from sonar.models.sonar_text import load_sonar_tokenizer
# Fix the tokenizer loading to properly get vocab size
def load_tokenizer(repo="text_sonar_basic_encoder"):
    print(f"Loading SONAR tokenizer from repo: {repo}")
    # create_encoder() returns the actual tokenizer interface
    orig_tokenizer = load_sonar_tokenizer(repo)
    tokenizer = orig_tokenizer.create_encoder()
    print(dir(orig_tokenizer))
    print(dir(tokenizer))
    print(orig_tokenizer.vocab_info)

    # Get vocab size from vocab_info in the original tokenizer
    vocab_size = orig_tokenizer.vocab_info.size

    # Create a wrapper class to add vocab_size attribute
    class TokenizerWrapper:
        def __init__(self, tokenizer, vocab_size):
            self.tokenizer = tokenizer
            self.vocab_size = vocab_size

        def __getattr__(self, name):
            # Delegate all other attributes/methods to the original tokenizer
            return getattr(self.tokenizer, name)

        def __call__(self, *args, **kwargs):
            # Delegate the __call__ method
            return self.tokenizer(*args, **kwargs)

        def encode(self, text):
            return self.tokenizer(text)

        def encode_as_tokens(self, text):
            return self.tokenizer.encode_as_tokens(text)

    wrapped_tokenizer = TokenizerWrapper(tokenizer, vocab_size)
    print(f"Tokenizer loaded. Vocab size: {wrapped_tokenizer.vocab_size}")
    return wrapped_tokenizer

# Reload the tokenizer with the wrapper
tokenizer = load_tokenizer()
VOCAB_SIZE = tokenizer.vocab_size

print(tokenizer.encode("Hello, world!"))
print(tokenizer.encode_as_tokens("Hello, world!"))

# Function to create target matrix for token presence probing
def create_token_presence_target_matrix(texts_list, tokenizer):
    """
    Create a sparse binary matrix where each row corresponds to a text sample
    and each column corresponds to a token in the vocabulary.
    A 1 indicates the token is present in the text.
    """
    print("Creating token presence target matrix...")
    num_samples = len(texts_list)

    # Initialize lists for sparse matrix construction
    row_indices = []
    col_indices = []
    data = []

    for i, text in enumerate(tqdm(texts_list)):
        # Get unique tokens in this text
        try:
            # Encode the text to get token IDs
            token_ids = tokenizer.encode_as_tokens(text)

            # Get unique token IDs (to avoid duplicates)
            unique_token_ids = set()
            for token_id in token_ids:
                if token_id < VOCAB_SIZE:  # Only consider tokens within our vocab size
                    unique_token_ids.add(token_id)

            # Add to sparse matrix construction lists
            for token_id in unique_token_ids:
                row_indices.append(i)
                col_indices.append(token_id)
                data.append(1)  # Binary indicator

        except Exception as e:
            print(f"Error processing text {i}: {e}")
            continue

    # Create sparse matrix
    Y_sparse = csr_matrix((data, (row_indices, col_indices)),
                          shape=(num_samples, VOCAB_SIZE))

    print(f"Created target matrix with shape {Y_sparse.shape}")
    print(f"Density: {Y_sparse.nnz / (Y_sparse.shape[0] * Y_sparse.shape[1]):.6f}")

    return Y_sparse

# Function to run token presence probes
def run_token_presence_probes(X, Y):
    """
    Train and evaluate a multi-label classifier to predict token presence.

    Args:
        X: Feature matrix (embeddings)
        Y: Target sparse matrix (token presence)

    Returns:
        Trained model
    """
    print("\n--- Token Presence Probe ---")
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")

    # 1. Split Data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Train shapes: X={X_train.shape}, Y={Y_train.shape}")
    print(f"Test shapes: X={X_test.shape}, Y={Y_test.shape}")

    # 2. Train Model
    # Use LogisticRegression as the base classifier
    base_classifier = LogisticRegression(
        C=1.0,
        solver='saga',  # Efficient for large, sparse datasets
        penalty='l2',
        max_iter=100,
        random_state=RANDOM_STATE,
        class_weight=None,  # 'balanced' could be used but may be too slow
        n_jobs=1  # Will be parallelized by OneVsRest
    )


# %%

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, hamming_loss, accuracy_score # Subset accuracy
from sklearn.metrics import classification_report # Can be very verbose
from scipy.sparse import csr_matrix # For efficient sparse matrix

# --- Configuration ---
MAX_FILES_TO_LOAD = 100 # Adjust as needed
PROBE_SUBSAMPLE_SIZE = 20000 # Subsample data points for probe training (adjust based on memory/time)
USE_PROBE_SUBSAMPLING = True
RANDOM_STATE = 42
# Limit classification report output (set to 0 or None to disable)
MAX_CLASSES_IN_REPORT = 20

# --- Load Tokenizer ---

# --- Data Loading ---
def load_data_for_probes(max_files=MAX_FILES_TO_LOAD):
    """Load vectors and texts for probing."""
    all_vecs = []
    all_texts = []

    print(f"Loading data from {max_files} files...")
    num_skipped = 0
    for i in tqdm(range(max_files)):
        try:
            vecs = load_embeds(i) # SONAR text to vec autoencoder
            texts = load_split_paragraphs(i) # Original text paragraphs
        except FileNotFoundError:
            print(f"Warning: File {i} not found, skipping.")
            continue

        for v, t in zip(vecs, texts):
            if len(t) <= 0:
                num_skipped += 1
                continue
            all_vecs.append(v.cpu()) # Move tensor to CPU
            all_texts.append(t)

    print(f"Loaded {len(all_vecs)} vectors/texts. Skipped {num_skipped} empty texts.")

    if not all_vecs:
        raise ValueError("No vectors were loaded. Check data paths and content.")

    vec_array = torch.stack(all_vecs).numpy()
    return vec_array, all_texts

# --- Target Matrix Creation ---
def create_token_presence_target_matrix(texts, tokenizer):
    """Creates a sparse matrix Y where Y[i,j]=1 if token j is in text i."""
    print(f"Creating sparse target matrix for {len(texts)} texts and vocab size {VOCAB_SIZE}...")
    rows, cols = [], []
    vocab = tokenizer.get_vocab() # Get token -> id mapping
    token_to_id = {token: i for token, i in vocab.items()} # Ensure it's token -> id

    for i, text in enumerate(tqdm(texts)):
        try:
            # Encode text to get token IDs
            # Note: SONAR's create_encoder might return different output format,
            # adjust if necessary. Assuming it returns a list of token IDs.
            token_ids = tokenizer.encode(text)
            # Get unique token IDs present in the text
            unique_ids = set(token_ids)
            for token_id in unique_ids:
                if 0 <= token_id < VOCAB_SIZE: # Sanity check
                    rows.append(i)
                    cols.append(token_id)
                # else: # Optional: Warn about out-of-range IDs if they occur
                #     print(f"Warning: Token ID {token_id} out of range [0, {VOCAB_SIZE})")

        except Exception as e:
            print(f"Warning: Error tokenizing text sample {i}: {e}")
            print(f"Text: '{text[:100]}...'")
            continue # Skip this sample if tokenization fails

    # Create the sparse matrix
    # Data is implicitly all 1s for presence
    y_sparse = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(texts), VOCAB_SIZE))
    print(f"Sparse target matrix created. Sparsity: {y_sparse.nnz / (y_sparse.shape[0] * y_sparse.shape[1]):.6f}")
    return y_sparse

# --- Probing Function ---
def run_token_presence_probes(X, Y, test_size=0.2):
    """Train and evaluate token presence linear probes."""
    print(f"\nRunning token presence probes on {X.shape[0]} samples...")

    # 1. Train/Test Split
    print(f"Splitting data (test_size={test_size})...")
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=RANDOM_STATE
    )
    print(f"Train shape: X={X_train.shape}, Y={Y_train.shape}")
    print(f"Test shape:  X={X_test.shape}, Y={Y_test.shape}")

    # 2. Define and Train Model
    # Using OneVsRestClassifier with Logistic Regression
    # Crucially use class_weight='balanced' due to sparsity
    print("Initializing OneVsRestClassifier with LogisticRegression...")
    base_classifier = LogisticRegression(
        solver='liblinear', # Often good for binary tasks and sparse data
        C=1.0,             # Regularization strength (default)
        random_state=RANDOM_STATE,
        class_weight='balanced', # IMPORTANT for sparse labels
        max_iter=1000      # Increase if convergence warnings appear
    )
    model = OneVsRestClassifier(base_classifier, n_jobs=-1) # Use all cores

    print("Training the probe model (this might take a while)...")
    start_time = time.time()
    model.fit(X_train, Y_train)
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")

    # 3. Evaluate Model
    print("Evaluating the probe model on the test set...")
    Y_pred = model.predict(X_test)

    # Calculate Metrics
    # Note: Y_test and Y_pred are sparse matrices
    hamming = hamming_loss(Y_test, Y_pred)
    # Subset accuracy: % of samples where all labels are correct (often low)
    subset_acc = accuracy_score(Y_test, Y_pred)
    # F1 scores
    f1_micro = f1_score(Y_test, Y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(Y_test, Y_pred, average='macro', zero_division=0)
    f1_samples = f1_score(Y_test, Y_pred, average='samples', zero_division=0)

    print("\n--- Probe Evaluation Results ---")
    print(f"Hamming Loss:         {hamming:.4f}")
    print(f"Subset Accuracy:      {subset_acc:.4f}")
    print(f"F1 Score (Micro):     {f1_micro:.4f}")
    print(f"F1 Score (Macro):     {f1_macro:.4f}")
    print(f"F1 Score (Samples):   {f1_samples:.4f}")

    # Optional: Detailed report (can be huge)
    if MAX_CLASSES_IN_REPORT is not None and MAX_CLASSES_IN_REPORT > 0:
        print(f"\nClassification Report (Top {MAX_CLASSES_IN_REPORT} classes by support):")
        try:
            # Get target names (token strings) - might be slow if vocab large
            target_names = [None] * VOCAB_SIZE
            vocab = tokenizer.get_vocab()
            for token, idx in vocab.items():
                if 0 <= idx < VOCAB_SIZE:
                    target_names[idx] = token
            target_names = [name if name is not None else f"UNK_{i}" for i, name in enumerate(target_names)]

            # Limit the report to a subset of classes for readability
            # Find classes with highest support in the test set
            support = np.array(Y_test.sum(axis=0)).flatten()
            sorted_indices = np.argsort(support)[::-1]
            top_indices = sorted_indices[:MAX_CLASSES_IN_REPORT]
            top_target_names = [target_names[i] for i in top_indices]

            report = classification_report(
                Y_test, Y_pred,
                target_names=top_target_names,
                labels=top_indices, # Only report on these labels
                zero_division=0,
                digits=3
            )
            print(report)
        except Exception as e:
            print(f"Could not generate detailed classification report: {e}")

    print("------------------------------")
    return model


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load Data
    vec_array, texts_list = load_data_for_probes(max_files=MAX_FILES_TO_LOAD)

    # 2. Create Target Matrix
    Y_sparse = create_token_presence_target_matrix(texts_list, tokenizer)

    # 3. Optional Subsampling for Probing
    X_probe = vec_array
    Y_probe = Y_sparse

    if USE_PROBE_SUBSAMPLING and len(vec_array) > PROBE_SUBSAMPLE_SIZE:
        print(f"\nSubsampling data from {len(vec_array)} to {PROBE_SUBSAMPLE_SIZE} points for probe training...")
        random.seed(RANDOM_STATE)
        indices = random.sample(range(len(vec_array)), PROBE_SUBSAMPLE_SIZE)
        X_probe = vec_array[indices]
        Y_probe = Y_sparse[indices] # Subsample sparse matrix
        print("Subsampling complete.")
    else:
        print("\nUsing full dataset for probe training (or dataset smaller than subsample size).")

    # 4. Run Probes
    probe_model = run_token_presence_probes(X_probe, Y_probe)

    print("\nToken presence probe analysis complete.")

# %%
