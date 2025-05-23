# %%
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random
import math

# Assuming these utils are in the same directory or accessible
from utils_load_data import load_embeds, load_split_paragraphs

# --- Configuration ---
# Paths determined from the previous training script output
SAE_MODEL_FILENAME = "sae_final_H8192_L1_8.0e-06.pt"
SAE_CONFIG_FILENAME = "sae_config_H8192_L1_8.0e-06.json"
SAE_MODEL_DIR = "./sae_models"

# Data Loading for Analysis (use same settings as training or a representative subset)
MAX_FILES_TO_LOAD_ANALYSIS = 100 # Use a reasonable number for analysis
ANALYSIS_SUBSAMPLE_SIZE = 1_000_000 # Subsample activations for analysis if needed (memory)
USE_ANALYSIS_SUBSAMPLING = True

# Analysis Parameters
NUM_FEATURES_TO_ANALYZE = 10 # How many features to find max activating examples for
TOP_K_EXAMPLES = 10       # How many top examples to show per feature
ANALYSIS_BATCH_SIZE = 2048 # Batch size for getting activations (adjust for GPU memory)

# System
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_STATE = 42 # Use the same seed if subsampling for consistency

# --- Set Seed ---
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

# --- Reuse SAE Model Definition ---
# (Copied from the training script for self-containment)
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, l1_coeff=0): # l1_coeff not needed for inference
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # self.l1_coeff = l1_coeff # Not needed for analysis part

        self.encoder = nn.Linear(input_dim, hidden_dim, bias=False)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)
        self.relu = nn.ReLU()

    # Only need encode for analysis
    @torch.no_grad() # Ensure no gradients are computed
    def encode(self, x):
        encoded = self.encoder(x)
        activations = self.relu(encoded)
        return activations

    # Forward pass for analysis (optional, just encoding needed)
    # def forward(self, x):
    #     activations = self.encode(x)
    #     reconstructed = self.decoder(activations)
    #     return activations, reconstructed

# --- Helper Functions ---
def load_sae_model_and_config(model_dir, config_filename, model_filename, device):
    """Loads the SAE config and model state dict."""
    config_path = os.path.join(model_dir, config_filename)
    model_path = os.path.join(model_dir, model_filename)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"Loading model state dict from: {model_path}")
    model = SparseAutoencoder(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim']
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set model to evaluation mode
    print("Model loaded successfully.")
    return model, config

def load_analysis_data(max_files, subsample_size=None):
    """Loads SONAR vectors and corresponding texts."""
    all_vecs = []
    all_texts = []
    print(f"Loading data from {max_files} files for analysis...")
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
            if isinstance(v, torch.Tensor):
                all_vecs.append(v.cpu())
                all_texts.append(t)
            else:
                 print(f"Warning: Unexpected data type from load_embeds: {type(v)}")
                 num_skipped += 1

    print(f"Loaded {len(all_vecs)} vectors/texts. Skipped {num_skipped} items.")
    if not all_vecs:
        raise ValueError("No vectors were loaded for analysis.")

    vec_tensor = torch.stack(all_vecs)
    print(f"Total items loaded: {vec_tensor.shape[0]}")

    # Optional Subsampling
    if subsample_size is not None and vec_tensor.shape[0] > subsample_size:
        print(f"Subsampling data from {vec_tensor.shape[0]} to {subsample_size} points...")
        # Use torch.randperm for consistency if using torch seed
        indices = torch.randperm(vec_tensor.shape[0])[:subsample_size]
        vec_tensor = vec_tensor[indices]
        # Subsample texts accordingly
        all_texts = [all_texts[i] for i in indices.tolist()]
        print("Subsampling complete.")

    return vec_tensor, all_texts

@torch.no_grad()
def get_sae_activations(model, data_tensor, batch_size, device):
    """Gets SAE hidden activations for the input data in batches."""
    model.eval()
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    all_activations = []
    print(f"Getting SAE activations for {len(data_tensor)} samples...")
    for (batch_x,) in tqdm(dataloader, desc="Encoding Batches"):
        batch_x = batch_x.to(device)
        activations = model.encode(batch_x)
        all_activations.append(activations.cpu()) # Move to CPU to save GPU memory
    return torch.cat(all_activations, dim=0) # Shape: [num_samples, hidden_dim]

def analyze_feature_activation_distribution(activations):
    """Analyzes and plots activation distributions."""
    print("\n--- Analyzing Activation Distributions ---")
    hidden_dim = activations.shape[1]
    num_samples = activations.shape[0]

    # 1. Overall Sparsity (L0 Norm)
    l0_norms = (activations > 1e-8).float().sum(dim=1) # Use small threshold
    avg_l0 = l0_norms.mean().item()
    print(f"Average L0 Norm (features active per sample): {avg_l0:.2f}")
    plt.figure(figsize=(10, 4))
    plt.hist(l0_norms.numpy(), bins=50, log=True)
    plt.title(f'Distribution of L0 Norms (Avg: {avg_l0:.2f})')
    plt.xlabel('Number of Active Features')
    plt.ylabel('Frequency (Log Scale)')
    plt.grid(True, alpha=0.3)
    plt.show()

    # 2. Distribution of Non-Zero Activation Values
    non_zero_activations = activations[activations > 1e-8].numpy()
    print(f"Number of non-zero activation values: {len(non_zero_activations)}")
    if len(non_zero_activations) > 0:
        plt.figure(figsize=(10, 4))
        plt.hist(non_zero_activations, bins=100, log=True)
        plt.title('Distribution of Non-Zero Activation Values')
        plt.xlabel('Activation Value')
        plt.ylabel('Frequency (Log Scale)')
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        print("No non-zero activations found.")


    # 3. Per-Feature Activation Frequency (How often each feature fires)
    feature_frequencies = (activations > 1e-8).float().mean(dim=0) # Avg over samples
    num_dead_features = (feature_frequencies == 0).sum().item()
    print(f"Number of 'dead' features (never active): {num_dead_features} out of {hidden_dim} ({num_dead_features/hidden_dim:.2%})")

    plt.figure(figsize=(10, 4))
    plt.hist(feature_frequencies.numpy() * 100, bins=50, log=True) # Plot as percentage
    plt.title('Distribution of Feature Activation Frequencies')
    plt.xlabel('Percentage of Samples Feature is Active (%)')
    plt.ylabel('Number of Features (Log Scale)')
    plt.grid(True, alpha=0.3)
    plt.show()

    return feature_frequencies # Return frequencies for selecting features later

def find_maximally_activating_examples(activations, texts, feature_frequencies, num_features, top_k):
    """Finds and prints texts that maximally activate selected features."""
    print(f"\n--- Finding Top {top_k} Maximally Activating Examples for {num_features} Features ---")
    hidden_dim = activations.shape[1]

    # Select features to analyze:
    # Strategy: Choose some frequent, some mid, some rare (but not dead)
    active_feature_indices = torch.where(feature_frequencies > 0)[0]
    if len(active_feature_indices) == 0:
        print("No active features found to analyze.")
        return

    # Sort active features by frequency
    sorted_freq_indices = torch.argsort(feature_frequencies[active_feature_indices], descending=True)
    sorted_active_indices = active_feature_indices[sorted_freq_indices]

    # Select indices spread across the frequency spectrum
    indices_to_analyze = []
    if len(sorted_active_indices) <= num_features:
        indices_to_analyze = sorted_active_indices.tolist()
    else:
        # Take some most frequent, some least frequent (but active), some middle
        step = len(sorted_active_indices) // (num_features -1) if num_features > 1 else 1
        indices_to_analyze = [sorted_active_indices[min(i * step, len(sorted_active_indices)-1)].item() for i in range(num_features)]
        # Ensure uniqueness if step size leads to duplicates
        indices_to_analyze = sorted(list(set(indices_to_analyze)))


    print(f"Analyzing features (indices): {indices_to_analyze}")

    for feature_idx in indices_to_analyze:
        feature_activation_values = activations[:, feature_idx]
        freq = feature_frequencies[feature_idx].item() * 100

        # Find top K activation values and their indices
        top_values, top_indices = torch.topk(feature_activation_values, k=top_k)

        print("-" * 60)
        print(f"Feature Index: {feature_idx} (Active in {freq:.3f}% of samples)")
        print("-" * 60)
        for i in range(top_k):
            sample_idx = top_indices[i].item()
            activation_value = top_values[i].item()
            text = texts[sample_idx]
            # Truncate long texts for display
            display_text = text[:250] + '...' if len(text) > 250 else text
            outstr = f"  Rk {i} | Act: {100*activation_value:.4f} |"
            print(outstr, {"text": display_text})
        print("-" * 20) # Separator between examples

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # 1. Load Model and Config
    model, config = load_sae_model_and_config(
        SAE_MODEL_DIR, SAE_CONFIG_FILENAME, SAE_MODEL_FILENAME, DEVICE
    )

    # 2. Load Data for Analysis
    sonar_activations_tensor, original_texts = load_analysis_data(
        max_files=MAX_FILES_TO_LOAD_ANALYSIS,
        subsample_size=ANALYSIS_SUBSAMPLE_SIZE if USE_ANALYSIS_SUBSAMPLING else None
    )

    # 3. Get SAE Activations
    sae_feature_activations = get_sae_activations(
        model, sonar_activations_tensor, ANALYSIS_BATCH_SIZE, DEVICE
    )
    # Ensure it's on CPU for numpy/matplotlib analysis
    sae_feature_activations = sae_feature_activations.cpu()

    # 4. Analyze Activation Distributions
    feature_frequencies = analyze_feature_activation_distribution(sae_feature_activations)

    # 5. Find Maximally Activating Examples
    find_maximally_activating_examples(
        sae_feature_activations,
        original_texts,
        feature_frequencies,
        num_features=NUM_FEATURES_TO_ANALYZE,
        top_k=TOP_K_EXAMPLES
    )

    print("\nSAE analysis script finished.")

# %%
