# %%
import os
import json
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.colors
import random # For subsampling

# Assuming these utils are in the same directory or accessible
from utils_load_data import load_embeds, load_split_paragraphs
from utils_sonar import load_tokenizer # Or use the alternative below

# --- Alternative Tokenizer Loading ---
# try:
#     from sonar.models.sonar_text import load_sonar_tokenizer
#     def load_tokenizer(repo="text_sonar_basic_encoder"):
#         return load_sonar_tokenizer(repo).create_encoder()
# except ImportError:
#     print("Warning: SONAR library not found for tokenizer loading.")
#     # Define a dummy tokenizer if needed, or ensure utils_sonar works
#     class DummyTokenizer:
#         def __call__(self, text): return len(text.split()) # Example
#     tokenizer = DummyTokenizer()
# # --- End Alternative ---


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap # Use umap-learn library
import phate # Import PHATE
import plotly.express as px # For interactive plots

# --- Configuration ---
unit = "Characters" # Or "Tokens"
MAX_FILES_TO_LOAD = 100 # Adjust as needed for memory/time
# Subsample for the main UMAP/t-SNE/PHATE calculation (can be larger)
CALCULATION_SUBSAMPLE_SIZE = 10000
USE_CALCULATION_SUBSAMPLING = True
# Subsample *again* for the interactive plot (should be smaller for performance)
INTERACTIVE_SUBSAMPLE_SIZE = 2000
RANDOM_STATE = 42 # For reproducibility

# --- Load Tokenizer ---
# tokenizer = load_tokenizer() # Uncomment if using the alternative loading

# --- Data Loading ---
def load_data_for_analysis(max_files=MAX_FILES_TO_LOAD):
    """Load vectors, texts, and metadata for analysis."""
    all_vecs = []
    all_texts = [] # Store original texts
    all_lengths = []
    # 0 for single sentence heuristic, 1 for double sentence heuristic
    all_labels = []
    label_names = {0: 'Single Sent (heuristic)', 1: 'Multi Sent (heuristic)'} # Map labels to names

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
            text_len = len(t)
            if text_len <= 0:
                num_skipped += 1
                continue

            # Determine length based on unit
            if unit == "Characters":
                current_len = text_len
            # elif unit == "Tokens": # Uncomment if using token length
            #     current_len = len(tokenizer(t)) # Requires tokenizer
            else:
                current_len = text_len # Default to characters if unit unknown

            # Simple heuristic for single vs multiple sentences
            label_code = 0 # Assume single
            if text_len > 30:
                if "." in t[10:-10] \
                    or ";" in t[10:-10] \
                    or "," in t[10:-10] \
                    or "!" in t[10:-10] \
                    or "?" in t[10:-10]:
                    label_code = 1 # Likely multiple sentences/clauses

            all_vecs.append(v.cpu()) # Move tensor to CPU
            all_texts.append(t) # Store the text
            all_lengths.append(current_len)
            all_labels.append(label_names[label_code]) # Store the descriptive label name

    print(f"Loaded {len(all_vecs)} vectors. Skipped {num_skipped} empty texts.")

    if not all_vecs:
        raise ValueError("No vectors were loaded. Check data paths and content.")

    # Stack vectors into a single tensor, then convert to numpy
    vec_tensor = torch.stack(all_vecs)
    vec_array = vec_tensor.numpy() # Shape: (num_samples, 1024)

    # Return numpy array for vectors, lists/numpy arrays for metadata
    return vec_array, all_texts, np.array(all_lengths), np.array(all_labels)


# --- Plotting Functions ---
def plot_dimensionality_reduction_static(results, labels, lengths, title):
    """Plots static results using Matplotlib."""
    plt.figure(figsize=(12, 10))
    unique_labels = np.unique(labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    label_map = {lbl: colors[i] for i, lbl in enumerate(unique_labels)}

    scatter_plots = []
    for label_val in unique_labels:
        idx = (labels == label_val)
        scatter = plt.scatter(
            results[idx, 0],
            results[idx, 1],
            alpha=0.3, s=5,
            color=label_map[label_val],
            label=label_val # Use the descriptive label name
        )
        scatter_plots.append(scatter)

    plt.title(f"{title} (Static)")
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(handles=scatter_plots)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.show()

def plot_dimensionality_reduction_interactive(results, texts, labels, lengths, title):
    """Plots interactive results using Plotly."""
    if len(results) == 0:
        print(f"Skipping interactive plot for {title}: No data points.")
        return

    print(f"Generating interactive plot for {title} with {len(results)} points...")

    # Truncate text for hover name to avoid overly large tooltips
    hover_texts = [t[:100] + '...' if len(t) > 100 else t for t in texts]

    fig = px.scatter(
        x=results[:, 0],
        y=results[:, 1],
        color=labels, # Color by the descriptive label
        title=title,
        labels={'x': 'Component 1', 'y': 'Component 2', 'color': 'Category'},
        opacity=0.6,
        hover_name=hover_texts, # Show truncated text on hover title
        hover_data={ # Add more info to the hover tooltip
            'Length': lengths,
            'Label': labels,
            # 'Full Text': texts # Commented out as requested
            # You could add X/Y coordinates too if desired:
            # 'X': results[:, 0],
            # 'Y': results[:, 1]
        }
    )

    fig.update_traces(marker=dict(size=5)) # Adjust marker size
    fig.update_layout(
        legend_title_text='Category',
        title_x=0.5 # Center title
    )
    fig.show()
    # Optionally save to HTML
    # safe_title = title.replace(" ", "_").replace("/", "_").lower()
    # fig.write_html(f"{safe_title}_interactive.html")
    # print(f"Saved interactive plot to {safe_title}_interactive.html")


# --- Analysis Functions ---
def run_pca_analysis(data, texts, labels, lengths):
    """Perform PCA and plot results."""
    print("\nRunning PCA...")
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    # Fit PCA on the full dataset (usually fast enough)
    pca_results = pca.fit_transform(data)
    print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained by 2 components: {pca.explained_variance_ratio_.sum():.4f}")

    # --- Plotting ---
    # Interactive plot (subsampled)
    if len(data) > INTERACTIVE_SUBSAMPLE_SIZE:
        print(f"Subsampling {INTERACTIVE_SUBSAMPLE_SIZE} points for PCA interactive plot...")
        random.seed(RANDOM_STATE)
        indices = random.sample(range(len(data)), INTERACTIVE_SUBSAMPLE_SIZE)
        plot_dimensionality_reduction_interactive(
            pca_results[indices],
            [texts[i] for i in indices],
            labels[indices],
            lengths[indices],
            'PCA of SONAR Embeddings (Interactive)'
        )
    else:
         plot_dimensionality_reduction_interactive(
            pca_results, texts, labels, lengths, 'PCA of SONAR Embeddings (Interactive)'
        )
    return pca_results # Return results if needed elsewhere


def run_umap_analysis(data, texts, labels, lengths):
    """Perform UMAP and plot results."""
    print("\nRunning UMAP...")
    umap_model = umap.UMAP(
        n_components=2, n_neighbors=15, min_dist=0.1,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=True
    )
    # Fit UMAP on the potentially pre-subsampled data passed to this function
    umap_results = umap_model.fit_transform(data)

    # --- Plotting ---
    # Interactive plot (further subsampled if needed)
    if len(data) > INTERACTIVE_SUBSAMPLE_SIZE:
        print(f"Subsampling {INTERACTIVE_SUBSAMPLE_SIZE} points for UMAP interactive plot...")
        random.seed(RANDOM_STATE)
        indices = random.sample(range(len(data)), INTERACTIVE_SUBSAMPLE_SIZE)
        plot_dimensionality_reduction_interactive(
            umap_results[indices],
            [texts[i] for i in indices],
            labels[indices],
            lengths[indices],
            'UMAP Projection of SONAR Embeddings (Interactive)'
        )
    else:
        plot_dimensionality_reduction_interactive(
            umap_results, texts, labels, lengths, 'UMAP Projection of SONAR Embeddings (Interactive)'
        )
    return umap_results


def run_tsne_analysis(data, texts, labels, lengths):
    """Perform t-SNE and plot results."""
    print("\nRunning t-SNE...")
    tsne_model = TSNE(
        n_components=2, perplexity=30, learning_rate='auto',
        n_iter=300, init='pca', random_state=RANDOM_STATE,
        n_jobs=-1, verbose=1
    )
    # Fit t-SNE on the potentially pre-subsampled data passed to this function
    tsne_results = tsne_model.fit_transform(data)

    # --- Plotting ---
    # Interactive plot (further subsampled if needed)
    if len(data) > INTERACTIVE_SUBSAMPLE_SIZE:
        print(f"Subsampling {INTERACTIVE_SUBSAMPLE_SIZE} points for t-SNE interactive plot...")
        random.seed(RANDOM_STATE)
        indices = random.sample(range(len(data)), INTERACTIVE_SUBSAMPLE_SIZE)
        plot_dimensionality_reduction_interactive(
            tsne_results[indices],
            [texts[i] for i in indices],
            labels[indices],
            lengths[indices],
            't-SNE Projection of SONAR Embeddings (Interactive)'
        )
    else:
        plot_dimensionality_reduction_interactive(
            tsne_results, texts, labels, lengths, 't-SNE Projection of SONAR Embeddings (Interactive)'
        )
    return tsne_results

def run_phate_analysis(data, texts, labels, lengths):
    """Perform PHATE and plot results."""
    print("\nRunning PHATE...")
    # PHATE hyperparameters (knn, decay) can be tuned. Defaults are often okay.
    phate_op = phate.PHATE(
        n_components=2,
        random_state=RANDOM_STATE,
        n_jobs=-1, # Use all cores
        verbose=True
    )
    # Fit PHATE on the potentially pre-subsampled data passed to this function
    phate_results = phate_op.fit_transform(data)

    # --- Plotting ---
    # Interactive plot (further subsampled if needed)
    if len(data) > INTERACTIVE_SUBSAMPLE_SIZE:
        print(f"Subsampling {INTERACTIVE_SUBSAMPLE_SIZE} points for PHATE interactive plot...")
        random.seed(RANDOM_STATE)
        indices = random.sample(range(len(data)), INTERACTIVE_SUBSAMPLE_SIZE)
        plot_dimensionality_reduction_interactive(
            phate_results[indices],
            [texts[i] for i in indices],
            labels[indices],
            lengths[indices],
            'PHATE Projection of SONAR Embeddings (Interactive)'
        )
    else:
        plot_dimensionality_reduction_interactive(
            phate_results, texts, labels, lengths, 'PHATE Projection of SONAR Embeddings (Interactive)'
        )
    return phate_results

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load Data (including texts now)
    vec_array, texts_list, lengths_array, labels_array = load_data_for_analysis(max_files=MAX_FILES_TO_LOAD)

    # 2. Data for Calculation (Optional Subsampling for UMAP/t-SNE/PHATE)
    data_for_calc = vec_array
    texts_for_calc = texts_list
    labels_for_calc = labels_array
    lengths_for_calc = lengths_array

    if USE_CALCULATION_SUBSAMPLING and len(vec_array) > CALCULATION_SUBSAMPLE_SIZE:
        print(f"\nSubsampling data from {len(vec_array)} to {CALCULATION_SUBSAMPLE_SIZE} points for UMAP/t-SNE/PHATE calculation...")
        random.seed(RANDOM_STATE) # Ensure reproducibility of subsampling
        indices = random.sample(range(len(vec_array)), CALCULATION_SUBSAMPLE_SIZE)
        data_for_calc = vec_array[indices]
        # Ensure metadata corresponds to the subsampled vectors
        texts_for_calc = [texts_list[i] for i in indices]
        labels_for_calc = labels_array[indices]
        lengths_for_calc = lengths_array[indices]
        print("Calculation subsampling complete.")
    else:
        print("\nUsing full dataset for UMAP/t-SNE/PHATE calculation (or dataset smaller than calc subsample size).")


    # 3. Run Analyses
    # PCA runs on full data, but interactive plot is subsampled inside the function
    run_pca_analysis(vec_array, texts_list, labels_array, lengths_array)

    # Run UMAP, t-SNE, and PHATE on potentially calculation-subsampled data
    # Interactive plots are further subsampled inside the functions if needed
    run_umap_analysis(data_for_calc, texts_for_calc, labels_for_calc, lengths_for_calc)
    run_tsne_analysis(data_for_calc, texts_for_calc, labels_for_calc, lengths_for_calc)
    run_phate_analysis(data_for_calc, texts_for_calc, labels_for_calc, lengths_for_calc) # Added PHATE call

    print("\nAnalysis complete.")

# %%

print("hello")

# %%
def plot_dimensionality_reduction_interactive_length(results, texts, labels, lengths, title):
    """Plots interactive results using Plotly, coloring by length."""
    if len(results) == 0:
        print(f"Skipping interactive plot for {title}: No data points.")
        return

    print(f"Generating interactive plot for {title} with {len(results)} points...")

    # Truncate text for hover name to avoid overly large tooltips
    hover_texts = [t[:100] + '...' if len(t) > 100 else t for t in texts]

    # Apply log transformation to lengths for color scale
    log_lengths = np.log1p(lengths)  # log1p to handle zero values safely

    fig = px.scatter(
        x=results[:, 0],
        y=results[:, 1],
        color=log_lengths,  # Use log-transformed lengths for coloring
        title=title,
        labels={'x': 'Component 1', 'y': 'Component 2', 'color': f'Log Length ({unit})'},
        opacity=0.6,
        hover_name=hover_texts, # Show truncated text on hover title
        hover_data={ # Add more info to the hover tooltip
            'Length': lengths,  # Show original lengths in hover
            'Label': labels,
            # 'Full Text': texts # Commented out as requested
            # You could add X/Y coordinates too if desired:
            # 'X': results[:, 0],
            # 'Y': results[:, 1]
        },
        color_continuous_scale='viridis'  # Use a continuous color scale
    )

    fig.update_traces(marker=dict(size=5)) # Adjust marker size
    fig.update_layout(
        coloraxis_colorbar=dict(title=f'Log Length ({unit})'),
        title_x=0.5 # Center title
    )
    fig.show()
    # Optionally save to HTML
    # safe_title = title.replace(" ", "_").replace("/", "_").lower()
    # fig.write_html(f"{safe_title}_interactive.html")
    # print(f"Saved interactive plot to {safe_title}_interactive.html")


def run_phate_analysis_length(data, texts, labels, lengths):
    """Perform PHATE and plot results with length-based coloring."""
    print("\nRunning PHATE (length-colored)...")
    # PHATE hyperparameters (knn, decay) can be tuned. Defaults are often okay.
    phate_op = phate.PHATE(
        n_components=2,
        random_state=RANDOM_STATE,
        n_jobs=-1, # Use all cores
        verbose=True
    )
    # Fit PHATE on the potentially pre-subsampled data passed to this function
    phate_results = phate_op.fit_transform(data)

    # --- Plotting ---
    # Interactive plot (further subsampled if needed)
    if len(data) > INTERACTIVE_SUBSAMPLE_SIZE:
        print(f"Subsampling {INTERACTIVE_SUBSAMPLE_SIZE} points for PHATE interactive plot...")
        random.seed(RANDOM_STATE)
        indices = random.sample(range(len(data)), INTERACTIVE_SUBSAMPLE_SIZE)
        plot_dimensionality_reduction_interactive_length(
            phate_results[indices],
            [texts[i] for i in indices],
            labels[indices],
            lengths[indices],
            f'PHATE Projection of SONAR Embeddings (Colored by {unit})'
        )
    else:
        plot_dimensionality_reduction_interactive_length(
            phate_results, texts, labels, lengths,
            f'PHATE Projection of SONAR Embeddings (Colored by {unit})'
        )
    return phate_results

if __name__ == "__main__":
    run_phate_analysis_length(data_for_calc, texts_for_calc, labels_for_calc, lengths_for_calc)
# %%
