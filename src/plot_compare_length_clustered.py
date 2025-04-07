# %%
import os
import json
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.colors
from utils_load_data import load_embeds, load_split_paragraphs
from utils_sonar import load_tokenizer

unit = "Characters"

tokenizer = load_tokenizer()

def load_data():
    """Load and process the data"""
    lens_single = []
    lens_double = []

    num_skipped = 0
    for i in tqdm(range(100)):
        vecs  = load_embeds(i) # SONAR text to vec autoencoder
        texts = load_split_paragraphs(i) # Original text paragraphs
        for v, t in zip(vecs, texts):
            if len(t) <= 0:
                num_skipped += 1
                continue
            if unit == "Characters":
                if len(t) > 30:
                    if "." in t[10:-10] \
                        or ";" in t[10:-10] \
                        or "," in t[10:-10] \
                        or "!" in t[10:-10] \
                        or "?" in t[10:-10]:
                        lens_double.append((len(t), torch.norm(v, dim=-1).item()))
                    else:
                        lens_single.append((len(t), torch.norm(v, dim=-1).item()))
                else:
                    lens_single.append((len(t), torch.norm(v, dim=-1).item()))
            elif unit == "Tokens":
                lens_single.append((len(tokenizer(t)), torch.norm(v, dim=-1).item()))

    return lens_single, lens_double

lens_single, lens_double = load_data()

# %%

def plot_scatter(lens_data, color='1'):
    """Create a simple scatter plot"""
    # Extract x and y values
    x_vals, y_vals = zip(*lens_data)

    # Plot scatter
    plt.scatter(x_vals, y_vals, alpha=0.05, s=1, label='Data points', color=color)

    # Set x-axis to log scale (y remains linear)
    plt.xscale('linear')
    plt.ylim(0.1, 0.4)
    if unit == "Tokens":
        plt.xlim(1, 300)
    if unit == "Characters":
        plt.xlim(1, 1200)

    # Add labels and legend
    plt.xlabel(f'Num {unit}')
    plt.ylabel('Vector Norm')
    plt.title(f'Number of {unit} vs Vector Norm')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)

# Main execution
if __name__ == "__main__":

    # Plot single-sentence data
    print("Plotting single-sentence data:")
    plt.figure(figsize=(10, 8))
    plot_scatter(lens_single, color='blue')
    plot_scatter(lens_double, color='orange')

    print(len(lens_single))
    print(len(lens_double))

    # plot_heatmap(lens_single)

    # # Plot double-sentence data if available
    # if lens_double:
    #     print("Plotting double-sentence data:")
    #     plot_scatter(lens_double)
    #     plot_heatmap(lens_double)

# %%
