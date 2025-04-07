# %%
import os
import json
import torch
from tqdm import tqdm
from utils_load_data import load_embeds, load_split_paragraphs
from utils_sonar import load_tokenizer

unit = "Characters"

tokenizer = load_tokenizer()

# Get the data
lens = []
file_name = f'num_{unit}_vs_norm.json'
if os.path.exists(f'../data/tae_data/{file_name}'):
    with open(f'../data/tae_data/{file_name}', 'r') as f:
        lens = json.load(f)
else:
    num_skipped = 0
    for i in tqdm(range(100)):
        vecs  = load_embeds(i) # SONAR text to vec autoencoder
        texts = load_split_paragraphs(i) # Original text paragraphs
        for v, t in zip(vecs, texts):
            if len(t) <= 0:
                num_skipped += 1
                continue
            if unit == "Characters":
                lens.append((len(t), torch.norm(v, dim=-1).item()))
            elif unit == "Tokens":
                lens.append((len(tokenizer(t)), torch.norm(v, dim=-1).item()))
    os.makedirs('../data/tae_data', exist_ok=True)
    with open(f'../data/tae_data/{file_name}', 'w') as f:
        json.dump(lens, f)
    print(f"Skipped {num_skipped} paragraphs")

# %%
from matplotlib import pyplot as plt
plt.scatter(*zip(*lens), alpha=0.1, s=1)
# semilogx
plt.semilogx()
plt.ylim(0.1, 0.4)
if unit == "Tokens":
    plt.xlim(1, 1000)

# %%
# Using hist2d with logarithmic bins for both x and y axes
import numpy as np
import matplotlib.colors

# Extract x and y values
x_vals, y_vals = zip(*lens)

# Create logarithmic bins for x-axis (token length)
xmin, xmax = min(x_vals), max(x_vals)
x_bins = np.logspace(np.log10(max(1, xmin)), np.log10(xmax), 100)  # 100 bins on log scale

# Create logarithmic bins for y-axis (vector norm)
ymin, ymax = min(y_vals), max(y_vals)
y_bins = np.logspace(np.log10(max(1e-10, ymin)), np.log10(ymax), 100)  # 100 bins on log scale

# Plot with logarithmic bins for both axes
plt.figure(figsize=(10, 8))
plt.hist2d(x_vals, y_vals, bins=[x_bins, y_bins], cmap='viridis', norm=matplotlib.colors.LogNorm())
plt.colorbar(label='Count')

# Set both axes to log scale
plt.xscale('log')
plt.yscale('log')

# Add labels
plt.xlabel(f'Num {unit} (log scale)')
plt.ylabel('Vector Norm (log scale)')
plt.title(f'Distribution of Number of {unit} vs Vector Norm (Log-Log Scale)')

# %%

# Calculate line of best fit using numpy's polyfit
import numpy as np
from scipy import stats

# Extract x and y values
x_vals, y_vals = zip(*lens)
x_vals_log = np.log10(x_vals)  # Convert x to log scale

# Calculate line of best fit (y vs log(x))
slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals_log, y_vals)

# Create points for the line of best fit
x_fit = np.logspace(np.log10(max(1, min(x_vals))), np.log10(max(x_vals)), 100)
y_fit = slope * np.log10(x_fit) + intercept  # y = m*log(x) + b

# Plot the line of best fit on a new figure
plt.figure(figsize=(10, 8))
plt.scatter(x_vals, y_vals, alpha=0.1, s=1, label='Data points')
plt.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Fit: y = {slope:.4f}*log(x) + {intercept:.4f}')

# Set x-axis to log scale (y remains linear)
plt.xscale('linear')
plt.ylim(0.1, 0.4)
if unit == "Tokens":
    plt.xlim(1, 300)
if unit == "Characters":
    plt.xlim(1, 1200)

# Add labels and legend
plt.xlabel(f'Num {unit} (log scale)')
plt.ylabel('Vector Norm')
plt.title(f'Number of {unit} vs Vector Norm with Line of Best Fit (y vs log x)')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)

# Print statistics
print(f"Slope: {slope:.4f}")
print(f"Intercept: {intercept:.4f}")
print(f"R-squared: {r_value**2:.4f}")
print(f"p-value: {p_value:.4e}")


# %%

# Using hist2d with logarithmic bins for both x and y axes
import numpy as np
import matplotlib.colors

# Extract x and y values
x_vals, y_vals = zip(*lens)

# Create logarithmic bins for x-axis (token length)
xmin, xmax = min(x_vals), max(x_vals)
x_bins = np.logspace(np.log10(max(1, xmin)), np.log10(xmax), 100)  # 100 bins on log scale

# Create logarithmic bins for y-axis (vector norm)
ymin, ymax = min(y_vals), max(y_vals)
y_bins = np.logspace(np.log10(max(1e-10, ymin)), np.log10(ymax), 100)  # 100 bins on log scale

# Plot with logarithmic bins for both axes
plt.figure(figsize=(10, 8))
plt.hist2d(x_vals, y_vals, bins=[x_bins, y_bins], cmap='viridis', norm=matplotlib.colors.LogNorm())
plt.colorbar(label='Count')

# Set both axes to log scale
plt.xscale('log')
plt.yscale('log')

# Add labels
plt.xlabel(f'Num {unit} (log scale)')
plt.ylabel('Vector Norm (log scale)')
plt.title(f'Distribution of Number of {unit} vs Vector Norm (Log-Log Scale)')

# Add line of best fit to the log-log plot
# Calculate line of best fit using numpy's polyfit for log-log relationship
x_vals_log = np.log10(x_vals)
y_vals_log = np.log10(y_vals)

# Calculate line of best fit (log(y) vs log(x))
slope_log, intercept_log, r_value_log, p_value_log, std_err_log = stats.linregress(x_vals_log, y_vals_log)

# Create points for the line of best fit
x_fit_log = np.logspace(np.log10(max(1, min(x_vals))), np.log10(max(x_vals)), 100)
y_fit_log = 10**(slope_log * np.log10(x_fit_log) + intercept_log)  # y = 10^(m*log(x) + b)

# Plot the line of best fit on the current figure
plt.plot(x_fit_log, y_fit_log, 'r-', linewidth=2,
         label=f'Fit: log(y) = {slope_log:.4f}*log(x) + {intercept_log:.4f}, R²: {r_value_log**2:.4f}')
plt.legend()


plt.ylim(None, 0.45)
if unit == "Tokens":
    plt.xlim(1, 512)

# Print log-log statistics
print("\nLog-Log Fit Statistics:")
print(f"Slope: {slope_log:.4f}")
print(f"Intercept: {intercept_log:.4f}")
print(f"R-squared: {r_value_log**2:.4f}")
print(f"p-value: {p_value_log:.4e}")

# %%

# %%
