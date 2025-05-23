# %%
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np
import random
import time
import math
import json
import warnings

# Optional: Weights & Biases for logging
try:
    import wandb
    USE_WANDB = True
except ImportError:
    USE_WANDB = False
    print("wandb not found. Skipping wandb logging.")


# Assuming these utils are in the same directory or accessible
from utils_load_data import load_embeds # Only need embeds now
torch.set_grad_enabled(True)

# --- Configuration ---
# Model Hyperparameters
INPUT_DIM = 1024 # Dimension of SONAR bottleneck vectors
EXPANSION_FACTOR = 8 # Ratio of hidden dim to input dim (e.g., 4, 8, 16, 32)
HIDDEN_DIM = INPUT_DIM * EXPANSION_FACTOR
# L1_COEFF: CRUCIAL hyperparameter. Needs tuning. Start low (e.g., 1e-4, 1e-3) and increase
# Aim for a target L0 norm (avg number of active features) e.g., 20-100.
L1_COEFF = 1e-5 # <--- *** TUNE THIS ***

# Training Hyperparameters
LEARNING_RATE = 3e-4
LR_WARMUP_STEPS = 500
LR_COSINE_DECAY_STEPS = 500000 # Adjust based on expected total steps
WEIGHT_DECAY = 0.01
BATCH_SIZE = 1024 # Use largest possible that fits memory
NUM_EPOCHS = 50 # Adjust as needed, SAEs can train relatively quickly
GRAD_CLIP_NORM = 1.0 # Max norm for gradient clipping

# Data Loading
MAX_FILES_TO_LOAD = 100 # Load more data if possible for better SAE training
SAE_SUBSAMPLE_SIZE = 1_000_000 # Optional: Subsample loaded activations (set to None to disable)
USE_SAE_SUBSAMPLING = False

# System & Reproducibility
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_STATE = 42

# Logging & Saving
MODEL_SAVE_DIR = "./sae_models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
LOG_INTERVAL = 100 # Log metrics every N steps

# --- Set Seed ---
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)
    # Might make things slower, but increases reproducibility
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

# --- SAE Model Definition ---
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, l1_coeff):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.l1_coeff = l1_coeff

        # Encoder: No bias term
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=False)
        # Decoder: Has bias term
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)

        # Ensure ReLU is NOT inplace
        self.relu = nn.ReLU(inplace=False) # Explicitly set inplace=False

        # Initialize weights (Kaiming for encoder, careful init for decoder)
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.decoder.weight, a=math.sqrt(5))
        nn.init.zeros_(self.decoder.bias)

        # --- Explicitly set dtype (usually not needed, but for debugging) ---
        # self.encoder.to(torch.float32)
        # self.decoder.to(torch.float32)
        # Note: Calling .to() inside __init__ might be less standard than
        # ensuring the model is moved to the correct device/dtype later.
        # The .to(DEVICE) call in the main script should handle this.
        # Let's rely on the main script's .to(DEVICE) for now.

        print(f"Initialized SAE: Input={input_dim}, Hidden={hidden_dim}, Expansion={hidden_dim/input_dim:.1f}x")
        print(f"  Encoder Weight dtype: {self.encoder.weight.dtype}")
        print(f"  Decoder Weight dtype: {self.decoder.weight.dtype}")
        print(f"  Decoder Bias dtype: {self.decoder.bias.dtype}")


    def forward(self, x):
        # Ensure input is float32 (should be from dataloader, but double-check)
        x = x.float()

        # Encode
        encoded = self.encoder(x)
        # Apply ReLU activation
        activations = self.relu(encoded)

        # Calculate L1 sparsity loss on activations
        l1_loss = self.l1_coeff * torch.linalg.vector_norm(activations, ord=1, dim=-1).mean()

        # Decode
        reconstructed = self.decoder(activations)

        # Calculate reconstruction loss (MSE)
        reconstruction_loss = nn.functional.mse_loss(reconstructed, x)

        # Total loss
        total_loss = reconstruction_loss + l1_loss

        # --- Add internal check ---
        if not total_loss.requires_grad:
             print("!!! WARNING: total_loss does not require grad INSIDE forward pass !!!")
             print(f"    x.requires_grad: {x.requires_grad}")
             print(f"    encoder.weight.requires_grad: {self.encoder.weight.requires_grad}")
             print(f"    encoded.requires_grad: {encoded.requires_grad}")
             print(f"    activations.requires_grad: {activations.requires_grad}")
             print(f"    reconstructed.requires_grad: {reconstructed.requires_grad}")
             print(f"    recon_loss.requires_grad: {reconstruction_loss.requires_grad}")
             print(f"    l1_loss.requires_grad: {l1_loss.requires_grad}")
        # --------------------------

        return total_loss, reconstruction_loss, l1_loss, activations, reconstructed

    @torch.no_grad() # No gradients needed for normalization
    def normalize_decoder_weights(self):
        """
        Constrains the decoder weight columns (features) to have unit L2 norm.
        """
        # Detach to avoid affecting gradients during training step
        # Clone to avoid modifying weights needed for gradient calculation before step
        w_dec = self.decoder.weight.data.clone()
        # Calculate norm over the input dimension (dim=1 for [out, in] -> [input_dim, hidden_dim] view)
        # Transpose to make columns features -> [hidden_dim, input_dim]
        w_dec_transposed = w_dec.t() # Shape: [hidden_dim, input_dim]
        norms = torch.linalg.vector_norm(w_dec_transposed, ord=2, dim=1, keepdim=True) # Shape: [hidden_dim, 1]
        # Avoid division by zero for potential zero-norm columns
        norms = torch.clamp(norms, min=1e-8)
        # Normalize
        normalized_w_dec = w_dec_transposed / norms
        # Transpose back and assign
        self.decoder.weight.data = normalized_w_dec.t()

    @torch.no_grad()
    def calculate_l0_norm(self, x):
        """Calculates the average L0 norm (number of non-zero activations)."""
        encoded = self.encoder(x)
        activations = self.relu(encoded)
        l0_norm = (activations > 0).float().sum(dim=-1).mean().item()
        return l0_norm

    def encode(self, x):
        encoded = self.encoder(x)
        activations = self.relu(encoded)
        return activations

    def decode(self, activations):
        return self.decoder(activations)

# --- Data Loading ---
def load_sae_data(max_files=MAX_FILES_TO_LOAD, subsample_size=None):
    """Load vectors for SAE training."""
    all_vecs = []
    print(f"Loading data from {max_files} files for SAE...")
    num_skipped = 0 # Although less likely for just vectors
    for i in tqdm(range(max_files), desc="Loading files"):
        try:
            vecs = load_embeds(i) # SONAR text to vec autoencoder
        except FileNotFoundError:
            print(f"Warning: File {i} not found, skipping.")
            continue
        # Assuming vecs is already a list/iterable of tensors
        for v in vecs:
             # Ensure it's a tensor and move to CPU if needed
            if isinstance(v, torch.Tensor):
                all_vecs.append(v.cpu())
            else:
                # Handle cases where load_embeds might return something else
                print(f"Warning: Unexpected data type from load_embeds: {type(v)}")
                num_skipped += 1


    print(f"Loaded {len(all_vecs)} vectors.")
    if not all_vecs:
        raise ValueError("No vectors were loaded.")

    vec_tensor = torch.stack(all_vecs) # Shape: [num_samples, input_dim]
    print(f"Total vectors loaded: {vec_tensor.shape[0]}")

    # Optional Subsampling
    if subsample_size is not None and vec_tensor.shape[0] > subsample_size:
        print(f"Subsampling data from {vec_tensor.shape[0]} to {subsample_size} points...")
        indices = torch.randperm(vec_tensor.shape[0])[:subsample_size]
        vec_tensor = vec_tensor[indices]
        print("Subsampling complete.")

    return vec_tensor

# --- Training Function ---
def train_sae(model, dataloader, optimizer, scheduler, num_epochs, device):
    model.train()
    total_steps = 0
    start_train_time = time.time()

    print("\n--- Starting SAE Training ---")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_l1_loss = 0.0
        epoch_l0_norm = 0.0
        epoch_start_time = time.time()

        # Wrap dataloader with tqdm for batch progress
        batch_iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for batch_idx, (batch_x,) in enumerate(batch_iterator):
            batch_x = batch_x.to(device)
            optimizer.zero_grad()

            # --- DEBUG: Check requires_grad status ---
            # Check a sample parameter BEFORE the forward pass
            enc_weight_req_grad_before = model.encoder.weight.requires_grad
            dec_weight_req_grad_before = model.decoder.weight.requires_grad
            dec_bias_req_grad_before = model.decoder.bias.requires_grad
            # -----------------------------------------

            # Forward pass
            total_loss, recon_loss, l1_loss, activations, reconstructed = model(batch_x)

            if not total_loss.requires_grad:
                # --- DEBUG: Check requires_grad status AFTER forward pass ---
                print(f"\n--- Debug Info (Step {total_steps}) ---")
                print(f"batch_x.requires_grad: {batch_x.requires_grad}")
                print(f"batch_x.dtype: {batch_x.dtype}") # Check data type too
                print(f"Encoder Weight requires_grad (Before): {enc_weight_req_grad_before}")
                print(f"Decoder Weight requires_grad (Before): {dec_weight_req_grad_before}")
                print(f"Decoder Bias requires_grad (Before): {dec_bias_req_grad_before}")
                print(f"Encoder Weight requires_grad (After): {model.encoder.weight.requires_grad}") # Should still be True
                print(f"activations.requires_grad: {activations.requires_grad}")
                print(f"reconstructed.requires_grad: {reconstructed.requires_grad}")
                print(f"recon_loss.requires_grad: {recon_loss.requires_grad}")
                print(f"l1_loss.requires_grad: {l1_loss.requires_grad}")
                print(f"total_loss.requires_grad: {total_loss.requires_grad}")
                print(f"total_loss.grad_fn: {total_loss.grad_fn}")
                print(f"--- End Debug Info ---")
                # ----------------------------------------------------------

            # Backward pass
            total_loss.backward() # Error occurs here

            # Gradient Clipping
            if GRAD_CLIP_NORM is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

            # Optimizer step
            optimizer.step()

            # Normalize decoder weights AFTER optimizer step
            model.normalize_decoder_weights()

            # Scheduler step (update LR)
            scheduler.step()

            # --- Logging ---
            current_lr = scheduler.get_last_lr()[0]
            # Calculate L0 norm (feature density) using a separate no_grad call
            with torch.no_grad():
                l0_norm = (activations > 0).float().sum(dim=-1).mean().item()

            epoch_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_l1_loss += l1_loss.item()
            epoch_l0_norm += l0_norm

            if USE_WANDB:
                wandb.log({
                    "train/step_loss": total_loss.item(),
                    "train/step_recon_loss": recon_loss.item(),
                    "train/step_l1_loss": l1_loss.item(),
                    "train/step_l0_norm": l0_norm,
                    "train/learning_rate": current_lr,
                    "epoch": epoch + (batch_idx / len(dataloader)), # Continuous epoch
                    "step": total_steps
                })

            # Update tqdm description
            if total_steps % LOG_INTERVAL == 0 or batch_idx == len(dataloader) - 1:
                 batch_iterator.set_postfix({
                     "Loss": f"{total_loss.item():.4f}",
                     "Recon": f"{recon_loss.item():.4f}",
                     "L1": f"{l1_loss.item():.4f}",
                     "L0": f"{l0_norm:.2f}",
                     "LR": f"{current_lr:.1e}"
                 })

            total_steps += 1

        # --- End of Epoch ---
        avg_epoch_loss = epoch_loss / len(dataloader)
        avg_epoch_recon_loss = epoch_recon_loss / len(dataloader)
        avg_epoch_l1_loss = epoch_l1_loss / len(dataloader)
        avg_epoch_l0_norm = epoch_l0_norm / len(dataloader)
        epoch_duration = time.time() - epoch_start_time

        print(f"Epoch {epoch+1}/{num_epochs} Summary | Time: {epoch_duration:.2f}s")
        print(f"  Avg Loss: {avg_epoch_loss:.6f} | Avg Recon Loss: {avg_epoch_recon_loss:.6f}")
        print(f"  Avg L1 Loss: {avg_epoch_l1_loss:.6f} | Avg L0 Norm: {avg_epoch_l0_norm:.2f}")

        if USE_WANDB:
            wandb.log({
                "train/epoch_loss": avg_epoch_loss,
                "train/epoch_recon_loss": avg_epoch_recon_loss,
                "train/epoch_l1_loss": avg_epoch_l1_loss,
                "train/epoch_l0_norm": avg_epoch_l0_norm,
                "epoch": epoch + 1 # Log epoch end
            })

        # Optional: Save checkpoint periodically
        # save_path = os.path.join(MODEL_SAVE_DIR, f"sae_epoch_{epoch+1}.pt")
        # torch.save(model.state_dict(), save_path)
        # print(f"Saved checkpoint to {save_path}")


    total_train_time = time.time() - start_train_time
    print(f"\n--- Training Finished ---")
    print(f"Total Training Time: {total_train_time:.2f} seconds")


# --- Cosine Annealing LR Scheduler ---
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """ Creates a cosine schedule with warmup. """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


# --- Main Execution ---
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # 1. Load Data
    activation_data = load_sae_data(
        max_files=MAX_FILES_TO_LOAD,
        subsample_size=SAE_SUBSAMPLE_SIZE if USE_SAE_SUBSAMPLING else None
    )
    dataset = TensorDataset(activation_data)
    print("Loaded dataset with len(dataset):", len(dataset))
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4, # Adjust based on your system
        pin_memory=True if DEVICE == torch.device("cuda") else False
    )
    print(f"Created DataLoader with {len(dataloader)} batches.")

    # 2. Initialize Model
    model = SparseAutoencoder(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        l1_coeff=L1_COEFF
    ).to(DEVICE)

    # 3. Initialize Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # Calculate total training steps for scheduler
    total_training_steps = len(dataloader) * NUM_EPOCHS
    # Adjust decay steps if needed, ensure it's >= total steps for full decay
    num_decay_steps = max(total_training_steps, LR_COSINE_DECAY_STEPS)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=LR_WARMUP_STEPS,
        num_training_steps=num_decay_steps # Use calculated total steps or config
    )

    # 4. Initialize Logging (Optional: Weights & Biases)
    if USE_WANDB:
        # Make run name more informative
        run_name = f"SAE_I{INPUT_DIM}_H{HIDDEN_DIM}_L1_{L1_COEFF:.1e}_B{BATCH_SIZE}_E{NUM_EPOCHS}"
        wandb.init(
            project="sae-sonar-activations", # Change project name if desired
            name=run_name,
            config={
                "input_dim": INPUT_DIM,
                "hidden_dim": HIDDEN_DIM,
                "expansion_factor": EXPANSION_FACTOR,
                "l1_coeff": L1_COEFF,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "batch_size": BATCH_SIZE,
                "num_epochs": NUM_EPOCHS,
                "lr_warmup_steps": LR_WARMUP_STEPS,
                "lr_cosine_decay_steps": num_decay_steps,
                "grad_clip_norm": GRAD_CLIP_NORM,
                "max_files_loaded": MAX_FILES_TO_LOAD,
                "subsample_size": SAE_SUBSAMPLE_SIZE if USE_SAE_SUBSAMPLING else "None",
                "random_state": RANDOM_STATE
            }
        )
        # Watch the model gradients (optional, can be resource intensive)
        # wandb.watch(model, log_freq=1000)

    # 5. Train the Model
    try:
        train_sae(model, dataloader, optimizer, scheduler, NUM_EPOCHS, DEVICE)

        # 6. Save Final Model and Config
        final_save_path = os.path.join(MODEL_SAVE_DIR, f"sae_final_H{HIDDEN_DIM}_L1_{L1_COEFF:.1e}.pt")
        config_save_path = os.path.join(MODEL_SAVE_DIR, f"sae_config_H{HIDDEN_DIM}_L1_{L1_COEFF:.1e}.json")

        print(f"\nSaving final model state dict to: {final_save_path}")
        torch.save(model.state_dict(), final_save_path)

        config = {
            "input_dim": INPUT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "l1_coeff": L1_COEFF,
            # Add any other relevant hyperparameters used during training
            "expansion_factor": EXPANSION_FACTOR,
            "trained_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "final_avg_l0_norm": model.calculate_l0_norm(activation_data[:1000].to(DEVICE)) # Estimate final L0
        }
        print(f"Saving model config to: {config_save_path}")
        with open(config_save_path, 'w') as f:
            json.dump(config, f, indent=4)

    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        # Potentially save a checkpoint even if training failed
        error_save_path = os.path.join(MODEL_SAVE_DIR, f"sae_ERROR_H{HIDDEN_DIM}_L1_{L1_COEFF:.1e}.pt")
        print(f"Saving model state dict due to error: {error_save_path}")
        torch.save(model.state_dict(), error_save_path)
        raise # Re-raise the exception

    finally:
        if USE_WANDB:
            wandb.finish()

    print("\nSAE training script finished.")

# %%
# TEST RECONSTRUCTION