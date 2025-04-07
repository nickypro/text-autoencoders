import torch
from torch import nn

class Linear(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.linear = nn.Linear(conf.d_res, conf.d_sonar)

    def forward(self, x):
        return self.linear(x)

class MLP(nn.Module):
    def __init__(self, conf):
        super().__init__()
        # More balanced hidden layer dimensions based on input/output sizes
        self.d_res = conf.d_res
        self.d_mlp = conf.d_mlp
        self.d_sonar = conf.d_sonar
        self.sequential = nn.Sequential(
            nn.Linear(self.d_res, self.d_mlp),
            nn.GELU(),
            nn.Dropout(0.1),  # Add dropout for regularization
            nn.Linear(self.d_mlp, self.d_mlp),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_mlp, self.d_sonar)
        )

    def forward(self, x):
        return self.sequential(x)

class SAE(nn.Module):
    def __init__(self, conf):
        super().__init__()
        # Store dimensions
        self.d_res = conf.d_res
        self.d_sae = conf.d_sae

        # Create encoder and decoder parts
        self.encoder = nn.Linear(self.d_res, self.d_sae, bias=True)
        self.decoder = nn.Linear(self.d_sae, self.d_res, bias=True)

        # Sparsity coefficient for L1 regularization
        self.sparsity_coeff = conf.sparsity_coeff if hasattr(conf, 'sparsity_coeff') else 0.1

        # Weight normalization epsilon (to avoid division by zero)
        self.weight_normalize_eps = conf.weight_normalize_eps if hasattr(conf, 'weight_normalize_eps') else 1e-8

        # Configuration for forward pass
        self.use_error_term = False  # Flag for controlling behavior in forward pass

    def encode(self, x):
        """Encode the input to get latent activations"""
        pre_acts = self.encoder(x)
        return torch.relu(pre_acts)  # Apply ReLU activation

    def decode(self, acts):
        """Decode the latent activations back to the input space"""
        # Use normalized decoder weights if configured
        return self.decoder(acts)

    @property
    def W_dec_normalized(self):
        """Returns decoder weights, normalized over the input dimension"""
        return self.decoder.weight / (self.decoder.weight.norm(dim=1, keepdim=True) + self.weight_normalize_eps)

    def forward(self, x):
        """
        Forward pass through the autoencoder

        Returns:
            tuple containing:
            - dict of loss terms
            - total loss
            - latent activations after ReLU
            - reconstructed input
        """
        # Encode the input to get latent activations
        pre_acts = self.encoder(x)
        acts = torch.relu(pre_acts)

        # Decode the latent activations back to the input space
        x_reconstructed = self.decoder(acts)

        # Calculate reconstruction loss
        reconstruction_loss = ((x_reconstructed - x) ** 2).mean(-1)

        # Calculate sparsity loss (L1 penalty on activations)
        sparsity_loss = acts.abs().sum(-1)

        # Combine losses
        loss_dict = {
            "reconstruction_loss": reconstruction_loss,
            "sparsity_loss": sparsity_loss
        }
        total_loss = reconstruction_loss + self.sparsity_coeff * sparsity_loss

        return loss_dict, total_loss, acts, x_reconstructed

    def run_with_cache(self, x):
        """Run the autoencoder and return a cache of intermediate values"""
        # Encode
        pre_acts = self.encoder(x)
        acts = torch.relu(pre_acts)

        # Decode
        x_reconstructed = self.decoder(acts)

        # Create cache dictionary
        cache = {
            "hook_sae_input": x,
            "hook_sae_pre_acts": pre_acts,
            "hook_sae_acts_post": acts,
            "hook_sae_output": x_reconstructed
        }

        return x_reconstructed, cache


