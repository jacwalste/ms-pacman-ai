"""
Noisy Linear Layer

Replaces standard linear layers with ones that have learnable noise.
This provides exploration without epsilon-greedy randomness.

Key idea: The network learns WHEN to explore (high noise) vs exploit (low noise).

Standard linear: y = Wx + b
Noisy linear:    y = (W + sigma_w * noise_w)x + (b + sigma_b * noise_b)

The sigma parameters are learned - the network controls its own exploration!
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """
    Linear layer with factorized Gaussian noise.

    Uses factorized noise for efficiency:
    - Instead of noise matrix (in_features x out_features)
    - We use two vectors: noise_in (in_features) and noise_out (out_features)
    - Combine them: noise_matrix = f(noise_out) @ f(noise_in).T
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        # Learnable parameters (mean)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))

        # Learnable parameters (noise scale)
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Factorized noise (not learnable, resampled each forward pass)
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Initialize parameters."""
        # Initialize mu with uniform distribution
        bound = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)

        # Initialize sigma to small constant
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def _scale_noise(self, size: int) -> torch.Tensor:
        """Generate factorized noise: f(x) = sign(x) * sqrt(|x|)"""
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        """Resample noise for next forward pass."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        # Outer product for factorized noise
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with noisy weights."""
        if self.training:
            # Use noisy weights during training
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # Use mean weights during evaluation (no exploration)
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)


# Test
if __name__ == "__main__":
    layer = NoisyLinear(128, 64)

    # Test forward pass
    x = torch.randn(32, 128)

    # Training mode (noisy)
    layer.train()
    out1 = layer(x)
    layer.reset_noise()
    out2 = layer(x)
    print(f"Training - outputs differ: {not torch.allclose(out1, out2)}")

    # Eval mode (deterministic)
    layer.eval()
    out3 = layer(x)
    out4 = layer(x)
    print(f"Eval - outputs same: {torch.allclose(out3, out4)}")

    print(f"Output shape: {out1.shape}")
