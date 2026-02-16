"""
Sparse Autoencoder (SAE) implementation for LLM interpretability.

Implements modern best practices from Anthropic's research:
- TopK activation for stable sparsity
- Geometric median initialization
- Tied encoder/decoder initialization
- Unit norm decoder constraint (prevents scaling cheat)
- Support for both TopK and L1 sparsity modes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, Literal

from .utils import geometric_median, unit_norm_decoder, init_tied_weights


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for extracting interpretable features from LLM activations.

    The SAE learns to represent high-dimensional activation vectors as sparse
    combinations of feature directions. Each feature corresponds to a human-
    interpretable concept or pattern in the model's representations.

    Architecture:
        - Encoder: Linear projection to expanded feature space
        - Sparsity: TopK (keeps only k largest activations) or L1 penalty
        - Decoder: Linear reconstruction from sparse features
        - Bias: Geometric median initialization for b_dec (robust centering)

    Key Features:
        - TopK activation: Stable, interpretable sparsity (exactly k features active)
        - Unit norm decoder: Prevents scaling cheat (decoder columns normalized to 1)
        - Geometric median: Robust initialization of reconstruction center
        - Tied weights: Encoder initialized as decoder transpose

    Attributes:
        W_enc: Encoder weight matrix (input_dim, feature_dim).
        W_dec: Decoder weight matrix (feature_dim, input_dim).
        b_enc: Encoder bias (feature_dim,).
        b_dec: Decoder bias / geometric median (input_dim,).
        k: Number of top features to keep (TopK mode).
        use_topk: Whether to use TopK or L1 sparsity.

    Example:
        >>> sae = SparseAutoencoder(input_dim=768, feature_dim=16384, k=128)
        >>> sample = torch.randn(100000, 768)
        >>> sae.initialize_geometric_median(sample)
        >>> reconstructed, features, pre_acts = sae(batch)
        >>> sae.normalize_decoder()  # Call after optimizer.step()
    """

    def __init__(
        self,
        input_dim: int,
        feature_dim: int,
        k: int = 128,
        sparsity_mode: Literal["topk", "l1"] = "topk",
        sparsity_coef: float = 1e-3,
        tied_weights: bool = True,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize the Sparse Autoencoder.

        Args:
            input_dim: Dimension of input activations (e.g., 768 for GPT-2 small).
            feature_dim: Number of sparse features (typically 8x-32x input_dim).
            k: Number of top features to keep in TopK mode.
            sparsity_mode: "topk" or "l1" for sparsity mechanism.
            sparsity_coef: Coefficient for L1 sparsity penalty (only used in L1 mode).
            tied_weights: Whether to tie encoder/decoder initialization.
            device: Device to place parameters on.
            dtype: Data type for parameters.
        """
        super().__init__()

        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.k = k
        self.sparsity_mode = sparsity_mode
        self.sparsity_coef = sparsity_coef
        self.tied_weights = tied_weights

        # Validate parameters (only for TopK mode)
        if sparsity_mode == "topk" and k > feature_dim:
            raise ValueError(f"k ({k}) must be <= feature_dim ({feature_dim})")

        # Initialize weights
        if tied_weights:
            W_enc, W_dec = init_tied_weights(input_dim, feature_dim, device=device)
            self.W_enc = nn.Parameter(W_enc.to(dtype))
            self.W_dec = nn.Parameter(W_dec.to(dtype))
        else:
            self.W_enc = nn.Parameter(torch.randn(input_dim, feature_dim, device=device, dtype=dtype))
            self.W_dec = nn.Parameter(torch.randn(feature_dim, input_dim, device=device, dtype=dtype))
            # Initialize with unit norm
            with torch.no_grad():
                self.W_dec.data = unit_norm_decoder(self.W_dec.data)

        # Biases
        self.b_enc = nn.Parameter(torch.zeros(feature_dim, device=device, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(input_dim, device=device, dtype=dtype))

        self.to(device)

    @property
    def use_topk(self) -> bool:
        """Check if using TopK sparsity mode."""
        return self.sparsity_mode == "topk"

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the SAE.

        Args:
            x: Input activations (batch_size, input_dim).

        Returns:
            Tuple of:
                - reconstructed: Reconstructed activations (batch_size, input_dim)
                - features: Sparse feature activations (batch_size, feature_dim)
                - pre_activations: Pre-sparsity activations (batch_size, feature_dim)
        """
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Input shape {x.shape} incompatible with input_dim {self.input_dim}")

        # Handle 3D input (batch, seq, hidden) by flattening
        original_shape = x.shape
        if x.ndim == 3:
            x = x.reshape(-1, x.shape[-1])

        # 1. Center input at geometric median (b_dec)
        x_centered = x - self.b_dec

        # 2. Encode to feature space
        z_pre = x_centered @ self.W_enc + self.b_enc

        # 3. Apply sparsity
        if self.use_topk:
            features = self._apply_topk(z_pre)
        else:
            # L1 mode: standard ReLU
            features = torch.relu(z_pre)

        # 4. Decode back to input space
        reconstructed = features @ self.W_dec + self.b_dec

        # Restore shape if input was 3D
        if len(original_shape) == 3:
            reconstructed = reconstructed.reshape(original_shape)

        return reconstructed, features, z_pre

    def _apply_topk(self, z_pre: torch.Tensor) -> torch.Tensor:
        """
        Apply TopK sparsity: keep only the k largest activations.

        Args:
            z_pre: Pre-activation values (batch_size, feature_dim).

        Returns:
            Sparse activations with exactly k non-zero values per sample.
        """
        # Get top k values and their indices
        vals, indices = torch.topk(z_pre, k=self.k, dim=-1)

        # Create sparse tensor with only top k values
        sparse = torch.zeros_like(z_pre)
        sparse.scatter_(-1, indices, vals)

        # Apply ReLU for non-negativity
        return torch.relu(sparse)

    def normalize_decoder(self):
        """
        Normalize decoder columns to unit norm in-place.

        This is a critical constraint to prevent the "scaling cheat" where
        the model could make features arbitrarily sparse by scaling up the
        decoder weights and down the encoder weights.

        Call this after each optimizer step during training.
        """
        with torch.no_grad():
            self.W_dec.data = unit_norm_decoder(self.W_dec.data)

    def get_decoder_norms(self) -> torch.Tensor:
        """
        Get L2 norms of decoder columns.

        Useful for monitoring training - norms should stay near 1.0.
        Significant deviations indicate the scaling cheat is occurring.

        Returns:
            Tensor of shape (feature_dim,) with decoder norms.
        """
        return torch.norm(self.W_dec, p=2, dim=1)

    def initialize_geometric_median(self, sample_activations: torch.Tensor):
        """
        Initialize b_dec to the geometric median of sample activations.

        This provides a robust centering point for the SAE, making
        reconstruction easier and training more stable.

        Args:
            sample_activations: Representative sample of activations
                                (n_samples, input_dim). Should be at least
                                ~100k samples for good estimation.
        """
        if sample_activations.ndim == 3:
            sample_activations = sample_activations.reshape(-1, sample_activations.shape[-1])

        if sample_activations.shape[-1] != self.input_dim:
            raise ValueError(
                f"Sample shape {sample_activations.shape} incompatible with "
                f"input_dim {self.input_dim}"
            )

        median = geometric_median(sample_activations)
        self.b_dec.data = median.to(self.b_dec.device, self.b_dec.dtype)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode activations to sparse features (inference only).

        Args:
            x: Input activations (batch_size, input_dim) or (batch, seq, input_dim).

        Returns:
            Sparse feature activations (batch_size, feature_dim).
        """
        with torch.no_grad():
            _, features, _ = self.forward(x)
        return features

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse features to activations (inference only).

        Args:
            features: Sparse features (batch_size, feature_dim).

        Returns:
            Reconstructed activations (batch_size, input_dim).
        """
        with torch.no_grad():
            return features @ self.W_dec + self.b_dec

    def get_feature_activations(self, x: torch.Tensor, feature_ids: Optional[list] = None) -> torch.Tensor:
        """
        Get activations for specific features.

        Args:
            x: Input activations (batch_size, input_dim).
            feature_ids: List of feature indices to compute. If None, compute all.

        Returns:
            Feature activations for requested features.
        """
        _, features, _ = self.forward(x)

        if feature_ids is not None:
            features = features[:, feature_ids]

        return features

    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration dict for serialization.

        Returns:
            Dictionary with all hyperparameters.
        """
        return {
            "input_dim": self.input_dim,
            "feature_dim": self.feature_dim,
            "k": self.k,
            "sparsity_mode": self.sparsity_mode,
            "sparsity_coef": self.sparsity_coef,
            "tied_weights": self.tied_weights,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any], device: str = "cpu") -> "SparseAutoencoder":
        """
        Create SAE from configuration dict.

        Args:
            config: Configuration dictionary (from get_config()).
            device: Device to place the model on.

        Returns:
            SparseAutoencoder instance.
        """
        return cls(
            input_dim=config["input_dim"],
            feature_dim=config["feature_dim"],
            k=config.get("k", 128),
            sparsity_mode=config.get("sparsity_mode", "topk"),
            sparsity_coef=config.get("sparsity_coef", 1e-3),
            tied_weights=config.get("tied_weights", True),
            device=device,
        )

    def save(self, path: str):
        """
        Save model state dict and config.

        Args:
            path: File path to save to.
        """
        state = {
            "config": self.get_config(),
            "state_dict": self.state_dict(),
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "SparseAutoencoder":
        """
        Load model from saved state.

        Args:
            path: File path to load from.
            device: Device to place the model on.

        Returns:
            SparseAutoencoder instance.
        """
        state = torch.load(path, map_location=device)
        sae = cls.from_config(state["config"], device=device)
        sae.load_state_dict(state["state_dict"])
        return sae

    def __repr__(self) -> str:
        return (
            f"SparseAutoencoder("
            f"input_dim={self.input_dim}, "
            f"feature_dim={self.feature_dim}, "
            f"k={self.k}, "
            f"mode={self.sparsity_mode})"
        )

    def __str__(self) -> str:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        lines = [
            f"SparseAutoencoder:",
            f"  Input dim: {self.input_dim}",
            f"  Feature dim: {self.feature_dim} ({self.feature_dim / self.input_dim:.1f}x expansion)",
            f"  Sparsity: k={self.k} ({self.k / self.feature_dim:.2%} active)",
            f"  Mode: {self.sparsity_mode}",
            f"  Parameters: {total_params:,} ({trainable_params:,} trainable)",
            f"  Device: {next(self.parameters()).device}",
        ]
        return "\n".join(lines)
