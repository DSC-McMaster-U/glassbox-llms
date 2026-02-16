"""
Mathematical utilities for Sparse Autoencoder training and analysis.

Includes:
- Geometric median computation (Weiszfeld algorithm)
- Sparsity metrics (L0 norm, explained variance)
- Dead neuron identification
"""

import torch
import torch.nn.functional as F
from typing import Optional


def geometric_median(X: torch.Tensor, eps: float = 1e-5, max_iter: int = 100) -> torch.Tensor:
    """
    Compute the geometric median of a set of points using the Weiszfeld algorithm.

    The geometric median is a robust estimator of central tendency that is less
    sensitive to outliers than the arithmetic mean. It's used to initialize the
    decoder bias (b_dec) in SAE training, providing a better centering point
    for activations.

    Args:
        X: Tensor of shape (n_samples, n_features) - the data points.
        eps: Convergence threshold for the iterative algorithm.
        max_iter: Maximum number of iterations.

    Returns:
        Tensor of shape (n_features,) - the geometric median.

    Reference:
        Weiszfeld, E. (1937). Sur le point pour lequel la somme des distances de
        n points donnes est minimum. Tohoku Mathematical Journal.
    """
    if X.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got shape {X.shape}")

    # Start with the mean as initial estimate
    median = X.mean(dim=0)

    for _ in range(max_iter):
        # Compute distances from current median to all points
        distances = torch.norm(X - median, dim=1, p=2)

        # Avoid division by zero - set very small distances to a small value
        distances = torch.clamp(distances, min=eps)

        # Compute weights (inverse of distance)
        weights = 1.0 / distances

        # Weighted average gives new median estimate
        new_median = (weights.unsqueeze(1) * X).sum(dim=0) / weights.sum()

        # Check for convergence
        if torch.norm(new_median - median, p=2) < eps:
            break

        median = new_median

    return median


def calc_l0_sparsity(activations: torch.Tensor) -> torch.Tensor:
    """
    Compute L0 sparsity (fraction of non-zero activations) per feature.

    Args:
        activations: Tensor of shape (n_samples, n_features) - feature activations.

    Returns:
        Tensor of shape (n_features,) - sparsity (fraction active) for each feature.
    """
    if activations.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got shape {activations.shape}")

    return (activations > 0).float().mean(dim=0)


def calc_explained_variance(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """
    Compute the fraction of variance explained by reconstruction.

    Formula: 1 - Var(original - reconstructed) / Var(original)

    Args:
        original: Original input tensor (n_samples, n_features).
        reconstructed: Reconstructed tensor (n_samples, n_features).

    Returns:
        float: Explained variance (1.0 = perfect reconstruction).
    """
    if original.shape != reconstructed.shape:
        raise ValueError(f"Shape mismatch: {original.shape} vs {reconstructed.shape}")

    # Compute residual variance
    residual = original - reconstructed
    var_residual = residual.var(dim=0).mean()
    var_original = original.var(dim=0).mean()

    # Avoid division by zero
    if var_original < 1e-10:
        return 1.0 if var_residual < 1e-10 else 0.0

    explained_var = 1.0 - var_residual / var_original
    return float(explained_var)


def calc_mse_loss(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """
    Compute mean squared error between original and reconstructed.

    Args:
        original: Original input tensor (n_samples, n_features).
        reconstructed: Reconstructed tensor (n_samples, n_features).

    Returns:
        float: MSE value.
    """
    return F.mse_loss(reconstructed, original).item()


def unit_norm_decoder(W_dec: torch.Tensor) -> torch.Tensor:
    """
    Normalize decoder columns to unit norm.

    This is a critical constraint in SAE training to prevent the "scaling cheat"
    where the model could make features arbitrarily sparse by scaling up the
    decoder weights and down the encoder weights.

    Args:
        W_dec: Decoder weight matrix (n_features, input_dim).

    Returns:
        Normalized decoder weight matrix.
    """
    return F.normalize(W_dec, p=2, dim=1)


def identify_dead_features(
    activation_history: torch.Tensor,
    threshold: int,
    current_step: Optional[int] = None
) -> torch.Tensor:
    """
    Identify features that have been inactive for a threshold number of steps.

    Args:
        activation_history: Boolean tensor of shape (n_features, history_length)
                           indicating whether each feature was active at each step.
        threshold: Number of consecutive inactive steps to consider a feature "dead".
        current_step: Current step index (if history is a rolling buffer).

    Returns:
        Boolean tensor of shape (n_features,) - True for dead features.
    """
    if activation_history.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got shape {activation_history.shape}")

    # Count consecutive inactive steps from the most recent history
    n_features, history_len = activation_history.shape

    # Look at the most recent 'threshold' steps
    recent_history = activation_history[:, -threshold:]

    # A feature is dead if it was never active in the recent history
    dead_mask = ~recent_history.any(dim=1)

    return dead_mask


def calc_mean_l0(activations: torch.Tensor) -> float:
    """
    Compute mean L0 norm (average number of active features per sample).

    Args:
        activations: Tensor of shape (n_samples, n_features).

    Returns:
        float: Average number of active features per sample.
    """
    l0_per_sample = (activations > 0).float().sum(dim=1)
    return float(l0_per_sample.mean())


def topk_activation(pre_activations: torch.Tensor, k: int) -> torch.Tensor:
    """
    Apply TopK sparsity to pre-activations.

    Keeps only the top k values (by magnitude) and sets the rest to zero.
    Also applies ReLU to ensure non-negativity.

    Args:
        pre_activations: Pre-activation values (n_samples, n_features).
        k: Number of top activations to keep.

    Returns:
        Sparse activations with exactly k non-zero values per sample.
    """
    # Get top k values and their indices
    vals, indices = torch.topk(pre_activations, k=k, dim=-1)

    # Create sparse tensor with only top k values
    sparse = torch.zeros_like(pre_activations)
    sparse.scatter_(-1, indices, vals)

    # Apply ReLU for non-negativity
    return torch.relu(sparse)


def init_tied_weights(input_dim: int, feature_dim: int, device: str = "cpu") -> tuple:
    """
    Initialize tied encoder/decoder weights.

    The decoder is initialized with random unit vectors, and the encoder
    is initialized as the transpose of the decoder (tied initialization).

    Args:
        input_dim: Dimension of input activations.
        feature_dim: Number of sparse features (expanded dimension).
        device: Device to place tensors on.

    Returns:
        Tuple of (W_enc, W_dec) initial weight tensors.
    """
    # Initialize decoder with random unit vectors
    W_dec = torch.randn(feature_dim, input_dim, device=device)
    W_dec = F.normalize(W_dec, p=2, dim=1)

    # Encoder is transpose of decoder (tied initialization)
    W_enc = W_dec.T.clone()

    return W_enc, W_dec
