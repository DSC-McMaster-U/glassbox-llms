"""
Sparse Autoencoder (SAE) module for LLM interpretability.

This module provides tools for training and analyzing sparse autoencoders
on LLM activations, enabling extraction of interpretable features.

Key components:
- SparseAutoencoder: Core SAE model with TopK and L1 sparsity modes
- SAETrainer: Training loop with dead neuron handling and metrics
- SAEFeature: Representation of a single learned SAE feature
- FeatureSet: Collection of features with SafeTensors serialization
- Utilities: Geometric median, sparsity metrics, explained variance

Example usage:
    >>> from glassboxllms.features import SparseAutoencoder, SAETrainer
    >>> from glassboxllms.features import geometric_median, FeatureSet
    >>>
    >>> # Initialize SAE
    >>> sae = SparseAutoencoder(input_dim=768, feature_dim=16384, k=128)
    >>>
    >>> # Initialize with geometric median
    >>> sample_acts = torch.randn(100000, 768)
    >>> sae.initialize_geometric_median(sample_acts)
    >>>
    >>> # Train
    >>> trainer = SAETrainer(sae, dataloader, aux_coef=1e-3)
    >>> trainer.train(n_epochs=10)
    >>>
    >>> # Export features
    >>> feature_set = FeatureSet.from_sae(sae, trainer.get_stats())
    >>> feature_set.save("features.safetensors")
"""

from .sae import SparseAutoencoder
from .trainer import SAETrainer, TrainerState
from .feature import SAEFeature
from .feature_set import FeatureSet
from .utils import (
    geometric_median,
    calc_l0_sparsity,
    calc_explained_variance,
    calc_mse_loss,
    calc_mean_l0,
    unit_norm_decoder,
    identify_dead_features,
    topk_activation,
    init_tied_weights,
)

__all__ = [
    # Core classes
    "SparseAutoencoder",
    "SAETrainer",
    "TrainerState",
    "SAEFeature",
    "FeatureSet",
    # Utility functions
    "geometric_median",
    "calc_l0_sparsity",
    "calc_explained_variance",
    "calc_mse_loss",
    "calc_mean_l0",
    "unit_norm_decoder",
    "identify_dead_features",
    "topk_activation",
    "init_tied_weights",
]

__version__ = "0.1.0"
