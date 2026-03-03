"""
SAE Feature Interpretation Examples

This package demonstrates two methods for extracting interpretable knowledge
from trained Sparse Autoencoder (SAE) features:

1. Max Activating Examples (max_activating_examples.py)
   - Find inputs that maximally activate each feature
   - Understand features through behavioral patterns

2. Decoder Analysis (decoder_analysis.py)
   - Analyze decoder weights using Logit Lens
   - Understand features through vocabulary predictions

Both analysis methods use the same diverse dataset from Hugging Face (dataset.py)
for consistent comparison.

Usage:
    python -m examples.sae_feature.max_activating_examples
    python -m examples.sae_feature.decoder_analysis
"""

from .dataset import create_diverse_dataset

__version__ = "1.0.0"
__all__ = ["create_diverse_dataset"]
