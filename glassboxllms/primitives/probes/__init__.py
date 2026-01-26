"""
glassbox-llms / probes
======================
Linear probing toolkit for interpreting neural network activations.

Provides:
  - LinearProbe: Logistic regression, linear regression, PCA projections
  - ActivationStore: Extract and cache activations from transformer layers
  - CAV (Concept Activation Vectors): Direction-based concept detection

"""

from .base import BaseProbe
from .linear import LinearProbe, ProbeResult
from .activation_store import ActivationStore

__all__ = ["BaseProbe", "LinearProbe", "ProbeResult", "ActivationStore"]
