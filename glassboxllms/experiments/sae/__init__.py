"""
SAE Experiments module for automated discovery of monosemantic features.

This module provides high-level experiment classes that orchestrate
the full pipeline: data collection → training → feature extraction → registration.
"""

from glassboxllms.experiments.sae.sae_experiment import SAEExperiment

__all__ = ["SAEExperiment"]
