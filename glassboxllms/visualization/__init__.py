"""
glassboxllms.visualization
==========================

Visualization utilities for interpretability outputs:
- static plots (`plots.py`)
- interactive dashboards (`interactive.py`)
- cinematic explainers (`manim_scenes/`)
"""

from .plots import (
    plot_attention_heatmap,
    plot_logit_lens,
    plot_sae_training_curves,
    plot_sae_sparsity,
    plot_probe_accuracy,
    plot_circuit_graph,
    plot_steering_effects,
)
from .interactive import (
    feature_browser,
    interactive_circuit_explorer,
    embedding_scatter_3d,
)

__all__ = [
    "plot_attention_heatmap",
    "plot_logit_lens",
    "plot_sae_training_curves",
    "plot_sae_sparsity",
    "plot_probe_accuracy",
    "plot_circuit_graph",
    "plot_steering_effects",
    "feature_browser",
    "interactive_circuit_explorer",
    "embedding_scatter_3d",
]
