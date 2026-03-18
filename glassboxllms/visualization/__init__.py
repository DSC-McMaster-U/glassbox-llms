"""
glassbox-llms / visualization
==============================

Lightweight interactive plotting layer for interpretability experiments.

Provides two complementary modules:

- **plots**: Matplotlib/seaborn static figures for attention heatmaps,
  logit lens predictions, SAE sparsity curves, probe accuracy, circuit
  graphs, and steering effect comparisons.
- **interactive**: Plotly-based interactive HTML dashboards for SAE feature
  browsing, circuit graph exploration, and embedding space visualization.

Example:
    >>> from glassboxllms.visualization import plot_attention_heatmap
    >>> fig = plot_attention_heatmap(attention_matrix, tokens)
    >>> fig.savefig("attention.png")

    >>> from glassboxllms.visualization import interactive_circuit_explorer
    >>> fig = interactive_circuit_explorer(circuit_graph)
    >>> fig.write_html("circuit.html")
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
    # Static plots
    "plot_attention_heatmap",
    "plot_logit_lens",
    "plot_sae_training_curves",
    "plot_sae_sparsity",
    "plot_probe_accuracy",
    "plot_circuit_graph",
    "plot_steering_effects",
    # Interactive
    "feature_browser",
    "interactive_circuit_explorer",
    "embedding_scatter_3d",
]
