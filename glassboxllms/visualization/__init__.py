"""
glassboxllms.visualization
==========================

Visualization layer for interpretability experiments.

Provides three complementary modules:

- **plots**: Matplotlib/seaborn static figures for attention heatmaps,
  logit lens predictions, SAE sparsity curves, probe accuracy, circuit
  graphs, and steering effect comparisons.
- **interactive**: Plotly-based interactive HTML dashboards for SAE feature
  browsing, circuit graph exploration, and embedding space visualization.
- **scenes**: Manim-based animated visualizations connected to real glassbox
  data objects (CircuitGraph, ProbeResult, SAEFeature, steering vectors).
- **adapters**: Converter functions from glassbox types to Manim-renderable
  data structures.

Example (static plots):
    >>> from glassboxllms.visualization import plot_attention_heatmap
    >>> fig = plot_attention_heatmap(attention_matrix, tokens)
    >>> fig.savefig("attention.png")

Example (interactive):
    >>> from glassboxllms.visualization import interactive_circuit_explorer
    >>> fig = interactive_circuit_explorer(circuit_graph)
    >>> fig.write_html("circuit.html")

Example (Manim scenes):
    >>> from glassboxllms.analysis.circuits import CircuitGraph
    >>> from glassboxllms.visualization import adapters, scenes
    >>>
    >>> graph = CircuitGraph.load("my_circuit.json")
    >>> scene_data = adapters.circuit_graph_to_scene_data(graph)
    >>> scene = scenes.CircuitDiscoveryScene(scene_data)
    >>> scene.render()
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

from .adapters import (
    circuit_graph_to_scene_data,
    cot_result_to_scene_data,
    probe_result_to_scene_data,
    sae_features_to_scene_data,
    steering_result_to_scene_data,
    pipeline_to_scene_data,
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
    # Manim adapters & scenes
    "adapters",
    "scenes",
    "circuit_graph_to_scene_data",
    "cot_result_to_scene_data",
    "probe_result_to_scene_data",
    "sae_features_to_scene_data",
    "steering_result_to_scene_data",
    "pipeline_to_scene_data",
]
