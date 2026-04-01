"""
Generate demo visualization artifacts using synthetic data.

Produces sample outputs showing what each visualization looks like,
without requiring a real model or the pipeline module. This demonstrates
the visualization capabilities standalone.

Once the pipeline and model integration PRs are merged, these
synthetic examples can be replaced with real model outputs using
``generate_demo_artifacts.py``.

Usage:
    python examples/generate_synthetic_demo.py

Requires: numpy, matplotlib, seaborn, networkx
Optional: plotly (for interactive HTML)
"""

import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "synthetic_demo")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Generating synthetic demo artifacts...")
    print(f"Output: {os.path.abspath(OUTPUT_DIR)}\n")

    # ------------------------------------------------------------------
    # 1. Logit Lens — synthetic layer-by-layer predictions
    # ------------------------------------------------------------------
    print("[1/5] Logit Lens (synthetic)...")
    np.random.seed(42)
    tokens = ["The", "capital", "of", "France", "is"]
    n_layers = 12
    seq_len = len(tokens)

    # Simulate: early layers predict generic tokens, later layers converge
    logit_lens_data = np.random.uniform(0.01, 0.05, (n_layers, seq_len))
    # "France" position gets higher probability in later layers
    for layer in range(n_layers):
        logit_lens_data[layer, 3] = 0.02 + (layer / n_layers) * 0.15
    # Last token "is" → predicting "Paris" gets stronger in later layers
    for layer in range(n_layers):
        logit_lens_data[layer, 4] = 0.01 + (layer / n_layers) ** 2 * 0.25

    # Top-1 predictions per layer per position (synthetic)
    top_predictions = []
    early_preds = ["the", "capital", "the", "'s", "now"]
    late_preds = ["the", "city", "the", ",", "Paris"]
    for layer in range(n_layers):
        blend = layer / (n_layers - 1)
        row = [late_preds[i] if np.random.random() < blend else early_preds[i]
               for i in range(seq_len)]
        top_predictions.append(row)

    try:
        from glassboxllms.visualization.plots import plot_logit_lens
        fig = plot_logit_lens(logit_lens_data, tokens, top_k_tokens=top_predictions)
        path = os.path.join(OUTPUT_DIR, "logit_lens.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved: {path}")
        import matplotlib.pyplot as plt
        plt.close(fig)
    except ImportError as e:
        print(f"  [skip] {e}")

    # ------------------------------------------------------------------
    # 2. Probe Accuracy — synthetic multi-layer comparison
    # ------------------------------------------------------------------
    print("[2/5] Probe Accuracy (synthetic)...")
    # Simulate: accuracy increases from early to late layers
    probe_metrics = {}
    for i, layer_idx in enumerate([0, 2, 4, 6, 8, 11]):
        # Accuracy ramps up from ~55% to ~95%
        acc = 0.50 + (i / 5) * 0.42 + np.random.uniform(-0.03, 0.03)
        acc = min(acc, 0.98)
        probe_metrics[f"h.{layer_idx}"] = {"accuracy": round(acc, 3)}
        print(f"    h.{layer_idx}: accuracy={acc:.3f}")

    try:
        from glassboxllms.visualization.plots import plot_probe_accuracy
        fig = plot_probe_accuracy(probe_metrics)
        path = os.path.join(OUTPUT_DIR, "probe.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved: {path}")
        import matplotlib.pyplot as plt
        plt.close(fig)
    except ImportError as e:
        print(f"  [skip] {e}")

    # ------------------------------------------------------------------
    # 3. Circuit Graph — synthetic layer importance scan
    # ------------------------------------------------------------------
    print("[3/5] Circuit Graph (synthetic)...")
    from glassboxllms.analysis.circuits.graph import CircuitGraph
    from glassboxllms.analysis.circuits.node import NodeType, EdgeType

    graph = CircuitGraph(
        model="gpt2",
        name="synthetic_circuit_scan",
        metadata={"strategy": "zero", "text": "The capital of France is"},
    )
    graph.add_node("input", node_type=NodeType.EMBEDDING, layer=0,
                   label="input")
    graph.add_node("output", node_type=NodeType.UNEMBEDDING,
                   label="output")

    # 6 layers with varying importance
    impacts = [0.8, 0.3, 0.15, 0.4, 0.6, 0.9]
    layer_ids = []
    for i, impact in enumerate(impacts):
        nid = f"layer.{i}"
        graph.add_node(nid, node_type=NodeType.RESIDUAL_STREAM, layer=i + 1,
                       label=f"h.{i}")
        graph.add_edge("input", nid, weight=impact, edge_type=EdgeType.INFERRED)
        graph.add_edge(nid, "output", weight=impact, edge_type=EdgeType.INFERRED)
        layer_ids.append((nid, impact))

    # Inter-layer edges
    for j in range(len(layer_ids) - 1):
        src_id, src_imp = layer_ids[j]
        tgt_id, tgt_imp = layer_ids[j + 1]
        if src_imp > 0.1 and tgt_imp > 0.1:
            graph.add_edge(src_id, tgt_id, weight=min(src_imp, tgt_imp),
                           edge_type=EdgeType.INFERRED)

    try:
        from glassboxllms.visualization.plots import plot_circuit_graph
        fig = plot_circuit_graph(graph, layout="layer", figsize=(16, 8))
        path = os.path.join(OUTPUT_DIR, "circuit.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved: {path}")
        import matplotlib.pyplot as plt
        plt.close(fig)
    except ImportError as e:
        print(f"  [skip] {e}")

    # Also save as JSON
    circuit_path = os.path.join(OUTPUT_DIR, "circuit.json")
    graph.save(circuit_path)
    print(f"  saved: {circuit_path}")

    # ------------------------------------------------------------------
    # 4. Steering Effect — synthetic before/after comparison
    # ------------------------------------------------------------------
    print("[4/5] Steering Effect (synthetic)...")
    # Simulate: steering shifts projection along sentiment direction
    steering_data = {
        "baseline": {"direction_projection": -5.8},
        "steered (strength=3.0)": {"direction_projection": -3.2},
    }

    try:
        from glassboxllms.visualization.plots import plot_steering_effects
        fig = plot_steering_effects(steering_data, metric="direction_projection")
        path = os.path.join(OUTPUT_DIR, "steering.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved: {path}")
        import matplotlib.pyplot as plt
        plt.close(fig)
    except ImportError as e:
        print(f"  [skip] {e}")

    # ------------------------------------------------------------------
    # 5. SAE Feature Browser — synthetic activation heatmap
    # ------------------------------------------------------------------
    print("[5/5] SAE Feature Browser (synthetic)...")
    np.random.seed(42)
    n_features, n_tokens = 50, 20
    # Sparse activation matrix (most entries near zero, some spikes)
    activation_matrix = np.abs(np.random.randn(n_features, n_tokens)) * 0.1
    # Add sparse spikes
    for _ in range(30):
        f, t = np.random.randint(n_features), np.random.randint(n_tokens)
        activation_matrix[f, t] = np.random.uniform(3, 12)

    try:
        from glassboxllms.visualization.interactive import feature_browser
        fig = feature_browser(activation_matrix, top_k=15)
        path = os.path.join(OUTPUT_DIR, "features.html")
        fig.write_html(path)
        print(f"  saved: {path}")
    except ImportError as e:
        print(f"  [skip] {e}")

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    metadata = {
        "type": "synthetic_demo",
        "note": "Generated with synthetic data to demonstrate visualization capabilities. "
                "Replace with real model outputs once pipeline integration PR is merged.",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  saved: {meta_path}")

    print(f"\n{'='*60}")
    print("Synthetic demo artifacts complete!")
    print(f"Output: {os.path.abspath(OUTPUT_DIR)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
