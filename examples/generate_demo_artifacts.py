"""
Generate the full demo artifact bundle from a real GPT-2 run.

Runs every pipeline stage on GPT-2 and saves visualization artifacts
to ``outputs/demo_run/``.

Usage:
    python examples/generate_demo_artifacts.py

Requires: torch, transformers, scikit-learn, matplotlib, plotly
"""

import json
import os
import time
from datetime import datetime, timezone

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "gpt2"
TARGET_LAYER = "transformer.h.5"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "demo_run")

LOGIT_LENS_TEXT = "The capital of France is"

POSITIVE_TEXTS = [
    "I love this movie, it was fantastic!",
    "This is wonderful and amazing.",
    "Great job, I'm so happy with the results.",
    "The food was delicious and the service excellent.",
    "What a beautiful day, everything is perfect!",
]

NEGATIVE_TEXTS = [
    "I hate this, it was terrible.",
    "This is awful and disappointing.",
    "Terrible work, I'm very upset.",
    "The food was disgusting and the service horrible.",
    "What an ugly mess, nothing works right.",
]

SAE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Scientists discovered a new species of deep-sea fish.",
    "The stock market reached an all-time high today.",
    "She played a beautiful melody on the piano.",
    "The algorithm processes data in logarithmic time.",
    "Rain is expected throughout the weekend.",
    "The team won the championship in overtime.",
    "Quantum computers can solve certain problems faster.",
]

CIRCUIT_TEXT = "The capital of France is"
CIRCUIT_LAYERS = [f"transformer.h.{i}" for i in range(6)]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metrics: dict = {
        "model": MODEL_NAME,
        "layer": TARGET_LAYER,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stages": {},
    }

    # ------------------------------------------------------------------
    # Load model ONCE
    # ------------------------------------------------------------------
    print("Loading model...")
    from glassboxllms.models.huggingface import TransformersModelWrapper

    model = TransformersModelWrapper(MODEL_NAME, model_class="causal_lm")
    print(f"  {model}")

    from glassboxllms.pipeline import (
        discover_circuit,
        extract_activations,
        run_logit_lens,
        steer_on_model,
        train_probe_on_model,
        train_sae_on_model,
    )
    from glassboxllms.primitives.probes.linear import LinearProbe

    # ------------------------------------------------------------------
    # 1. Logit Lens
    # ------------------------------------------------------------------
    print("\n[1/6] Logit Lens...")
    t0 = time.time()
    logit_results = run_logit_lens(MODEL_NAME, LOGIT_LENS_TEXT, top_k=5, model=model)
    elapsed = time.time() - t0
    print(f"  tokens: {logit_results['tokens']}")
    print(f"  shape:  {logit_results['logit_lens_data'].shape}")
    print(f"  time:   {elapsed:.1f}s")

    try:
        from glassboxllms.visualization.plots import plot_logit_lens

        fig = plot_logit_lens(
            logit_results["logit_lens_data"],
            logit_results["tokens"],
            top_k_tokens=logit_results["top_k_tokens"],
        )
        path = os.path.join(OUTPUT_DIR, "logit_lens.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved:  {path}")
        import matplotlib.pyplot as plt
        plt.close(fig)
    except ImportError as e:
        print(f"  [skip plot] {e}")

    metrics["stages"]["logit_lens"] = {
        "text": LOGIT_LENS_TEXT,
        "n_layers": int(logit_results["logit_lens_data"].shape[0]),
        "seq_len": int(logit_results["logit_lens_data"].shape[1]),
        "top_prediction_final_layer": logit_results["top_k_tokens"][-1][-1][:3],
        "elapsed_s": round(elapsed, 2),
    }

    # ------------------------------------------------------------------
    # 2. Probe
    # ------------------------------------------------------------------
    print("\n[2/6] Linear Probe...")
    t0 = time.time()
    probe, direction = train_probe_on_model(
        MODEL_NAME, POSITIVE_TEXTS, NEGATIVE_TEXTS,
        layer=TARGET_LAYER, model=model,
    )
    elapsed = time.time() - t0
    print(f"  fitted:    {probe.is_fitted}")
    print(f"  direction: shape={direction.shape}, norm={np.linalg.norm(direction):.4f}")
    print(f"  time:      {elapsed:.1f}s")

    # Extract activations for probe scatter plot
    all_probe_texts = POSITIVE_TEXTS + NEGATIVE_TEXTS
    probe_acts = extract_activations(
        MODEL_NAME, all_probe_texts, [TARGET_LAYER],
        return_type="numpy", model=model,
    )[TARGET_LAYER]
    if probe_acts.ndim == 3:
        probe_acts = probe_acts.mean(axis=1)
    probe_labels = np.array([1] * len(POSITIVE_TEXTS) + [0] * len(NEGATIVE_TEXTS))

    try:
        from glassboxllms.visualization.plots import plot_probe_accuracy

        # Probe across multiple layers to show where sentiment is encoded
        probe_layers = [f"transformer.h.{i}" for i in [0, 2, 4, 5, 8, 11]]
        probe_metrics = {}
        for pl in probe_layers:
            pl_acts = extract_activations(
                MODEL_NAME, all_probe_texts, [pl],
                return_type="numpy", model=model,
            )[pl]
            if pl_acts.ndim == 3:
                pl_acts = pl_acts.mean(axis=1)
            p = LinearProbe(layer=pl, direction="sentiment", model_type="logistic")
            p.fit(pl_acts, probe_labels)
            result = p.evaluate(pl_acts, probe_labels)
            acc = result.accuracy if result else 0.0
            # Use short label for display
            short_label = f"h.{pl.split('.')[-1]}"
            probe_metrics[short_label] = {"accuracy": acc}
            print(f"    layer {short_label}: accuracy={acc:.2f}")

        fig = plot_probe_accuracy(probe_metrics)
        path = os.path.join(OUTPUT_DIR, "probe.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved:     {path}")
        import matplotlib.pyplot as plt
        plt.close(fig)
    except ImportError as e:
        print(f"  [skip plot] {e}")

    metrics["stages"]["probe"] = {
        "layer": TARGET_LAYER,
        "n_positive": len(POSITIVE_TEXTS),
        "n_negative": len(NEGATIVE_TEXTS),
        "direction_norm": round(float(np.linalg.norm(direction)), 4),
        "elapsed_s": round(elapsed, 2),
    }

    # ------------------------------------------------------------------
    # 3. SAE
    # ------------------------------------------------------------------
    print("\n[3/6] Sparse Autoencoder...")
    t0 = time.time()
    sae, feature_set = train_sae_on_model(
        MODEL_NAME, SAE_TEXTS, layer=TARGET_LAYER,
        feature_dim=512, k=32, n_epochs=3, batch_size=16, model=model,
    )
    elapsed = time.time() - t0
    print(f"  features:  {len(feature_set)}")
    print(f"  time:      {elapsed:.1f}s")

    try:
        from glassboxllms.visualization.interactive import feature_browser

        # Build activation matrix for top features
        sae_acts = extract_activations(
            MODEL_NAME, SAE_TEXTS[:4], [TARGET_LAYER],
            return_type="torch", model=model,
        )[TARGET_LAYER]
        import torch
        if sae_acts.ndim == 3:
            sae_acts = sae_acts.reshape(-1, sae_acts.shape[-1])
        with torch.no_grad():
            _, feat_acts, _ = sae(sae_acts.to(torch.float32))
        activation_matrix = feat_acts.detach().cpu().numpy().T  # (features, samples)

        fig = feature_browser(activation_matrix[:50], top_k=20)
        path = os.path.join(OUTPUT_DIR, "features.html")
        fig.write_html(path)
        print(f"  saved:     {path}")
    except ImportError as e:
        print(f"  [skip plot] {e}")

    metrics["stages"]["sae"] = {
        "layer": TARGET_LAYER,
        "feature_dim": 512,
        "k": 32,
        "n_features": len(feature_set),
        "elapsed_s": round(elapsed, 2),
    }

    # ------------------------------------------------------------------
    # 4. Circuit Discovery
    # ------------------------------------------------------------------
    print("\n[4/6] Circuit Discovery...")
    t0 = time.time()

    def output_norm_metric(output):
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state[0, -1].norm()
        return output[0, -1].norm()

    graph = discover_circuit(
        MODEL_NAME, CIRCUIT_TEXT, output_norm_metric,
        layers=CIRCUIT_LAYERS, strategy="zero", model=model,
    )
    elapsed = time.time() - t0
    summary = graph.summary()
    print(f"  nodes: {summary['num_nodes']}, edges: {summary['num_edges']}")
    print(f"  time:  {elapsed:.1f}s")

    try:
        from glassboxllms.visualization.plots import plot_circuit_graph

        # Shorten node labels for readability
        for node in graph.nodes:
            short = node.id.replace("transformer.", "").replace("layer.", "L")
            node.metadata["label"] = short

        fig = plot_circuit_graph(graph, layout="layer", figsize=(16, 8))
        path = os.path.join(OUTPUT_DIR, "circuit.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved: {path}")
        import matplotlib.pyplot as plt
        plt.close(fig)
    except ImportError as e:
        print(f"  [skip plot] {e}")

    # Also save circuit as JSON
    circuit_path = os.path.join(OUTPUT_DIR, "circuit.json")
    graph.save(circuit_path)
    print(f"  saved: {circuit_path}")

    metrics["stages"]["circuit"] = {
        "text": CIRCUIT_TEXT,
        "strategy": "zero",
        "n_layers_scanned": len(CIRCUIT_LAYERS),
        **summary,
        "elapsed_s": round(elapsed, 2),
    }

    # ------------------------------------------------------------------
    # 5. Steering
    # ------------------------------------------------------------------
    print("\n[5/6] Activation Steering...")
    t0 = time.time()
    steer_results = steer_on_model(
        MODEL_NAME, POSITIVE_TEXTS + NEGATIVE_TEXTS,
        layer=TARGET_LAYER, direction=direction, strength=3.0,
        model=model,
    )
    elapsed = time.time() - t0
    print(f"  before shape: {steer_results['activations_before'].shape}")
    print(f"  after shape:  {steer_results['activations_after'].shape}")
    print(f"  time:         {elapsed:.1f}s")

    try:
        from glassboxllms.visualization.plots import plot_steering_effects

        # Project activations onto steering direction to show meaningful shift.
        # The dot product with the direction vector measures how much each
        # sample aligns with the sentiment concept.
        dir_unit = steer_results["direction"] / (np.linalg.norm(steer_results["direction"]) + 1e-8)
        proj_before = steer_results["activations_before"] @ dir_unit
        proj_after = steer_results["activations_after"] @ dir_unit

        steering_data = {
            "baseline": {"direction_projection": float(proj_before.mean())},
            f"steered (strength={steer_results['strength']})": {
                "direction_projection": float(proj_after.mean()),
            },
        }
        print(f"  projection before: {proj_before.mean():.3f}")
        print(f"  projection after:  {proj_after.mean():.3f}")
        fig = plot_steering_effects(steering_data, metric="direction_projection")
        path = os.path.join(OUTPUT_DIR, "steering.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved:        {path}")
        import matplotlib.pyplot as plt
        plt.close(fig)
    except ImportError as e:
        print(f"  [skip plot] {e}")

    metrics["stages"]["steering"] = {
        "layer": TARGET_LAYER,
        "strength": steer_results["strength"],
        "n_samples": steer_results["activations_before"].shape[0],
        "elapsed_s": round(elapsed, 2),
    }

    # ------------------------------------------------------------------
    # 6. Save metrics JSON
    # ------------------------------------------------------------------
    print("\n[6/6] Saving metrics...")
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  saved: {metrics_path}")

    # ------------------------------------------------------------------
    # Summary markdown
    # ------------------------------------------------------------------
    summary_path = os.path.join(OUTPUT_DIR, "demo_summary.md")
    with open(summary_path, "w") as f:
        f.write(f"# Glassbox LLMs Demo Results\n\n")
        f.write(f"**Model:** {MODEL_NAME}  \n")
        f.write(f"**Target Layer:** {TARGET_LAYER}  \n")
        f.write(f"**Generated:** {metrics['generated_at']}  \n\n")
        f.write("## Pipeline Stages\n\n")
        for stage_name, stage_data in metrics["stages"].items():
            f.write(f"### {stage_name.replace('_', ' ').title()}\n")
            for k, v in stage_data.items():
                f.write(f"- **{k}:** {v}\n")
            f.write("\n")
        f.write("## Artifacts\n\n")
        for fname in sorted(os.listdir(OUTPUT_DIR)):
            f.write(f"- `{fname}`\n")
    print(f"  saved: {summary_path}")

    print("\n" + "=" * 60)
    print("Demo artifact bundle complete!")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
