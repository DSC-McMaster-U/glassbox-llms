"""
Example: Animate the full interpretability pipeline with Manim.

Shows how to combine results from multiple analysis stages (SAE feature
discovery, probing, circuit discovery, steering) into a single pipeline
overview animation.

Usage:
    manim -ql full_pipeline_animation.py FullPipelineScene
"""

import numpy as np
import torch

from glassboxllms.analysis.circuits import CircuitGraph
from glassboxllms.features.feature import SAEFeature
from glassboxllms.primitives.probes.base import ProbeResult
from glassboxllms.visualization.adapters import pipeline_to_scene_data
from glassboxllms.visualization.scenes import FullPipelineScene


def main():
    # --- Stage 1: SAE features ---
    features = []
    for i in range(50):
        vec = torch.randn(768)
        vec = vec / vec.norm()
        features.append(SAEFeature(
            id=i, layer=6, model_name="gpt2", decoder_vector=vec,
            activation_stats={"sparsity": 0.01, "max_activation": float(np.random.uniform(1, 10))},
        ))

    # --- Stage 2: Probing results ---
    probe_results = [
        ("mlp.6", "sentiment", ProbeResult(accuracy=0.87, f1=0.85)),
        ("mlp.10", "tense", ProbeResult(accuracy=0.92, f1=0.91)),
        ("attn.8", "subject_number", ProbeResult(accuracy=0.78, f1=0.76)),
    ]

    # --- Stage 3: Circuit graph ---
    graph = CircuitGraph(model="gpt2", name="IOI Circuit")
    graph.add_node("embed", node_type="embedding", layer=0)
    graph.add_node("attn.1.h3", node_type="attention_head", layer=1, index=3)
    graph.add_node("mlp.4.n42", node_type="neuron", layer=4, index=42)
    graph.add_node("attn.8.h11", node_type="attention_head", layer=8, index=11)
    graph.add_node("unembed", node_type="unembedding", layer=12)
    graph.add_edge("embed", "attn.1.h3", weight=0.9)
    graph.add_edge("attn.1.h3", "mlp.4.n42", weight=0.7)
    graph.add_edge("mlp.4.n42", "attn.8.h11", weight=0.8)
    graph.add_edge("attn.8.h11", "unembed", weight=0.95)

    # --- Stage 4: Steering ---
    steering_info = {
        "direction": "truthfulness",
        "layer": "residual.12",
        "strength": 3.0,
    }

    # Combine into pipeline
    scene_data = pipeline_to_scene_data(
        model_name="gpt2",
        circuit_graph=graph,
        probe_results=probe_results,
        sae_features=features,
        steering_results=steering_info,
    )

    scene = FullPipelineScene()
    scene.scene_data = scene_data
    scene.render()


if __name__ == "__main__":
    main()
