"""
Example: Animate SAE feature discovery with Manim.

Shows how to take SAEFeature objects (from a trained SparseAutoencoder)
and render their activation statistics and decoder directions.

Usage:
    manim -ql sae_feature_animation.py SAEFeatureDiscoveryScene
"""

import torch
import numpy as np

from glassboxllms.features.feature import SAEFeature
from glassboxllms.visualization.adapters import sae_features_to_scene_data
from glassboxllms.visualization.scenes import SAEFeatureDiscoveryScene


def build_example_features():
    """Create example SAE features with realistic statistics."""
    rng = np.random.default_rng(42)
    input_dim = 768
    n_features = 20

    features = []
    for i in range(n_features):
        # Random decoder direction (unit norm)
        vec = torch.randn(input_dim)
        vec = vec / vec.norm()

        sparsity = rng.uniform(0.001, 0.05)
        max_act = rng.uniform(1.0, 15.0)
        mean_act = max_act * rng.uniform(0.05, 0.3)

        feature = SAEFeature(
            id=i,
            layer=6,
            model_name="gpt2",
            decoder_vector=vec,
            activation_stats={
                "sparsity": sparsity,
                "max_activation": max_act,
                "mean_activation": mean_act,
            },
        )
        features.append(feature)

    # Optional: build a small activation grid (features x samples)
    n_samples = 50
    activation_matrix = np.zeros((n_features, n_samples))
    for i in range(n_features):
        # Sparse activations
        active_mask = rng.random(n_samples) < features[i].sparsity * 10
        activation_matrix[i, active_mask] = rng.uniform(0, features[i].max_activation,
                                                         size=active_mask.sum())

    return features, activation_matrix


def main():
    features, activation_matrix = build_example_features()
    print(f"Created {len(features)} example features")

    scene_data = sae_features_to_scene_data(
        features=features,
        activation_matrix=activation_matrix,
        top_k=10,
    )

    scene = SAEFeatureDiscoveryScene()
    scene.scene_data = scene_data
    scene.render()


if __name__ == "__main__":
    main()
