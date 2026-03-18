"""
Example: Animate steering vector effects with Manim.

Shows how to visualize the shift in model representations when
a steering vector is added to activations at a specific layer.

Usage:
    manim -ql steering_vector_animation.py SteeringVectorScene
"""

import numpy as np

from glassboxllms.visualization.adapters import steering_result_to_scene_data
from glassboxllms.visualization.scenes import SteeringVectorScene


def build_example_steering():
    """Simulate a steering experiment (e.g., truth-telling direction)."""
    rng = np.random.default_rng(42)
    n_features = 768
    n_samples = 150
    strength = 3.0

    # Steering vector (a learned direction, e.g., from probing or CAV)
    steering_vector = rng.standard_normal(n_features)
    steering_vector = steering_vector / np.linalg.norm(steering_vector)

    # Baseline activations (before steering)
    activations_before = rng.standard_normal((n_samples, n_features))

    # After steering: shift activations along the steering direction
    activations_after = activations_before + strength * steering_vector

    return steering_vector, activations_before, activations_after, strength


def main():
    steering_vec, before, after, strength = build_example_steering()
    print(f"Steering vector shape: {steering_vec.shape}")
    print(f"Samples: {before.shape[0]}, strength: {strength}")

    scene_data = steering_result_to_scene_data(
        steering_vector=steering_vec,
        activations_before=before,
        activations_after=after,
        direction_name="truthfulness",
        layer="residual.12",
        strength=strength,
    )

    scene = SteeringVectorScene()
    scene.scene_data = scene_data
    scene.render()


if __name__ == "__main__":
    main()
