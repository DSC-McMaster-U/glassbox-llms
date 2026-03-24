"""
Example: Animate probe decision boundaries with Manim.

Shows how to take a trained LinearProbe's results and visualize
the learned hyperplane separating classes in activation space.

Usage:
    manim -ql probe_hyperplane_animation.py ProbingHyperplaneScene
"""

import numpy as np

from glassboxllms.primitives.probes.base import ProbeResult
from glassboxllms.visualization.adapters import probe_result_to_scene_data
from glassboxllms.visualization.scenes import ProbingHyperplaneScene


def build_example_probe_result():
    """Simulate a trained sentiment probe with synthetic activations."""
    rng = np.random.default_rng(42)
    n_features = 768
    n_samples = 200

    # Simulate probe coefficients (learned direction)
    coefficients = rng.standard_normal(n_features)
    coefficients = coefficients / np.linalg.norm(coefficients)

    # Simulate activations with a separable structure
    positive_acts = rng.standard_normal((n_samples // 2, n_features)) + coefficients * 2
    negative_acts = rng.standard_normal((n_samples // 2, n_features)) - coefficients * 2
    activations = np.vstack([positive_acts, negative_acts])
    labels = np.array([1] * (n_samples // 2) + [0] * (n_samples // 2))

    result = ProbeResult(
        accuracy=0.87,
        precision=0.88,
        recall=0.86,
        f1=0.87,
        coefficients=coefficients,
        metadata={"n_samples": n_samples, "n_classes": 2},
    )

    return result, activations, labels


def main():
    result, activations, labels = build_example_probe_result()
    print(result)

    scene_data = probe_result_to_scene_data(
        probe_result=result,
        activations=activations,
        labels=labels,
        class_names=["Negative", "Positive"],
        layer="mlp.6",
        direction_name="sentiment",
    )

    scene = ProbingHyperplaneScene()
    scene.scene_data = scene_data
    scene.render()


if __name__ == "__main__":
    main()
