"""
Tests for visualization adapter functions.

These tests verify that adapter functions correctly convert
glassbox data objects into scene-renderable data structures,
without requiring Manim to be installed.
"""

import numpy as np
import pytest
import torch

from glassboxllms.analysis.circuits import CircuitGraph
from glassboxllms.features.feature import SAEFeature
from glassboxllms.primitives.probes.base import ProbeResult
from glassboxllms.visualization.adapters import (
    CircuitSceneData,
    PipelineSceneData,
    ProbeSceneData,
    SAESceneData,
    SteeringSceneData,
    circuit_graph_to_scene_data,
    pipeline_to_scene_data,
    probe_result_to_scene_data,
    sae_features_to_scene_data,
    steering_result_to_scene_data,
)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def sample_circuit_graph():
    graph = CircuitGraph(model="gpt2", name="test-circuit")
    graph.add_node("embed", node_type="embedding", layer=0)
    graph.add_node("attn.1.h0", node_type="attention_head", layer=1, index=0)
    graph.add_node("mlp.2.n5", node_type="neuron", layer=2, index=5)
    graph.add_node("unembed", node_type="unembedding", layer=3)
    graph.add_edge("embed", "attn.1.h0", weight=0.9)
    graph.add_edge("attn.1.h0", "mlp.2.n5", weight=0.7)
    graph.add_edge("mlp.2.n5", "unembed", weight=0.8)
    return graph


@pytest.fixture
def sample_probe_result():
    rng = np.random.default_rng(42)
    coefficients = rng.standard_normal(768)
    return ProbeResult(
        accuracy=0.85,
        precision=0.84,
        recall=0.86,
        f1=0.85,
        coefficients=coefficients,
        metadata={"n_samples": 100, "n_classes": 2},
    )


@pytest.fixture
def sample_sae_features():
    features = []
    for i in range(5):
        vec = torch.randn(768)
        vec = vec / vec.norm()
        features.append(SAEFeature(
            id=i,
            layer=6,
            model_name="gpt2",
            decoder_vector=vec,
            activation_stats={
                "sparsity": 0.01 * (i + 1),
                "max_activation": float(10 - i),
                "mean_activation": float(2 - i * 0.3),
            },
        ))
    return features


# ======================================================================
# CircuitGraph adapter tests
# ======================================================================


class TestCircuitGraphAdapter:
    def test_basic_conversion(self, sample_circuit_graph):
        data = circuit_graph_to_scene_data(sample_circuit_graph)

        assert isinstance(data, CircuitSceneData)
        assert data.model_name == "gpt2"
        assert data.circuit_name == "test-circuit"
        assert len(data.nodes) == 4
        assert len(data.edges) == 3

    def test_layers_sorted(self, sample_circuit_graph):
        data = circuit_graph_to_scene_data(sample_circuit_graph)
        assert data.layers == [0, 1, 2, 3]

    def test_node_fields(self, sample_circuit_graph):
        data = circuit_graph_to_scene_data(sample_circuit_graph)
        node_ids = {n["id"] for n in data.nodes}
        assert "embed" in node_ids
        assert "unembed" in node_ids

        embed_node = next(n for n in data.nodes if n["id"] == "embed")
        assert embed_node["type"] == "embedding"
        assert embed_node["layer"] == 0

    def test_edge_fields(self, sample_circuit_graph):
        data = circuit_graph_to_scene_data(sample_circuit_graph)
        edge = data.edges[0]
        assert "source" in edge
        assert "target" in edge
        assert "weight" in edge
        assert edge["weight"] is not None

    def test_metadata_has_summary(self, sample_circuit_graph):
        data = circuit_graph_to_scene_data(sample_circuit_graph)
        assert "num_nodes" in data.metadata
        assert data.metadata["num_nodes"] == 4

    def test_empty_graph(self):
        graph = CircuitGraph(model="empty")
        data = circuit_graph_to_scene_data(graph)
        assert len(data.nodes) == 0
        assert len(data.edges) == 0
        assert data.layers == []


# ======================================================================
# ProbeResult adapter tests
# ======================================================================


class TestProbeResultAdapter:
    def test_basic_conversion(self, sample_probe_result):
        data = probe_result_to_scene_data(
            sample_probe_result,
            layer="mlp.6",
            direction_name="sentiment",
        )
        assert isinstance(data, ProbeSceneData)
        assert data.accuracy == 0.85
        assert data.f1 == 0.85
        assert data.layer == "mlp.6"
        assert data.direction_name == "sentiment"

    def test_with_activations(self, sample_probe_result):
        rng = np.random.default_rng(42)
        activations = rng.standard_normal((100, 768))
        labels = np.array([0] * 50 + [1] * 50)

        data = probe_result_to_scene_data(
            sample_probe_result,
            activations=activations,
            labels=labels,
            class_names=["neg", "pos"],
        )

        assert data.points_2d is not None
        assert data.points_2d.shape == (100, 2)
        assert data.labels is not None
        assert data.class_names == ["neg", "pos"]

    def test_subsampling(self, sample_probe_result):
        rng = np.random.default_rng(42)
        activations = rng.standard_normal((1000, 768))
        labels = np.array([0] * 500 + [1] * 500)

        data = probe_result_to_scene_data(
            sample_probe_result,
            activations=activations,
            labels=labels,
            max_samples=100,
        )

        assert data.points_2d.shape[0] == 100

    def test_no_activations(self, sample_probe_result):
        data = probe_result_to_scene_data(sample_probe_result)
        assert data.points_2d is None

    def test_none_coefficients(self):
        result = ProbeResult(accuracy=0.5, coefficients=None)
        data = probe_result_to_scene_data(result)
        assert data.coefficients is not None  # Should default to zeros


# ======================================================================
# SAEFeature adapter tests
# ======================================================================


class TestSAEFeatureAdapter:
    def test_basic_conversion(self, sample_sae_features):
        data = sae_features_to_scene_data(sample_sae_features, top_k=3)

        assert isinstance(data, SAESceneData)
        assert data.model_name == "gpt2"
        assert data.layer == 6
        assert len(data.features) == 3

    def test_sorted_by_max_activation(self, sample_sae_features):
        data = sae_features_to_scene_data(sample_sae_features, top_k=5)

        max_acts = [f["max_activation"] for f in data.features]
        assert max_acts == sorted(max_acts, reverse=True)

    def test_decoder_2d_present(self, sample_sae_features):
        data = sae_features_to_scene_data(sample_sae_features, top_k=3)

        for feat in data.features:
            assert "decoder_2d" in feat
            assert len(feat["decoder_2d"]) == 2

    def test_with_activation_matrix(self, sample_sae_features):
        matrix = np.random.rand(5, 50)
        data = sae_features_to_scene_data(
            sample_sae_features,
            activation_matrix=matrix,
        )
        assert data.activation_grid is not None
        assert data.activation_grid.shape == (5, 50)

    def test_empty_features(self):
        data = sae_features_to_scene_data([], model_name="test", layer=0)
        assert len(data.features) == 0

    def test_single_feature(self):
        vec = torch.randn(768)
        feature = SAEFeature(
            id=0, layer=6, model_name="gpt2", decoder_vector=vec,
            activation_stats={"max_activation": 5.0},
        )
        data = sae_features_to_scene_data([feature], top_k=1)
        assert len(data.features) == 1


# ======================================================================
# SteeringVector adapter tests
# ======================================================================


class TestSteeringResultAdapter:
    def test_basic_conversion(self):
        rng = np.random.default_rng(42)
        vec = rng.standard_normal(768)
        before = rng.standard_normal((50, 768))
        after = before + vec * 3

        data = steering_result_to_scene_data(
            steering_vector=vec,
            activations_before=before,
            activations_after=after,
            direction_name="truth",
            layer="res.12",
            strength=3.0,
        )

        assert isinstance(data, SteeringSceneData)
        assert data.direction_name == "truth"
        assert data.layer == "res.12"
        assert data.strength == 3.0
        assert data.points_before_2d.shape == (50, 2)
        assert data.points_after_2d.shape == (50, 2)
        assert data.steering_vector_2d.shape == (2,)

    def test_subsampling(self):
        rng = np.random.default_rng(42)
        vec = rng.standard_normal(768)
        before = rng.standard_normal((1000, 768))
        after = before + vec

        data = steering_result_to_scene_data(
            steering_vector=vec,
            activations_before=before,
            activations_after=after,
            max_samples=100,
        )

        assert data.points_before_2d.shape[0] == 100


# ======================================================================
# Pipeline adapter tests
# ======================================================================


class TestPipelineAdapter:
    def test_full_pipeline(self, sample_circuit_graph, sample_sae_features):
        probe_results = [
            ("mlp.6", "sentiment", ProbeResult(accuracy=0.85, f1=0.83)),
        ]
        steering = {"direction": "truth", "strength": 3.0}

        data = pipeline_to_scene_data(
            model_name="gpt2",
            circuit_graph=sample_circuit_graph,
            probe_results=probe_results,
            sae_features=sample_sae_features,
            steering_results=steering,
        )

        assert isinstance(data, PipelineSceneData)
        assert data.model_name == "gpt2"
        assert len(data.stages) == 4

        stage_types = [s["type"] for s in data.stages]
        assert "sae" in stage_types
        assert "probe" in stage_types
        assert "circuit" in stage_types
        assert "steering" in stage_types

    def test_partial_pipeline(self, sample_circuit_graph):
        data = pipeline_to_scene_data(
            model_name="gpt2",
            circuit_graph=sample_circuit_graph,
        )
        assert len(data.stages) == 1
        assert data.stages[0]["type"] == "circuit"

    def test_empty_pipeline(self):
        data = pipeline_to_scene_data(model_name="gpt2")
        assert len(data.stages) == 0
