"""
Unit tests for the visualization module.

Tests cover:
- Static plot functions (matplotlib-based)
- Interactive plot functions (plotly-based)
- Edge cases (empty data, missing labels)
- Save-to-file functionality
- Integration with project data structures (CircuitGraph, TrainerState)
"""

import os
import shutil
import tempfile

import numpy as np
import pytest

from glassboxllms.visualization.plots import (
    plot_attention_heatmap,
    plot_logit_lens,
    plot_sae_training_curves,
    plot_sae_sparsity,
    plot_probe_accuracy,
    plot_circuit_graph,
    plot_steering_effects,
)
from glassboxllms.visualization.interactive import (
    feature_browser,
    interactive_circuit_explorer,
    embedding_scatter_3d,
)
from glassboxllms.analysis.circuits import CircuitGraph


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    dirpath = tempfile.mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath)


@pytest.fixture
def sample_attention():
    np.random.seed(42)
    return np.random.rand(8, 8)


@pytest.fixture
def sample_tokens():
    return ["The", "cat", "sat", "on", "the", "mat", ".", "<eos>"]


@pytest.fixture
def sample_circuit_graph():
    g = CircuitGraph(model="gpt2-small", name="test_viz_circuit")
    g.add_node("mlp.0.neuron.1", node_type="neuron", layer=0, index=1)
    g.add_node("mlp.0.neuron.2", node_type="neuron", layer=0, index=2)
    g.add_node("attn.1.head.0", node_type="attention_head", layer=1, index=0)
    g.add_node("mlp.2.neuron.5", node_type="neuron", layer=2, index=5)
    g.add_edge("mlp.0.neuron.1", "attn.1.head.0", weight=0.7)
    g.add_edge("mlp.0.neuron.2", "attn.1.head.0", weight=0.3)
    g.add_edge("attn.1.head.0", "mlp.2.neuron.5", weight=0.9)
    return g


@pytest.fixture
def sample_training_history():
    np.random.seed(42)
    n = 100
    return {
        "loss": list(np.exp(-np.linspace(0, 2, n)) + np.random.randn(n) * 0.01),
        "mse_loss": list(np.exp(-np.linspace(0, 2, n)) + np.random.randn(n) * 0.005),
        "explained_variance": list(1 - np.exp(-np.linspace(0, 2, n))),
        "mean_l0": list(np.linspace(50, 128, n)),
        "dead_feature_count": list(np.linspace(500, 10, n).astype(int)),
        "aux_loss": [],
        "l1_loss": [],
    }


# =============================================================================
# Static Plot Tests
# =============================================================================

class TestAttentionHeatmap:

    def test_basic(self, sample_attention, sample_tokens):
        import matplotlib
        matplotlib.use("Agg")
        fig = plot_attention_heatmap(sample_attention, sample_tokens)
        assert fig is not None

    def test_without_tokens(self, sample_attention):
        import matplotlib
        matplotlib.use("Agg")
        fig = plot_attention_heatmap(sample_attention)
        assert fig is not None

    def test_with_layer_head(self, sample_attention, sample_tokens):
        import matplotlib
        matplotlib.use("Agg")
        fig = plot_attention_heatmap(
            sample_attention, sample_tokens, layer=3, head=7,
        )
        assert fig is not None

    def test_save_to_file(self, sample_attention, temp_dir):
        import matplotlib
        matplotlib.use("Agg")
        path = os.path.join(temp_dir, "attention.png")
        fig = plot_attention_heatmap(sample_attention, save_path=path)
        assert os.path.exists(path)
        assert fig is not None


class TestLogitLens:

    def test_basic(self, sample_tokens):
        import matplotlib
        matplotlib.use("Agg")
        data = np.random.rand(12, len(sample_tokens))
        fig = plot_logit_lens(data, sample_tokens)
        assert fig is not None

    def test_save(self, sample_tokens, temp_dir):
        import matplotlib
        matplotlib.use("Agg")
        data = np.random.rand(6, len(sample_tokens))
        path = os.path.join(temp_dir, "logit_lens.png")
        fig = plot_logit_lens(data, sample_tokens, save_path=path)
        assert os.path.exists(path)


class TestSAETrainingCurves:

    def test_basic(self, sample_training_history):
        import matplotlib
        matplotlib.use("Agg")
        fig = plot_sae_training_curves(sample_training_history)
        assert fig is not None

    def test_subset_metrics(self, sample_training_history):
        import matplotlib
        matplotlib.use("Agg")
        fig = plot_sae_training_curves(
            sample_training_history, metrics=["loss", "explained_variance"],
        )
        assert fig is not None

    def test_empty_history(self):
        import matplotlib
        matplotlib.use("Agg")
        fig = plot_sae_training_curves({"loss": [], "mse_loss": []})
        assert fig is not None


class TestSAESparsity:

    def test_sparsity_only(self):
        import matplotlib
        matplotlib.use("Agg")
        sparsity = np.random.rand(1000)
        fig = plot_sae_sparsity(sparsity)
        assert fig is not None

    def test_with_reconstruction(self):
        import matplotlib
        matplotlib.use("Agg")
        sparsity = np.random.rand(1000)
        mse = np.random.rand(500) * 0.1
        fig = plot_sae_sparsity(sparsity, reconstruction_mse=mse)
        assert fig is not None

    def test_save(self, temp_dir):
        import matplotlib
        matplotlib.use("Agg")
        path = os.path.join(temp_dir, "sparsity.png")
        fig = plot_sae_sparsity(np.random.rand(100), save_path=path)
        assert os.path.exists(path)


class TestProbeAccuracy:

    def test_dict_input(self):
        import matplotlib
        matplotlib.use("Agg")
        results = {
            "layer_0": {"accuracy": 0.55, "f1": 0.50},
            "layer_5": {"accuracy": 0.72, "f1": 0.68},
            "layer_11": {"accuracy": 0.91, "f1": 0.89},
        }
        fig = plot_probe_accuracy(results)
        assert fig is not None

    def test_layers_scores_format(self):
        import matplotlib
        matplotlib.use("Agg")
        results = {
            "layers": list(range(12)),
            "scores": np.linspace(0.5, 0.95, 12),
        }
        fig = plot_probe_accuracy(results)
        assert fig is not None

    def test_custom_metric(self):
        import matplotlib
        matplotlib.use("Agg")
        results = {
            "layer_0": {"accuracy": 0.55, "f1": 0.50},
            "layer_5": {"accuracy": 0.72, "f1": 0.68},
        }
        fig = plot_probe_accuracy(results, metric="f1")
        assert fig is not None


class TestCircuitGraph:

    def test_basic(self, sample_circuit_graph):
        import matplotlib
        matplotlib.use("Agg")
        fig = plot_circuit_graph(sample_circuit_graph)
        assert fig is not None

    def test_spring_layout(self, sample_circuit_graph):
        import matplotlib
        matplotlib.use("Agg")
        fig = plot_circuit_graph(sample_circuit_graph, layout="spring")
        assert fig is not None

    def test_circular_layout(self, sample_circuit_graph):
        import matplotlib
        matplotlib.use("Agg")
        fig = plot_circuit_graph(sample_circuit_graph, layout="circular")
        assert fig is not None

    def test_save(self, sample_circuit_graph, temp_dir):
        import matplotlib
        matplotlib.use("Agg")
        path = os.path.join(temp_dir, "circuit.png")
        fig = plot_circuit_graph(sample_circuit_graph, save_path=path)
        assert os.path.exists(path)


class TestSteeringEffects:

    def test_basic(self):
        import matplotlib
        matplotlib.use("Agg")
        effects = {
            "baseline": {"score": 0.2},
            "+love": {"score": 0.8},
            "+hate": {"score": -0.4},
        }
        fig = plot_steering_effects(effects)
        assert fig is not None

    def test_custom_metric(self):
        import matplotlib
        matplotlib.use("Agg")
        effects = {
            "baseline": {"score": 0.2, "perplexity": 15.3},
            "+love": {"score": 0.8, "perplexity": 18.1},
        }
        fig = plot_steering_effects(effects, metric="perplexity")
        assert fig is not None


# =============================================================================
# Interactive Plot Tests
# =============================================================================

class TestFeatureBrowser:

    def test_basic(self):
        activations = np.random.rand(50, 20)
        fig = feature_browser(activations)
        assert fig is not None

    def test_with_labels(self):
        activations = np.random.rand(50, 20)
        feature_labels = [f"feat_{i}" for i in range(50)]
        token_labels = [f"tok_{i}" for i in range(20)]
        fig = feature_browser(
            activations,
            feature_labels=feature_labels,
            token_labels=token_labels,
            top_k=5,
        )
        assert fig is not None

    def test_save_html(self, temp_dir):
        path = os.path.join(temp_dir, "features.html")
        fig = feature_browser(np.random.rand(20, 10), save_path=path)
        assert os.path.exists(path)


class TestInteractiveCircuitExplorer:

    def test_basic(self, sample_circuit_graph):
        fig = interactive_circuit_explorer(sample_circuit_graph)
        assert fig is not None

    def test_custom_title(self, sample_circuit_graph):
        fig = interactive_circuit_explorer(
            sample_circuit_graph, title="My Circuit",
        )
        assert fig is not None

    def test_save_html(self, sample_circuit_graph, temp_dir):
        path = os.path.join(temp_dir, "circuit.html")
        fig = interactive_circuit_explorer(
            sample_circuit_graph, save_path=path,
        )
        assert os.path.exists(path)


class TestEmbeddingScatter3d:

    def test_pca(self):
        embeddings = np.random.rand(100, 64)
        fig = embedding_scatter_3d(embeddings, method="pca")
        assert fig is not None

    def test_with_labels(self):
        embeddings = np.random.rand(100, 64)
        labels = [f"class_{i % 5}" for i in range(100)]
        fig = embedding_scatter_3d(embeddings, labels=labels, method="pca")
        assert fig is not None

    def test_with_color_values(self):
        embeddings = np.random.rand(100, 64)
        color = np.random.rand(100)
        fig = embedding_scatter_3d(
            embeddings, color_values=color, method="pca",
        )
        assert fig is not None

    def test_low_dim_input(self):
        embeddings = np.random.rand(50, 2)
        fig = embedding_scatter_3d(embeddings, method="pca")
        assert fig is not None

    def test_save_html(self, temp_dir):
        path = os.path.join(temp_dir, "embedding.html")
        embeddings = np.random.rand(50, 32)
        fig = embedding_scatter_3d(embeddings, method="pca", save_path=path)
        assert os.path.exists(path)

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="Unknown reduction method"):
            embedding_scatter_3d(np.random.rand(50, 32), method="invalid")
