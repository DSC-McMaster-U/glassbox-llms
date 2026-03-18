"""
Tests for the pipeline glue layer.

Uses mocks for heavy model loading to keep tests fast and portable.
"""

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch, PropertyMock

from glassboxllms.pipeline import (
    _to_torch,
    _to_numpy,
    extract_activations,
    train_sae_on_model,
    train_probe_on_model,
    discover_circuit,
)
from glassboxllms.features import SparseAutoencoder, SAETrainer, FeatureSet
from glassboxllms.analysis.circuits.graph import CircuitGraph
from glassboxllms.analysis.circuits.node import NodeType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HIDDEN_DIM = 64
SEQ_LEN = 8
BATCH_SIZE = 4


def _fake_activations(n_texts, seq_len=SEQ_LEN, hidden_dim=HIDDEN_DIM):
    """Return fake activation tensor of shape (n_texts, seq_len, hidden_dim)."""
    return torch.randn(n_texts, seq_len, hidden_dim)


def _mock_model(layer_names=None):
    """Create a mock ModelWrapper with sensible defaults."""
    if layer_names is None:
        layer_names = [f"layer.{i}" for i in range(4)]

    model = MagicMock()
    model.layer_names = layer_names
    model.model = MagicMock()

    def fake_get_activations(texts, layers, return_type="numpy"):
        n = len(texts) if isinstance(texts, list) else 1
        acts = {}
        for layer in layers:
            t = torch.randn(n, SEQ_LEN, HIDDEN_DIM)
            if return_type == "numpy":
                acts[layer] = t.numpy()
            else:
                acts[layer] = t
        return acts

    model.get_activations = MagicMock(side_effect=fake_get_activations)

    # Mock tokenizer
    model.tokenizer = MagicMock()
    model.tokenizer.return_value = {
        "input_ids": torch.randint(0, 1000, (1, SEQ_LEN)),
        "attention_mask": torch.ones(1, SEQ_LEN, dtype=torch.long),
    }

    # Mock get_layer_module
    def fake_get_layer_module(layer_name):
        module = MagicMock()
        module.register_forward_hook = MagicMock(return_value=MagicMock())
        return module

    model.get_layer_module = MagicMock(side_effect=fake_get_layer_module)

    return model


# ---------------------------------------------------------------------------
# Type conversion tests
# ---------------------------------------------------------------------------


class TestTypeConversions:
    def test_numpy_to_torch(self):
        arr = np.random.randn(3, 4).astype(np.float32)
        tensor = _to_torch(arr)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 4)
        np.testing.assert_allclose(tensor.numpy(), arr, atol=1e-6)

    def test_torch_to_torch(self):
        tensor = torch.randn(3, 4)
        result = _to_torch(tensor)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32

    def test_torch_to_numpy(self):
        tensor = torch.randn(3, 4)
        arr = _to_numpy(tensor)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3, 4)

    def test_numpy_to_numpy(self):
        arr = np.random.randn(3, 4)
        result = _to_numpy(arr)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# extract_activations tests
# ---------------------------------------------------------------------------


class TestExtractActivations:
    def test_returns_torch_by_default(self):
        model = _mock_model()
        result = extract_activations(
            "gpt2", ["hello"], ["layer.0"], return_type="torch", model=model
        )
        assert "layer.0" in result
        assert isinstance(result["layer.0"], torch.Tensor)

    def test_returns_numpy(self):
        model = _mock_model()
        result = extract_activations(
            "gpt2", ["hello"], ["layer.0"], return_type="numpy", model=model
        )
        assert isinstance(result["layer.0"], np.ndarray)

    def test_multiple_layers(self):
        model = _mock_model()
        layers = ["layer.0", "layer.1", "layer.2"]
        result = extract_activations(
            "gpt2", ["hello", "world"], layers, return_type="torch", model=model
        )
        assert len(result) == 3
        for layer in layers:
            assert layer in result
            assert result[layer].shape[0] == 2  # batch size


# ---------------------------------------------------------------------------
# train_sae_on_model tests
# ---------------------------------------------------------------------------


class TestTrainSaeOnModel:
    @patch("glassboxllms.pipeline._load_model")
    def test_returns_sae_and_feature_set(self, mock_load):
        model = _mock_model()
        mock_load.return_value = model

        sae, feature_set = train_sae_on_model(
            "gpt2",
            texts=["text1", "text2", "text3"],
            layer="layer.0",
            feature_dim=128,
            k=16,
            n_epochs=1,
            batch_size=8,
        )

        assert isinstance(sae, SparseAutoencoder)
        assert isinstance(feature_set, FeatureSet)
        assert sae.feature_dim == 128
        assert sae.k == 16
        assert len(feature_set) == 128

    @patch("glassboxllms.pipeline._load_model")
    def test_feature_set_has_metadata(self, mock_load):
        model = _mock_model()
        mock_load.return_value = model

        _, feature_set = train_sae_on_model(
            "gpt2",
            texts=["text1", "text2"],
            layer="layer.0",
            feature_dim=64,
            k=8,
            n_epochs=1,
        )

        assert feature_set.config["model_name"] == "gpt2"
        assert feature_set.config["layer"] == "layer.0"

    @patch("glassboxllms.pipeline._load_model")
    def test_passes_model_through(self, mock_load):
        model = _mock_model()

        sae, _ = train_sae_on_model(
            "gpt2",
            texts=["text1"],
            layer="layer.0",
            feature_dim=64,
            k=8,
            n_epochs=1,
            model=model,
        )

        # _load_model should NOT be called when model is passed
        mock_load.assert_not_called()


# ---------------------------------------------------------------------------
# train_probe_on_model tests
# ---------------------------------------------------------------------------


class TestTrainProbeOnModel:
    @patch("glassboxllms.pipeline._load_model")
    def test_returns_probe_and_direction(self, mock_load):
        model = _mock_model()
        mock_load.return_value = model

        probe, direction = train_probe_on_model(
            "gpt2",
            positive_texts=["good", "great", "excellent"],
            negative_texts=["bad", "terrible", "awful"],
            layer="layer.0",
        )

        assert probe.is_fitted
        assert isinstance(direction, np.ndarray)
        assert direction.shape == (HIDDEN_DIM,)

    @patch("glassboxllms.pipeline._load_model")
    def test_probe_type_cav(self, mock_load):
        model = _mock_model()
        mock_load.return_value = model

        probe, direction = train_probe_on_model(
            "gpt2",
            positive_texts=["good", "great"],
            negative_texts=["bad", "awful"],
            layer="layer.0",
            probe_type="cav",
        )

        assert probe.is_fitted
        assert probe.model_type == "cav"


# ---------------------------------------------------------------------------
# discover_circuit tests
# ---------------------------------------------------------------------------


class TestDiscoverCircuit:
    @patch("glassboxllms.pipeline._load_model")
    def test_returns_circuit_graph(self, mock_load):
        model = _mock_model(layer_names=["layer.0", "layer.1"])
        # Make model.model callable and return something with last_hidden_state
        output_mock = MagicMock()
        output_mock.last_hidden_state = torch.randn(1, SEQ_LEN, HIDDEN_DIM)
        model.model.return_value = output_mock
        mock_load.return_value = model

        def metric(output):
            return output[0, -1].norm()

        graph = discover_circuit(
            "gpt2",
            text="hello world",
            metric_fn=metric,
            layers=["layer.0", "layer.1"],
        )

        assert isinstance(graph, CircuitGraph)
        assert graph.model == "gpt2"
        assert graph.has_node("input")
        assert graph.has_node("output")

    @patch("glassboxllms.pipeline._load_model")
    def test_graph_has_layer_nodes(self, mock_load):
        layers = ["layer.0", "attn.1", "mlp.2"]
        model = _mock_model(layer_names=layers)
        output_mock = MagicMock()
        output_mock.last_hidden_state = torch.randn(1, SEQ_LEN, HIDDEN_DIM)
        model.model.return_value = output_mock
        mock_load.return_value = model

        def metric(output):
            return output[0, -1].norm()

        graph = discover_circuit(
            "gpt2",
            text="test",
            metric_fn=metric,
            layers=layers,
        )

        # Should have input + output + 3 layer nodes = 5
        assert len(graph) == 5

    @patch("glassboxllms.pipeline._load_model")
    def test_graph_metadata(self, mock_load):
        model = _mock_model(layer_names=["layer.0"])
        output_mock = MagicMock()
        output_mock.last_hidden_state = torch.randn(1, SEQ_LEN, HIDDEN_DIM)
        model.model.return_value = output_mock
        mock_load.return_value = model

        graph = discover_circuit(
            "gpt2",
            text="test input",
            metric_fn=lambda x: x[0, -1].norm(),
            layers=["layer.0"],
            strategy="zero",
        )

        assert graph.metadata["strategy"] == "zero"
        assert graph.metadata["text"] == "test input"
        assert "baseline_metric" in graph.metadata


# ---------------------------------------------------------------------------
# SAETrainer.get_feature_set tests
# ---------------------------------------------------------------------------


class TestTrainerGetFeatureSet:
    def test_get_feature_set_basic(self):
        input_dim = 32
        feature_dim = 64
        sae = SparseAutoencoder(input_dim=input_dim, feature_dim=feature_dim, k=8)

        data = torch.randn(100, input_dim)
        dataset = torch.utils.data.TensorDataset(data)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)

        trainer = SAETrainer(sae, dataloader)
        trainer.train(n_epochs=1)

        fs = trainer.get_feature_set(model_name="test-model", layer="layer.5")

        assert isinstance(fs, FeatureSet)
        assert len(fs) == feature_dim
        assert fs.config["model_name"] == "test-model"
        assert fs.config["layer"] == "layer.5"
        assert fs.W_dec.shape == (feature_dim, input_dim)
        assert fs.W_enc.shape == (input_dim, feature_dim)

    def test_get_feature_set_has_stats(self):
        sae = SparseAutoencoder(input_dim=16, feature_dim=32, k=4)
        data = torch.randn(50, 16)
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data), batch_size=16
        )

        trainer = SAETrainer(sae, dataloader)
        trainer.train(n_epochs=1)

        fs = trainer.get_feature_set()
        assert "final_explained_variance" in fs.stats
        assert "final_mse" in fs.stats
