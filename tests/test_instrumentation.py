"""
Tests for the instrumentation package:
  - HookManager
  - ActivationStore
  - ActivationExtractor
  - patch_activation
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import os


# ── Fixtures ─────────────────────────────────────────────────────


class TinyModel(nn.Module):
    """Minimal two-layer model for testing hooks."""

    def __init__(self):
        super().__init__()
        self.layer_a = nn.Linear(8, 8)
        self.layer_b = nn.Linear(8, 4)

    def forward(self, x):
        x = self.layer_a(x)
        x = self.layer_b(x)
        return x


class TransformerLikeModel(nn.Module):
    """Model that accepts ``input_ids`` kwarg (like HuggingFace models)."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(8, 8)
        self.block = nn.Linear(8, 8)
        self.head = nn.Linear(8, 4)

    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids.float())
        x = self.block(x)
        x = self.head(x)
        return x


@pytest.fixture
def model():
    return TinyModel()


@pytest.fixture
def hf_model():
    return TransformerLikeModel()


@pytest.fixture
def sample_input():
    return torch.randn(2, 8)


# ── HookManager ─────────────────────────────────────────────────


class TestHookManager:
    def test_attach_and_capture(self, model, sample_input):
        from glassboxllms.instrumentation import HookManager

        mgr = HookManager(model)
        mgr.attach_hook("layer_a")

        model(sample_input)

        assert "layer_a" in mgr.activations
        assert mgr.activations["layer_a"].shape == (2, 8)

        mgr.remove_hooks()

    def test_get_returns_none_for_missing(self, model):
        from glassboxllms.instrumentation import HookManager

        mgr = HookManager(model)
        assert mgr.get("nonexistent") is None

    def test_custom_hook(self, model, sample_input):
        from glassboxllms.instrumentation import HookManager

        captured = {}

        def my_hook(module, inp, output):
            captured["out"] = output.detach()

        mgr = HookManager(model)
        mgr.attach_hook("layer_b", hook_fn=my_hook)

        model(sample_input)

        assert "out" in captured
        assert captured["out"].shape == (2, 4)

        mgr.remove_hooks()

    def test_invalid_layer_raises(self, model):
        from glassboxllms.instrumentation import HookManager

        mgr = HookManager(model)
        with pytest.raises(ValueError, match="not found"):
            mgr.attach_hook("nonexistent_layer")

    def test_add_hook_alias(self, model, sample_input):
        from glassboxllms.instrumentation import HookManager

        called = {"count": 0}

        def hook(module, inp, output):
            called["count"] += 1

        mgr = HookManager(model)
        mgr.add_hook("layer_a", hook)
        model(sample_input)

        assert called["count"] == 1
        mgr.remove_all_hooks()

    def test_clear_activations(self, model, sample_input):
        from glassboxllms.instrumentation import HookManager

        mgr = HookManager(model)
        mgr.attach_hook("layer_a")
        model(sample_input)
        assert len(mgr.activations) == 1

        mgr.clear_activations()
        assert len(mgr.activations) == 0
        mgr.remove_hooks()


# ── ActivationStore ──────────────────────────────────────────────


class TestActivationStore:
    def test_save_and_get_all(self):
        from glassboxllms.instrumentation import ActivationStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = ActivationStore(storage_dir=tmpdir, buffer_size=100)
            t1 = torch.randn(1, 16)
            t2 = torch.randn(1, 16)

            store.save("layer.0", t1)
            store.save("layer.0", t2)

            result = store.get_all("layer.0")
            assert result.shape == (2, 1, 16)

    def test_buffer_flush(self):
        from glassboxllms.instrumentation import ActivationStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = ActivationStore(storage_dir=tmpdir, buffer_size=3)

            for _ in range(5):
                store.save("layer.0", torch.randn(1, 8))

            # 3 flushed to disk, 2 in buffer
            assert len(store._disk_manifest["layer.0"]) == 1
            assert len(store._buffer["layer.0"]) == 2

            result = store.get_all("layer.0")
            assert result.shape[0] == 5

    def test_create_hook(self, model, sample_input):
        from glassboxllms.instrumentation import ActivationStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = ActivationStore(storage_dir=tmpdir)
            hook = store.create_hook("layer_a")
            handle = model.layer_a.register_forward_hook(hook)

            model(sample_input)
            handle.remove()

            result = store.get_all("layer_a")
            assert result.shape[0] == 1  # one forward pass
            assert result.shape[-1] == 8  # hidden dim

    def test_get_by_token(self):
        from glassboxllms.instrumentation import ActivationStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = ActivationStore(storage_dir=tmpdir)
            store.save("layer.0", torch.randn(1, 8), token_idx=3)
            store.save("layer.0", torch.randn(1, 8), token_idx=5)

            tok3 = store.get_by_token("layer.0", 3)
            assert tok3.shape[0] == 1

            tok5 = store.get_by_token("layer.0", 5)
            assert tok5.shape[0] == 1

            missing = store.get_by_token("layer.0", 99)
            assert missing.numel() == 0

    def test_layer_names_property(self):
        from glassboxllms.instrumentation import ActivationStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = ActivationStore(storage_dir=tmpdir)
            store.save("layer.0", torch.randn(1, 8))
            store.save("layer.5", torch.randn(1, 8))

            assert store.layer_names == ["layer.0", "layer.5"]

    def test_repr_and_str(self):
        from glassboxllms.instrumentation import ActivationStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = ActivationStore(storage_dir=tmpdir)
            assert "ActivationStore" in repr(store)
            assert "empty" in str(store)

            store.save("layer.0", torch.randn(1, 8))
            assert "layer.0" in str(store)


# ── ActivationExtractor ─────────────────────────────────────────


class TestActivationExtractor:
    def test_extract_from_tensors(self, hf_model, sample_input):
        from glassboxllms.instrumentation import ActivationExtractor

        extractor = ActivationExtractor(hf_model)
        result = extractor.extract_from_tensors(
            input_ids=sample_input,
            layers=["block"],
            pooling="none",
            return_type="torch",
        )

        assert "block" in result
        assert result["block"].shape == (2, 8)

    def test_invalid_layer_raises(self, hf_model, sample_input):
        from glassboxllms.instrumentation import ActivationExtractor

        extractor = ActivationExtractor(hf_model)
        with pytest.raises(ValueError, match="not found"):
            extractor.extract_from_tensors(
                input_ids=sample_input,
                layers=["nonexistent"],
                pooling="none",
            )

    def test_list_layers(self, model):
        from glassboxllms.instrumentation import ActivationExtractor

        extractor = ActivationExtractor(model)
        layers = extractor.list_layers()
        assert "layer_a" in layers
        assert "layer_b" in layers

    def test_list_layers_with_pattern(self, model):
        from glassboxllms.instrumentation import ActivationExtractor

        extractor = ActivationExtractor(model)
        layers = extractor.list_layers(pattern="layer_b")
        assert layers == ["layer_b"]

    def test_with_backing_store(self, hf_model, sample_input):
        from glassboxllms.instrumentation import ActivationExtractor, ActivationStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = ActivationStore(storage_dir=tmpdir)
            extractor = ActivationExtractor(hf_model, store=store)

            extractor.extract_from_tensors(
                input_ids=sample_input,
                layers=["block"],
                pooling="none",
                return_type="torch",
            )

            # Store should also have the activations
            stored = store.get_all("block")
            assert stored.shape[0] == 1  # one forward pass


# ── patch_activation ─────────────────────────────────────────────


class TestPatchActivation:
    def test_patch_replaces_output(self, model, sample_input):
        from glassboxllms.instrumentation import patch_activation

        replacement = torch.ones(2, 8) * 42.0
        out = patch_activation(model, "layer_a", replacement, sample_input)

        # The output should reflect the patched layer_a
        assert out is not None

    def test_patch_with_dict_input(self, model):
        from glassboxllms.instrumentation import patch_activation

        x = torch.randn(2, 8)
        replacement = torch.zeros(2, 8)

        # patch_activation accepts raw tensors too
        out = patch_activation(model, "layer_a", replacement, x)
        assert out is not None

    def test_patch_invalid_layer(self, model, sample_input):
        from glassboxllms.instrumentation import patch_activation

        with pytest.raises(ValueError, match="not found"):
            patch_activation(
                model, "nonexistent", torch.zeros(2, 8), sample_input
            )


# ── get_layer_names ──────────────────────────────────────────────


class TestGetLayerNames:
    def test_all_layers(self, model):
        from glassboxllms.instrumentation import get_layer_names

        names = get_layer_names(model, "all")
        assert "layer_a" in names
        assert "layer_b" in names

    def test_invalid_type_raises(self, model):
        from glassboxllms.instrumentation import get_layer_names

        with pytest.raises(ValueError):
            get_layer_names(model, "invalid_type")
