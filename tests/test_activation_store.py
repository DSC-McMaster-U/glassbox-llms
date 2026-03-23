import os
import shutil

import pytest
import torch

from glassboxllms.instrumentation.activations import ActivationStore


TEST_DIR = "./test_activations"
LAYER_NAME = "transformer.layer.0.mlp"


@pytest.fixture(autouse=True)
def cleanup_storage():
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    yield
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)


@pytest.fixture
def store():
    return ActivationStore(device="cpu", storage_dir=TEST_DIR, buffer_size=5)


def test_initialization(store):
    assert store.device == "cpu"
    assert os.path.exists(TEST_DIR)
    assert len(store._buffer) == 0


def test_save_and_retrieve_ram(store):
    data = torch.randn(1, 10)
    store.save(LAYER_NAME, data)

    retrieved = store.get_all(LAYER_NAME)
    assert retrieved.shape == (1, 1, 10)
    assert torch.equal(retrieved[0], data)


def test_buffer_flush_to_disk(store):
    for i in range(5):
        store.save(LAYER_NAME, torch.full((1, 5), float(i)))

    assert len(store._buffer[LAYER_NAME]) == 0
    assert len(store._disk_manifest[LAYER_NAME]) == 1

    filename = store._disk_manifest[LAYER_NAME][0]
    assert os.path.exists(filename)

    all_acts = store.get_all(LAYER_NAME)
    assert all_acts.shape == (5, 1, 5)


def test_get_by_token(store):
    data1 = torch.randn(1, 10)
    data2 = torch.randn(1, 10)

    store.save(LAYER_NAME, data1, token_idx=10)
    store.save(LAYER_NAME, data2, token_idx=10)
    store.save(LAYER_NAME, torch.randn(1, 10), token_idx=20)

    token_10_acts = store.get_by_token(LAYER_NAME, 10)
    assert token_10_acts.shape == (2, 1, 10)
    assert torch.equal(token_10_acts[0], data1)


def test_hook_functionality(store):
    hook = store.create_hook(LAYER_NAME, token_idx=None)
    mock_output = torch.randn(1, 128)

    hook(None, None, mock_output)

    assert len(store._buffer[LAYER_NAME]) == 1
    assert store.get_all(LAYER_NAME).shape == (1, 1, 128)


def test_clear(store):
    store.save(LAYER_NAME, torch.randn(1, 5), token_idx=2)
    store.clear()
    assert len(store._buffer) == 0
    assert len(store._token_index) == 0


def test_empty_retrieval(store):
    res = store.get_all("non_existent")
    assert isinstance(res, torch.Tensor)
    assert res.numel() == 0


def test_persist_method(store):
    store.save(LAYER_NAME, torch.randn(1, 5))
    save_name = "state.pt"
    full_path = store.persist(save_name)

    assert os.path.exists(full_path)

    loaded = torch.load(full_path, weights_only=True)
    assert "data" in loaded
    assert LAYER_NAME in loaded["data"]


def test_string_representation(store):
    store.save("layer_a", torch.randn(1, 5))
    assert "layer_a" in str(store)
    assert "ActivationStore" in repr(store)

