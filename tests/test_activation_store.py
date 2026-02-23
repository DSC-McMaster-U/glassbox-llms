import pytest
import torch
import os
import shutil
from tests.test_activation_store import ActivationStore, HAS_SAFETENSORS

TEST_DIR = "./test_activations"
LAYER_NAME = "transformer.layer.0.mlp"

@pytest.fixture(autouse=True)
def cleanup_storage():
    """Ensures the test directory is clean before and after each test."""
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    yield
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)

@pytest.fixture
def store():
    """Initializes the ActivationStore with a small buffer for testing flushes."""
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
    # Buffer size is 5. We trigger a flush by saving 5 items.
    for i in range(5):
        store.save(LAYER_NAME, torch.full((1, 5), float(i)))

    # Buffer should be cleared, manifest should have 1 entry
    assert len(store._buffer[LAYER_NAME]) == 0
    assert len(store._disk_manifest[LAYER_NAME]) == 1

    # Check if file exists on disk
    filename = store._disk_manifest[LAYER_NAME][0]
    assert os.path.exists(filename)

    # Retrieve all (disk + potential ram)
    all_acts = store.get_all(LAYER_NAME)
    assert all_acts.shape == (5, 1, 5)

def test_get_by_token(store):
    # !! metadata indexing only works for RAM in current implementation !!
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

    # Mock PyTorch Module call
    mock_module = None
    mock_input = None
    mock_output = torch.randn(1, 128)

    hook(mock_module, mock_input, mock_output)

    assert len(store._buffer[LAYER_NAME]) == 1
    assert store.get_all(LAYER_NAME).shape == (1, 1, 128)

def test_clear(store):
    store.save(LAYER_NAME, torch.randn(1, 5))
    store.clear()
    assert len(store._buffer) == 0
    assert len(store._metadata_index) == 0

def test_empty_retrieval(store):
    # Retrieving non-existent layer should return empty tensor
    res = store.get_all("non_existent")
    assert isinstance(res, torch.Tensor)
    assert res.numel() == 0

def test_persistence_method(store):
    store.save(LAYER_NAME, torch.randn(1, 5))
    save_path = "state.pt"
    store.persist_to_disk(save_path)

    full_path = os.path.join(TEST_DIR, save_path)
    assert os.path.exists(full_path)

    loaded = torch.load(full_path)
    assert "data" in loaded
    assert LAYER_NAME in loaded["data"]

def test_string_representation(store):
    store.save("layer_a", torch.randn(1, 5))
    # Test that __str__ and __repr__ don't crash and contain info
    assert "layer_a" in str(store)
    assert "ActivationStore" in repr(store)
