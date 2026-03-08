"""
Tests for SAEExperiment class.

Tests the full pipeline: activation collection, SAE training, feature extraction,
and Atlas registration.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from glassboxllms.experiments import SAEExperiment
from glassboxllms.analysis.feature_atlas import Atlas, FeatureType


class DummyModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, hidden_size=64):
        super().__init__()
        self.embedding = nn.Embedding(1000, hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.output = nn.Linear(hidden_size, 1000)
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.mlp(x)
        return self.output(x)


@pytest.fixture
def dummy_model():
    """Create a dummy model for testing."""
    return DummyModel(hidden_size=64)


@pytest.fixture
def dummy_dataloader():
    """Create a dummy dataloader."""
    # Generate random token IDs
    data = torch.randint(0, 1000, (100, 10))  # 100 sequences of length 10
    dataset = TensorDataset(data)
    return DataLoader(dataset, batch_size=8)


def test_sae_experiment_initialization(dummy_model):
    """Test SAEExperiment initialization."""
    experiment = SAEExperiment(
        model=dummy_model,
        layer="mlp.0",  # First layer of MLP
        sparsity_alpha=0.1,
        d_sae=512,
        k=32,
        device="cpu"
    )
    
    assert experiment.model == dummy_model
    assert experiment.layer == "mlp.0"
    assert experiment.d_sae == 512
    assert experiment.k == 32
    assert not experiment.trained
    assert experiment.input_dim == 256  # mlp.0 out_features = hidden_size * 4 = 64 * 4


def test_activation_collection(dummy_model, dummy_dataloader):
    """Test activation collection from model."""
    experiment = SAEExperiment(
        model=dummy_model,
        layer="mlp.2",  # Output of MLP
        d_sae=256,
        k=16,
        device="cpu"
    )
    
    activations = experiment.collect_activations(
        dummy_dataloader,
        num_samples=50,
        pooling="mean"
    )
    
    assert activations.shape[0] == 50
    assert activations.shape[1] == 64  # hidden_size
    assert experiment.activations_collected


def test_sae_training_small():
    """Test SAE training on synthetic data."""
    # Create synthetic activations
    n_samples = 500
    input_dim = 32
    activations = torch.randn(n_samples, input_dim)
    
    # Create experiment with small SAE
    experiment = SAEExperiment(
        model=DummyModel(hidden_size=input_dim),
        layer="mlp.0",
        d_sae=128,
        k=8,
        sparsity_mode="topk",
        device="cpu"
    )
    
    # Manually set input_dim to match synthetic data
    experiment.input_dim = input_dim
    experiment.sae = experiment.sae.__class__(
        input_dim=input_dim,
        feature_dim=128,
        k=8,
        sparsity_mode="topk",
        device="cpu"
    )
    
    # Train
    stats = experiment.train(
        activations,
        n_epochs=3,
        batch_size=32,
        learning_rate=1e-3
    )
    
    assert experiment.trained
    assert "final_explained_variance" in stats
    assert "mean_l0" in stats
    assert stats["final_explained_variance"] >= 0.0


def test_validation_criteria():
    """Test success criteria validation."""
    n_samples = 500
    input_dim = 32
    activations = torch.randn(n_samples, input_dim)
    
    experiment = SAEExperiment(
        model=DummyModel(hidden_size=input_dim),
        layer="mlp.0",
        d_sae=128,
        k=8,
        device="cpu"
    )
    
    experiment.input_dim = input_dim
    experiment.sae = experiment.sae.__class__(
        input_dim=input_dim,
        feature_dim=128,
        k=8,
        sparsity_mode="topk",
        device="cpu"
    )
    
    # Train
    experiment.train(activations, n_epochs=2, batch_size=32)
    
    # Validate
    criteria = experiment.validate_training()
    
    assert "high_reconstruction" in criteria
    assert "sparse_activations" in criteria
    assert "low_dead_features" in criteria
    assert isinstance(criteria["high_reconstruction"], bool)


def test_feature_extraction():
    """Test feature extraction to Atlas Feature objects."""
    n_samples = 500
    input_dim = 32
    activations = torch.randn(n_samples, input_dim)
    
    experiment = SAEExperiment(
        model=DummyModel(hidden_size=input_dim),
        layer="mlp.0",
        d_sae=64,
        k=8,
        model_name="test_model",
        device="cpu"
    )
    
    experiment.input_dim = input_dim
    experiment.sae = experiment.sae.__class__(
        input_dim=input_dim,
        feature_dim=64,
        k=8,
        sparsity_mode="topk",
        device="cpu"
    )
    
    # Train
    experiment.train(activations, n_epochs=2, batch_size=32)
    
    # Extract features
    features = experiment.extract_features(skip_dead=True, dataset_name="test_data")
    
    assert len(features) > 0
    assert all(f.feature_type == FeatureType.SAE_LATENT for f in features)
    assert all(f.location.model_name == "test_model" for f in features)
    assert all(f.location.layer == "mlp.0" for f in features)
    assert all("decoder_norm" in f.metadata for f in features)


def test_atlas_registration():
    """Test feature registration to Atlas."""
    n_samples = 500
    input_dim = 32
    activations = torch.randn(n_samples, input_dim)
    
    experiment = SAEExperiment(
        model=DummyModel(hidden_size=input_dim),
        layer="mlp.0",
        d_sae=64,
        k=8,
        model_name="test_model",
        device="cpu"
    )
    
    experiment.input_dim = input_dim
    experiment.sae = experiment.sae.__class__(
        input_dim=input_dim,
        feature_dim=64,
        k=8,
        sparsity_mode="topk",
        device="cpu"
    )
    
    # Train
    experiment.train(activations, n_epochs=2, batch_size=32)
    
    # Register to new Atlas
    atlas = experiment.register_features(atlas_name="test_atlas", dataset_name="test_data")
    
    assert isinstance(atlas, Atlas)
    assert len(atlas) > 0
    assert atlas.metadata.name == "test_atlas"
    
    # Check features are searchable
    layer_features = atlas.find_by_layer("mlp.0")
    assert len(layer_features) > 0
    
    sae_features = atlas.find_by_type(FeatureType.SAE_LATENT)
    assert len(sae_features) > 0


def test_full_pipeline_integration(dummy_model, dummy_dataloader):
    """Integration test of full pipeline."""
    experiment = SAEExperiment(
        model=dummy_model,
        layer="mlp.2",
        d_sae=128,
        k=16,
        sparsity_alpha=0.1,
        model_name="dummy_model",
        device="cpu"
    )
    
    # Step 1: Collect activations
    activations = experiment.collect_activations(
        dummy_dataloader,
        num_samples=100,
        pooling="mean"
    )
    assert activations.shape[0] == 100
    
    # Step 2: Train SAE
    stats = experiment.train(
        activations,
        n_epochs=2,
        batch_size=16
    )
    assert experiment.trained
    assert stats["final_explained_variance"] >= 0.0
    
    # Step 3: Validate
    criteria = experiment.validate_training()
    assert "high_reconstruction" in criteria
    
    # Step 4: Register features
    atlas = experiment.register_features(
        atlas_name="integration_test",
        dataset_name="dummy_data"
    )
    assert len(atlas) > 0
    
    # Step 5: Get stats
    full_stats = experiment.get_stats()
    assert full_stats["trained"]
    assert "stats" in full_stats
    assert "validation" in full_stats


def test_checkpoint_save_load(tmp_path):
    """Test checkpoint saving."""
    n_samples = 500
    input_dim = 32
    activations = torch.randn(n_samples, input_dim)
    
    experiment = SAEExperiment(
        model=DummyModel(hidden_size=input_dim),
        layer="mlp.0",
        d_sae=64,
        k=8,
        device="cpu"
    )
    
    experiment.input_dim = input_dim
    experiment.sae = experiment.sae.__class__(
        input_dim=input_dim,
        feature_dim=64,
        k=8,
        sparsity_mode="topk",
        device="cpu"
    )
    
    # Train and save
    experiment.train(activations, n_epochs=2, batch_size=32)
    checkpoint_path = tmp_path / "test_checkpoint.pt"
    experiment.save_checkpoint(checkpoint_path)
    
    assert checkpoint_path.exists()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    assert "sae_state_dict" in checkpoint
    assert "config" in checkpoint
    assert "stats" in checkpoint


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
