"""
Unit tests for Sparse Autoencoder module.

Tests cover:
- Initialization (weight shapes, tied weights, geometric median)
- Forward pass (TopK, L1, reconstruction)
- Decoder normalization
- Geometric median computation
- FeatureSet serialization
- SAEFeature dataclass
"""

import pytest
import torch
import torch.nn.functional as F
import os
import tempfile
import shutil

from glassboxllms.features import (
    SparseAutoencoder,
    SAETrainer,
    SAEFeature,
    FeatureSet,
    geometric_median,
    calc_explained_variance,
    calc_mean_l0,
    unit_norm_decoder,
    topk_activation,
)


class TestSparseAutoencoder:
    """Tests for the SparseAutoencoder class."""

    def test_initialization(self):
        """Test that SAE initializes with correct shapes."""
        sae = SparseAutoencoder(input_dim=768, feature_dim=16384, k=128)

        assert sae.input_dim == 768
        assert sae.feature_dim == 16384
        assert sae.k == 128

        # Check weight shapes
        assert sae.W_enc.shape == (768, 16384)
        assert sae.W_dec.shape == (16384, 768)
        assert sae.b_enc.shape == (16384,)
        assert sae.b_dec.shape == (768,)

    def test_tied_weights_initialization(self):
        """Test that tied weights are correctly initialized."""
        sae = SparseAutoencoder(input_dim=100, feature_dim=200, tied_weights=True)

        # Check that encoder is roughly transpose of decoder (accounting for normalization)
        W_enc_from_dec = sae.W_dec.T.clone()

        # They should be close (though not exact due to normalization)
        similarity = F.cosine_similarity(
            sae.W_enc.flatten(),
            W_enc_from_dec.flatten(),
            dim=0
        )
        assert similarity > 0.99

    def test_decoder_unit_norm(self):
        """Test that decoder columns are initialized with unit norm."""
        sae = SparseAutoencoder(input_dim=100, feature_dim=200)

        norms = torch.norm(sae.W_dec, p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shapes."""
        sae = SparseAutoencoder(input_dim=768, feature_dim=16384, k=128)
        batch = torch.randn(32, 768)

        reconstructed, features, pre_acts = sae(batch)

        assert reconstructed.shape == (32, 768)
        assert features.shape == (32, 16384)
        assert pre_acts.shape == (32, 16384)

    def test_topk_sparsity(self):
        """Test that TopK mode produces exactly k active features per sample."""
        sae = SparseAutoencoder(input_dim=100, feature_dim=200, k=10, sparsity_mode="topk")
        batch = torch.randn(16, 100)

        _, features, _ = sae(batch)

        # Count active features per sample
        active_counts = (features > 0).sum(dim=1)

        # Should be exactly k (or fewer if ReLU zeros out negative values)
        assert (active_counts <= 10).all()

    def test_l1_mode(self):
        """Test that L1 mode produces ReLU activations."""
        sae = SparseAutoencoder(input_dim=100, feature_dim=200, sparsity_mode="l1")
        batch = torch.randn(16, 100)

        _, features, pre_acts = sae(batch)

        # Features should be non-negative (ReLU applied)
        assert (features >= 0).all()

        # Should match manual ReLU
        expected = torch.relu(pre_acts)
        assert torch.allclose(features, expected)

    def test_reconstruction_3d_input(self):
        """Test that 3D input (batch, seq, hidden) is handled correctly."""
        sae = SparseAutoencoder(input_dim=768, feature_dim=2000, k=64)
        batch = torch.randn(4, 128, 768)  # (batch, seq, hidden)

        reconstructed, features, _ = sae(batch)

        # Should restore original shape
        assert reconstructed.shape == (4, 128, 768)
        assert features.shape == (4 * 128, 2000)

    def test_normalize_decoder(self):
        """Test that normalize_decoder maintains unit norm."""
        sae = SparseAutoencoder(input_dim=100, feature_dim=200)

        # Perturb decoder weights
        with torch.no_grad():
            sae.W_dec.data *= 2.0

        norms_before = torch.norm(sae.W_dec, p=2, dim=1)
        assert not torch.allclose(norms_before, torch.ones_like(norms_before))

        # Normalize
        sae.normalize_decoder()

        norms_after = torch.norm(sae.W_dec, p=2, dim=1)
        assert torch.allclose(norms_after, torch.ones_like(norms_after), atol=1e-6)

    def test_get_decoder_norms(self):
        """Test get_decoder_norms returns correct values."""
        sae = SparseAutoencoder(input_dim=100, feature_dim=200)

        norms = sae.get_decoder_norms()
        assert norms.shape == (200,)
        assert torch.allclose(norms, torch.ones(200), atol=1e-6)

    def test_encode_decode(self):
        """Test encode and decode methods."""
        sae = SparseAutoencoder(input_dim=100, feature_dim=200, k=10)
        batch = torch.randn(16, 100)

        # Encode
        features = sae.encode(batch)
        assert features.shape == (16, 200)

        # Decode
        reconstructed = sae.decode(features)
        assert reconstructed.shape == (16, 100)

    def test_config_roundtrip(self):
        """Test that config serialization/deserialization works."""
        original = SparseAutoencoder(
            input_dim=768,
            feature_dim=16384,
            k=128,
            sparsity_mode="topk",
            sparsity_coef=1e-3,
            tied_weights=True,
        )

        config = original.get_config()
        restored = SparseAutoencoder.from_config(config)

        assert restored.input_dim == original.input_dim
        assert restored.feature_dim == original.feature_dim
        assert restored.k == original.k
        assert restored.sparsity_mode == original.sparsity_mode
        assert restored.sparsity_coef == original.sparsity_coef
        assert restored.tied_weights == original.tied_weights

    def test_save_load(self):
        """Test model save and load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original = SparseAutoencoder(input_dim=100, feature_dim=200, k=10)
            original.b_dec.data = torch.randn(100)  # Set non-zero bias

            path = os.path.join(tmpdir, "sae.pt")
            original.save(path)

            loaded = SparseAutoencoder.load(path)

            # Check weights match
            assert torch.allclose(loaded.W_dec, original.W_dec)
            assert torch.allclose(loaded.b_dec, original.b_dec)


class TestGeometricMedian:
    """Tests for geometric median computation."""

    def test_convergence_2d(self):
        """Test that geometric median converges on 2D data."""
        # Create simple 2D data centered at (3, 4)
        X = torch.tensor([[3.0, 4.0], [3.1, 4.1], [2.9, 3.9], [3.0, 4.0]])

        median = geometric_median(X)

        # Should be close to the center
        assert median.shape == (2,)
        assert 2.8 < median[0] < 3.2
        assert 3.8 < median[1] < 4.2

    def test_robust_to_outliers(self):
        """Test that geometric median is robust to outliers (vs mean)."""
        # Data with one outlier
        X = torch.cat([
            torch.randn(100, 10),  # Normal data centered at 0
            torch.ones(1, 10) * 100,  # One outlier at 100
        ])

        median = geometric_median(X)
        mean = X.mean(dim=0)

        # Median should be closer to origin than mean
        median_norm = torch.norm(median, p=2)
        mean_norm = torch.norm(mean, p=2)

        assert median_norm < mean_norm

    def test_high_dimensional(self):
        """Test on high-dimensional data."""
        X = torch.randn(1000, 768)

        median = geometric_median(X, max_iter=50)

        assert median.shape == (768,)
        assert not torch.isnan(median).any()

    def test_initialize_sae_bias(self):
        """Test using geometric median to initialize SAE bias."""
        sae = SparseAutoencoder(input_dim=100, feature_dim=200)

        # Create data centered at 5.0
        sample = torch.ones(1000, 100) * 5.0 + torch.randn(1000, 100) * 0.1

        sae.initialize_geometric_median(sample)

        # Bias should be close to 5.0
        assert 4.5 < sae.b_dec.mean() < 5.5


class TestUtilities:
    """Tests for utility functions."""

    def test_calc_explained_variance(self):
        """Test explained variance calculation."""
        # Perfect reconstruction
        original = torch.randn(100, 50)
        perfect = original.clone()

        ev_perfect = calc_explained_variance(original, perfect)
        assert abs(ev_perfect - 1.0) < 1e-5

        # Zero reconstruction
        zero = torch.zeros_like(original)
        ev_zero = calc_explained_variance(original, zero)
        assert ev_zero < 0.01  # Should be near 0

        # Noisy reconstruction
        noisy = original + torch.randn_like(original) * 0.1
        ev_noisy = calc_explained_variance(original, noisy)
        assert 0 < ev_noisy < 1

    def test_calc_mean_l0(self):
        """Test L0 norm calculation."""
        # Create features with known sparsity
        features = torch.zeros(100, 50)
        features[:50, :10] = 1.0  # First 50 samples, 10 features active

        mean_l0 = calc_mean_l0(features)
        assert mean_l0 == 5.0  # Average of 10 for first 50, 0 for rest

    def test_unit_norm_decoder(self):
        """Test decoder normalization."""
        W = torch.randn(100, 50)

        W_norm = unit_norm_decoder(W)

        norms = torch.norm(W_norm, p=2, dim=1)
        assert torch.allclose(norms, torch.ones(100), atol=1e-6)

    def test_topk_activation(self):
        """Test TopK activation function."""
        z = torch.randn(16, 100)

        sparse = topk_activation(z, k=10)

        # Should have exactly 10 non-zero entries per row
        active_counts = (sparse > 0).sum(dim=1)
        assert (active_counts == 10).all()

        # Values should match top 10 from input
        for i in range(16):
            top_10_vals = torch.topk(z[i], k=10).values
            active_vals = sparse[i][sparse[i] > 0]
            assert torch.allclose(active_vals.sort().values, top_10_vals.sort().values)


class TestSAEFeature:
    """Tests for the SAEFeature dataclass."""

    def test_initialization(self):
        """Test SAEFeature initialization."""
        decoder = torch.randn(768)

        feature = SAEFeature(
            id=42,
            layer=6,
            model_name="gpt2",
            decoder_vector=decoder,
        )

        assert feature.id == 42
        assert feature.layer == 6
        assert feature.model_name == "gpt2"
        assert torch.equal(feature.decoder_vector, decoder)

    def test_similarity(self):
        """Test cosine similarity between features."""
        f1 = SAEFeature(
            id=0,
            layer=0,
            model_name="test",
            decoder_vector=torch.tensor([1.0, 0.0, 0.0]),
        )
        f2 = SAEFeature(
            id=1,
            layer=0,
            model_name="test",
            decoder_vector=torch.tensor([0.0, 1.0, 0.0]),
        )
        f3 = SAEFeature(
            id=2,
            layer=0,
            model_name="test",
            decoder_vector=torch.tensor([1.0, 0.0, 0.0]),
        )

        # Orthogonal features
        assert abs(f1.similarity(f2)) < 0.01

        # Identical features
        assert abs(f1.similarity(f3) - 1.0) < 0.01

    def test_activation_on(self):
        """Test computing feature activation."""
        decoder = torch.tensor([1.0, 2.0, 3.0])
        encoder = torch.tensor([0.5, 1.0, 1.5])

        feature = SAEFeature(
            id=0,
            layer=0,
            model_name="test",
            decoder_vector=decoder,
            encoder_vector=encoder,
        )

        activation = torch.tensor([1.0, 1.0, 1.0])
        result = feature.activation_on(activation)

        # Should use encoder: dot([1,1,1], [0.5, 1.0, 1.5]) = 3.0
        expected = 3.0
        assert abs(result - expected) < 0.01

    def test_to_device(self):
        """Test moving feature to device."""
        feature = SAEFeature(
            id=0,
            layer=0,
            model_name="test",
            decoder_vector=torch.randn(768),
        )

        # This is a no-op on CPU but tests the method
        feature.to("cpu")
        assert feature.decoder_vector.device.type == "cpu"

    def test_numpy_roundtrip(self):
        """Test conversion to/from numpy format."""
        original = SAEFeature(
            id=5,
            layer=3,
            model_name="gpt2",
            decoder_vector=torch.randn(768),
            encoder_vector=torch.randn(768),
            activation_stats={"sparsity": 0.001, "max_activation": 5.0},
            metadata={"custom": "value"},
        )

        numpy_data = original.as_numpy()
        restored = SAEFeature.from_numpy(numpy_data)

        assert restored.id == original.id
        assert restored.layer == original.layer
        assert restored.model_name == original.model_name
        assert torch.allclose(restored.decoder_vector, original.decoder_vector)
        assert torch.allclose(restored.encoder_vector, original.encoder_vector)
        assert restored.activation_stats == original.activation_stats
        assert restored.metadata == original.metadata


class TestFeatureSet:
    """Tests for the FeatureSet class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir)

    def test_initialization(self):
        """Test FeatureSet initialization."""
        W_dec = torch.randn(100, 50)

        feature_set = FeatureSet(
            config={"model_name": "gpt2", "layer": 6, "k": 10},
            stats={"explained_variance": 0.85},
            W_dec=W_dec,
        )

        assert len(feature_set) == 100
        assert feature_set.config["model_name"] == "gpt2"

    def test_get_feature(self):
        """Test retrieving a single feature."""
        W_dec = torch.randn(100, 50)

        feature_set = FeatureSet(
            config={"model_name": "gpt2", "layer": 6},
            stats={},
            W_dec=W_dec,
        )

        feature = feature_set.get(0)

        assert isinstance(feature, SAEFeature)
        assert feature.id == 0
        assert feature.layer == 6
        assert feature.model_name == "gpt2"
        assert torch.equal(feature.decoder_vector, W_dec[0])

    def test_iteration(self):
        """Test iterating over features."""
        W_dec = torch.randn(10, 50)

        feature_set = FeatureSet(
            config={},
            stats={},
            W_dec=W_dec,
        )

        features = list(feature_set)
        assert len(features) == 10
        assert all(isinstance(f, SAEFeature) for f in features)

    def test_filter(self):
        """Test filtering features."""
        W_dec = torch.randn(100, 50)
        stats = [{"sparsity": 0.001 * i} for i in range(100)]

        feature_set = FeatureSet(
            config={},
            stats={},
            W_dec=W_dec,
            feature_stats=stats,
        )

        # Filter by sparsity
        filtered = feature_set.filter(sparsity_max=0.01)
        assert len(filtered) == 11  # 0.000 to 0.010

    def test_save_load(self, temp_dir):
        """Test FeatureSet save and load with SafeTensors."""
        W_dec = torch.randn(100, 50)
        W_enc = torch.randn(50, 100)

        original = FeatureSet(
            config={"model_name": "gpt2", "layer": 6, "k": 10},
            stats={"explained_variance": 0.85, "dead_features": 5},
            W_dec=W_dec,
            W_enc=W_enc,
            feature_stats=[{"id": i, "sparsity": 0.01} for i in range(100)],
            metadata={"version": "1.0"},
        )

        path = os.path.join(temp_dir, "features.safetensors")
        original.save(path)

        loaded = FeatureSet.load(path)

        assert loaded.config == original.config
        assert loaded.stats == original.stats
        assert torch.allclose(loaded.W_dec, original.W_dec)
        assert torch.allclose(loaded.W_enc, original.W_enc)
        assert loaded.feature_stats == original.feature_stats
        assert loaded.metadata == original.metadata

    def test_from_sae(self):
        """Test creating FeatureSet from trained SAE."""
        sae = SparseAutoencoder(input_dim=50, feature_dim=100, k=10)
        stats = {"explained_variance": 0.9}

        feature_set = FeatureSet.from_sae(sae, stats)

        assert len(feature_set) == 100
        assert feature_set.W_dec.shape == (100, 50)
        assert feature_set.config["input_dim"] == 50
        assert feature_set.config["feature_dim"] == 100

    def test_get_decoder_norms(self):
        """Test getting decoder norms."""
        # Create normalized decoder weights
        W_dec = torch.randn(100, 50)
        W_dec = F.normalize(W_dec, p=2, dim=1)

        feature_set = FeatureSet(
            config={},
            stats={},
            W_dec=W_dec,
        )

        norms = feature_set.get_decoder_norms()
        assert torch.allclose(norms, torch.ones(100), atol=1e-5)


class TestSAETrainer:
    """Tests for the SAETrainer class."""

    def test_initialization(self):
        """Test trainer initialization."""
        sae = SparseAutoencoder(input_dim=50, feature_dim=100, k=10)

        # Create simple dataloader
        data = torch.randn(100, 50)
        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=10)

        trainer = SAETrainer(sae, dataloader, aux_coef=1e-3)

        assert trainer.sae == sae
        assert trainer.aux_coef == 1e-3

    def test_train_step_topk(self):
        """Test a single training step in TopK mode."""
        sae = SparseAutoencoder(input_dim=50, feature_dim=100, k=10, sparsity_mode="topk")

        data = torch.randn(32, 50)
        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=16)

        trainer = SAETrainer(sae, dataloader, aux_coef=1e-3)

        batch = next(iter(dataloader))[0]
        metrics = trainer.train_step(batch)

        assert "loss" in metrics
        assert "mse_loss" in metrics
        assert metrics["loss"] > 0

    def test_train_step_l1(self):
        """Test a single training step in L1 mode."""
        sae = SparseAutoencoder(
            input_dim=50, feature_dim=100, sparsity_mode="l1", sparsity_coef=1e-3
        )

        data = torch.randn(32, 50)
        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=16)

        trainer = SAETrainer(sae, dataloader)

        batch = next(iter(dataloader))[0]
        metrics = trainer.train_step(batch)

        assert "loss" in metrics
        assert "mse_loss" in metrics
        assert "l1_loss" in metrics

    def test_decoder_normalized_after_step(self):
        """Test that decoder is normalized after each training step."""
        sae = SparseAutoencoder(input_dim=50, feature_dim=100, k=10)

        data = torch.randn(32, 50)
        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=16)

        trainer = SAETrainer(sae, dataloader)

        batch = next(iter(dataloader))[0]
        trainer.train_step(batch)

        norms = sae.get_decoder_norms()
        assert torch.allclose(norms, torch.ones(100), atol=1e-5)

    def test_get_stats(self):
        """Test getting training statistics."""
        sae = SparseAutoencoder(input_dim=50, feature_dim=100, k=10)

        data = torch.randn(100, 50)
        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=20)

        trainer = SAETrainer(sae, dataloader, log_every=50)

        # Train for a few steps
        trainer.train(n_epochs=1, n_steps=5)

        stats = trainer.get_stats()

        assert "final_explained_variance" in stats
        assert "final_mse" in stats
        assert "mean_l0" in stats
        assert "dead_features" in stats
        assert "total_steps" in stats
