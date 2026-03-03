"""
SAETrainer for training Sparse Autoencoders on LLM activations.

Implements modern training techniques:
- Auxiliary loss for dead neurons (TopK mode)
- L1 sparsity penalty (L1 mode)
- Dead neuron tracking and resampling
- Gradient accumulation for large effective batch sizes
- Mixed precision support
- Comprehensive metrics logging
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional, Callable, Literal
from dataclasses import dataclass, field
import time

from .sae import SparseAutoencoder
from .utils import (
    calc_explained_variance,
    calc_mse_loss,
    calc_mean_l0,
    calc_l0_sparsity,
    identify_dead_features,
)


@dataclass
class TrainerState:
    """Tracks the training state and metrics."""

    step: int = 0
    epoch: int = 0
    best_loss: float = float('inf')

    # Running metrics
    metrics: Dict[str, List[float]] = field(default_factory=lambda: {
        "loss": [],
        "mse_loss": [],
        "aux_loss": [],
        "l1_loss": [],
        "explained_variance": [],
        "dead_feature_count": [],
        "mean_l0": [],
    })

    # Dead feature tracking (for TopK mode)
    activation_history: Optional[torch.Tensor] = None
    dead_feature_count: int = 0

    def update(self, metrics: Dict[str, float]):
        """Update metrics with new values."""
        for key, value in metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)

    def get_recent(self, key: str, n: int = 100) -> float:
        """Get average of recent n values for a metric."""
        if key not in self.metrics or not self.metrics[key]:
            return 0.0
        values = self.metrics[key][-n:]
        return sum(values) / len(values)


class SAETrainer:
    """
    Trainer for Sparse Autoencoder models.

    Handles the training loop, loss computation, dead neuron management,
    and metrics collection. Supports both TopK and L1 sparsity modes.

    TopK Mode:
        - Uses auxiliary loss on residual for dead neurons
        - Tracks feature activation history
        - Supports resampling of dead neurons

    L1 Mode:
        - Uses L1 penalty on feature activations
        - Simpler training without auxiliary loss

    Attributes:
        sae: SparseAutoencoder model to train.
        optimizer: PyTorch optimizer.
        dataloader: Training data loader.
        aux_coef: Coefficient for auxiliary loss (TopK mode).
        resample_dead_every: Resample dead neurons every N steps (0 to disable).
        log_every: Log metrics every N steps.
        device: Training device.

    Example:
        >>> sae = SparseAutoencoder(input_dim=768, feature_dim=16384, k=128)
        >>> trainer = SAETrainer(sae, dataloader, aux_coef=1e-3)
        >>> trainer.train(n_epochs=10)
        >>> stats = trainer.get_stats()
    """

    def __init__(
        self,
        sae: SparseAutoencoder,
        dataloader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        aux_coef: float = 1e-3,
        resample_dead_every: int = 0,
        resample_threshold: int = 500,
        log_every: int = 100,
        save_every: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
        device: Optional[str] = None,
        use_amp: bool = False,
        grad_accum_steps: int = 1,
    ):
        """
        Initialize the SAE trainer.

        Args:
            sae: SparseAutoencoder model to train.
            dataloader: DataLoader providing training activations.
            optimizer: Optional optimizer (default: Adam with lr=1e-4).
            aux_coef: Coefficient for auxiliary loss (TopK mode only).
            resample_dead_every: Resample dead neurons every N steps (0 to disable).
            resample_threshold: Steps of inactivity to consider a feature "dead".
            log_every: Log metrics every N steps.
            save_every: Save checkpoint every N steps (None to disable).
            checkpoint_dir: Directory to save checkpoints.
            device: Device to train on (inferred from model if not specified).
            use_amp: Whether to use mixed precision training.
            grad_accum_steps: Gradient accumulation steps for effective batch size.
        """
        self.sae = sae
        self.dataloader = dataloader

        # Device
        if device is None:
            self.device = next(sae.parameters()).device
        else:
            self.device = torch.device(device)
            self.sae.to(self.device)

        # Optimizer
        if optimizer is None:
            self.optimizer = torch.optim.Adam(sae.parameters(), lr=1e-4, betas=(0.9, 0.999))
        else:
            self.optimizer = optimizer

        # Training hyperparameters
        self.aux_coef = aux_coef
        self.resample_dead_every = resample_dead_every
        self.resample_threshold = resample_threshold
        self.log_every = log_every
        self.save_every = save_every
        self.checkpoint_dir = checkpoint_dir
        self.use_amp = use_amp
        self.grad_accum_steps = grad_accum_steps

        # Training state
        self.state = TrainerState()

        # Initialize activation history for dead neuron tracking
        if self.sae.use_topk:
            self.state.activation_history = torch.zeros(
                self.sae.feature_dim, resample_threshold,
                dtype=torch.bool, device=self.device
            )

        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None

    def compute_loss(
        self,
        batch: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute training loss for a batch.

        Args:
            batch: Input activations (batch_size, input_dim).
            return_components: If True, return dict of loss components.

        Returns:
            Total loss (or tuple of (loss, components) if return_components=True).
        """
        batch = batch.to(self.device)

        # Forward pass with mixed precision if enabled
        if self.use_amp:
            with torch.cuda.amp.autocast():
                reconstructed, features, z_pre = self.sae(batch)
        else:
            reconstructed, features, z_pre = self.sae(batch)

        # Main loss: MSE reconstruction
        mse_loss = F.mse_loss(reconstructed, batch)

        loss = mse_loss
        aux_loss = torch.tensor(0.0, device=self.device)
        l1_loss = torch.tensor(0.0, device=self.device)

        # Auxiliary loss for dead neurons (TopK mode)
        if self.sae.use_topk:
            # Identify dead features
            dead_features = self._get_dead_features(features)

            if dead_features.any() and self.aux_coef > 0:
                # Train dead features on residual
                residual = batch - reconstructed
                dead_indices = torch.where(dead_features)[0]

                # Compute activations for dead features only
                z_dead = z_pre[:, dead_indices]
                f_dead = torch.relu(z_dead)

                # Reconstruct using only dead features
                recon_dead = f_dead @ self.sae.W_dec[dead_indices] + self.sae.b_dec

                # Auxiliary loss: how well can dead features explain the residual?
                aux_loss = F.mse_loss(recon_dead, residual)
                loss = mse_loss + self.aux_coef * aux_loss

            # Update activation history
            self._update_activation_history(features)

        else:
            # L1 sparsity penalty for ReLU mode
            if self.sae.sparsity_coef > 0:
                l1_loss = self.sae.sparsity_coef * features.abs().sum() / batch.shape[0]
                loss = mse_loss + l1_loss

        if return_components:
            return loss, {
                "loss": loss.item(),
                "mse_loss": mse_loss.item(),
                "aux_loss": aux_loss.item(),
                "l1_loss": l1_loss.item(),
            }

        return loss

    def _get_dead_features(self, features: torch.Tensor) -> torch.Tensor:
        """Get mask of dead features based on activation history."""
        if self.state.activation_history is None:
            return torch.zeros(self.sae.feature_dim, dtype=torch.bool, device=self.device)

        # Features are dead if never active in recent history
        recent_activity = self.state.activation_history.sum(dim=1)
        dead_mask = recent_activity == 0
        return dead_mask

    def _update_activation_history(self, features: torch.Tensor):
        """Update rolling activation history for dead neuron tracking."""
        if self.state.activation_history is None:
            return

        # Roll history and add new step
        self.state.activation_history = torch.roll(
            self.state.activation_history, shifts=-1, dims=1
        )

        # Mark features as active if they fired in this batch (any sample)
        active = (features > 0).any(dim=0)
        self.state.activation_history[:, -1] = active

    def _resample_dead_neurons(self, sample_activations: torch.Tensor):
        """
        Resample dead neurons from high-loss samples.

        Re-initializes dead feature directions to point towards samples
        with high reconstruction error.

        Args:
            sample_activations: Sample of activations to compute loss on.
        """
        dead_mask = self._get_dead_features(torch.zeros(1, self.sae.feature_dim, device=self.device))

        if not dead_mask.any():
            return

        dead_indices = torch.where(dead_mask)[0]

        with torch.no_grad():
            # Compute reconstruction errors
            sample_activations = sample_activations.to(self.device)
            reconstructed, _, _ = self.sae(sample_activations)
            errors = (sample_activations - reconstructed).norm(dim=-1)

            # Get top-k highest loss samples (one per dead neuron)
            n_dead = len(dead_indices)
            top_indices = errors.topk(min(n_dead, len(errors))).indices

            # Set new decoder directions to high-loss samples (normalized)
            new_directions = sample_activations[top_indices[:n_dead]]
            new_directions = F.normalize(new_directions, p=2, dim=-1)

            self.sae.W_dec.data[dead_indices] = new_directions

            # Reset encoder to match (transposed)
            self.sae.W_enc.data[:, dead_indices] = new_directions.T

            # Reset biases for dead features
            self.sae.b_enc.data[dead_indices] = 0.0

        print(f"Resampled {n_dead} dead neurons at step {self.state.step}")

    def train_step(self, batch: torch.Tensor, is_accumulation_step: bool = False) -> Dict[str, float]:
        """
        Execute a single training step.

        Args:
            batch: Input batch of activations.
            is_accumulation_step: If True, don't step optimizer (gradient accumulation).

        Returns:
            Dictionary of metrics for this step.
        """
        # Compute loss
        loss, components = self.compute_loss(batch, return_components=True)

        # Backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optimizer step (if not accumulating)
        if not is_accumulation_step:
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            # Normalize decoder (prevent scaling cheat)
            self.sae.normalize_decoder()

            # Zero gradients
            self.optimizer.zero_grad()

            # Increment step counter
            self.state.step += 1

        return components

    def train(
        self,
        n_epochs: int,
        n_steps: Optional[int] = None,
        callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run the training loop.

        Args:
            n_epochs: Number of epochs to train.
            n_steps: Optional max steps (early termination if reached).
            callback: Optional callback(step, metrics) called at log_every intervals.

        Returns:
            Dictionary of final training statistics.
        """
        self.sae.train()
        start_time = time.time()

        total_steps = 0

        for epoch in range(n_epochs):
            self.state.epoch = epoch

            for batch_idx, batch in enumerate(self.dataloader):
                # Handle different dataloader return types
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]  # Assume first element is data

                # Determine if this is an accumulation step
                is_accumulation = (batch_idx + 1) % self.grad_accum_steps != 0

                # Training step
                metrics = self.train_step(batch, is_accumulation_step=is_accumulation)

                # Scale loss for gradient accumulation
                if self.grad_accum_steps > 1 and not is_accumulation:
                    for key in metrics:
                        metrics[key] /= self.grad_accum_steps

                # Update running metrics
                self.state.update(metrics)

                total_steps += 1

                # Check for early termination
                if n_steps is not None and total_steps >= n_steps:
                    break

                # Logging
                if self.state.step % self.log_every == 0 and not is_accumulation:
                    recent_metrics = {
                        "loss": self.state.get_recent("loss"),
                        "mse_loss": self.state.get_recent("mse_loss"),
                        "explained_variance": self.state.get_recent("explained_variance"),
                        "dead_features": self._get_dead_features(torch.zeros(1, self.sae.feature_dim, device=self.device)).sum().item(),
                        "mean_l0": self.state.get_recent("mean_l0"),
                    }

                    # Compute explained variance on recent batch
                    with torch.no_grad():
                        batch_device = batch.to(self.device)
                        recon, features, _ = self.sae(batch_device)
                        recent_metrics["explained_variance"] = calc_explained_variance(batch_device, recon)
                        recent_metrics["mean_l0"] = calc_mean_l0(features)

                    self._log_step(recent_metrics)

                    if callback:
                        callback(self.state.step, recent_metrics)

                # Resample dead neurons
                if self.resample_dead_every > 0 and self.state.step % self.resample_dead_every == 0:
                    # Get a sample from dataloader for resampling
                    sample_batch = next(iter(self.dataloader))
                    if isinstance(sample_batch, (list, tuple)):
                        sample_batch = sample_batch[0]
                    self._resample_dead_neurons(sample_batch[:1000])

                # Save checkpoint
                if self.save_every is not None and self.state.step % self.save_every == 0:
                    self._save_checkpoint()

            if n_steps is not None and total_steps >= n_steps:
                break

        # Final stats
        elapsed = time.time() - start_time
        print(f"\nTraining completed: {self.state.step} steps in {elapsed:.1f}s ({self.state.step/elapsed:.1f} steps/s)")

        return self.get_stats()

    def _log_step(self, metrics: Dict[str, float]):
        """Print training metrics."""
        dead = int(metrics.get("dead_features", 0))
        print(
            f"Step {self.state.step:6d} | "
            f"Loss: {metrics['loss']:.4f} | "
            f"MSE: {metrics['mse_loss']:.4f} | "
            f"ExpVar: {metrics['explained_variance']:.3f} | "
            f"L0: {metrics['mean_l0']:.1f} | "
            f"Dead: {dead}"
        )

    def _save_checkpoint(self):
        """Save training checkpoint."""
        if self.checkpoint_dir is None:
            return

        import os
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        path = os.path.join(self.checkpoint_dir, f"checkpoint_step_{self.state.step}.pt")
        self.sae.save(path)
        print(f"Saved checkpoint to {path}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive training statistics.

        Returns:
            Dictionary with training metrics and final statistics.
        """
        # Compute final statistics
        with torch.no_grad():
            # Get a sample batch for final metrics
            sample_batch = next(iter(self.dataloader))
            if isinstance(sample_batch, (list, tuple)):
                sample_batch = sample_batch[0]
            sample_batch = sample_batch.to(self.device)

            recon, features, _ = self.sae(sample_batch)

            explained_var = calc_explained_variance(sample_batch, recon)
            mse = calc_mse_loss(sample_batch, recon)
            mean_l0 = calc_mean_l0(features)
            l0_per_feature = calc_l0_sparsity(features)

        return {
            "final_explained_variance": explained_var,
            "final_mse": mse,
            "mean_l0": mean_l0,
            "mean_sparsity": float(l0_per_feature.mean()),
            "dead_features": self._get_dead_features(features).sum().item(),
            "total_steps": self.state.step,
            "total_epochs": self.state.epoch,
            "training_history": {
                k: v for k, v in self.state.metrics.items() if v
            },
            "per_feature_stats": [
                {
                    "id": i,
                    "sparsity": float(l0_per_feature[i]),
                }
                for i in range(len(l0_per_feature))
            ],
        }

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the SAE on a validation set.

        Args:
            dataloader: Validation data loader.

        Returns:
            Dictionary of evaluation metrics.
        """
        self.sae.eval()

        total_mse = 0.0
        total_explained_var = 0.0
        total_l0 = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]

                batch = batch.to(self.device)
                recon, features, _ = self.sae(batch)

                total_mse += calc_mse_loss(batch, recon)
                total_explained_var += calc_explained_variance(batch, recon)
                total_l0 += calc_mean_l0(features)
                n_batches += 1

        self.sae.train()

        return {
            "mse": total_mse / n_batches,
            "explained_variance": total_explained_var / n_batches,
            "mean_l0": total_l0 / n_batches,
        }
