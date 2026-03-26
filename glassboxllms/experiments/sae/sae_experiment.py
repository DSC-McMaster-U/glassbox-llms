"""
SAE Experiment: Automated pipeline for discovering monosemantic features.

Implements the full workflow:
1. Collect activations from target layer via hooks
2. Train Sparse Autoencoder to learn interpretable features
3. Extract and register features to Atlas
4. Validate training quality (R^2, sparsity, dead neurons)
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Any, List, Optional, Literal, Union
from pathlib import Path

# Import SAE infrastructure
from glassboxllms.features import SparseAutoencoder, SAETrainer
from glassboxllms.features.utils import identify_dead_features

# Import instrumentation
from glassboxllms.instrumentation.hook_manager import HookManager
from glassboxllms.instrumentation.activations import ActivationStore

# Import Atlas integration
from glassboxllms.analysis.feature_atlas import Atlas, Feature, FeatureType, Location, History


class SAEExperiment:
    """
    Automated experiment for discovering monosemantic features using Sparse Autoencoders.
    
    This class orchestrates the full pipeline:
    - Activation collection from specified model layer
    - SAE training with modern best practices
    - Feature extraction and registration to Atlas
    - Success criteria validation
    
    The goal is to move beyond polysemantic neurons by learning sparse,
    interpretable feature directions that each represent a single concept.
    
    Attributes:
        model: PyTorch model to analyze
        layer: Target layer name (e.g., "transformer.h.11.mlp")
        sae: Trained SparseAutoencoder instance
        stats: Training statistics (R^2, L0, dead features, etc.)
        
    Example:
        >>> experiment = SAEExperiment(
        ...     model=gpt2,
        ...     layer="transformer.h.11.mlp",
        ...     sparsity_alpha=0.1,
        ...     d_sae=32768
        ... )
        >>> activations = experiment.collect_activations(dataloader, num_samples=1000000)
        >>> stats = experiment.train(activations, n_epochs=10)
        >>> atlas = experiment.register_features(Atlas(name="gpt2-features"))
    """
    
    def __init__(
        self,
        model: nn.Module,
        layer: str,
        sparsity_alpha: float = 0.1,
        d_sae: int = 32768,
        k: Optional[int] = None,
        sparsity_mode: Literal["topk", "l1"] = "topk",
        model_name: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        Initialize SAE Experiment.
        
        Args:
            model: PyTorch model to analyze
            layer: Target layer name for activation extraction
            sparsity_alpha: Sparsity coefficient (aux_coef for TopK, L1 weight for L1 mode)
            d_sae: Number of SAE features (typically 8x-32x hidden_dim)
            k: Number of active features for TopK mode (auto-computed if None)
            sparsity_mode: "topk" or "l1" sparsity mechanism
            model_name: Model identifier for metadata (optional)
            device: Device for training ("cpu", "cuda", "mps")
        """
        self.model = model
        self.layer = layer
        self.sparsity_alpha = sparsity_alpha
        self.d_sae = d_sae
        self.sparsity_mode = sparsity_mode
        self.device = device
        self.model_name = model_name or str(type(model).__name__)
        
        # Infer input dimension from model
        self.input_dim = self._infer_input_dim()
        
        # Auto-compute k if not provided (TopK mode)
        if k is None and sparsity_mode == "topk":
            # Default: ~1-2% of features active (common in literature)
            self.k = max(32, min(128, d_sae // 64))
        else:
            self.k = k
        
        # Initialize SAE
        self.sae = SparseAutoencoder(
            input_dim=self.input_dim,
            feature_dim=d_sae,
            k=self.k,
            sparsity_mode=sparsity_mode,
            sparsity_coef=sparsity_alpha if sparsity_mode == "l1" else 0.0,
            device=device
        )
        
        # Training state
        self.stats: Dict[str, Any] = {}
        self.activations_collected = False
        self.trained = False
        
    def _infer_input_dim(self) -> int:
        """
        Infer the activation dimension of the target layer.
        
        Returns:
            Input dimension for SAE
            
        Raises:
            RuntimeError: If layer not found or dimension cannot be inferred
        """
        # Try to get layer from model
        try:
            layer_dict = dict(self.model.named_modules())
            if self.layer not in layer_dict:
                raise KeyError(f"Layer '{self.layer}' not found in model")
            
            target_layer = layer_dict[self.layer]
            
            # Try to infer from common attributes
            if hasattr(target_layer, 'out_features'):
                return target_layer.out_features
            elif hasattr(target_layer, 'hidden_size'):
                return target_layer.hidden_size
            elif hasattr(target_layer, 'embed_dim'):
                return target_layer.embed_dim
            else:
                # Last resort: run a dummy forward pass
                return self._infer_from_forward()
                
        except Exception as e:
            raise RuntimeError(
                f"Could not infer input dimension for layer '{self.layer}'. "
                f"Please specify manually or check layer name. Error: {e}"
            )
    
    def _infer_from_forward(self) -> int:
        """Infer dimension by running a dummy forward pass."""
        hook_manager = HookManager(self.model)
        activations = {}
        
        def capture_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                activations[name] = output
            return hook
        
        hook_manager.attach_hook(self.layer, capture_hook(self.layer))
        
        # Create dummy input (assuming text model with token embeddings)
        with torch.no_grad():
            dummy_input = torch.randint(0, 1000, (1, 10)).to(self.device)
            try:
                self.model(dummy_input)
            except:
                # Try with input_ids kwarg
                self.model(input_ids=dummy_input)
        
        hook_manager.remove_hooks()
        
        if self.layer in activations:
            act = activations[self.layer]
            return act.shape[-1]  # Last dimension is typically hidden_dim
        
        raise RuntimeError(f"Could not capture activations for layer '{self.layer}'")
    
    def collect_activations(
        self,
        dataloader: DataLoader,
        num_samples: int = 100000,
        pooling: Literal["mean", "last", "none"] = "mean"
    ) -> torch.Tensor:
        """
        Collect activations from target layer.
        
        Args:
            dataloader: DataLoader providing input batches
            num_samples: Number of activation samples to collect
            pooling: How to handle sequence dimension ("mean", "last", or "none")
        
        Returns:
            Activations tensor of shape (num_samples, input_dim)
        """
        print(f"Collecting {num_samples} activations from layer '{self.layer}'...")
        
        hook_manager = HookManager(self.model)
        store = ActivationStore(device=self.device, buffer_size=1000)
        
        # Attach hook
        hook_manager.attach_hook(self.layer, store.create_hook(self.layer))
        
        # Collect activations
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Handle different batch formats
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    self.model(**batch)
                elif isinstance(batch, (tuple, list)):
                    batch = tuple(b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch)
                    self.model(*batch)
                else:
                    batch = batch.to(self.device)
                    self.model(batch)
                
                # Check if we have enough samples
                current_count = len(store.get_all(self.layer))
                if current_count >= num_samples:
                    print(f"    Collected {current_count} samples after {batch_idx + 1} batches")
                    break
                
                if (batch_idx + 1) % 100 == 0:
                    print(f"    Progress: {current_count}/{num_samples} samples...")
        
        hook_manager.remove_hooks()
        
        # Get all activations
        activations = store.get_all(self.layer)
        
        # Handle 3D activations (batch, seq, hidden)
        if activations.ndim == 3:
            if pooling == "mean":
                activations = activations.mean(dim=1)
            elif pooling == "last":
                activations = activations[:, -1, :]
            # "none" keeps 3D, will be flattened in SAE
        
        # Trim to exact number requested
        activations = activations[:num_samples]
        
        print(f"    ✓ Collected activations: {activations.shape}")
        self.activations_collected = True
        
        return activations
    
    def train(
        self,
        activations: torch.Tensor,
        n_epochs: int = 10,
        batch_size: int = 256,
        learning_rate: float = 3e-4,
        log_every: int = 100,
    ) -> Dict[str, Any]:
        """
        Train the Sparse Autoencoder on collected activations.
        
        Args:
            activations: Activation tensor from collect_activations()
            n_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for Adam optimizer
            log_every: Log metrics every N steps
            
        Returns:
            Training statistics dictionary
        """
        print(f"Training SAE ({n_epochs} epochs, batch_size={batch_size})...")
        
        # Initialize with geometric median
        print("    Initializing with geometric median...")
        sample_size = min(10000, len(activations))
        self.sae.initialize_geometric_median(activations[:sample_size])
        
        # Create DataLoader
        dataset = TensorDataset(activations)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Create optimizer with custom learning rate
        optimizer = torch.optim.Adam(self.sae.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        
        # Initialize trainer
        trainer = SAETrainer(
            self.sae,
            dataloader,
            optimizer=optimizer,
            aux_coef=self.sparsity_alpha if self.sparsity_mode == "topk" else 0.0,
            device=self.device,
            log_every=log_every
        )
        
        # Train
        self.stats = trainer.train(n_epochs=n_epochs)
        self.trained = True
        
        print(f"    ✓ Training complete")
        print(f"      Explained variance: {self.stats['final_explained_variance']:.3f}")
        print(f"      Mean L0: {self.stats.get('mean_l0', 'N/A')}")
        print(f"      Dead features: {self.stats.get('dead_features', 'N/A')}")
        
        return self.stats
    
    def validate_training(self) -> Dict[str, bool]:
        """
        Validate that training meets success criteria.
        
        Success criteria:
        - High reconstruction: R^2 > 0.7
        - Sparse activations: Mean L0 < 50
        - Low dead features: < 10% dead
        
        Returns:
            Dictionary of criteria with pass/fail status
        """
        if not self.trained:
            raise RuntimeError("Must train SAE before validation. Call train() first.")
        
        r2 = self.stats.get("final_explained_variance", 0.0)
        mean_l0 = self.stats.get("mean_l0", float('inf'))
        dead_count = self.stats.get("dead_feature_count", 0)  
        
        if "dead_feature_count" not in self.stats:
            with torch.no_grad():
                W_dec = self.sae.W_dec.detach().cpu()
                norms = torch.norm(W_dec, dim=1, p=2)
                dead_count = (norms < 1e-6).sum().item()
                self.stats["dead_feature_count"] = dead_count
        
        dead_percentage = dead_count / self.d_sae
        print(f"    Dead features: {dead_count}/{self.d_sae} ({dead_percentage*100:.1f}%)")
        
        criteria = {
            "high_reconstruction": r2 > 0.7,
            "sparse_activations": mean_l0 < 50,
            "low_dead_features": dead_percentage < 0.1  # Less than 10%
        }
        
        return criteria
    
    def extract_features(
        self,
        skip_dead: bool = True,
        dataset_name: str = "unknown"
    ) -> List[Feature]:
        """
        Extract features from trained SAE and convert to Atlas Feature objects.
        
        Args:
            skip_dead: Whether to skip dead features (decoder norm < 1e-6)
            dataset_name: Dataset name for History metadata
            
        Returns:
            List of Atlas Feature objects
        """
        if not self.trained:
            raise RuntimeError("Must train SAE before extracting features. Call train() first.")
        
        print(f"Extracting features from trained SAE...")
        
        features = []
        W_dec = self.sae.W_dec.detach().cpu()
        
        # Calculate per-feature statistics
        norms = torch.norm(W_dec, dim=1, p=2)
        dead_mask = (norms < 1e-6).cpu().numpy()
        
        dead_count = dead_mask.sum()
        alive_count = (~dead_mask).sum()
        
        print(f"    Total features: {self.d_sae}")
        print(f"    Alive features: {alive_count}")
        print(f"    Dead features: {dead_count}")
        
        for i in range(self.sae.feature_dim):
            if skip_dead and dead_mask[i]:
                continue
            
            feature = Feature(
                feature_type=FeatureType.SAE_LATENT,
                location=Location(
                    model_name=self.model_name,
                    layer=self.layer,
                    neuron_idx=i
                ),
                label=f"sae_latent_{i}",
                description=f"SAE latent feature {i} from {self.layer} (d_sae={self.d_sae}, k={self.k})",
                history=History(
                    method="sparse_autoencoder",
                    dataset=dataset_name,
                    hyperparameters={
                        "d_sae": self.d_sae,
                        "sparsity_alpha": self.sparsity_alpha,
                        "k": self.k,
                        "sparsity_mode": self.sparsity_mode,
                        "n_epochs": self.stats.get("n_epochs", "unknown")
                    }
                ),
                metadata={
                    "decoder_norm": norms[i].item(),
                    "explained_variance": self.stats["final_explained_variance"],
                    "mean_l0": self.stats.get("mean_l0", 0.0),
                    "is_dead": bool(dead_mask[i])
                }
            )
            features.append(feature)
        
        print(f"    ✓ Extracted {len(features)} features")
        return features
    
    def register_features(
        self,
        atlas: Optional[Atlas] = None,
        atlas_name: Optional[str] = None,
        dataset_name: str = "unknown",
        skip_dead: bool = True
    ) -> Atlas:
        """
        Extract features and register them to an Atlas.
        
        Args:
            atlas: Existing Atlas instance (if None, creates new one)
            atlas_name: Name for new Atlas (used if atlas is None)
            dataset_name: Dataset name for History metadata
            skip_dead: Whether to skip dead features
            
        Returns:
            Atlas with registered features
        """
        print(f"Registering features to Atlas...")
        
        # Extract features
        features = self.extract_features(skip_dead=skip_dead, dataset_name=dataset_name)
        
        # Create or use Atlas
        if atlas is None:
            atlas_name = atlas_name or f"{self.model_name}_{self.layer}_features"
            atlas = Atlas(
                name=atlas_name,
                description=f"SAE features from {self.layer}",
                model_name=self.model_name
            )
        
        # Register all features
        for feature in features:
            atlas.add(feature)
        
        print(f"    ✓ Registered {len(features)} features to Atlas '{atlas.metadata.name}'")
        print(f"    Total features in Atlas: {len(atlas)}")
        
        return atlas
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get training statistics and validation results.
        
        Returns:
            Dictionary with stats and validation criteria
        """
        if not self.trained:
            return {"trained": False}
        
        result = {
            "trained": True,
            "stats": self.stats,
        }
        
        # Add validation if possible
        try:
            result["validation"] = self.validate_training()
            result["success"] = all(result["validation"].values())
        except:
            pass
        
        return result
    
    def save_checkpoint(self, path: Union[str, Path]):
        """
        Save trained SAE checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        if not self.trained:
            raise RuntimeError("Cannot save untrained SAE")
        
        path = Path(path)
        checkpoint = {
            "sae_state_dict": self.sae.state_dict(),
            "config": {
                "input_dim": self.input_dim,
                "feature_dim": self.d_sae,
                "k": self.k,
                "sparsity_mode": self.sparsity_mode,
                "sparsity_alpha": self.sparsity_alpha,
            },
            "stats": self.stats,
            "model_name": self.model_name,
            "layer": self.layer,
        }
        torch.save(checkpoint, path)
    
    def __repr__(self) -> str:
        status = "trained" if self.trained else "untrained"
        return (
            f"SAEExperiment(model='{self.model_name}', layer='{self.layer}', "
            f"d_sae={self.d_sae}, k={self.k}, {status})"
        )
