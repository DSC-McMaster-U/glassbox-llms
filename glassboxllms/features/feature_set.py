"""
FeatureSet for managing collections of SAE features with SafeTensors serialization.
"""

import json
import os
from typing import Dict, Any, List, Optional, Iterator, Union
from dataclasses import dataclass, field
import torch

try:
    HAS_SAFETENSORS = True
    from safetensors.torch import save_file, load_file
except ImportError:
    HAS_SAFETENSORS = False

from .feature import SAEFeature


@dataclass
class FeatureSet:
    """
    Container for all features from a single SAE training run.

    Handles serialization to SafeTensors format for secure, memory-mapped
    storage of learned sparse features.

    Attributes:
        config: Training configuration (model, layer, k, feature_dim, etc.).
        stats: Aggregate statistics from training.
        W_dec: All decoder vectors (feature_dim, input_dim).
        W_enc: Optional encoder vectors (input_dim, feature_dim).
        feature_stats: Per-feature statistics list.
        metadata: Additional metadata.

    Example:
        >>> feature_set = FeatureSet.from_sae(sae, stats)
        >>> feature_set.save("features.safetensors")
        >>> loaded = FeatureSet.load("features.safetensors")
        >>> for feature in loaded:
        ...     print(feature.id, feature.sparsity)
    """

    config: Dict[str, Any]
    stats: Dict[str, Any]
    W_dec: torch.Tensor
    W_enc: Optional[torch.Tensor] = None
    feature_stats: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate tensor shapes after initialization."""
        if self.W_dec.ndim != 2:
            raise ValueError(f"W_dec must be 2D, got shape {self.W_dec.shape}")

        feature_dim, input_dim = self.W_dec.shape

        if self.W_enc is not None:
            if self.W_enc.shape != (input_dim, feature_dim):
                raise ValueError(
                    f"W_enc shape {self.W_enc.shape} incompatible with "
                    f"W_dec shape {self.W_dec.shape}"
                )

        # Ensure feature_stats length matches
        if self.feature_stats and len(self.feature_stats) != feature_dim:
            raise ValueError(
                f"feature_stats length {len(self.feature_stats)} does not match "
                f"feature_dim {feature_dim}"
            )

    def __len__(self) -> int:
        """Return the number of features."""
        return self.W_dec.shape[0]

    def __iter__(self) -> Iterator[SAEFeature]:
        """Iterate over all features."""
        for i in range(len(self)):
            yield self.get(i)

    def __getitem__(self, idx: Union[int, slice]) -> Union[SAEFeature, List[SAEFeature]]:
        """Get feature(s) by index or slice."""
        if isinstance(idx, slice):
            return [self.get(i) for i in range(*idx.indices(len(self)))]
        return self.get(idx)

    def get(self, feature_id: int) -> SAEFeature:
        """
        Get a single feature by ID.

        Args:
            feature_id: Index of the feature (0 to feature_dim-1).

        Returns:
            SAEFeature instance.

        Raises:
            IndexError: If feature_id is out of range.
        """
        if not 0 <= feature_id < len(self):
            raise IndexError(f"Feature ID {feature_id} out of range [0, {len(self)})")

        decoder_vector = self.W_dec[feature_id]

        encoder_vector = None
        if self.W_enc is not None:
            encoder_vector = self.W_enc[:, feature_id]

        stats = {}
        if self.feature_stats and feature_id < len(self.feature_stats):
            stats = self.feature_stats[feature_id]

        return SAEFeature(
            id=feature_id,
            layer=self.config.get("layer", 0),
            model_name=self.config.get("model_name", "unknown"),
            decoder_vector=decoder_vector.clone(),
            encoder_vector=encoder_vector.clone() if encoder_vector is not None else None,
            activation_stats=stats.copy() if stats else {},
            metadata={"feature_set_config": self.config},
        )

    def filter(
        self,
        sparsity_min: Optional[float] = None,
        sparsity_max: Optional[float] = None,
        max_activation_min: Optional[float] = None,
        max_activation_max: Optional[float] = None,
    ) -> List[SAEFeature]:
        """
        Filter features based on criteria.

        Args:
            sparsity_min: Minimum sparsity (inclusive).
            sparsity_max: Maximum sparsity (inclusive).
            max_activation_min: Minimum max activation (inclusive).
            max_activation_max: Maximum max activation (inclusive).

        Returns:
            List of SAEFeature instances matching all criteria.
        """
        if not self.feature_stats:
            return []

        matching = []
        for i, stats in enumerate(self.feature_stats):
            # Check sparsity criteria
            sparsity = stats.get("sparsity", float('inf'))
            if sparsity_min is not None and sparsity < sparsity_min:
                continue
            if sparsity_max is not None and sparsity > sparsity_max:
                continue

            # Check max activation criteria
            max_act = stats.get("max_activation", float('inf'))
            if max_activation_min is not None and max_act < max_activation_min:
                continue
            if max_activation_max is not None and max_act > max_activation_max:
                continue

            matching.append(self.get(i))

        return matching

    def save(self, path: str, metadata_only: bool = False):
        """
        Serialize FeatureSet to SafeTensors format.

        Args:
            path: File path to save to.
            metadata_only: If True, only save metadata (no tensors).

        Raises:
            RuntimeError: If safetensors is not installed.
            ValueError: If path does not end with .safetensors.
        """
        if not HAS_SAFETENSORS:
            raise RuntimeError("safetensors is required for serialization")

        if not path.endswith(".safetensors"):
            raise ValueError("Path must end with .safetensors")

        # Prepare tensors (ensure contiguous for safetensors)
        tensors = {}
        if not metadata_only:
            tensors["decoder_weights"] = self.W_dec.contiguous()
            if self.W_enc is not None:
                tensors["encoder_weights"] = self.W_enc.contiguous()

        # Prepare metadata as JSON
        metadata_json = {
            "config": self.config,
            "stats": self.stats,
            "feature_stats": self.feature_stats,
            "metadata": self.metadata,
        }

        # Convert metadata to string format for safetensors
        metadata_str = {k: json.dumps(v) for k, v in metadata_json.items()}

        # Save with safetensors
        save_file(tensors, path, metadata=metadata_str)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "FeatureSet":
        """
        Deserialize FeatureSet from SafeTensors format.

        Args:
            path: File path to load from.
            device: Device to load tensors to.

        Returns:
            FeatureSet instance.

        Raises:
            RuntimeError: If safetensors is not installed.
            FileNotFoundError: If file does not exist.
        """
        if not HAS_SAFETENSORS:
            raise RuntimeError("safetensors is required for deserialization")

        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        # Load tensors and metadata
        result = load_file(path, device=device)

        # Load metadata from JSON strings
        from safetensors import safe_open
        with safe_open(path, framework="pt", device=device) as f:
            metadata = f.metadata()

        # Parse JSON metadata
        config = json.loads(metadata.get("config", "{}"))
        stats = json.loads(metadata.get("stats", "{}"))
        feature_stats = json.loads(metadata.get("feature_stats", "[]"))
        extra_metadata = json.loads(metadata.get("metadata", "{}"))

        return cls(
            config=config,
            stats=stats,
            W_dec=result["decoder_weights"],
            W_enc=result.get("encoder_weights"),
            feature_stats=feature_stats,
            metadata=extra_metadata,
        )

    @classmethod
    def from_sae(
        cls,
        sae,
        stats: Optional[Dict[str, Any]] = None,
        per_feature_stats: Optional[List[Dict[str, Any]]] = None,
    ) -> "FeatureSet":
        """
        Create a FeatureSet from a trained SparseAutoencoder.

        Args:
            sae: Trained SparseAutoencoder instance.
            stats: Training statistics dict.
            per_feature_stats: Per-feature statistics list.

        Returns:
            FeatureSet instance.
        """
        config = sae.get_config()

        # Get weights (clone to avoid reference issues)
        W_dec = sae.W_dec.data.clone()

        W_enc = None
        if hasattr(sae, 'W_enc') and sae.W_enc is not None:
            W_enc = sae.W_enc.data.clone()

        return cls(
            config=config,
            stats=stats or {},
            W_dec=W_dec,
            W_enc=W_enc,
            feature_stats=per_feature_stats or [],
            metadata={"source": "SparseAutoencoder"},
        )

    def to(self, device: Union[str, torch.device]) -> "FeatureSet":
        """
        Move all tensors to a different device.

        Args:
            device: Target device (e.g., "cuda", "cpu").

        Returns:
            FeatureSet with tensors on the new device.
        """
        self.W_dec = self.W_dec.to(device)
        if self.W_enc is not None:
            self.W_enc = self.W_enc.to(device)
        return self

    def get_decoder_norms(self) -> torch.Tensor:
        """
        Get L2 norms of all decoder vectors.

        Useful for monitoring "scaling cheat" - norms should stay near 1.

        Returns:
            Tensor of shape (n_features,) with decoder norms.
        """
        return torch.norm(self.W_dec, p=2, dim=1)

    def get_feature_activations(self, activation: torch.Tensor) -> torch.Tensor:
        """
        Compute activations of all features on an input activation.

        Args:
            activation: Input activation (input_dim,) or (1, input_dim).

        Returns:
            Feature activations (n_features,).
        """
        if activation.ndim == 2:
            activation = activation.squeeze(0)

        # Compute activations using encoder weights
        if self.W_enc is not None:
            return torch.matmul(activation, self.W_enc)
        else:
            # Use decoder weights as fallback
            return torch.matmul(self.W_dec, activation)

    def __repr__(self) -> str:
        model = self.config.get("model_name", "unknown")
        layer = self.config.get("layer", "?")
        n_features = len(self)
        return f"FeatureSet(model='{model}', layer={layer}, n_features={n_features})"

    def __str__(self) -> str:
        lines = [
            f"FeatureSet:",
            f"  Model: {self.config.get('model_name', 'unknown')}",
            f"  Layer: {self.config.get('layer', 'unknown')}",
            f"  Features: {len(self)}",
            f"  Input dim: {self.W_dec.shape[1]}",
        ]

        if self.stats:
            lines.append("  Stats:")
            for key, value in self.stats.items():
                if isinstance(value, float):
                    lines.append(f"    {key}: {value:.4f}")
                else:
                    lines.append(f"    {key}: {value}")

        return "\n".join(lines)
