"""
SAEFeature dataclass for representing individual sparse autoencoder features.

An SAEFeature represents a single learned sparse direction in activation space,
along with its statistics and metadata from SAE training.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union
import torch
import numpy as np


@dataclass
class SAEFeature:
    """
    Represents a single sparse autoencoder feature.

    Each feature consists of a decoder vector (direction in activation space)
    and optionally an encoder vector. Features include statistics about their
    activation patterns computed during training or analysis.

    Attributes:
        id: Feature index in the SAE.
        layer: Source model layer name (e.g., "layer.6").
        model_name: Source model identifier (e.g., "gpt2").
        decoder_vector: Direction in activation space (input_dim,).
        encoder_vector: Optional encoder direction (input_dim,).
        activation_stats: Statistics about feature activations.
        metadata: Additional metadata about the feature.

    Example:
        >>> feature = SAEFeature(
        ...     id=0,
        ...     layer="layer.6",
        ...     model_name="gpt2",
        ...     decoder_vector=torch.randn(768),
        ... )
        >>> print(feature)
        SAEFeature(id=0, layer='layer.6', model='gpt2', sparsity=None)
    """

    id: int
    layer: int
    model_name: str
    decoder_vector: torch.Tensor
    encoder_vector: Optional[torch.Tensor] = None

    # Activation statistics (computed during training or analysis)
    activation_stats: Dict[str, float] = field(default_factory=dict)

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate tensor shapes and types after initialization."""
        if not isinstance(self.decoder_vector, torch.Tensor):
            raise TypeError(f"decoder_vector must be a torch.Tensor, got {type(self.decoder_vector)}")

        if self.decoder_vector.ndim != 1:
            raise ValueError(f"decoder_vector must be 1D, got shape {self.decoder_vector.shape}")

        if self.encoder_vector is not None:
            if not isinstance(self.encoder_vector, torch.Tensor):
                raise TypeError(f"encoder_vector must be a torch.Tensor, got {type(self.encoder_vector)}")
            if self.encoder_vector.shape != self.decoder_vector.shape:
                raise ValueError(
                    f"encoder_vector shape {self.encoder_vector.shape} does not match "
                    f"decoder_vector shape {self.decoder_vector.shape}"
                )

    def to(self, device: Union[str, torch.device]) -> "SAEFeature":
        """
        Move feature tensors to a different device.

        Args:
            device: Target device (e.g., "cuda", "cpu").

        Returns:
            SAEFeature with tensors on the new device.
        """
        self.decoder_vector = self.decoder_vector.to(device)
        if self.encoder_vector is not None:
            self.encoder_vector = self.encoder_vector.to(device)
        return self

    def similarity(self, other: "SAEFeature") -> float:
        """
        Compute cosine similarity with another feature.

        Uses the decoder vectors for comparison.

        Args:
            other: Another SAEFeature to compare with.

        Returns:
            Cosine similarity (-1 to 1, where 1 is identical direction).
        """
        v1 = self.decoder_vector
        v2 = other.decoder_vector

        # Compute cosine similarity
        dot = torch.dot(v1, v2)
        norm1 = torch.norm(v1, p=2)
        norm2 = torch.norm(v2, p=2)

        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0

        return float(dot / (norm1 * norm2))

    def activation_on(self, activation: torch.Tensor) -> float:
        """
        Compute feature activation on a given activation vector.

        The activation is computed as the dot product of the encoder vector
        with the input activation (or decoder if encoder not available).

        Args:
            activation: Input activation tensor (input_dim,) or (1, input_dim).

        Returns:
            Feature activation value (non-negative due to ReLU in SAE).
        """
        if activation.ndim == 2:
            activation = activation.squeeze(0)

        if activation.shape != self.decoder_vector.shape:
            raise ValueError(
                f"Activation shape {activation.shape} does not match "
                f"decoder shape {self.decoder_vector.shape}"
            )

        # Use encoder if available, otherwise use decoder
        direction = self.encoder_vector if self.encoder_vector is not None else self.decoder_vector

        # Compute activation
        return float(torch.dot(activation, direction))

    @property
    def sparsity(self) -> Optional[float]:
        """Return feature sparsity (fraction of samples where active), if available."""
        return self.activation_stats.get("sparsity")

    @property
    def max_activation(self) -> Optional[float]:
        """Return maximum activation observed, if available."""
        return self.activation_stats.get("max_activation")

    @property
    def mean_activation(self) -> Optional[float]:
        """Return mean activation, if available."""
        return self.activation_stats.get("mean_activation")

    @property
    def frequency(self) -> Optional[float]:
        """Return activation frequency (alias for sparsity), if available."""
        return self.sparsity

    def as_numpy(self) -> Dict[str, Any]:
        """
        Convert feature to numpy format for serialization.

        Returns:
            Dictionary with numpy arrays and metadata.
        """
        result = {
            "id": self.id,
            "layer": self.layer,
            "model_name": self.model_name,
            "decoder_vector": self.decoder_vector.cpu().numpy(),
            "activation_stats": self.activation_stats.copy(),
            "metadata": self.metadata.copy(),
        }

        if self.encoder_vector is not None:
            result["encoder_vector"] = self.encoder_vector.cpu().numpy()

        return result

    @classmethod
    def from_numpy(cls, data: Dict[str, Any]) -> "SAEFeature":
        """
        Create an SAEFeature from numpy format.

        Args:
            data: Dictionary with numpy arrays and metadata.

        Returns:
            SAEFeature instance.
        """
        decoder_vector = torch.from_numpy(data["decoder_vector"])

        encoder_vector = None
        if "encoder_vector" in data:
            encoder_vector = torch.from_numpy(data["encoder_vector"])

        return cls(
            id=data["id"],
            layer=data["layer"],
            model_name=data["model_name"],
            decoder_vector=decoder_vector,
            encoder_vector=encoder_vector,
            activation_stats=data.get("activation_stats", {}),
            metadata=data.get("metadata", {}),
        )

    def __repr__(self) -> str:
        sparsity_str = f"{self.sparsity:.6f}" if self.sparsity is not None else "None"
        return (
            f"SAEFeature(id={self.id}, layer={self.layer}, model='{self.model_name}', "
            f"sparsity={sparsity_str})"
        )

    def __str__(self) -> str:
        lines = [
            f"SAEFeature {self.id}:",
            f"  Layer: {self.layer}",
            f"  Model: {self.model_name}",
            f"  Decoder shape: {tuple(self.decoder_vector.shape)}",
        ]

        if self.encoder_vector is not None:
            lines.append(f"  Encoder shape: {tuple(self.encoder_vector.shape)}")

        if self.activation_stats:
            lines.append("  Activation stats:")
            for key, value in self.activation_stats.items():
                if isinstance(value, float):
                    lines.append(f"    {key}: {value:.6f}")
                else:
                    lines.append(f"    {key}: {value}")

        return "\n".join(lines)
