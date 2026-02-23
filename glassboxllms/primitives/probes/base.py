"""
Base class for all probing modules.

Probes are simple models trained on frozen activations to test
whether specific concepts are linearly encoded in neural representations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Union
import numpy as np


@dataclass
class ProbeResult:
    """Container for probe evaluation results."""
    accuracy: float
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    explained_variance: Optional[float] = None  # For PCA/regression
    coefficients: Optional[np.ndarray] = None   # Learned direction
    metadata: Optional[dict] = None

    def __repr__(self) -> str:
        parts = [f"accuracy={self.accuracy:.3f}"]
        if self.f1 is not None:
            parts.append(f"f1={self.f1:.3f}")
        if self.explained_variance is not None:
            parts.append(f"explained_var={self.explained_variance:.3f}")
        return f"ProbeResult({', '.join(parts)})"


class BaseProbe(ABC):
    """
    Abstract base class for all probing classifiers.

    Probes train a simple model on frozen activations to determine
    if a concept (e.g., tense, gender, sentiment) is linearly encoded.

    Key principle: The probe should be simple enough that it cannot
    learn the task on its own — it can only succeed if the activations
    already encode the target concept.

    Attributes:
        layer (str): Target layer identifier (e.g., "mlp.10", "attention.5").
        direction (str): Name of the concept being probed (e.g., "tense").
        is_fitted (bool): Whether the probe has been trained.
    """

    def __init__(self, layer: str, direction: str):
        self.layer = layer
        self.direction = direction
        self.is_fitted = False

    @abstractmethod
    def fit(
        self,
        activations: np.ndarray,
        labels: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "BaseProbe":
        """
        Train the probe on activations.

        Args:
            activations: Shape (n_samples, n_features) — frozen activations.
            labels: Shape (n_samples,) — target labels/values.
            sample_weight: Optional per-sample weights.

        Returns:
            self (fitted probe).
        """
        ...

    @abstractmethod
    def predict(self, activations: np.ndarray) -> np.ndarray:
        """
        Predict labels from activations.

        Args:
            activations: Shape (n_samples, n_features).

        Returns:
            Predictions of shape (n_samples,).
        """
        ...

    @abstractmethod
    def evaluate(
        self,
        activations: np.ndarray,
        labels: np.ndarray,
    ) -> ProbeResult:
        """
        Evaluate probe performance on held-out data.

        Args:
            activations: Shape (n_samples, n_features).
            labels: Shape (n_samples,).

        Returns:
            ProbeResult with metrics.
        """
        ...

    def get_direction(self) -> np.ndarray:
        """
        Return the learned linear direction (coefficients).

        For classification probes, this is the weight vector.
        For PCA, this is the principal component(s).

        Returns:
            Array of shape (n_features,) or (n_components, n_features).
        """
        raise NotImplementedError("Subclass must implement get_direction()")

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "unfitted"
        return f"{self.__class__.__name__}(layer='{self.layer}', direction='{self.direction}', {status})"
