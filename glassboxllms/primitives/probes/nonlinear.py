"""
Non-Linear Probes for Neural Network Interpretability.

Implements:
  - NonLinearProbe: A Multi-Layer Perceptron (MLP) probe to detect if 
    information is present in activations even if not linearly accessible.
"""

from typing import List, Optional, Union, Dict, Any
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    r2_score,
    mean_squared_error,
)
import warnings

from .base import BaseProbe, ProbeResult


class NonLinearProbe(BaseProbe):
    """
    Multi-Layer Perceptron probe for detecting non-linearly encoded information.
    
    Supports two API patterns:
    
    1. Standard pattern (compatible with LinearProbe):
        >>> activations = store.extract(...)[layer]
        >>> probe.fit(activations, labels)
        >>> results = probe.evaluate(test_activations, test_labels)
    
    2. Convenience pattern (from ticket example):
        >>> probe.fit(dataset, activation_store)
        >>> results = probe.evaluate(dataset)
    
    Args:
        layer: Target layer identifier (e.g., "mlp.10", "attention.5")
        direction: Name of the concept being probed (e.g., "tense")
        target: Alias for 'direction' (for ticket API compatibility)
        hidden_dim: Single hidden dimension (alternative to hidden_dims)
        hidden_dims: List of hidden layer sizes (e.g., [128, 64])
        is_classification: Whether to perform classification (True) or regression (False)
        learning_rate_init: Initial learning rate for Adam optimizer
        max_iter: Maximum number of iterations
        early_stopping: Whether to use early stopping
        validation_fraction: Fraction of training data for validation
        n_iter_no_change: Number of iterations with no improvement
        random_state: Random seed for reproducibility
        normalize: Whether to standardize activations
        robust_scaling: Whether to use RobustScaler (less sensitive to outliers)
        **kwargs: Additional arguments passed to sklearn MLP estimator
    """
    
    def __init__(
        self,
        layer: str,
        direction: str,
        target: Optional[str] = None,
        hidden_dim: Optional[int] = None,
        hidden_dims: List[int] = [512, 256],
        is_classification: bool = True,
        learning_rate_init: float = 1e-3,
        max_iter: int = 200,
        early_stopping: bool = True,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 10,
        random_state: int = 42,
        normalize: bool = True,
        robust_scaling: bool = True,
        **kwargs,
    ):
        # Support target as alias for direction (from ticket example)
        if target is not None:
            direction = target
        
        super().__init__(layer, direction)
        
        # Support hidden_dim (singular) for simpler API (from ticket example)
        if hidden_dim is not None:
            hidden_dims = [hidden_dim]
        
        self.hidden_dims = hidden_dims
        self.is_classification = is_classification
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state
        self.normalize = normalize
        self.robust_scaling = robust_scaling
        self.kwargs = kwargs
        
        # Initialize scaler and estimator
        if robust_scaling:
            self._scaler = RobustScaler() if normalize else None
        else:
            self._scaler = StandardScaler() if normalize else None
        self._estimator = self._init_estimator()

    # ==================== ACTIVATIONSTORE INTEGRATION ====================
    
    def fit(
        self,
        *args,
        **kwargs
    ) -> "NonLinearProbe":
        """
        Flexible fit method that supports multiple signatures:
        
        1. Standard: fit(activations, labels, sample_weight=None)
        2. Ticket API: fit(dataset, activation_store)
        
        Args:
            activations: numpy array of shape (n_samples, n_features)
            labels: numpy array of shape (n_samples,)
            dataset: Object with .texts and .labels attributes
            activation_store: ActivationStore instance
            sample_weight: Optional per-sample weights
            
        Returns:
            self (fitted probe)
            
        Example (Ticket API):
            >>> probe = NonLinearProbe(layer="mlp.10", hidden_dim=128, target="is_plural")
            >>> probe.fit(dataset, activation_store)
        """
        # Check if we're using the ticket API: fit(dataset, activation_store)
        if len(args) == 2 and hasattr(args[0], 'texts') and hasattr(args[0], 'labels'):
            # This matches the ticket API
            dataset, activation_store = args[0], args[1]
            return self._fit_from_store(dataset, activation_store, **kwargs)
        
        # Otherwise use standard API
        if len(args) < 2:
            raise ValueError("fit() requires at least 2 arguments: activations and labels")
        
        activations, labels = args[0], args[1]
        sample_weight = kwargs.get('sample_weight', None)
        return self._fit_standard(activations, labels, sample_weight)

    def _fit_from_store(
        self,
        dataset: Any,
        activation_store: Any,
        pooling: str = "mean",
        **extract_kwargs
    ) -> "NonLinearProbe":
        """
        Internal method to fit from ActivationStore (ticket API).
        
        Args:
            dataset: Object with .texts and .labels attributes
            activation_store: ActivationStore instance
            pooling: How to pool sequence activations
            **extract_kwargs: Additional args for activation_store.extract()
            
        Returns:
            self (fitted probe)
        """
        # Validate dataset structure
        if not hasattr(dataset, 'texts') or not hasattr(dataset, 'labels'):
            raise ValueError(
                "Dataset must have 'texts' and 'labels' attributes. "
                f"Got attributes: {dir(dataset)}"
            )
        
        # Extract activations using ActivationStore
        # Note: activation_store should be imported from instrumentation/activation_store.py
        try:
            # Extract activations for the probe's target layer
            activations_dict = activation_store.extract(
                texts=dataset.texts,
                tokenizer=getattr(dataset, 'tokenizer', None),
                layers=[self.layer],
                pooling=pooling,
                **extract_kwargs
            )
        except Exception as e:
            raise ValueError(
                f"Failed to extract activations from ActivationStore: {e}\n"
                "Make sure activation_store is properly initialized and "
                "dataset has required attributes."
            )
        
        # Check if layer exists in extracted activations
        if self.layer not in activations_dict:
            available_layers = list(activations_dict.keys())
            raise ValueError(
                f"Layer '{self.layer}' not found in extracted activations. "
                f"Available layers: {available_layers}"
            )
        
        # Get activations and labels
        activations = activations_dict[self.layer]
        labels = dataset.labels
        
        # Validate shapes
        if len(activations) != len(labels):
            raise ValueError(
                f"Mismatch between activations ({len(activations)}) "
                f"and labels ({len(labels)})"
            )
        
        # Fit using standard method
        return self._fit_standard(activations, labels)

    def _fit_standard(
        self,
        activations: np.ndarray,
        labels: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "NonLinearProbe":
        """
        Internal standard fit method.
        """
        X = self._prepare_activations(activations, fit_scaler=True)
        y = np.asarray(labels)

        # Check for NaNs or extreme values
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            warnings.warn(
                f"Activations contain NaN or Inf values. "
                f"Min: {np.nanmin(X):.2e}, Max: {np.nanmax(X):.2e}",
                RuntimeWarning
            )
            # Replace NaN/Inf with finite values
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        if sample_weight is not None:
            self._estimator.fit(X, y, sample_weight=sample_weight)
        else:
            self._estimator.fit(X, y)

        self.is_fitted = True
        return self

    def evaluate(
        self,
        *args,
        **kwargs
    ) -> ProbeResult:
        """
        Flexible evaluate method that supports multiple signatures:
        
        1. Standard: evaluate(activations, labels)
        2. Ticket API: evaluate(dataset)
        
        Args:
            activations: numpy array of shape (n_samples, n_features)
            labels: numpy array of shape (n_samples,)
            dataset: Object with .texts and .labels attributes
            
        Returns:
            ProbeResult with evaluation metrics
            
        Example (Ticket API):
            >>> metrics = probe.evaluate(dataset)
        """
        # Check if we're using the ticket API: evaluate(dataset)
        if len(args) == 1 and hasattr(args[0], 'texts') and hasattr(args[0], 'labels'):
            dataset = args[0]
            # For evaluation, we need activations - user must have fitted first
            # and provided activation_store in fit() or we need it here
            # Since ticket example shows evaluate(dataset), we assume
            # activation_store was provided during fit and stored
            if not hasattr(self, '_last_activation_store'):
                raise ValueError(
                    "For evaluate(dataset), activation_store must be provided during fit(). "
                    "Alternatively, use evaluate(activations, labels)."
                )
            return self._evaluate_from_store(dataset, self._last_activation_store, **kwargs)
        
        # Otherwise use standard API
        if len(args) < 2:
            raise ValueError("evaluate() requires at least 2 arguments: activations and labels")
        
        activations, labels = args[0], args[1]
        return self._evaluate_standard(activations, labels)

    def _evaluate_from_store(
        self,
        dataset: Any,
        activation_store: Any,
        pooling: str = "mean",
        **extract_kwargs
    ) -> ProbeResult:
        """
        Internal method to evaluate from ActivationStore.
        """
        if not self.is_fitted:
            raise RuntimeError("Probe must be fitted before evaluation")

        # Extract activations
        activations_dict = activation_store.extract(
            texts=dataset.texts,
            tokenizer=getattr(dataset, 'tokenizer', None),
            layers=[self.layer],
            pooling=pooling,
            **extract_kwargs
        )
        
        if self.layer not in activations_dict:
            raise ValueError(
                f"Layer '{self.layer}' not found in extracted activations. "
                f"Available layers: {list(activations_dict.keys())}"
            )
        
        activations = activations_dict[self.layer]
        labels = dataset.labels
        
        return self._evaluate_standard(activations, labels)

    def _evaluate_standard(
        self,
        activations: np.ndarray,
        labels: np.ndarray,
    ) -> ProbeResult:
        """
        Internal standard evaluate method.
        """
        self._check_fitted()
        X = self._prepare_activations(activations, fit_scaler=False)
        y = np.asarray(labels)

        y_pred = self._estimator.predict(X)

        if self.is_classification:
            acc = accuracy_score(y, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y, y_pred, average="weighted", zero_division=0
            )
            
            metadata = {
                "type": "mlp_classification", 
                "hidden_layers": self.hidden_dims,
                "iterations": self._estimator.n_iter_,
            }
            
            if hasattr(self._estimator, 'loss_curve_'):
                metadata["loss_curve"] = self._estimator.loss_curve_
            
            if self.early_stopping and hasattr(self._estimator, 'validation_scores_'):
                metadata["validation_scores"] = self._estimator.validation_scores_
            
            if self.early_stopping and hasattr(self._estimator, 'best_loss_'):
                metadata["best_loss"] = self._estimator.best_loss_
            
            if hasattr(self._estimator, 'coefs_'):
                metadata["n_layers"] = len(self._estimator.coefs_)
            
            return ProbeResult(
                accuracy=acc,
                precision=precision,
                recall=recall,
                f1=f1,
                coefficients=self.get_direction(),
                metadata=metadata,
            )
        else:
            # Regression
            if len(np.unique(y)) < 2:
                warnings.warn(
                    f"Regression target has only {len(np.unique(y))} unique values. "
                    f"R² score may be undefined.",
                    RuntimeWarning
                )
                r2 = 0.0
            else:
                try:
                    r2 = r2_score(y, y_pred)
                except Exception as e:
                    warnings.warn(f"Could not compute R²: {e}. Using 0.0")
                    r2 = 0.0
            
            mse = mean_squared_error(y, y_pred)
            
            metadata = {
                "mse": mse,
                "rmse": np.sqrt(mse),
                "type": "mlp_regression",
                "iterations": self._estimator.n_iter_,
            }
            
            if hasattr(self._estimator, 'loss_curve_'):
                metadata["loss_curve"] = self._estimator.loss_curve_
            
            if self.early_stopping and hasattr(self._estimator, 'validation_scores_'):
                metadata["validation_scores"] = self._estimator.validation_scores_
            
            if self.early_stopping and hasattr(self._estimator, 'best_loss_'):
                metadata["best_loss"] = self._estimator.best_loss_
            
            if hasattr(self._estimator, 'coefs_'):
                metadata["n_layers"] = len(self._estimator.coefs_)
            
            return ProbeResult(
                accuracy=r2,
                explained_variance=r2,
                coefficients=self.get_direction(),
                metadata=metadata,
            )

    # ==================== REST OF THE METHODS (unchanged) ====================

    def _init_estimator(self):
        """Initialize the underlying sklearn MLP estimator."""
        common_params = {
            "hidden_layer_sizes": tuple(self.hidden_dims),
            "learning_rate_init": self.learning_rate_init,
            "max_iter": self.max_iter,
            "early_stopping": self.early_stopping,
            "validation_fraction": self.validation_fraction,
            "n_iter_no_change": self.n_iter_no_change,
            "random_state": self.random_state,
            "solver": "adam",
            "batch_size": "auto",
            **self.kwargs,
        }

        if self.is_classification:
            return MLPClassifier(**common_params)
        else:
            common_params["learning_rate"] = "adaptive"
            common_params["alpha"] = common_params.get("alpha", 1e-4)
            return MLPRegressor(**common_params)

    def predict(self, activations: np.ndarray) -> np.ndarray:
        """Predict labels/values from activations."""
        self._check_fitted()
        X = self._prepare_activations(activations, fit_scaler=False)
        return self._estimator.predict(X)

    def predict_proba(self, activations: np.ndarray) -> np.ndarray:
        """Predict class probabilities (classification only)."""
        if not self.is_classification:
            raise ValueError("predict_proba only available for classification probes")
        self._check_fitted()
        X = self._prepare_activations(activations, fit_scaler=False)
        return self._estimator.predict_proba(X)

    def get_direction(self) -> np.ndarray:
        """Return the learned non-linear direction (first layer weights)."""
        self._check_fitted()
        if hasattr(self._estimator, "coefs_") and self._estimator.coefs_:
            return self._estimator.coefs_[0]
        else:
            raise AttributeError(
                "Estimator does not have accessible weights. "
                "This may occur if the model hasn't converged."
            )

    def score_activation(self, activation: np.ndarray) -> np.ndarray:
        """Score a single activation through the learned MLP."""
        self._check_fitted()
        x = np.atleast_2d(activation)
        if self._scaler is not None:
            x = self._scaler.transform(x)

        if hasattr(self._estimator, "coefs_") and self._estimator.coefs_:
            W1 = self._estimator.coefs_[0]
            b1 = self._estimator.intercepts_[0]
            z = np.dot(x, W1) + b1
            return np.maximum(0, z).squeeze()
        else:
            raise AttributeError("Estimator does not have accessible weights.")

    def _prepare_activations(
        self, activations: np.ndarray, fit_scaler: bool
    ) -> np.ndarray:
        """Flatten 3D activations and optionally normalize."""
        X = np.asarray(activations)
        
        if np.any(np.isnan(X)):
            X = np.nan_to_num(X, nan=0.0)
        
        if np.any(np.isinf(X)):
            X = np.where(np.isinf(X), np.sign(X) * 1e6, X)
        
        if X.size > 0:
            abs_max = np.max(np.abs(X))
            if abs_max > 1e6:
                X = np.clip(X, -1e6, 1e6)
                warnings.warn(
                    f"Extreme values detected in activations (max abs: {abs_max:.2e}). "
                    f"Clipped to [-1e6, 1e6].",
                    RuntimeWarning
                )
        
        if X.ndim == 3:
            X = X.mean(axis=1)
        
        if X.ndim != 2:
            raise ValueError(
                f"Expected 2D or 3D activations, got shape {X.shape}. "
                "For 3D activations, the sequence dimension will be averaged."
            )
        
        if self._scaler is not None:
            if fit_scaler:
                X = self._scaler.fit_transform(X)
            else:
                X = self._scaler.transform(X)
        
        return X

    def _check_fitted(self):
        """Raise an error if the probe has not been fitted."""
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} not fitted. Call fit() first."
            )

    def __repr__(self) -> str:
        """String representation of the probe."""
        status = "fitted" if self.is_fitted else "unfitted"
        hidden_str = "x".join(str(d) for d in self.hidden_dims)
        task = "classification" if self.is_classification else "regression"
        return (
            f"{self.__class__.__name__}(layer='{self.layer}', "
            f"direction='{self.direction}', hidden={hidden_str}, "
            f"task={task}, {status})"
        )


# Convenience aliases
MLPProbe = NonLinearProbe
MLPClassifierProbe = lambda layer, direction, **kw: NonLinearProbe(
    layer, direction, is_classification=True, **kw
)
MLPRegressionProbe = lambda layer, direction, **kw: NonLinearProbe(
    layer, direction, is_classification=False, **kw
)