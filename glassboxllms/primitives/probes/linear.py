"""
Linear Probes for Neural Network Interpretability.

Implements:
  - LogisticProbe: Binary/multiclass classification on activations
  - LinearRegressionProbe: Continuous target prediction
  - PCAProbe: Principal component analysis for activation structure
  - CAVProbe: Concept Activation Vectors (directional sensitivity)
"""

from typing import Literal, Optional, Union
import numpy as np
from sklearn.linear_model import SGDClassifier, SGDRegressor, LogisticRegression, Ridge
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    r2_score,
    mean_squared_error,
)

from .base import BaseProbe, ProbeResult


ModelType = Literal["logistic", "linear", "pca", "cav"]


class LinearProbe(BaseProbe):
    """
    Unified interface for fitting linear models to frozen activations.

    Supports multiple probe types:
      - 'logistic': Logistic regression for classification tasks
      - 'linear': Ridge regression for continuous targets
      - 'pca': PCA for finding principal directions in activation space
      - 'cav': Concept Activation Vector (binary direction classifier)

    Key idea: We never modify the neural network's weights. We train a
    simple linear model on frozen activations to test if concepts are
    linearly separable at a given layer.

    Attributes:
        layer (str): Target layer identifier (e.g., "mlp.10").
        direction (str): Name of the concept being probed (e.g., "tense").
        model_type (ModelType): Type of linear model to use.
        normalize (bool): Whether to standardize activations before fitting.

    Example:
        >>> from glassbox.probes.linear import LinearProbe
        >>> probe = LinearProbe(layer="mlp.10", direction="tense")
        >>> probe.fit(activations, labels)
        >>> results = probe.evaluate(test_activations, test_labels)
        >>> print(results)
        ProbeResult(accuracy=0.847, f1=0.831)
    """

    def __init__(
        self,
        layer: str,
        direction: str,
        model_type: ModelType = "logistic",
        normalize: bool = True,
        n_components: int = 1,  # For PCA
        regularization: float = 1.0,  # C for logistic, alpha for ridge
        max_iter: int = 1000,
        random_state: int = 42,
        incremental: bool = False,  # For large datasets
        **kwargs,
    ):
        super().__init__(layer, direction)
        self.model_type = model_type
        self.normalize = normalize
        self.n_components = n_components
        self.regularization = regularization
        self.max_iter = max_iter
        self.random_state = random_state
        self.incremental = incremental
        self.kwargs = kwargs

        self._scaler = StandardScaler() if normalize else None
        self._estimator = self._init_estimator()

    def _init_estimator(self):
        """Initialize the underlying sklearn estimator."""
        if self.model_type == "logistic":
            if self.incremental:
                return SGDClassifier(
                    loss="log_loss",
                    penalty="l2",
                    alpha=1.0 / self.regularization,
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                    **self.kwargs,
                )
            return LogisticRegression(
                C=self.regularization,
                max_iter=self.max_iter,
                random_state=self.random_state,
                solver="lbfgs",
                **self.kwargs,
            )

        elif self.model_type == "linear":
            if self.incremental:
                return SGDRegressor(
                    penalty="l2",
                    alpha=self.regularization,
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                    **self.kwargs,
                )
            return Ridge(
                alpha=self.regularization,
                max_iter=self.max_iter,
                random_state=self.random_state,
                **self.kwargs,
            )

        elif self.model_type == "pca":
            if self.incremental:
                return IncrementalPCA(n_components=self.n_components, **self.kwargs)
            return PCA(
                n_components=self.n_components,
                random_state=self.random_state,
                **self.kwargs,
            )

        elif self.model_type == "cav":
            # CAV uses a linear SVM or logistic regression to find direction
            return LogisticRegression(
                C=self.regularization,
                max_iter=self.max_iter,
                random_state=self.random_state,
                solver="lbfgs",
                **self.kwargs,
            )

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(
        self,
        activations: np.ndarray,
        labels: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "LinearProbe":
        """
        Train the probe on frozen activations.

        Args:
            activations: Shape (n_samples, n_features) or (n_samples, seq_len, n_features).
                         If 3D, will average over seq_len dimension.
            labels: Shape (n_samples,). Required for classification/regression.
                    Not used for PCA.
            sample_weight: Optional per-sample weights.

        Returns:
            self (fitted probe).
        """
        X = self._prepare_activations(activations, fit_scaler=True)

        if self.model_type == "pca":
            self._estimator.fit(X)
        else:
            if labels is None:
                raise ValueError(f"labels required for model_type='{self.model_type}'")
            y = np.asarray(labels)
            if sample_weight is not None:
                self._estimator.fit(X, y, sample_weight=sample_weight)
            else:
                self._estimator.fit(X, y)

        self.is_fitted = True
        return self

    def partial_fit(
        self,
        activations: np.ndarray,
        labels: Optional[np.ndarray] = None,
        classes: Optional[np.ndarray] = None,
    ) -> "LinearProbe":
        """
        Incrementally fit the probe (for large datasets).

        Only works if incremental=True was set during initialization.

        Args:
            activations: Batch of activations (n_samples, n_features).
            labels: Batch of labels (n_samples,).
            classes: All possible class labels (required on first call for classification).

        Returns:
            self.
        """
        if not self.incremental:
            raise RuntimeError("partial_fit requires incremental=True")

        X = self._prepare_activations(activations, fit_scaler=not self.is_fitted)

        if self.model_type == "pca":
            self._estimator.partial_fit(X)
        else:
            if labels is None:
                raise ValueError("labels required for incremental classification/regression")
            y = np.asarray(labels)
            if hasattr(self._estimator, "partial_fit"):
                if classes is not None:
                    self._estimator.partial_fit(X, y, classes=classes)
                else:
                    self._estimator.partial_fit(X, y)

        self.is_fitted = True
        return self

    def predict(self, activations: np.ndarray) -> np.ndarray:
        """
        Predict labels/values from activations.

        Args:
            activations: Shape (n_samples, n_features) or 3D.

        Returns:
            Predictions of shape (n_samples,).
            For PCA, returns transformed coordinates.
        """
        self._check_fitted()
        X = self._prepare_activations(activations, fit_scaler=False)

        if self.model_type == "pca":
            return self._estimator.transform(X)
        return self._estimator.predict(X)

    def predict_proba(self, activations: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (classification only).

        Args:
            activations: Shape (n_samples, n_features).

        Returns:
            Probabilities of shape (n_samples, n_classes).
        """
        if self.model_type not in ("logistic", "cav"):
            raise ValueError("predict_proba only available for logistic/cav probes")
        self._check_fitted()
        X = self._prepare_activations(activations, fit_scaler=False)
        return self._estimator.predict_proba(X)

    def evaluate(
        self,
        activations: np.ndarray,
        labels: np.ndarray,
    ) -> ProbeResult:
        """
        Evaluate probe on held-out data.

        Args:
            activations: Test activations (n_samples, n_features).
            labels: True labels (n_samples,).

        Returns:
            ProbeResult with relevant metrics.
        """
        self._check_fitted()
        X = self._prepare_activations(activations, fit_scaler=False)
        y = np.asarray(labels)

        if self.model_type == "pca":
            # For PCA, report explained variance
            X_transformed = self._estimator.transform(X)
            X_reconstructed = self._estimator.inverse_transform(X_transformed)
            reconstruction_error = np.mean((X - X_reconstructed) ** 2)
            explained_var = self._estimator.explained_variance_ratio_.sum()
            return ProbeResult(
                accuracy=1.0 - reconstruction_error / np.var(X),
                explained_variance=explained_var,
                coefficients=self._estimator.components_,
                metadata={"reconstruction_mse": reconstruction_error},
            )

        elif self.model_type == "linear":
            y_pred = self._estimator.predict(X)
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            return ProbeResult(
                accuracy=r2,
                explained_variance=r2,
                coefficients=self._estimator.coef_,
                metadata={"mse": mse, "rmse": np.sqrt(mse)},
            )

        else:  # logistic or cav
            y_pred = self._estimator.predict(X)
            acc = accuracy_score(y, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y, y_pred, average="weighted", zero_division=0
            )
            return ProbeResult(
                accuracy=acc,
                precision=precision,
                recall=recall,
                f1=f1,
                coefficients=self._estimator.coef_.squeeze(),
                metadata={"n_samples": len(y), "n_classes": len(np.unique(y))},
            )

    def get_direction(self) -> np.ndarray:
        """
        Return the learned linear direction.

        For classification: the weight vector (coef_).
        For PCA: the principal components.
        For CAV: the concept direction vector.

        Returns:
            Array of shape (n_features,) or (n_components, n_features).
        """
        self._check_fitted()

        if self.model_type == "pca":
            return self._estimator.components_

        return self._estimator.coef_.squeeze()

    def score_activation(self, activation: np.ndarray) -> float:
        """
        Score a single activation along the learned direction.

        Useful for CAV analysis: how strongly does this activation
        express the target concept?

        Args:
            activation: Shape (n_features,) or (1, n_features).

        Returns:
            Scalar projection onto the concept direction.
        """
        self._check_fitted()
        direction = self.get_direction()
        if direction.ndim > 1:
            direction = direction[0]  # Use first component/class

        x = np.atleast_2d(activation)
        if self._scaler is not None:
            x = self._scaler.transform(x)

        return float(np.dot(x.squeeze(), direction))

    def _prepare_activations(
        self, activations: np.ndarray, fit_scaler: bool
    ) -> np.ndarray:
        """Flatten 3D activations and optionally normalize."""
        X = np.asarray(activations)

        # If 3D (batch, seq_len, features), average over sequence
        if X.ndim == 3:
            X = X.mean(axis=1)

        if X.ndim != 2:
            raise ValueError(f"Expected 2D or 3D activations, got shape {X.shape}")

        if self._scaler is not None:
            if fit_scaler:
                X = self._scaler.fit_transform(X)
            else:
                X = self._scaler.transform(X)

        return X

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Probe not fitted. Call fit() first.")


# Convenience aliases
LogisticProbe = lambda layer, direction, **kw: LinearProbe(layer, direction, model_type="logistic", **kw)
RegressionProbe = lambda layer, direction, **kw: LinearProbe(layer, direction, model_type="linear", **kw)
PCAProbe = lambda layer, direction, **kw: LinearProbe(layer, direction, model_type="pca", **kw)
CAVProbe = lambda layer, direction, **kw: LinearProbe(layer, direction, model_type="cav", **kw)
