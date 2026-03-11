"""
Probing experiment wrapped in the unified BaseExperiment interface.

Trains a linear or nonlinear probe on frozen activations to detect
linearly-encoded concepts in a model's representation space.

Usage:
    from glassboxllms.experiments import run_experiment

    result = run_experiment("probing", {
        "activations_train": X_train,   # np.ndarray
        "labels_train": y_train,        # np.ndarray
        "activations_test": X_test,
        "labels_test": y_test,
        "probe_type": "logistic",       # logistic | linear | pca | cav | nonlinear
        "model_name": "gpt2",
        "layer": "transformer.h.5",
    })
"""

from typing import Any, Dict

from glassboxllms.experiments.base import BaseExperiment, ExperimentResult


class ProbingExperiment(BaseExperiment):
    """Train a probe on frozen activations and evaluate."""

    @property
    def name(self) -> str:
        return "probing"

    @property
    def default_config(self) -> Dict[str, Any]:
        return {
            "probe_type": "logistic",
            "model_name": "unknown",
            "layer": "unknown",
            "direction": "concept",
        }

    def validate_config(self, config: Dict[str, Any]) -> bool:
        required = ["activations_train", "labels_train", "activations_test", "labels_test"]
        return all(k in config for k in required)

    def run(self, config: Dict[str, Any]) -> ExperimentResult:
        merged = {**self.default_config, **config}

        if not self.validate_config(merged):
            return ExperimentResult(
                experiment_type=self.name,
                model_name=merged.get("model_name", "unknown"),
                status="failed",
                metrics={},
                config={k: str(type(v)) for k, v in merged.items()},
                details={"error": "Missing required keys: activations_train/test, labels_train/test"},
            )

        probe_type = merged["probe_type"]

        try:
            if probe_type == "nonlinear":
                from glassboxllms.primitives.probes.nonlinear import NonLinearProbe
                probe = NonLinearProbe(
                    layer=merged["layer"],
                    direction=merged["direction"],
                )
            else:
                from glassboxllms.primitives.probes.linear import LinearProbe
                probe = LinearProbe(
                    layer=merged["layer"],
                    direction=merged["direction"],
                    model_type=probe_type,
                )

            probe.fit(merged["activations_train"], merged["labels_train"])
            result = probe.evaluate(merged["activations_test"], merged["labels_test"])

            metrics = {
                "accuracy": result.accuracy,
            }
            if result.precision is not None:
                metrics["precision"] = result.precision
            if result.recall is not None:
                metrics["recall"] = result.recall
            if result.f1 is not None:
                metrics["f1"] = result.f1
            if result.explained_variance is not None:
                metrics["explained_variance"] = result.explained_variance

            artifacts = {}
            if result.coefficients is not None:
                artifacts["direction_vector"] = result.coefficients

            return ExperimentResult(
                experiment_type=self.name,
                model_name=merged["model_name"],
                status="success",
                metrics=metrics,
                artifacts=artifacts,
                config={k: str(v) if not isinstance(v, (str, int, float, bool)) else v
                        for k, v in merged.items()},
            )

        except Exception as e:
            return ExperimentResult(
                experiment_type=self.name,
                model_name=merged.get("model_name", "unknown"),
                status="failed",
                metrics={},
                config={k: str(type(v)) for k, v in merged.items()},
                details={"error": str(e)},
            )
