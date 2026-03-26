"""
CoT Faithfulness wrapped in the unified BaseExperiment interface.

Usage:
    from glassboxllms.experiments import run_experiment

    result = run_experiment("cot_faithfulness", {
        "generate_fn": my_llm_generate,
        "model_name": "gpt2",
        "dataset": "arc",
        "n_samples": 20,
    })
"""

from typing import Any, Dict

from glassboxllms.experiments.base import BaseExperiment, ExperimentResult
from glassboxllms.experiments.cot_faithfulness.evaluator import (
    CoTFaithfulnessEvaluator,
)


class CoTFaithfulnessExperiment(BaseExperiment):
    """Evaluate whether a model's chain-of-thought actually drives its answers."""

    @property
    def name(self) -> str:
        return "cot_faithfulness"

    @property
    def default_config(self) -> Dict[str, Any]:
        return {
            "model_name": "unknown",
            "dataset": "arc",
            "n_samples": 20,
            "seed": 42,
            "verbose": True,
        }

    def validate_config(self, config: Dict[str, Any]) -> bool:
        return "generate_fn" in config

    def run(self, config: Dict[str, Any]) -> ExperimentResult:
        merged = {**self.default_config, **config}
        if not self.validate_config(merged):
            return ExperimentResult(
                experiment_type=self.name,
                model_name=merged.get("model_name", "unknown"),
                status="failed",
                metrics={},
                config=merged,
                details={"error": "config must include 'generate_fn'"},
            )

        evaluator = CoTFaithfulnessEvaluator(seed=merged["seed"])
        result = evaluator.evaluate(
            generate_fn=merged["generate_fn"],
            model_name=merged["model_name"],
            dataset=merged["dataset"],
            n_samples=merged["n_samples"],
            verbose=merged["verbose"],
        )

        return ExperimentResult(
            experiment_type=self.name,
            model_name=result.model_name,
            status="success",
            metrics={
                "truncation_faithfulness": result.truncation_faithfulness,
                "error_following": result.error_following,
                "avg_faithfulness": result.avg_faithfulness,
            },
            artifacts={
                "truncation_details": result.truncation_details,
                "error_details": result.error_details,
            },
            config=merged,
        )
