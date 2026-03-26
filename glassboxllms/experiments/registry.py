"""
Experiment registry — discover and run any experiment by name.

Usage:
    from glassboxllms.experiments import run_experiment, list_experiments

    print(list_experiments())
    result = run_experiment("cot_faithfulness", {"generate_fn": fn, ...})
"""

from typing import Any, Dict, List, Type

from glassboxllms.experiments.base import BaseExperiment, ExperimentResult
from glassboxllms.experiments.cot_faithfulness.experiment import CoTFaithfulnessExperiment
from glassboxllms.experiments.logit_lens import LogitLensExperiment
from glassboxllms.experiments.probing import ProbingExperiment

# Central registry — add new experiments here
_REGISTRY: Dict[str, Type[BaseExperiment]] = {
    "cot_faithfulness": CoTFaithfulnessExperiment,
    "logit_lens": LogitLensExperiment,
    "probing": ProbingExperiment,
}


def register_experiment(name: str, cls: Type[BaseExperiment]) -> None:
    """Register a new experiment class under *name*."""
    _REGISTRY[name] = cls


def list_experiments() -> List[str]:
    """Return all registered experiment names."""
    return list(_REGISTRY.keys())


def get_experiment(name: str) -> BaseExperiment:
    """Instantiate and return the experiment registered under *name*."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown experiment '{name}'. Available: {list_experiments()}"
        )
    return _REGISTRY[name]()


def run_experiment(name: str, config: Dict[str, Any]) -> ExperimentResult:
    """One-liner: look up an experiment by name and run it with *config*."""
    return get_experiment(name).run(config)
