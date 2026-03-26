"""
glassboxllms.experiments — Unified experiment framework.

Run any registered experiment with a single call::

    from glassboxllms.experiments import run_experiment

    result = run_experiment("cot_faithfulness", {
        "generate_fn": my_fn,
        "model_name": "gpt2",
    })
    print(result.summary())
"""

from glassboxllms.experiments.base import BaseExperiment, ExperimentResult
from glassboxllms.experiments.registry import (
    run_experiment,
    list_experiments,
    get_experiment,
    register_experiment,
)

__all__ = [
    "BaseExperiment",
    "ExperimentResult",
    "run_experiment",
    "list_experiments",
    "get_experiment",
    "register_experiment",
]
