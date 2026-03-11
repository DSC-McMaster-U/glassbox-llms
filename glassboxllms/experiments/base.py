"""
Unified experiment interface for glassbox-llms.

Every experiment in the library implements ``BaseExperiment`` so that it
can be discovered, configured, and executed through a single API::

    from glassboxllms.experiments import run_experiment

    result = run_experiment("cot_faithfulness", {
        "generate_fn": my_fn,
        "model_name": "gpt2",
        "dataset": "arc",
    })
    print(result.summary())
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentResult:
    """Standardised result container returned by every experiment."""

    experiment_type: str
    model_name: str
    status: str  # "success" | "failed" | "partial"
    metrics: Dict[str, Any]
    artifacts: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    details: Optional[Dict[str, Any]] = None

    def summary(self) -> str:
        """Pretty-print a one-screen overview."""
        lines = [
            f"\n{'='*60}",
            f"  {self.experiment_type.upper()} — {self.model_name}",
            f"  Status: {self.status}  |  {self.timestamp[:19]}",
            f"{'='*60}",
        ]
        for k, v in self.metrics.items():
            if isinstance(v, float):
                lines.append(f"  {k:<30} {v:>8.2f}")
            else:
                lines.append(f"  {k:<30} {v!s:>8}")
        lines.append(f"{'='*60}\n")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseExperiment(ABC):
    """
    Abstract base for all glassbox-llms experiments.

    Subclasses must implement ``run``, ``validate_config``, ``name``,
    and ``default_config``.
    """

    @abstractmethod
    def run(self, config: Dict[str, Any]) -> ExperimentResult:
        """Execute the experiment with *config* and return a result."""
        ...

    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Return True if *config* has every required key."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable experiment name."""
        ...

    @property
    @abstractmethod
    def default_config(self) -> Dict[str, Any]:
        """Sensible defaults — users merge their overrides on top."""
        ...
