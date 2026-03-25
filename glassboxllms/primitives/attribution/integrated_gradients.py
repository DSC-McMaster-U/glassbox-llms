"""
Integrated Gradients attribution primitives.

This module intentionally provides a lightweight, model-agnostic scaffold that
can be used by CircuitDiscoveryExperiment. Real model-specific gradient capture
should be implemented by wrappers/callables that return NodeAttribution values.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class NodeAttribution:
    """Attribution score for a candidate circuit node."""

    node_id: str
    score: float
    node_type: str = "feature"
    layer: Optional[int] = None
    index: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "score": float(self.score),
            "node_type": self.node_type,
            "layer": self.layer,
            "index": self.index,
            "metadata": dict(self.metadata),
        }


def select_high_importance_nodes(
    attributions: Iterable[NodeAttribution],
    threshold: float,
    max_nodes: int = 64,
) -> List[NodeAttribution]:
    """
    Keep top nodes whose attribution score is >= threshold.

    This utility is useful for building deterministic tests and for integrating
    model-specific Integrated Gradients implementations.
    """
    ranked = sorted(attributions, key=lambda item: float(item.score), reverse=True)
    selected = [item for item in ranked if float(item.score) >= threshold]
    return selected[:max_nodes]


class IntegratedGradientsAttributor:
    """
    Adapter for model wrappers that expose integrated gradients at node level.

    Expected model method:
      model.integrated_gradients(task: str) -> Iterable[dict or NodeAttribution]
    """

    def __call__(self, model: Any, task: str) -> List[Dict[str, Any]]:
        if not hasattr(model, "integrated_gradients"):
            raise NotImplementedError(
                "Model does not implement integrated_gradients(task). "
                "Inject a custom attribution_fn into CircuitDiscoveryExperiment "
                "or implement model.integrated_gradients."
            )

        results = model.integrated_gradients(task)
        normalized: List[Dict[str, Any]] = []
        for item in results:
            if isinstance(item, NodeAttribution):
                normalized.append(item.to_dict())
            elif isinstance(item, dict):
                normalized.append(item)
            else:
                raise TypeError(
                    "integrated_gradients(task) must return dicts or NodeAttribution objects"
                )
        return normalized
