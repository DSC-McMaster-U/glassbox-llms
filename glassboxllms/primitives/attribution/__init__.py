"""Attribution primitives for interpretability workflows."""

from .integrated_gradients import (
    IntegratedGradientsAttributor,
    NodeAttribution,
    select_high_importance_nodes,
)

__all__ = [
    "IntegratedGradientsAttributor",
    "NodeAttribution",
    "select_high_importance_nodes",
]
