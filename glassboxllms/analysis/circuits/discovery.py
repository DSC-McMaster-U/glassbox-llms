"""
Circuit discovery orchestration.

This module implements CircuitDiscoveryExperiment, which discovers a task-relevant
circuit by combining:
1) saliency-based node selection,
2) connectivity verification (e.g., path patching), and
3) edge pruning by performance impact.

The experiment returns a CircuitGraph, annotated with discovery metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import Any, Callable, Dict, Iterable, List, Optional

from glassboxllms.analysis.feature_atlas import Atlas, FeatureType

from .graph import CircuitGraph


AttributionFn = Callable[[Any, str], List[Dict[str, Any]]]
ConnectivityFn = Callable[[Any, str, str, str], float]
PerformanceFn = Callable[[Any, str, Optional[CircuitGraph]], float]


@dataclass
class DiscoveryMetrics:
    """Summary metrics produced during circuit discovery."""

    task: str
    threshold: float
    candidate_nodes: int
    selected_nodes: int
    candidate_edges: int
    selected_edges_pre_prune: int
    pruned_edges: int
    baseline_performance: float
    isolated_performance: float
    retention_ratio: float


@dataclass
class CircuitDiscoveryExperiment:
    """
    Discover a task-specific circuit as a directed subgraph.

    The class is model-agnostic and relies on three injected callables:
      - attribution_fn: scores candidate nodes
      - connectivity_fn: estimates directed edge significance
      - performance_fn: evaluates task performance with/without a circuit

    API target:
      circuit = CircuitDiscoveryExperiment(...).discover()
    """

    model: Any
    task: str
    threshold: float = 0.05
    attribution_fn: Optional[AttributionFn] = None
    connectivity_fn: Optional[ConnectivityFn] = None
    performance_fn: Optional[PerformanceFn] = None
    feature_atlas: Optional[Atlas] = None
    model_name: Optional[str] = None
    output_path: Optional[str] = None
    max_nodes: int = 64
    metadata: Dict[str, Any] = field(default_factory=dict)

    def discover(self) -> CircuitGraph:
        """Run attribution -> connectivity -> pruning and return discovered CircuitGraph."""
        self._validate_callables()

        circuit = CircuitGraph(
            model=self.model_name or getattr(self.model, "name", "unknown_model"),
            name=f"{self.task}_discovered_circuit",
            metadata={
                "discovery_method": "attribution_plus_patching",
                "task": self.task,
                "threshold": self.threshold,
                **self.metadata,
            },
        )

        selected_nodes = self._select_nodes_by_saliency(circuit)
        candidate_edges, selected_edges = self._build_connectivity_edges(circuit)
        pruned_edges = self._prune_edges_by_impact(circuit)

        baseline_score = float(self.performance_fn(self.model, self.task, None))
        isolated_score = float(self.performance_fn(self.model, self.task, circuit))
        retention_ratio = (isolated_score / baseline_score) if baseline_score > 0 else 0.0

        if self.feature_atlas is not None:
            self._cross_reference_feature_atlas(circuit)

        metrics = DiscoveryMetrics(
            task=self.task,
            threshold=self.threshold,
            candidate_nodes=len(selected_nodes),
            selected_nodes=len(circuit.nodes),
            candidate_edges=candidate_edges,
            selected_edges_pre_prune=selected_edges,
            pruned_edges=pruned_edges,
            baseline_performance=baseline_score,
            isolated_performance=isolated_score,
            retention_ratio=retention_ratio,
        )

        circuit.metadata["discovery_metrics"] = metrics.__dict__
        circuit.metadata["success_criteria"] = {
            "retention_target": 0.90,
            "retention_met": retention_ratio >= 0.90,
            "feature_atlas_cross_referenced": self.feature_atlas is not None,
        }

        if self.output_path:
            circuit.save(self.output_path)

        return circuit

    def _validate_callables(self) -> None:
        missing = []
        if self.attribution_fn is None:
            missing.append("attribution_fn")
        if self.connectivity_fn is None:
            missing.append("connectivity_fn")
        if self.performance_fn is None:
            missing.append("performance_fn")

        if missing:
            raise ValueError(
                "CircuitDiscoveryExperiment requires injected callables: "
                + ", ".join(missing)
            )

    def _select_nodes_by_saliency(self, circuit: CircuitGraph) -> List[Dict[str, Any]]:
        attributions = list(self.attribution_fn(self.model, self.task))

        if not attributions:
            return []

        sorted_candidates = sorted(
            attributions,
            key=lambda candidate: float(candidate.get("score", 0.0)),
            reverse=True,
        )

        selected: List[Dict[str, Any]] = []
        for candidate in sorted_candidates:
            score = float(candidate.get("score", 0.0))
            if score < self.threshold:
                continue

            node_id = str(candidate["node_id"])
            node_type = str(candidate.get("node_type", "feature"))
            layer = candidate.get("layer")
            index = candidate.get("index")
            node_metadata = dict(candidate.get("metadata", {}))
            node_metadata["saliency_score"] = score

            circuit.add_node(
                node_id=node_id,
                node_type=node_type,
                layer=layer,
                index=index,
                **node_metadata,
            )
            selected.append(candidate)

            if len(selected) >= self.max_nodes:
                break

        return selected

    def _build_connectivity_edges(self, circuit: CircuitGraph) -> tuple[int, int]:
        node_ids = circuit.node_ids
        if len(node_ids) < 2:
            return 0, 0

        candidate_edges = 0
        selected_edges = 0
        for source, target in permutations(node_ids, 2):
            candidate_edges += 1
            effect = float(self.connectivity_fn(self.model, self.task, source, target))
            if abs(effect) < self.threshold:
                continue

            circuit.add_edge(
                source=source,
                target=target,
                weight=effect,
                edge_type="direct",
                connectivity_score=effect,
            )
            selected_edges += 1

        return candidate_edges, selected_edges

    def _prune_edges_by_impact(self, circuit: CircuitGraph) -> int:
        if not circuit.edges:
            return 0

        current_score = float(self.performance_fn(self.model, self.task, circuit))
        pruned = 0

        # Snapshot to avoid mutating during iteration
        candidate_edges = list(circuit.edges)

        for edge in candidate_edges:
            removed = circuit.remove_edge(edge.source, edge.target)
            if not removed:
                continue

            new_score = float(self.performance_fn(self.model, self.task, circuit))
            impact = current_score - new_score

            if impact > self.threshold:
                circuit.add_edge(
                    source=edge.source,
                    target=edge.target,
                    weight=edge.weight,
                    edge_type=edge.edge_type,
                    **edge.metadata,
                )
            else:
                pruned += 1
                current_score = new_score

        return pruned

    def _cross_reference_feature_atlas(self, circuit: CircuitGraph) -> None:
        for node in circuit.nodes:
            matches: List[str] = []

            if node.node_type.value == "feature":
                if self.feature_atlas.get(node.id) is not None:
                    matches.append(node.id)
            elif node.node_type.value == "neuron":
                matches.extend(
                    self._match_by_location(
                        feature_type=FeatureType.NEURON,
                        layer=node.layer,
                        index=node.index,
                    )
                )
            elif node.node_type.value == "attention_head":
                matches.extend(
                    self._match_by_location(
                        feature_type=FeatureType.HEAD,
                        layer=node.layer,
                        index=node.index,
                    )
                )

            node.metadata["feature_atlas_ids"] = sorted(set(matches))

    def _match_by_location(
        self,
        feature_type: FeatureType,
        layer: Optional[int],
        index: Optional[int],
    ) -> List[str]:
        if layer is None:
            return []

        layer_matches = self.feature_atlas.find_by_layer(str(layer), exact=False)
        matched_ids = []

        for feature in layer_matches:
            if feature.feature_type != feature_type:
                continue

            if feature_type == FeatureType.NEURON and feature.location.neuron_idx == index:
                matched_ids.append(feature.id)
            elif feature_type == FeatureType.HEAD and feature.location.head_idx == index:
                matched_ids.append(feature.id)

        return matched_ids
