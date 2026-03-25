"""
Validation script for CircuitDiscoveryExperiment.

Runs a deterministic end-to-end check aligned with ticket deliverables:
- attribution-based node selection
- connectivity + pruning
- retention ratio >= 0.90
- FeatureAtlas cross-reference
- JSON serialization roundtrip

Output:
    Discovered circuit artifact: `outputs/representations/circuits/discovered_circuit.json`

Usage:
    python -m pytest tests/test_circuit_discovery_validation.py -v -s
    OR
    python tests/test_circuit_discovery_validation.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from glassboxllms.analysis.circuits import CircuitDiscoveryExperiment, CircuitGraph
from glassboxllms.analysis.feature_atlas import Atlas, Feature, FeatureType, Location


class DummyModel:
    name = "dummy-transformer"


def attribution_fn(_model, _task):
    return [
        {
            "node_id": "mlp.5.neuron.42",
            "node_type": "neuron",
            "layer": 5,
            "index": 42,
            "score": 0.40,
        },
        {
            "node_id": "attn.6.head.3",
            "node_type": "attention_head",
            "layer": 6,
            "index": 3,
            "score": 0.32,
        },
        {
            "node_id": "mlp.2.neuron.1",
            "node_type": "neuron",
            "layer": 2,
            "index": 1,
            "score": 0.01,
        },
    ]


def connectivity_fn(_model, _task, source, target):
    if source == "mlp.5.neuron.42" and target == "attn.6.head.3":
        return 0.20
    if source == "attn.6.head.3" and target == "mlp.5.neuron.42":
        return 0.02
    return 0.0


def performance_fn(_model, _task, circuit):
    if circuit is None:
        return 1.0

    score = 0.0
    if circuit.has_edge("mlp.5.neuron.42", "attn.6.head.3"):
        score += 0.95

    for edge in circuit.edges:
        if (edge.source, edge.target) != ("mlp.5.neuron.42", "attn.6.head.3"):
            score += 0.01

    return min(score, 1.0)


def build_atlas():
    atlas = Atlas(name="validate-atlas", model_name="dummy-transformer")
    atlas.add(
        Feature(
            feature_type=FeatureType.NEURON,
            label="Neuron 42",
            location=Location(
                model_name="dummy-transformer",
                layer="mlp.5",
                neuron_idx=42,
            ),
        )
    )
    atlas.add(
        Feature(
            feature_type=FeatureType.HEAD,
            label="Head 3",
            location=Location(
                model_name="dummy-transformer",
                layer="attn.6",
                head_idx=3,
            ),
        )
    )
    return atlas


def main() -> int:
    print("[1/4] Building atlas and running discovery...")
    atlas = build_atlas()

    output_path = ROOT / "outputs" / "representations" / "circuits" / "discovered_circuit.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    experiment = CircuitDiscoveryExperiment(
        model=DummyModel(),
        task="indirect_object_identification",
        threshold=0.05,
        attribution_fn=attribution_fn,
        connectivity_fn=connectivity_fn,
        performance_fn=performance_fn,
        feature_atlas=atlas,
        output_path=str(output_path),
    )

    circuit = experiment.discover()

    print("[2/4] Checking core graph shape...")
    assert circuit.has_node("mlp.5.neuron.42")
    assert circuit.has_node("attn.6.head.3")
    assert not circuit.has_node("mlp.2.neuron.1")
    assert circuit.has_edge("mlp.5.neuron.42", "attn.6.head.3")

    print("[3/4] Checking ticket success criteria...")
    metrics = circuit.metadata["discovery_metrics"]
    retention = float(metrics["retention_ratio"])
    assert retention >= 0.90, f"Retention below threshold: {retention:.3f}"
    assert circuit.metadata["success_criteria"]["retention_met"] is True

    neuron_node = circuit.get_node("mlp.5.neuron.42")
    head_node = circuit.get_node("attn.6.head.3")
    assert len(neuron_node.metadata.get("feature_atlas_ids", [])) >= 1
    assert len(head_node.metadata.get("feature_atlas_ids", [])) >= 1

    print("[4/4] Checking JSON serialization roundtrip...")
    assert output_path.exists(), "Output JSON was not written"
    loaded = CircuitGraph.load(output_path)
    assert loaded.has_edge("mlp.5.neuron.42", "attn.6.head.3")

    payload = loaded.to_dict()
    json.dumps(payload)

    print("\n--- Discovered Subgraph ---")
    print(f"Saved JSON: {output_path}")
    print(f"Nodes ({len(loaded.nodes)}):")
    for node in loaded.nodes:
        print(f"  - {node.id} [{node.node_type.value}] atlas={node.metadata.get('feature_atlas_ids', [])}")
    print(f"Edges ({len(loaded.edges)}):")
    for edge in loaded.edges:
        print(f"  - {edge.source} -> {edge.target} (weight={edge.weight})")

    print("✅ Circuit discovery validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
