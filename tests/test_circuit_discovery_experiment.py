import os

from glassboxllms.analysis.circuits import CircuitDiscoveryExperiment
from glassboxllms.analysis.feature_atlas import Atlas, Feature, FeatureType, Location


class DummyModel:
    name = "dummy-transformer"


def _attribution_fn(_model, _task):
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


def _connectivity_fn(_model, _task, source, target):
    if source == "mlp.5.neuron.42" and target == "attn.6.head.3":
        return 0.20
    if source == "attn.6.head.3" and target == "mlp.5.neuron.42":
        return 0.02
    return 0.0


def _performance_fn(_model, _task, circuit):
    if circuit is None:
        return 1.0

    # Simple deterministic scoring model:
    # - key edge contributes +0.95
    # - any extra edge contributes +0.01
    score = 0.0
    if circuit.has_edge("mlp.5.neuron.42", "attn.6.head.3"):
        score += 0.95

    for edge in circuit.edges:
        if (edge.source, edge.target) != ("mlp.5.neuron.42", "attn.6.head.3"):
            score += 0.01

    return min(score, 1.0)


def test_discover_returns_circuit_with_expected_nodes_edges():
    experiment = CircuitDiscoveryExperiment(
        model=DummyModel(),
        task="indirect_object_identification",
        threshold=0.05,
        attribution_fn=_attribution_fn,
        connectivity_fn=_connectivity_fn,
        performance_fn=_performance_fn,
    )

    circuit = experiment.discover()

    assert circuit.model == "dummy-transformer"
    assert circuit.name == "indirect_object_identification_discovered_circuit"
    assert circuit.has_node("mlp.5.neuron.42")
    assert circuit.has_node("attn.6.head.3")
    assert not circuit.has_node("mlp.2.neuron.1")  # filtered by threshold
    assert circuit.has_edge("mlp.5.neuron.42", "attn.6.head.3")
    assert not circuit.has_edge("attn.6.head.3", "mlp.5.neuron.42")


def test_discovery_metadata_contains_success_criteria():
    experiment = CircuitDiscoveryExperiment(
        model=DummyModel(),
        task="indirect_object_identification",
        threshold=0.05,
        attribution_fn=_attribution_fn,
        connectivity_fn=_connectivity_fn,
        performance_fn=_performance_fn,
    )

    circuit = experiment.discover()
    metrics = circuit.metadata["discovery_metrics"]

    assert metrics["baseline_performance"] == 1.0
    assert metrics["isolated_performance"] >= 0.90
    assert metrics["retention_ratio"] >= 0.90
    assert circuit.metadata["success_criteria"]["retention_met"] is True


def test_discovery_serialization_output(tmp_path):
    output_path = os.path.join(tmp_path, "discovered_circuit.json")

    experiment = CircuitDiscoveryExperiment(
        model=DummyModel(),
        task="indirect_object_identification",
        threshold=0.05,
        attribution_fn=_attribution_fn,
        connectivity_fn=_connectivity_fn,
        performance_fn=_performance_fn,
        output_path=output_path,
    )

    circuit = experiment.discover()

    assert os.path.exists(output_path)
    loaded = circuit.load(output_path)
    assert loaded.has_edge("mlp.5.neuron.42", "attn.6.head.3")


def test_feature_atlas_cross_reference_is_attached():
    atlas = Atlas(name="test-atlas", model_name="dummy-transformer")
    atlas.add(
        Feature(
            feature_type=FeatureType.NEURON,
            label="Neuron 42 feature",
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
            label="Head 3 feature",
            location=Location(
                model_name="dummy-transformer",
                layer="attn.6",
                head_idx=3,
            ),
        )
    )

    experiment = CircuitDiscoveryExperiment(
        model=DummyModel(),
        task="indirect_object_identification",
        threshold=0.05,
        attribution_fn=_attribution_fn,
        connectivity_fn=_connectivity_fn,
        performance_fn=_performance_fn,
        feature_atlas=atlas,
    )

    circuit = experiment.discover()

    neuron_node = circuit.get_node("mlp.5.neuron.42")
    head_node = circuit.get_node("attn.6.head.3")

    assert len(neuron_node.metadata["feature_atlas_ids"]) >= 1
    assert len(head_node.metadata["feature_atlas_ids"]) >= 1
    assert circuit.metadata["success_criteria"]["feature_atlas_cross_referenced"] is True


def test_discovery_requires_injected_callables():
    experiment = CircuitDiscoveryExperiment(model=DummyModel(), task="ioi")

    try:
        experiment.discover()
        raise AssertionError("discover() should fail without callables")
    except ValueError as error:
        message = str(error)
        assert "attribution_fn" in message
        assert "connectivity_fn" in message
        assert "performance_fn" in message
