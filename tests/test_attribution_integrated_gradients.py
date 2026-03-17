from glassboxllms.primitives.attribution import (
    IntegratedGradientsAttributor,
    NodeAttribution,
    select_high_importance_nodes,
)


def test_select_high_importance_nodes_filters_and_sorts():
    attributions = [
        NodeAttribution(node_id="a", score=0.1),
        NodeAttribution(node_id="b", score=0.8),
        NodeAttribution(node_id="c", score=0.4),
        NodeAttribution(node_id="d", score=0.01),
    ]

    selected = select_high_importance_nodes(attributions, threshold=0.1, max_nodes=2)

    assert [node.node_id for node in selected] == ["b", "c"]


def test_integrated_gradients_attributor_normalizes_output():
    class DummyModel:
        def integrated_gradients(self, _task):
            return [
                NodeAttribution(node_id="x", score=0.7, node_type="neuron").to_dict(),
                NodeAttribution(node_id="y", score=0.5, node_type="attention_head"),
            ]

    attributor = IntegratedGradientsAttributor()
    output = attributor(DummyModel(), "ioi")

    assert isinstance(output, list)
    assert output[0]["node_id"] == "x"
    assert output[1]["node_id"] == "y"


def test_integrated_gradients_attributor_raises_for_missing_model_method():
    class DummyModel:
        pass

    attributor = IntegratedGradientsAttributor()

    try:
        attributor(DummyModel(), "ioi")
        raise AssertionError("Expected NotImplementedError")
    except NotImplementedError:
        pass
