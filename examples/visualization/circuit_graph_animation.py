"""
Example: Animate a CircuitGraph with Manim.

Shows how to build a CircuitGraph from real circuit discovery results
and render it as an animated layered directed graph.

Usage:
    manim -ql circuit_graph_animation.py CircuitDiscoveryScene
"""

from glassboxllms.analysis.circuits import CircuitGraph
from glassboxllms.visualization.adapters import circuit_graph_to_scene_data
from glassboxllms.visualization.scenes import CircuitDiscoveryScene


def build_example_circuit() -> CircuitGraph:
    """Build an example induction circuit (IOI-style)."""
    graph = CircuitGraph(model="gpt2-small", name="Induction Circuit")

    # Embedding layer
    graph.add_node("embed.token", node_type="embedding", layer=0, label="Token Embed")

    # Early attention heads (duplicate token detection)
    graph.add_node("attn.0.head.1", node_type="attention_head", layer=1, index=1,
                   label="Dup Token H1")
    graph.add_node("attn.0.head.7", node_type="attention_head", layer=1, index=7,
                   label="Prev Token H7")

    # MLP neurons
    graph.add_node("mlp.2.neuron.42", node_type="neuron", layer=2, index=42,
                   label="N42")
    graph.add_node("mlp.2.neuron.108", node_type="neuron", layer=2, index=108,
                   label="N108")

    # Induction heads
    graph.add_node("attn.5.head.3", node_type="attention_head", layer=5, index=3,
                   label="Induction H3")
    graph.add_node("attn.5.head.11", node_type="attention_head", layer=5, index=11,
                   label="Induction H11")

    # Output
    graph.add_node("unembed.logits", node_type="unembedding", layer=6, label="Logits")

    # Edges with causal weights
    graph.add_edge("embed.token", "attn.0.head.1", weight=0.9, edge_type="direct")
    graph.add_edge("embed.token", "attn.0.head.7", weight=0.85, edge_type="direct")
    graph.add_edge("attn.0.head.1", "mlp.2.neuron.42", weight=0.7, edge_type="attention")
    graph.add_edge("attn.0.head.7", "mlp.2.neuron.108", weight=0.65, edge_type="attention")
    graph.add_edge("mlp.2.neuron.42", "attn.5.head.3", weight=0.8, edge_type="residual")
    graph.add_edge("mlp.2.neuron.108", "attn.5.head.3", weight=0.4, edge_type="residual")
    graph.add_edge("mlp.2.neuron.42", "attn.5.head.11", weight=0.3, edge_type="residual")
    graph.add_edge("mlp.2.neuron.108", "attn.5.head.11", weight=0.75, edge_type="residual")
    graph.add_edge("attn.5.head.3", "unembed.logits", weight=0.95, edge_type="direct")
    graph.add_edge("attn.5.head.11", "unembed.logits", weight=0.6, edge_type="direct")

    return graph


def main():
    graph = build_example_circuit()
    print(graph)
    print(graph.summary())

    # Convert to scene data
    scene_data = circuit_graph_to_scene_data(graph)

    # Render with Manim
    scene = CircuitDiscoveryScene()
    scene.scene_data = scene_data
    scene.render()


if __name__ == "__main__":
    main()
