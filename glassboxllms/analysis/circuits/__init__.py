"""
glassbox-llms / analysis / circuits
===================================

Graph-based representation for model circuits.

A circuit is a structured subnetwork within a model that implements a 
specific behavior. This module provides data structures for representing
circuits as directed graphs where:

- **Nodes** represent internal components (neurons, attention heads, features)
- **Edges** represent functional or causal connections

This is a representation module - it does not perform automatic circuit 
discovery, causal testing, or attribution. Those belong in separate modules.

Example:
    >>> from glassboxllms.analysis.circuits import CircuitGraph
    >>>
    >>> graph = CircuitGraph(model="gpt2-xl")
    >>> graph.add_node("mlp.5.neuron.42", node_type="neuron")
    >>> graph.add_node("attn.10.head.3", node_type="attention_head")
    >>> graph.add_edge("mlp.5.neuron.42", "attn.10.head.3", weight=0.8)
    >>>
    >>> graph.save("induction_circuit.json")

See Also:
    - Anthropic's Transformer Circuits: https://transformer-circuits.pub
    - Mechanistic Interpretability research for circuit discovery methods
"""

from .graph import CircuitGraph
from .node import CircuitEdge, CircuitNode, EdgeType, NodeType

__all__ = [
    "CircuitGraph",
    "CircuitNode",
    "CircuitEdge",
    "NodeType",
    "EdgeType",
]
