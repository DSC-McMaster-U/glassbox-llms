# Circuit Graph Module

A graph-based representation for model circuits in GlassboxLLMs.

## Overview

This module provides data structures for representing neural network circuits as directed graphs. A **circuit** is a structured subnetwork within a model that implements a specific behavior (e.g., induction heads, indirect object identification).

**Key components:**
- **Nodes** represent internal model components (neurons, attention heads, SAE features)
- **Edges** represent functional or causal connections between components

> **Note:** This is a *representation* module. It does not perform circuit discovery, causal testing, or attribution—those belong in separate analysis modules.

## Installation

This module is part of GlassboxLLMs and requires no additional dependencies beyond the standard library.

## Quick Start

```python
from glassboxllms.analysis.circuits import CircuitGraph

# Create a new circuit graph
graph = CircuitGraph(model="gpt2-xl", name="induction_circuit")

# Add nodes (neurons, attention heads, features)
graph.add_node("mlp.5.neuron.42", node_type="neuron", layer=5, index=42)
graph.add_node("attn.10.head.3", node_type="attention_head", layer=10, index=3)
graph.add_node("feature.sae_v1.1234", node_type="feature")

# Add directed edges with optional weights
graph.add_edge(
    source="mlp.5.neuron.42",
    target="attn.10.head.3",
    weight=0.8,
    discovered_by="activation_patching"
)

# Save to JSON
graph.save("circuits/induction_circuit.json")

# Load from JSON
loaded = CircuitGraph.load("circuits/induction_circuit.json")
```

## Node Types

| Type | Description | Example ID |
|------|-------------|------------|
| `neuron` | MLP layer neuron | `mlp.5.neuron.42` |
| `attention_head` | Attention head | `attn.10.head.3` |
| `feature` | SAE/Feature Atlas feature | `feature.sae_v1.1234` |
| `mlp_layer` | Entire MLP layer | `mlp.5` |
| `residual_stream` | Residual stream position | `resid.post.5` |
| `embedding` | Input embedding | `embed.input` |
| `unembedding` | Output unembedding | `unembed.output` |

## Edge Types

| Type | Description |
|------|-------------|
| `direct` | Direct causal connection |
| `attention` | Attention-mediated connection |
| `residual` | Through residual stream |
| `inferred` | Discovered via analysis (e.g., patching) |
| `manual` | User-annotated connection |

## API Reference

### CircuitGraph

```python
class CircuitGraph:
    def __init__(self, model: str, name: str = None, metadata: dict = None)
```

#### Node Operations

| Method | Description |
|--------|-------------|
| `add_node(id, node_type, layer=None, index=None, **metadata)` | Add a node |
| `get_node(id)` | Get node by ID (or None) |
| `remove_node(id)` | Remove node and connected edges |
| `has_node(id)` | Check if node exists |
| `nodes` | List all nodes |
| `node_ids` | List all node IDs |
| `nodes_by_type(type)` | Filter nodes by type |
| `nodes_by_layer(layer)` | Filter nodes by layer |

#### Edge Operations

| Method | Description |
|--------|-------------|
| `add_edge(source, target, weight=None, edge_type="direct", **metadata)` | Add directed edge |
| `get_edge(source, target)` | Get edge (or None) |
| `remove_edge(source, target)` | Remove edge |
| `has_edge(source, target)` | Check if edge exists |
| `edges` | List all edges |
| `edges_from(node_id)` | Edges originating from node |
| `edges_to(node_id)` | Edges pointing to node |

#### Graph Traversal

| Method | Description |
|--------|-------------|
| `get_neighbors(id, direction="out")` | Get neighboring nodes |
| `predecessors(id)` | Nodes pointing to this node |
| `successors(id)` | Nodes this node points to |
| `in_degree(id)` | Count of incoming edges |
| `out_degree(id)` | Count of outgoing edges |
| `sources()` | Nodes with no incoming edges |
| `sinks()` | Nodes with no outgoing edges |

#### Subgraph Operations

| Method | Description |
|--------|-------------|
| `get_subgraph(node_ids)` | Extract induced subgraph |
| `merge(other, overwrite=False)` | Merge another graph |

#### Serialization

| Method | Description |
|--------|-------------|
| `to_dict()` | Serialize to dictionary |
| `from_dict(data)` | Deserialize from dictionary |
| `save(path)` | Save to JSON file |
| `load(path)` | Load from JSON file |

#### Statistics

| Method | Description |
|--------|-------------|
| `summary()` | Get graph statistics |

## JSON Schema

```json
{
  "version": "1.0",
  "model": "gpt2-xl",
  "name": "induction_circuit",
  "metadata": {
    "source": "manual_annotation",
    "serialized_at": "2026-02-15T10:30:00"
  },
  "nodes": [
    {
      "id": "mlp.5.neuron.42",
      "type": "neuron",
      "layer": 5,
      "index": 42,
      "metadata": {"label": "key_mover"}
    }
  ],
  "edges": [
    {
      "source": "mlp.5.neuron.42",
      "target": "attn.10.head.3",
      "weight": 0.8,
      "edge_type": "inferred",
      "metadata": {"discovered_by": "patching"}
    }
  ]
}
```

## Examples

### Building an Induction Circuit

```python
from glassboxllms.analysis.circuits import CircuitGraph

# Induction heads copy patterns from earlier in context
graph = CircuitGraph(model="gpt2-small", name="induction_circuit")

# Previous token head (attends to token before the current one)
graph.add_node("attn.5.head.1", node_type="attention_head", layer=5, index=1,
               role="previous_token_head")

# Induction head (copies pattern)
graph.add_node("attn.6.head.9", node_type="attention_head", layer=6, index=9,
               role="induction_head")

# The induction head uses info from previous token head
graph.add_edge("attn.5.head.1", "attn.6.head.9", 
               weight=0.92, edge_type="inferred",
               evidence="activation_patching")

print(graph.summary())
```

### Extracting a Subcircuit

```python
# Get just the attention-related components
attention_nodes = graph.nodes_by_type("attention_head")
attention_ids = [n.id for n in attention_nodes]
attention_subgraph = graph.get_subgraph(attention_ids)
```

### Analyzing Circuit Structure

```python
# Find entry and exit points
entry_points = graph.sources()
exit_points = graph.sinks()

# Get statistics
stats = graph.summary()
print(f"Circuit has {stats['num_nodes']} components")
print(f"Spans layers {stats['layer_range'][0]} to {stats['layer_range'][1]}")
```

## Related Modules

- **`glassboxllms.instrumentation.activation_patching`** - Discover circuits via causal interventions
- **`glassboxllms.primitives.probes`** - Test what information is encoded in circuit components

## References

- [Anthropic Transformer Circuits Thread](https://transformer-circuits.pub)
- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
- [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
