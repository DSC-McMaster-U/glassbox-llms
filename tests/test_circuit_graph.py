"""
Unit tests for the CircuitGraph module.

Tests cover:
- Node operations (add, remove, query)
- Edge operations (add, remove, query)
- Graph traversal (neighbors, predecessors, successors)
- Subgraph extraction
- Serialization and deserialization (dict and JSON file)
- Edge cases and error handling
"""

import json
import os
import shutil
import tempfile

import pytest
# run pip install pytest>=7.0.0

from glassboxllms.analysis.circuits import (
    CircuitGraph,
    CircuitNode,
    CircuitEdge,
    NodeType,
    EdgeType,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def empty_graph():
    """Create an empty circuit graph."""
    return CircuitGraph(model="gpt2-small")


@pytest.fixture
def sample_graph():
    """Create a sample graph with nodes and edges."""
    g = CircuitGraph(model="gpt2-small", name="test_circuit")
    
    # Add neurons
    g.add_node("mlp.0.neuron.1", node_type="neuron", layer=0, index=1)
    g.add_node("mlp.0.neuron.2", node_type="neuron", layer=0, index=2)
    g.add_node("mlp.1.neuron.1", node_type="neuron", layer=1, index=1)
    
    # Add attention heads
    g.add_node("attn.1.head.0", node_type="attention_head", layer=1, index=0)
    g.add_node("attn.2.head.3", node_type="attention_head", layer=2, index=3)
    
    # Add edges
    g.add_edge("mlp.0.neuron.1", "attn.1.head.0", weight=0.5)
    g.add_edge("mlp.0.neuron.2", "attn.1.head.0", weight=0.3)
    g.add_edge("attn.1.head.0", "mlp.1.neuron.1", weight=0.8)
    g.add_edge("mlp.1.neuron.1", "attn.2.head.3", weight=0.6)
    
    return g


@pytest.fixture
def temp_dir():
    """Create a temporary directory for file tests."""
    dirpath = tempfile.mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath)


# =============================================================================
# Node Operation Tests
# =============================================================================

class TestNodeOperations:
    
    def test_add_node_basic(self, empty_graph):
        """Test adding a basic node."""
        node = empty_graph.add_node("mlp.5.neuron.42", node_type="neuron")
        
        assert node.id == "mlp.5.neuron.42"
        assert node.node_type == NodeType.NEURON
        assert len(empty_graph) == 1
        assert "mlp.5.neuron.42" in empty_graph
    
    def test_add_node_with_metadata(self, empty_graph):
        """Test adding a node with layer, index, and custom metadata."""
        node = empty_graph.add_node(
            "attn.10.head.3",
            node_type="attention_head",
            layer=10,
            index=3,
            label="key_mover",
            importance=0.95,
        )
        
        assert node.layer == 10
        assert node.index == 3
        assert node.metadata["label"] == "key_mover"
        assert node.metadata["importance"] == 0.95
    
    def test_add_node_string_type_conversion(self, empty_graph):
        """Test that string node types are converted to enums."""
        node = empty_graph.add_node("test.node", node_type="ATTENTION_HEAD")
        assert node.node_type == NodeType.ATTENTION_HEAD
        
        node2 = empty_graph.add_node("test.node2", node_type="feature")
        assert node2.node_type == NodeType.FEATURE
    
    def test_add_duplicate_node_raises(self, empty_graph):
        """Test that adding a duplicate node raises ValueError."""
        empty_graph.add_node("mlp.0.neuron.1", node_type="neuron")
        
        with pytest.raises(ValueError, match="already exists"):
            empty_graph.add_node("mlp.0.neuron.1", node_type="neuron")
    
    def test_get_node(self, sample_graph):
        """Test retrieving a node by ID."""
        node = sample_graph.get_node("mlp.0.neuron.1")
        
        assert node is not None
        assert node.id == "mlp.0.neuron.1"
        assert node.node_type == NodeType.NEURON
    
    def test_get_node_not_found(self, sample_graph):
        """Test that get_node returns None for non-existent node."""
        node = sample_graph.get_node("nonexistent.node")
        assert node is None
    
    def test_remove_node(self, sample_graph):
        """Test removing a node."""
        initial_count = len(sample_graph)
        result = sample_graph.remove_node("mlp.0.neuron.2")
        
        assert result is True
        assert len(sample_graph) == initial_count - 1
        assert "mlp.0.neuron.2" not in sample_graph
    
    def test_remove_node_removes_connected_edges(self, sample_graph):
        """Test that removing a node also removes its edges."""
        # attn.1.head.0 has 2 incoming and 1 outgoing edge
        assert sample_graph.has_edge("mlp.0.neuron.1", "attn.1.head.0")
        assert sample_graph.has_edge("attn.1.head.0", "mlp.1.neuron.1")
        
        sample_graph.remove_node("attn.1.head.0")
        
        assert not sample_graph.has_edge("mlp.0.neuron.1", "attn.1.head.0")
        assert not sample_graph.has_edge("attn.1.head.0", "mlp.1.neuron.1")
    
    def test_remove_nonexistent_node(self, sample_graph):
        """Test removing a non-existent node returns False."""
        result = sample_graph.remove_node("nonexistent")
        assert result is False
    
    def test_has_node(self, sample_graph):
        """Test has_node method."""
        assert sample_graph.has_node("mlp.0.neuron.1")
        assert not sample_graph.has_node("nonexistent")
    
    def test_nodes_property(self, sample_graph):
        """Test nodes property returns all nodes."""
        nodes = sample_graph.nodes
        assert len(nodes) == 5
        assert all(isinstance(n, CircuitNode) for n in nodes)
    
    def test_node_ids_property(self, sample_graph):
        """Test node_ids property."""
        ids = sample_graph.node_ids
        assert "mlp.0.neuron.1" in ids
        assert "attn.1.head.0" in ids
    
    def test_nodes_by_type(self, sample_graph):
        """Test filtering nodes by type."""
        neurons = sample_graph.nodes_by_type("neuron")
        heads = sample_graph.nodes_by_type(NodeType.ATTENTION_HEAD)
        
        assert len(neurons) == 3
        assert len(heads) == 2
        assert all(n.node_type == NodeType.NEURON for n in neurons)
    
    def test_nodes_by_layer(self, sample_graph):
        """Test filtering nodes by layer."""
        layer0 = sample_graph.nodes_by_layer(0)
        layer1 = sample_graph.nodes_by_layer(1)
        
        assert len(layer0) == 2
        assert len(layer1) == 2


# =============================================================================
# Edge Operation Tests
# =============================================================================

class TestEdgeOperations:
    
    def test_add_edge_basic(self, empty_graph):
        """Test adding a basic edge."""
        empty_graph.add_node("a", node_type="neuron")
        empty_graph.add_node("b", node_type="neuron")
        
        edge = empty_graph.add_edge("a", "b")
        
        assert edge.source == "a"
        assert edge.target == "b"
        assert empty_graph.has_edge("a", "b")
    
    def test_add_edge_with_weight(self, empty_graph):
        """Test adding an edge with weight."""
        empty_graph.add_node("a", node_type="neuron")
        empty_graph.add_node("b", node_type="neuron")
        
        edge = empty_graph.add_edge("a", "b", weight=0.75)
        
        assert edge.weight == 0.75
    
    def test_add_edge_with_type_and_metadata(self, empty_graph):
        """Test adding an edge with type and metadata."""
        empty_graph.add_node("a", node_type="neuron")
        empty_graph.add_node("b", node_type="attention_head")
        
        edge = empty_graph.add_edge(
            "a", "b",
            weight=0.5,
            edge_type="attention",
            discovered_by="patching",
        )
        
        assert edge.edge_type == EdgeType.ATTENTION
        assert edge.metadata["discovered_by"] == "patching"
    
    def test_add_edge_nonexistent_source_raises(self, empty_graph):
        """Test that adding edge with non-existent source raises."""
        empty_graph.add_node("b", node_type="neuron")
        
        with pytest.raises(ValueError, match="Source node.*does not exist"):
            empty_graph.add_edge("a", "b")
    
    def test_add_edge_nonexistent_target_raises(self, empty_graph):
        """Test that adding edge with non-existent target raises."""
        empty_graph.add_node("a", node_type="neuron")
        
        with pytest.raises(ValueError, match="Target node.*does not exist"):
            empty_graph.add_edge("a", "b")
    
    def test_add_duplicate_edge_raises(self, empty_graph):
        """Test that adding duplicate edge raises ValueError."""
        empty_graph.add_node("a", node_type="neuron")
        empty_graph.add_node("b", node_type="neuron")
        empty_graph.add_edge("a", "b")
        
        with pytest.raises(ValueError, match="already exists"):
            empty_graph.add_edge("a", "b")
    
    def test_get_edge(self, sample_graph):
        """Test retrieving an edge."""
        edge = sample_graph.get_edge("mlp.0.neuron.1", "attn.1.head.0")
        
        assert edge is not None
        assert edge.weight == 0.5
    
    def test_get_edge_not_found(self, sample_graph):
        """Test that get_edge returns None for non-existent edge."""
        edge = sample_graph.get_edge("mlp.0.neuron.1", "mlp.0.neuron.2")
        assert edge is None
    
    def test_remove_edge(self, sample_graph):
        """Test removing an edge."""
        assert sample_graph.has_edge("mlp.0.neuron.1", "attn.1.head.0")
        
        result = sample_graph.remove_edge("mlp.0.neuron.1", "attn.1.head.0")
        
        assert result is True
        assert not sample_graph.has_edge("mlp.0.neuron.1", "attn.1.head.0")
    
    def test_remove_nonexistent_edge(self, sample_graph):
        """Test removing non-existent edge returns False."""
        result = sample_graph.remove_edge("mlp.0.neuron.1", "mlp.0.neuron.2")
        assert result is False
    
    def test_edges_property(self, sample_graph):
        """Test edges property returns all edges."""
        edges = sample_graph.edges
        assert len(edges) == 4
        assert all(isinstance(e, CircuitEdge) for e in edges)
    
    def test_edges_from(self, sample_graph):
        """Test getting edges from a specific node."""
        edges = sample_graph.edges_from("attn.1.head.0")
        
        assert len(edges) == 1
        assert edges[0].target == "mlp.1.neuron.1"
    
    def test_edges_to(self, sample_graph):
        """Test getting edges to a specific node."""
        edges = sample_graph.edges_to("attn.1.head.0")
        
        assert len(edges) == 2
        sources = {e.source for e in edges}
        assert sources == {"mlp.0.neuron.1", "mlp.0.neuron.2"}
    
    def test_self_loop_allowed(self, empty_graph):
        """Test that self-loops are allowed."""
        empty_graph.add_node("a", node_type="neuron")
        edge = empty_graph.add_edge("a", "a")
        
        assert edge.source == "a"
        assert edge.target == "a"


# =============================================================================
# Graph Traversal Tests
# =============================================================================

class TestGraphTraversal:
    
    def test_get_neighbors_out(self, sample_graph):
        """Test getting outgoing neighbors."""
        neighbors = sample_graph.get_neighbors("attn.1.head.0", direction="out")
        
        assert len(neighbors) == 1
        assert neighbors[0].id == "mlp.1.neuron.1"
    
    def test_get_neighbors_in(self, sample_graph):
        """Test getting incoming neighbors."""
        neighbors = sample_graph.get_neighbors("attn.1.head.0", direction="in")
        
        assert len(neighbors) == 2
        ids = {n.id for n in neighbors}
        assert ids == {"mlp.0.neuron.1", "mlp.0.neuron.2"}
    
    def test_get_neighbors_both(self, sample_graph):
        """Test getting all neighbors."""
        neighbors = sample_graph.get_neighbors("attn.1.head.0", direction="both")
        
        assert len(neighbors) == 3
    
    def test_get_neighbors_nonexistent_raises(self, sample_graph):
        """Test that get_neighbors raises for non-existent node."""
        with pytest.raises(ValueError, match="does not exist"):
            sample_graph.get_neighbors("nonexistent")
    
    def test_predecessors(self, sample_graph):
        """Test predecessors method."""
        preds = sample_graph.predecessors("attn.1.head.0")
        
        ids = {n.id for n in preds}
        assert ids == {"mlp.0.neuron.1", "mlp.0.neuron.2"}
    
    def test_successors(self, sample_graph):
        """Test successors method."""
        succs = sample_graph.successors("attn.1.head.0")
        
        assert len(succs) == 1
        assert succs[0].id == "mlp.1.neuron.1"
    
    def test_in_degree(self, sample_graph):
        """Test in_degree calculation."""
        assert sample_graph.in_degree("mlp.0.neuron.1") == 0
        assert sample_graph.in_degree("attn.1.head.0") == 2
        assert sample_graph.in_degree("attn.2.head.3") == 1
    
    def test_out_degree(self, sample_graph):
        """Test out_degree calculation."""
        assert sample_graph.out_degree("mlp.0.neuron.1") == 1
        assert sample_graph.out_degree("attn.1.head.0") == 1
        assert sample_graph.out_degree("attn.2.head.3") == 0
    
    def test_sources(self, sample_graph):
        """Test finding source nodes (no incoming edges)."""
        sources = sample_graph.sources()
        
        ids = {n.id for n in sources}
        assert ids == {"mlp.0.neuron.1", "mlp.0.neuron.2"}
    
    def test_sinks(self, sample_graph):
        """Test finding sink nodes (no outgoing edges)."""
        sinks = sample_graph.sinks()
        
        ids = {n.id for n in sinks}
        assert ids == {"attn.2.head.3"}


# =============================================================================
# Subgraph Tests
# =============================================================================

class TestSubgraph:
    
    def test_get_subgraph_basic(self, sample_graph):
        """Test extracting a subgraph."""
        subgraph = sample_graph.get_subgraph([
            "mlp.0.neuron.1",
            "attn.1.head.0",
            "mlp.1.neuron.1",
        ])
        
        assert len(subgraph) == 3
        assert len(subgraph.edges) == 2
        assert subgraph.has_edge("mlp.0.neuron.1", "attn.1.head.0")
        assert subgraph.has_edge("attn.1.head.0", "mlp.1.neuron.1")
    
    def test_get_subgraph_preserves_model(self, sample_graph):
        """Test that subgraph preserves model name."""
        subgraph = sample_graph.get_subgraph(["mlp.0.neuron.1"])
        assert subgraph.model == sample_graph.model
    
    def test_get_subgraph_excludes_external_edges(self, sample_graph):
        """Test that subgraph excludes edges to nodes outside the subgraph."""
        subgraph = sample_graph.get_subgraph([
            "mlp.0.neuron.1",
            "mlp.0.neuron.2",
        ])
        
        # These nodes have edges to attn.1.head.0 which is not in subgraph
        assert len(subgraph.edges) == 0
    
    def test_get_subgraph_nonexistent_nodes_ignored(self, sample_graph):
        """Test that non-existent node IDs are silently ignored."""
        subgraph = sample_graph.get_subgraph([
            "mlp.0.neuron.1",
            "nonexistent.node",
        ])
        
        assert len(subgraph) == 1
    
    def test_merge_graphs(self, sample_graph, empty_graph):
        """Test merging two graphs."""
        empty_graph.add_node("new.node.1", node_type="feature")
        empty_graph.add_node("mlp.0.neuron.1", node_type="neuron")  # Duplicate
        
        initial_count = len(sample_graph)
        sample_graph.merge(empty_graph)
        
        # Should have one new node (duplicate not added)
        assert len(sample_graph) == initial_count + 1
        assert sample_graph.has_node("new.node.1")


# =============================================================================
# Serialization Tests
# =============================================================================

class TestSerialization:
    
    def test_to_dict(self, sample_graph):
        """Test serializing graph to dictionary."""
        data = sample_graph.to_dict()
        
        assert data["version"] == "1.0"
        assert data["model"] == "gpt2-small"
        assert data["name"] == "test_circuit"
        assert len(data["nodes"]) == 5
        assert len(data["edges"]) == 4
    
    def test_from_dict(self, sample_graph):
        """Test deserializing graph from dictionary."""
        data = sample_graph.to_dict()
        restored = CircuitGraph.from_dict(data)
        
        assert restored.model == sample_graph.model
        assert restored.name == sample_graph.name
        assert len(restored) == len(sample_graph)
        assert len(restored.edges) == len(sample_graph.edges)
    
    def test_roundtrip_preserves_data(self, sample_graph):
        """Test that serialization roundtrip preserves all data."""
        data = sample_graph.to_dict()
        restored = CircuitGraph.from_dict(data)
        
        # Check nodes
        for node in sample_graph.nodes:
            restored_node = restored.get_node(node.id)
            assert restored_node is not None
            assert restored_node.node_type == node.node_type
            assert restored_node.layer == node.layer
            assert restored_node.index == node.index
        
        # Check edges
        for edge in sample_graph.edges:
            restored_edge = restored.get_edge(edge.source, edge.target)
            assert restored_edge is not None
            assert restored_edge.weight == edge.weight
    
    def test_save_and_load(self, sample_graph, temp_dir):
        """Test saving and loading from JSON file."""
        filepath = os.path.join(temp_dir, "circuit.json")
        
        sample_graph.save(filepath)
        assert os.path.exists(filepath)
        
        loaded = CircuitGraph.load(filepath)
        
        assert loaded.model == sample_graph.model
        assert len(loaded) == len(sample_graph)
    
    def test_save_creates_parent_dirs(self, sample_graph, temp_dir):
        """Test that save creates parent directories."""
        filepath = os.path.join(temp_dir, "nested", "deep", "circuit.json")
        
        sample_graph.save(filepath)
        assert os.path.exists(filepath)
    
    def test_saved_file_is_valid_json(self, sample_graph, temp_dir):
        """Test that saved file is valid JSON."""
        filepath = os.path.join(temp_dir, "circuit.json")
        sample_graph.save(filepath)
        
        with open(filepath, "r") as f:
            data = json.load(f)
        
        assert "nodes" in data
        assert "edges" in data


# =============================================================================
# Magic Method Tests
# =============================================================================

class TestMagicMethods:
    
    def test_len(self, sample_graph, empty_graph):
        """Test __len__ returns node count."""
        assert len(sample_graph) == 5
        assert len(empty_graph) == 0
    
    def test_contains(self, sample_graph):
        """Test __contains__ for node lookup."""
        assert "mlp.0.neuron.1" in sample_graph
        assert "nonexistent" not in sample_graph
    
    def test_iter(self, sample_graph):
        """Test __iter__ iterates over nodes."""
        nodes = list(sample_graph)
        
        assert len(nodes) == 5
        assert all(isinstance(n, CircuitNode) for n in nodes)
    
    def test_repr(self, sample_graph):
        """Test __repr__ output."""
        repr_str = repr(sample_graph)
        
        assert "gpt2-small" in repr_str
        assert "test_circuit" in repr_str
        assert "nodes=5" in repr_str
        assert "edges=4" in repr_str


# =============================================================================
# Statistics Tests
# =============================================================================

class TestStatistics:
    
    def test_summary(self, sample_graph):
        """Test summary method."""
        summary = sample_graph.summary()
        
        assert summary["model"] == "gpt2-small"
        assert summary["num_nodes"] == 5
        assert summary["num_edges"] == 4
        assert summary["node_types"]["neuron"] == 3
        assert summary["node_types"]["attention_head"] == 2
        assert summary["layer_range"] == (0, 2)
        assert summary["num_sources"] == 2
        assert summary["num_sinks"] == 1


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    
    def test_empty_graph_operations(self, empty_graph):
        """Test operations on empty graph don't crash."""
        assert len(empty_graph) == 0
        assert empty_graph.nodes == []
        assert empty_graph.edges == []
        assert empty_graph.sources() == []
        assert empty_graph.sinks() == []
    
    def test_special_characters_in_id(self, empty_graph):
        """Test that special characters in node IDs work."""
        empty_graph.add_node("sae:v1.feature[123]", node_type="feature")
        empty_graph.add_node("layer-5/mlp/neuron_42", node_type="neuron")
        
        assert empty_graph.has_node("sae:v1.feature[123]")
        assert empty_graph.has_node("layer-5/mlp/neuron_42")
    
    def test_node_type_invalid_raises(self, empty_graph):
        """Test that invalid node type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown node type"):
            empty_graph.add_node("test", node_type="invalid_type")
    
    def test_disconnected_nodes(self, empty_graph):
        """Test graph with disconnected components."""
        empty_graph.add_node("a", node_type="neuron")
        empty_graph.add_node("b", node_type="neuron")
        empty_graph.add_node("c", node_type="neuron")
        empty_graph.add_node("d", node_type="neuron")
        
        empty_graph.add_edge("a", "b")
        empty_graph.add_edge("c", "d")
        
        # All four are either sources or sinks depending on perspective
        sources = empty_graph.sources()
        sinks = empty_graph.sinks()
        
        assert len(sources) == 2  # a, c
        assert len(sinks) == 2    # b, d
