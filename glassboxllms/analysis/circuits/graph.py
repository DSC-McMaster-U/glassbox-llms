"""
CircuitGraph: A directed graph representation for model circuits.

This module provides a graph-based abstraction for representing the structural
anatomy of neural network circuits, where nodes are internal components and
edges represent functional or causal connections.

This is a representation module - it does not perform circuit discovery,
causal testing, or attribution. Those belong in separate analysis modules.
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Union

from .node import CircuitEdge, CircuitNode, EdgeType, NodeType


# Schema version for serialization compatibility
SCHEMA_VERSION = "1.0"


class CircuitGraph:
    """
    A directed graph representing a model circuit.
    
    Nodes represent internal model components (neurons, attention heads, features).
    Edges represent directed functional or causal connections between components.
    
    Attributes:
        model: Identifier of the model this circuit belongs to (e.g., "gpt2-xl")
        name: Optional human-readable name for this circuit
        metadata: Graph-level metadata (source, creation date, etc.)
    
    Example:
        >>> graph = CircuitGraph(model="gpt2-xl")
        >>> graph.add_node("mlp.5.neuron.42", node_type="neuron")
        >>> graph.add_node("attn.10.head.3", node_type="attention_head")
        >>> graph.add_edge("mlp.5.neuron.42", "attn.10.head.3", weight=0.8)
        >>> graph.save("induction_circuit.json")
    """
    
    def __init__(
        self,
        model: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a new circuit graph.
        
        Args:
            model: Model identifier (e.g., "gpt2-xl", "llama-7b")
            name: Optional human-readable circuit name
            metadata: Optional graph-level metadata
        """
        self.model = model
        self.name = name
        self.metadata = metadata or {}
        
        # Internal storage
        self._nodes: Dict[str, CircuitNode] = {}
        self._edges: Dict[tuple, CircuitEdge] = {}  # (source, target) -> edge
        
        # Adjacency lists for efficient traversal
        self._outgoing: Dict[str, Set[str]] = defaultdict(set)  # node_id -> {target_ids}
        self._incoming: Dict[str, Set[str]] = defaultdict(set)  # node_id -> {source_ids}
    
    # =========================================================================
    # Node Operations
    # =========================================================================
    
    def add_node(
        self,
        node_id: str,
        node_type: Union[str, NodeType],
        layer: Optional[int] = None,
        index: Optional[int] = None,
        **metadata,
    ) -> CircuitNode:
        """
        Add a node to the circuit graph.
        
        Args:
            node_id: Unique identifier (e.g., "mlp.5.neuron.42")
            node_type: Type of component ("neuron", "attention_head", "feature", etc.)
            layer: Optional layer index
            index: Optional component index within the layer
            **metadata: Additional key-value metadata
        
        Returns:
            The created CircuitNode
        
        Raises:
            ValueError: If a node with this ID already exists
        """
        if node_id in self._nodes:
            raise ValueError(f"Node '{node_id}' already exists in the graph")
        
        if isinstance(node_type, str):
            node_type = NodeType.from_string(node_type)
        
        node = CircuitNode(
            id=node_id,
            node_type=node_type,
            layer=layer,
            index=index,
            metadata=metadata,
        )
        self._nodes[node_id] = node
        return node
    
    def get_node(self, node_id: str) -> Optional[CircuitNode]:
        """Get a node by ID, or None if not found."""
        return self._nodes.get(node_id)
    
    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node and all its connected edges.
        
        Args:
            node_id: ID of the node to remove
        
        Returns:
            True if the node was removed, False if it didn't exist
        """
        if node_id not in self._nodes:
            return False
        
        # Remove all edges connected to this node
        edges_to_remove = []
        for (src, tgt) in self._edges:
            if src == node_id or tgt == node_id:
                edges_to_remove.append((src, tgt))
        
        for edge_key in edges_to_remove:
            self._remove_edge_internal(*edge_key)
        
        # Remove from adjacency lists
        del self._nodes[node_id]
        self._outgoing.pop(node_id, None)
        self._incoming.pop(node_id, None)
        
        return True
    
    def has_node(self, node_id: str) -> bool:
        """Check if a node exists in the graph."""
        return node_id in self._nodes
    
    @property
    def nodes(self) -> List[CircuitNode]:
        """Return all nodes in the graph."""
        return list(self._nodes.values())
    
    @property
    def node_ids(self) -> List[str]:
        """Return all node IDs."""
        return list(self._nodes.keys())
    
    def nodes_by_type(self, node_type: Union[str, NodeType]) -> List[CircuitNode]:
        """Get all nodes of a specific type."""
        if isinstance(node_type, str):
            node_type = NodeType.from_string(node_type)
        return [n for n in self._nodes.values() if n.node_type == node_type]
    
    def nodes_by_layer(self, layer: int) -> List[CircuitNode]:
        """Get all nodes at a specific layer."""
        return [n for n in self._nodes.values() if n.layer == layer]
    
    # =========================================================================
    # Edge Operations
    # =========================================================================
    
    def add_edge(
        self,
        source: str,
        target: str,
        weight: Optional[float] = None,
        edge_type: Union[str, EdgeType] = EdgeType.DIRECT,
        **metadata,
    ) -> CircuitEdge:
        """
        Add a directed edge between two nodes.
        
        Args:
            source: ID of the source node
            target: ID of the target node
            weight: Optional edge weight/strength
            edge_type: Type of connection
            **metadata: Additional edge metadata
        
        Returns:
            The created CircuitEdge
        
        Raises:
            ValueError: If source or target node doesn't exist, or edge already exists
        """
        if source not in self._nodes:
            raise ValueError(f"Source node '{source}' does not exist")
        if target not in self._nodes:
            raise ValueError(f"Target node '{target}' does not exist")
        
        edge_key = (source, target)
        if edge_key in self._edges:
            raise ValueError(f"Edge from '{source}' to '{target}' already exists")
        
        if isinstance(edge_type, str):
            edge_type = EdgeType(edge_type.lower())
        
        edge = CircuitEdge(
            source=source,
            target=target,
            weight=weight,
            edge_type=edge_type,
            metadata=metadata,
        )
        
        self._edges[edge_key] = edge
        self._outgoing[source].add(target)
        self._incoming[target].add(source)
        
        return edge
    
    def get_edge(self, source: str, target: str) -> Optional[CircuitEdge]:
        """Get an edge by source and target, or None if not found."""
        return self._edges.get((source, target))
    
    def remove_edge(self, source: str, target: str) -> bool:
        """
        Remove an edge from the graph.
        
        Args:
            source: Source node ID
            target: Target node ID
        
        Returns:
            True if the edge was removed, False if it didn't exist
        """
        edge_key = (source, target)
        if edge_key not in self._edges:
            return False
        
        self._remove_edge_internal(source, target)
        return True
    
    def _remove_edge_internal(self, source: str, target: str):
        """Internal edge removal (no existence check)."""
        del self._edges[(source, target)]
        self._outgoing[source].discard(target)
        self._incoming[target].discard(source)
    
    def has_edge(self, source: str, target: str) -> bool:
        """Check if an edge exists."""
        return (source, target) in self._edges
    
    @property
    def edges(self) -> List[CircuitEdge]:
        """Return all edges in the graph."""
        return list(self._edges.values())
    
    def edges_from(self, node_id: str) -> List[CircuitEdge]:
        """Get all edges originating from a node."""
        return [self._edges[(node_id, t)] for t in self._outgoing.get(node_id, set())]
    
    def edges_to(self, node_id: str) -> List[CircuitEdge]:
        """Get all edges pointing to a node."""
        return [self._edges[(s, node_id)] for s in self._incoming.get(node_id, set())]
    
    # =========================================================================
    # Graph Traversal
    # =========================================================================
    
    def get_neighbors(
        self,
        node_id: str,
        direction: str = "out",
    ) -> List[CircuitNode]:
        """
        Get neighboring nodes.
        
        Args:
            node_id: The node to find neighbors for
            direction: "out" for successors, "in" for predecessors, "both" for all
        
        Returns:
            List of neighboring nodes
        """
        if node_id not in self._nodes:
            raise ValueError(f"Node '{node_id}' does not exist")
        
        neighbor_ids: Set[str] = set()
        
        if direction in ("out", "both"):
            neighbor_ids.update(self._outgoing.get(node_id, set()))
        if direction in ("in", "both"):
            neighbor_ids.update(self._incoming.get(node_id, set()))
        
        return [self._nodes[nid] for nid in neighbor_ids]
    
    def predecessors(self, node_id: str) -> List[CircuitNode]:
        """Get all nodes that have edges pointing to this node."""
        return self.get_neighbors(node_id, direction="in")
    
    def successors(self, node_id: str) -> List[CircuitNode]:
        """Get all nodes that this node points to."""
        return self.get_neighbors(node_id, direction="out")
    
    def in_degree(self, node_id: str) -> int:
        """Number of incoming edges to a node."""
        return len(self._incoming.get(node_id, set()))
    
    def out_degree(self, node_id: str) -> int:
        """Number of outgoing edges from a node."""
        return len(self._outgoing.get(node_id, set()))
    
    def sources(self) -> List[CircuitNode]:
        """Get all nodes with no incoming edges (entry points)."""
        return [n for n in self._nodes.values() if self.in_degree(n.id) == 0]
    
    def sinks(self) -> List[CircuitNode]:
        """Get all nodes with no outgoing edges (endpoints)."""
        return [n for n in self._nodes.values() if self.out_degree(n.id) == 0]
    
    # =========================================================================
    # Subgraph Operations
    # =========================================================================
    
    def get_subgraph(self, node_ids: List[str]) -> "CircuitGraph":
        """
        Extract an induced subgraph containing only the specified nodes.
        
        Args:
            node_ids: List of node IDs to include
        
        Returns:
            A new CircuitGraph containing only the specified nodes and
            edges between them
        """
        subgraph = CircuitGraph(
            model=self.model,
            name=f"{self.name}_subgraph" if self.name else None,
            metadata={"parent_graph": self.name, **self.metadata},
        )
        
        node_set = set(node_ids)
        
        # Add nodes
        for node_id in node_ids:
            if node_id in self._nodes:
                node = self._nodes[node_id]
                subgraph._nodes[node_id] = CircuitNode(
                    id=node.id,
                    node_type=node.node_type,
                    layer=node.layer,
                    index=node.index,
                    metadata=node.metadata.copy(),
                )
        
        # Add edges (only between nodes in subgraph)
        for (src, tgt), edge in self._edges.items():
            if src in node_set and tgt in node_set:
                subgraph._edges[(src, tgt)] = CircuitEdge(
                    source=edge.source,
                    target=edge.target,
                    weight=edge.weight,
                    edge_type=edge.edge_type,
                    metadata=edge.metadata.copy(),
                )
                subgraph._outgoing[src].add(tgt)
                subgraph._incoming[tgt].add(src)
        
        return subgraph
    
    def merge(self, other: "CircuitGraph", overwrite: bool = False) -> "CircuitGraph":
        """
        Merge another graph into this one.
        
        Args:
            other: Graph to merge
            overwrite: If True, overwrite existing nodes/edges with new ones
        
        Returns:
            self (for chaining)
        """
        for node in other.nodes:
            if node.id not in self._nodes:
                self._nodes[node.id] = CircuitNode(
                    id=node.id,
                    node_type=node.node_type,
                    layer=node.layer,
                    index=node.index,
                    metadata=node.metadata.copy(),
                )
            elif overwrite:
                self._nodes[node.id] = node
        
        for edge in other.edges:
            edge_key = (edge.source, edge.target)
            if edge_key not in self._edges:
                self._edges[edge_key] = CircuitEdge(
                    source=edge.source,
                    target=edge.target,
                    weight=edge.weight,
                    edge_type=edge.edge_type,
                    metadata=edge.metadata.copy(),
                )
                self._outgoing[edge.source].add(edge.target)
                self._incoming[edge.target].add(edge.source)
            elif overwrite:
                self._edges[edge_key] = edge
        
        return self
    
    # =========================================================================
    # Serialization
    # =========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the graph to a dictionary.
        
        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            "version": SCHEMA_VERSION,
            "model": self.model,
            "name": self.name,
            "metadata": {
                **self.metadata,
                "serialized_at": datetime.now().isoformat(),
            },
            "nodes": [node.to_dict() for node in self._nodes.values()],
            "edges": [edge.to_dict() for edge in self._edges.values()],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CircuitGraph":
        """
        Deserialize a graph from a dictionary.
        
        Args:
            data: Dictionary representation of the graph
        
        Returns:
            Reconstructed CircuitGraph
        """
        version = data.get("version", "1.0")
        if version != SCHEMA_VERSION:
            # Future: handle version migrations
            pass
        
        graph = cls(
            model=data["model"],
            name=data.get("name"),
            metadata=data.get("metadata", {}),
        )
        
        # Add nodes first
        for node_data in data.get("nodes", []):
            node = CircuitNode.from_dict(node_data)
            graph._nodes[node.id] = node
        
        # Then add edges
        for edge_data in data.get("edges", []):
            edge = CircuitEdge.from_dict(edge_data)
            edge_key = (edge.source, edge.target)
            graph._edges[edge_key] = edge
            graph._outgoing[edge.source].add(edge.target)
            graph._incoming[edge.target].add(edge.source)
        
        return graph
    
    def save(self, path: Union[str, Path], indent: int = 2) -> None:
        """
        Save the graph to a JSON file.
        
        Args:
            path: File path to save to
            indent: JSON indentation level (set to None for compact output)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "CircuitGraph":
        """
        Load a graph from a JSON file.
        
        Args:
            path: File path to load from
        
        Returns:
            Loaded CircuitGraph
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    # =========================================================================
    # Python Magic Methods
    # =========================================================================
    
    def __len__(self) -> int:
        """Return the number of nodes in the graph."""
        return len(self._nodes)
    
    def __contains__(self, node_id: str) -> bool:
        """Check if a node ID is in the graph."""
        return node_id in self._nodes
    
    def __iter__(self) -> Iterator[CircuitNode]:
        """Iterate over nodes in the graph."""
        return iter(self._nodes.values())
    
    def __repr__(self) -> str:
        name_str = f", name={self.name!r}" if self.name else ""
        return f"CircuitGraph(model={self.model!r}{name_str}, nodes={len(self._nodes)}, edges={len(self._edges)})"
    
    def __str__(self) -> str:
        return self.__repr__()
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of the graph structure.
        
        Returns:
            Dictionary with graph statistics
        """
        type_counts = defaultdict(int)
        for node in self._nodes.values():
            type_counts[node.node_type.value] += 1
        
        layers = [n.layer for n in self._nodes.values() if n.layer is not None]
        
        return {
            "model": self.model,
            "name": self.name,
            "num_nodes": len(self._nodes),
            "num_edges": len(self._edges),
            "node_types": dict(type_counts),
            "layer_range": (min(layers), max(layers)) if layers else None,
            "num_sources": len(self.sources()),
            "num_sinks": len(self.sinks()),
        }
