"""
Node and Edge abstractions for circuit graphs.

Defines the structural components used to represent model circuits:
- CircuitNode: Individual model components (neurons, attention heads, features)
- CircuitEdge: Directed connections between nodes
- NodeType: Enumeration of supported component types
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class NodeType(Enum):
    """
    Supported types of circuit nodes.
    
    These represent the fundamental computational units that can
    participate in a circuit.
    """
    NEURON = "neuron"
    ATTENTION_HEAD = "attention_head"
    FEATURE = "feature"  # SAE/Feature Atlas features
    MLP_LAYER = "mlp_layer"
    RESIDUAL_STREAM = "residual_stream"
    EMBEDDING = "embedding"
    UNEMBEDDING = "unembedding"
    
    @classmethod
    def from_string(cls, s: str) -> "NodeType":
        """Convert string to NodeType, case-insensitive."""
        normalized = s.lower().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(f"Unknown node type: {s}. Valid types: {[m.value for m in cls]}")


class EdgeType(Enum):
    """
    Types of edges representing different connection semantics.
    """
    DIRECT = "direct"  # Direct causal connection
    ATTENTION = "attention"  # Attention-mediated connection
    RESIDUAL = "residual"  # Through residual stream
    INFERRED = "inferred"  # Discovered via analysis (e.g., patching)
    MANUAL = "manual"  # User-annotated connection


@dataclass
class CircuitNode:
    """
    Represents a single component in a circuit graph.
    
    Attributes:
        id: Unique identifier following convention: {component}.{layer}.{subtype}.{index}
            Examples: "mlp.5.neuron.42", "attn.10.head.3", "feature.sae_v1.1234"
        node_type: The type of component (neuron, attention_head, etc.)
        layer: Optional layer index (None for layer-agnostic nodes)
        index: Optional component index within the layer
        metadata: Arbitrary additional information (labels, annotations, etc.)
    """
    id: str
    node_type: NodeType
    layer: Optional[int] = None
    index: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Convert string node_type to enum if necessary
        if isinstance(self.node_type, str):
            self.node_type = NodeType.from_string(self.node_type)
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, CircuitNode):
            return self.id == other.id
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize node to dictionary."""
        return {
            "id": self.id,
            "type": self.node_type.value,
            "layer": self.layer,
            "index": self.index,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CircuitNode":
        """Deserialize node from dictionary."""
        return cls(
            id=data["id"],
            node_type=NodeType.from_string(data["type"]),
            layer=data.get("layer"),
            index=data.get("index"),
            metadata=data.get("metadata", {}),
        )
    
    def __repr__(self) -> str:
        return f"CircuitNode(id={self.id!r}, type={self.node_type.value})"


@dataclass
class CircuitEdge:
    """
    Represents a directed connection between two circuit nodes.
    
    Attributes:
        source: ID of the source node
        target: ID of the target node
        weight: Optional connection strength/importance (e.g., from patching)
        edge_type: Semantic type of the connection
        metadata: Arbitrary additional information
    """
    source: str
    target: str
    weight: Optional[float] = None
    edge_type: EdgeType = EdgeType.DIRECT
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Convert string edge_type to enum if necessary
        if isinstance(self.edge_type, str):
            self.edge_type = EdgeType(self.edge_type.lower())
    
    def __hash__(self) -> int:
        return hash((self.source, self.target))
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, CircuitEdge):
            return self.source == other.source and self.target == other.target
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize edge to dictionary."""
        return {
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
            "edge_type": self.edge_type.value,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CircuitEdge":
        """Deserialize edge from dictionary."""
        return cls(
            source=data["source"],
            target=data["target"],
            weight=data.get("weight"),
            edge_type=EdgeType(data.get("edge_type", "direct")),
            metadata=data.get("metadata", {}),
        )
    
    def __repr__(self) -> str:
        weight_str = f", weight={self.weight:.3f}" if self.weight is not None else ""
        return f"CircuitEdge({self.source!r} -> {self.target!r}{weight_str})"
