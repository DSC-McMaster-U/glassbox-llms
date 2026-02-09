from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from typing import Any, Optional, Dict, List, Union
import uuid

class FeatureType(Enum):
    """
    Types of interpretable features that can be discovered in a model.

    Various types included of varying depth. Part of Feature.
    """
    NEURON = "neuron"
    HEAD = "head"
    SAE_LATENT = "sae_latent"
    PROBE_DIRECTION = "probe_direction"
    CIRCUIT = "circuit"

    def __repr__(self) -> str:
        return f"FeatureType.{self.name}"


@dataclass
class Location:
    """
    The location of a feature within the specific model architecture

    USAGE
        model_name: name/identifier of the model (e.g. "llama-2-7b")
        layer: layer identifier (e.g. "attention.5")
        sublayer: optional sublayer specification (e.g. "attn_pattern")
        neuron_idx: optional neuron idx for neuron-level things
        head_idx: optional head idx for attention features
        position: optional token position if it's position-specific (e.g., "last")
    """
    model_name: str
    layer: str
    sublayer: Optional[str] = None
    neuron_idx: Optional[int] = None
    head_idx: Optional[int] = None
    position: Optional[str] = None

    def __repr__(self) -> str:
        parts = [f"model='{self.model_name}'", f"layer='{self.layer}'"]
        if self.sublayer:
            parts.append(f"sublayer='{self.sublayer}'")
        if self.neuron_idx is not None:
            parts.append(f"neuron={self.neuron_idx}")
        if self.head_idx is not None:
            parts.append(f"head={self.head_idx}")
        if self.position:
            parts.append(f"position='{self.position}'")
        return f"<Location object {', '.join(parts)}>"


@dataclass
class History:
    """
    Record for how a feature was found

    USAGE
        method: technique used to discover/analyze the feature (e.g. "activation_patching")
        dataset: dataset used during analysis (e.g. "pile-subset-10k")
        timestamp: time of analysis (ISO timestamp)
        tool_version: tool used (optional)
        hyperparameters: dict of hyperparameters (optional)
        author: (optional)
    """
    method: str
    dataset: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    tool_version: Optional[str] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    author: Optional[str] = None

    def __repr__(self) -> str:
        parts = [f"method='{self.method}'", f"dataset='{self.dataset}'"]
        if self.tool_version:
            parts.append(f"version='{self.tool_version}'")
        if self.author:
            parts.append(f"by='{self.author}'")
        return f"<History object {', '.join(parts)}>"

# class Evidence:
# todo: do we need this?

@dataclass
class Feature:
    """
    Class for a discovered feature. Stored in an Atlas.

    Attributes:
        id: UUID
        feature_type: Type of feature (neuron, latent, etc.)
        location: Precise location within architecture
        history: Info about the history of the feature
        label: A more human-readable name/label
        description: Detailed human-friendly description (optional)
        tags: List of tags (the user will define these tags/handle sorting by them or whatever) (optional)
        metadata: User-defined metadata dict (optional)

    Example:
         feature = Feature(
             feature_type=FeatureType.PROBE_DIRECTION,
             location=Location(model_name="gpt2", layer="mlp.10"),
             label="latent_found_2",
             description="Latent from forward pass",
             history=History(method="linear_probe", dataset="pile"),)
    """
    feature_type: FeatureType
    location: Location
    label: str
    history: Optional[History] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Validates
        if not self.label or not isinstance(self.label, str):
            raise ValueError("Feature label must be a non-empty string")
        if not isinstance(self.feature_type, FeatureType):
            raise ValueError(f"feature_type must be a FeatureType enum: got {type(self.feature_type)}")

    def to_dict(self) -> Dict[str, Any]:
        # serializes Feature to dict
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Feature":
        # Loads Feature from dict
        if isinstance(data.get("feature_type"), str):
            data["feature_type"] = FeatureType(data["feature_type"])

        if isinstance(data.get("location"), dict):
            data["location"] = Location(**data["location"])
        if isinstance(data.get("history"), dict):
            data["history"] = History(**data["history"])

        return cls(**data)

    def __repr__(self) -> str:
        return (f"<Feature id='{self.id[:8]}...', type={self.feature_type!r}, "
                f"label='{self.label}', location={self.location!r}>")

    def __str__(self) -> str:
        lines = [
            f"Feature: {self.label}",
            f"  ID: {self.id}",
            f"  Type: {self.feature_type.value}",
            f"  Location: {self.location}",
        ]
        if self.description:
            lines.append(f"  Description: {self.description[:60]}{'...' if len(self.description) > 60 else ''}")
        if self.history:
            lines.append(f"  history: {self.history}")
        if self.tags:
            lines.append(f"  Tags: {', '.join(self.tags)}")
        return "\n".join(lines)

    def __hash__(self) -> int:
        # Allows using Features in dicts and sets
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        # Overloaded equality checks. Note this checks only ID!!
        if not isinstance(other, Feature):
            return NotImplemented
        else:
            return self.id == other.id
