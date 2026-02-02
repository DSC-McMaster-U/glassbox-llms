"""
Feature data model for the Feature Atlas.

This module defines the core data structures for representing discovered
features in neural networks, including neurons, attention heads, SAE latents,
probe directions, and circuits.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from typing import Any, Optional, Dict, List, Union

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

    ATTRIBUTES
        model_name: Name/identifier of the model (e.g. "llama-2-7b")
        layer: Layer identifier (e.g. "attention.5")
        sublayer: Optional sublayer specification (e.g. "attn_pattern")
        neuron_idx: Optional neuron idx for neuron-level things
        head_idx: Optional head idx for attention features
        position: Optional token position if it's position-specific (e.g., "last")
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
        return f"Location({', '.join(parts)})"

@dataclass
class History:
    # todo



@dataclass
class Feature:
    # todo
