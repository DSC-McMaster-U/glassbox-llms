"""
Feature data model for the Feature Atlas.

This module defines the core data structures for representing discovered
features in neural networks, including neurons, attention heads, SAE latents,
probe directions, and circuits.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto

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
    # todo








@dataclass
class Feature:
    # todo
