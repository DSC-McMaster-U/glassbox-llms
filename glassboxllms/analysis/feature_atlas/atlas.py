from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Iterator
import json

from .feature import Feature, FeatureType

SCHEMA_VERSION = "1.0.0"

# todo: make sure no other dunder methods are missing

@dataclass
class AtlasMetadata:
    """
    Simple helper class to store some Atlas metadata. Handled automatically.
    """
    schema_version: str = SCHEMA_VERSION
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    modified: str = field(default_factory=lambda: datetime.now().isoformat())
    # these three shouldn't strictly be here
    name: Optional[str] = None
    description: Optional[str] = None
    model_name: Optional[str] = None

    def touch(self) -> None:
        # named after the unix command
        self.modified = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AtlasMetadata":
        return cls(**data)


class Atlas:
    """
    Container and registry for Features

    Attributes:
        features: Dictionary mapping feature IDs to Feature objects
        metadata: Atlas-level metadata

    Example:
        atlas = Atlas(name="gpt2-features")
        atlas.add(feature)
        atlas.save("features.json")

        # And then later you can do
        atlas = Atlas.load("features.json")
        features = atlas.find_by_layer("mlp.10")
    """

    def __init__(self, name: Optional[str] = None, description: Optional[str] = None,
        model_name: Optional[str] = None, features: Optional[List[Feature]] = None,
    ):
        """
        Atlas()

        Args:
            name: Human-readable name (optional)
            description: Atlas's purpose or some other descriptor (optional)
            model_name: Default model name (can be overwritten later) (optional)
            features: List of features to add initially (optional)
        """
        self._features: Dict[str, Feature] = {}
        self.metadata = AtlasMetadata(
            name=name,
            description=description,
            model_name=model_name,
        )
        if features:
            for feature in features:
                self.add(feature)

    def add(self, feature: Feature) -> str:
        """
        Add a feature to the atlas. Returns feature ID
        """
        if feature.id in self._features:
            raise ValueError(f"Feature with ID '{feature.id}' already exists in atlas")
        self._features[feature.id] = feature
        self.metadata.touch()
        return feature.id

    def get(self, feature_id: str) -> Optional[Feature]:
        """
        Retrieve a feature by ID.
        """
        return self._features.get(feature_id)

    def __getitem__(self, feature_id: str) -> Feature:
        """
        some_atlas["some uuid"]

        We don't really need this, but it's nice since .get's return is kind of vague if an id is not found
        And it's convenient
        """
        if feature_id not in self._features:
            raise KeyError(f"No feature with ID '{feature_id}' found in atlas")
        return self._features[feature_id]

    def __contains__(self, feature_id: str) -> bool:
        return feature_id in self._features

    def remove(self, feature_id: str) -> bool:
        if feature_id in self._features:
            del self._features[feature_id]
            self.metadata.touch()
            return True
        return False

    def list(self) -> List[Feature]:
        return list(self._features.values())

    def __len__(self) -> int:
        return len(self._features)

    def __iter__(self) -> Iterator[Feature]:
        return iter(self._features.values())

    # --- SEARCH AND FILTER METHODS ---

    def find_by_id(self, feature_id: str) -> Optional[Feature]:
        return self.get(feature_id)

    def find_by_label(self, label: str, exact: bool = False) -> List[Feature]:
        """
        Set exact to match exactly, otherwise it does it by substring
        """
        results = []
        for feature in self._features.values():
            if exact:
                if feature.label == label:
                    results.append(feature)
            else:
                if label.lower() in feature.label.lower():
                    results.append(feature)
        return results

    def find_by_type(self, feature_type: Union[FeatureType, str]) -> List[Feature]:
        if isinstance(feature_type, str):
            feature_type = FeatureType(feature_type)

        return [
            f for f in self._features.values()
            if f.feature_type == feature_type
        ]

    def find_by_layer(self, layer: str, exact: bool = False) -> List[Feature]:
        """
        Set exact to match exactly, otherwise it does it by substring
        """
        results = []
        for feature in self._features.values():
            if exact:
                if feature.location.layer == layer:
                    results.append(feature)
            else:
                if layer.lower() in feature.location.layer.lower():
                    results.append(feature)
        return results

    def find_by_model(self, model_name: str, exact: bool = False) -> List[Feature]:
        """
        Set exact to match exactly, otherwise it does it by substring
        """
        results = []
        for feature in self._features.values():
            if exact:
                if feature.location.model_name == model_name:
                    results.append(feature)
            else:
                if model_name.lower() in feature.location.model_name.lower():
                    results.append(feature)
        return results

    def find_by_tag(self, tag: str) -> List[Feature]:
        """
        The user should define their own tag system
        """
        results = []
        for feature in self._features.values():
            if tag in feature.tags:
                results.append(feature)
        return results

    def filter(self, feature_type: Optional[Union[FeatureType, str]] = None, layer: Optional[str] = None,
        model_name: Optional[str] = None, label: Optional[str] = None, tag: Optional[str] = None) -> List[Feature]:
        """
        Advanced filter to filter by criteria and logic. Uses AND logic
        todo: add OR/NOT logic

        Args (all optional):
            feature_type: FeatureType or type string to filter by
            layer: Layer identifier (substring)
            model_name: Model name (substring)
            label: Label (substring)
            tag: Tags
        """
        results = list(self._features.values())

        if feature_type is not None:
            if isinstance(feature_type, str):
                feature_type = FeatureType(feature_type)
            results = [f for f in results if f.feature_type == feature_type]

        if layer is not None:
            results = [
                f for f in results
                if layer.lower() in f.location.layer.lower()
            ]

        if model_name is not None:
            results = [
                f for f in results
                if model_name.lower() in f.location.model_name.lower()
            ]

        if label is not None:
            results = [
                f for f in results
                if label.lower() in f.label.lower()
            ]

        if tag is not None:
            results = [f for f in results if tag in f.tags]

        return results

    # --- PERSISTENCE/SAVE/LOADING ---

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "features": [f.to_dict() for f in self._features.values()],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Atlas":
        if "features" not in data:
            raise ValueError("An Atlas object must contain the 'features' key")

        metadata_data = data.get("metadata", {})
        metadata = AtlasMetadata.from_dict(metadata_data)

        features = []
        for feature_data in data["features"]:
            feature = Feature.from_dict(feature_data)
            features.append(feature)

        atlas = cls(
            name=metadata.name,
            description=metadata.description,
            model_name=metadata.model_name,
            features=features,
        )
        atlas.metadata = metadata

        return atlas

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the atlas to a file (JSON or YAML based on extension).
        Either use a string or a Pathlib.path for the path arg (yaml or json)

        Raises:
            ValueError: If the file extension is not supported
        """
        path = Path(path)
        data = self.to_dict()

        if path.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError:
                raise ImportError(
                    "PyYAML was not found in your environment. Install it: pip install pyyaml"
                )
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        elif path.suffix.lower() == ".json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(
                f"Unsupported file extension: {path.suffix}. "
                "Try using .json or .yaml instead"
            )

        self.metadata.touch()

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Atlas":
        """
        Load an atlas from a file (JSON or YAML based on extension)
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Atlas file not found: {path}")

        if path.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError:
                raise ImportError(
                    "PyYAML was not found in your environment. Install it: pip install pyyaml"
            )
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

        elif path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

        else:
            raise ValueError(
                f"Unsupported file extension: {path.suffix}. "
                "Try using .json or .yaml instead"
            )

        return cls.from_dict(data)

    # --- UTILITY ---

    def summary(self) -> str:
        """
        Makes a pretty formatted version of the atlas. It's like __repr__ but prettier.
        """
        type_counts: Dict[str, int] = {}
        model_counts: Dict[str, int] = {}
        layer_counts: Dict[str, int] = {}

        for feature in self._features.values():
            ft = feature.feature_type.value
            type_counts[ft] = type_counts.get(ft, 0) + 1

            model = feature.location.model_name
            model_counts[model] = model_counts.get(model, 0) + 1

            layer = feature.location.layer
            layer_counts[layer] = layer_counts.get(layer, 0) + 1

        lines = [
            "-- Atlas Object --",
            f"Atlas: {self.metadata.name or 'Unnamed'}",
            f"  Schema Version: {self.metadata.schema_version}",
            f"  Created: {self.metadata.created}",
            f"  Modified: {self.metadata.modified}",
            f"  Total Features: {len(self._features)}",
        ]

        if self.metadata.description:
            lines.append(f"  Description: {self.metadata.description}")

        if type_counts:
            lines.append("  By Type:")
            for ft, count in sorted(type_counts.items()):
                lines.append(f"    {ft}: {count}")

        if model_counts:
            lines.append("  By Model:")
            for model, count in sorted(model_counts.items()):
                lines.append(f"    {model}: {count}")

        if layer_counts and len(layer_counts) <= 10:
            lines.append("  By Layer:")
            for layer, count in sorted(layer_counts.items()):
                lines.append(f"    {layer}: {count}")
        elif layer_counts:
            lines.append(f"  Layers: {len(layer_counts)} unique layers")

        return "\n".join(lines)

    def __repr__(self) -> str:
        name = self.metadata.name or "Unnaned"
        return f"<Atlas '{name}' with {len(self._features)} features>"

    def __str__(self) -> str:
        # This could technically just be the same as __repr__
        return self.summary()
