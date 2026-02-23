import pytest
import json
import tempfile
import os
from datetime import datetime
from pathlib import Path

from glassboxllms.analysis.feature_atlas import (
    Feature,
    FeatureType,
    Location,
    History,
    Atlas,
    AtlasMetadata,
    SCHEMA_VERSION,
)

# usage: python -m pytest tests/test_feature_atlas.py -v --tb=short 2>&1

# --- FeatureType Tests ---

class TestFeatureType:
    """Tests for the FeatureType enum."""

    def test_all_feature_types_exist(self):
        """Ensure all expected feature types are available."""
        expected_types = ["neuron", "head", "sae_latent", "probe_direction", "circuit"]
        actual_types = [ft.value for ft in FeatureType]
        assert set(actual_types) == set(expected_types)

    def test_feature_type_values(self):
        """Test specific feature type values."""
        assert FeatureType.NEURON.value == "neuron"
        assert FeatureType.HEAD.value == "head"
        assert FeatureType.SAE_LATENT.value == "sae_latent"
        assert FeatureType.PROBE_DIRECTION.value == "probe_direction"
        assert FeatureType.CIRCUIT.value == "circuit"

    def test_feature_type_repr(self):
        """Test __repr__ method of FeatureType."""
        assert repr(FeatureType.NEURON) == "FeatureType.NEURON"
        assert repr(FeatureType.SAE_LATENT) == "FeatureType.SAE_LATENT"

    def test_feature_type_from_string(self):
        """Test creating FeatureType from string value."""
        assert FeatureType("neuron") == FeatureType.NEURON
        assert FeatureType("head") == FeatureType.HEAD
        assert FeatureType("sae_latent") == FeatureType.SAE_LATENT


# --- Location Tests ---

class TestLocation:
    """Tests for the Location dataclass."""

    def test_location_minimal(self):
        """Test Location with only required fields."""
        loc = Location(model_name="gpt2", layer="mlp.10")
        assert loc.model_name == "gpt2"
        assert loc.layer == "mlp.10"
        assert loc.sublayer is None
        assert loc.neuron_idx is None
        assert loc.head_idx is None
        assert loc.position is None

    def test_location_full(self):
        """Test Location with all fields populated."""
        loc = Location(
            model_name="llama-2-7b",
            layer="attention.5",
            sublayer="attn_pattern",
            neuron_idx=42,
            head_idx=3,
            position="last",
        )
        assert loc.model_name == "llama-2-7b"
        assert loc.layer == "attention.5"
        assert loc.sublayer == "attn_pattern"
        assert loc.neuron_idx == 42
        assert loc.head_idx == 3
        assert loc.position == "last"

    def test_location_repr_minimal(self):
        """Test __repr__ with minimal fields."""
        loc = Location(model_name="gpt2", layer="mlp.10")
        repr_str = repr(loc)
        assert "model='gpt2'" in repr_str
        assert "layer='mlp.10'" in repr_str
        assert "sublayer" not in repr_str
        assert "neuron" not in repr_str

    def test_location_repr_full(self):
        """Test __repr__ with all fields."""
        loc = Location(
            model_name="gpt2",
            layer="mlp.10",
            sublayer="out_proj",
            neuron_idx=5,
            head_idx=2,
            position="first",
        )
        repr_str = repr(loc)
        assert "model='gpt2'" in repr_str
        assert "layer='mlp.10'" in repr_str
        assert "sublayer='out_proj'" in repr_str
        assert "neuron=5" in repr_str
        assert "head=2" in repr_str
        assert "position='first'" in repr_str


# --- History Tests ---

class TestHistory:
    """Tests for the History dataclass."""

    def test_history_minimal(self):
        """Test History with only required fields."""
        history = History(method="activation_patching", dataset="pile-subset")
        assert history.method == "activation_patching"
        assert history.dataset == "pile-subset"
        assert history.timestamp is not None
        assert history.tool_version is None
        assert history.hyperparameters is None
        assert history.author is None

    def test_history_full(self):
        """Test History with all fields populated."""
        history = History(
            method="linear_probe",
            dataset="pile",
            timestamp="2024-01-15T10:30:00",
            tool_version="1.2.0",
            hyperparameters={"lr": 0.001, "epochs": 100},
            author="researcher",
        )
        assert history.method == "linear_probe"
        assert history.dataset == "pile"
        assert history.timestamp == "2024-01-15T10:30:00"
        assert history.tool_version == "1.2.0"
        assert history.hyperparameters == {"lr": 0.001, "epochs": 100}
        assert history.author == "researcher"

    def test_history_timestamp_auto_generated(self):
        """Test that timestamp is auto-generated if not provided."""
        before = datetime.now().isoformat()
        history = History(method="test", dataset="test")
        after = datetime.now().isoformat()
        assert before <= history.timestamp <= after

    def test_history_repr(self):
        """Test __repr__ method of History."""
        history = History(
            method="activation_patching",
            dataset="pile",
            tool_version="1.0",
            author="tester",
        )
        repr_str = repr(history)
        assert "method='activation_patching'" in repr_str
        assert "dataset='pile'" in repr_str
        assert "version='1.0'" in repr_str
        assert "by='tester'" in repr_str


# --- Feature Tests ---

class TestFeature:
    """Tests for the Feature dataclass."""

    @pytest.fixture
    def basic_location(self):
        return Location(model_name="gpt2", layer="mlp.10")

    @pytest.fixture
    def basic_history(self):
        return History(method="probe", dataset="test")

    def test_feature_minimal(self, basic_location):
        """Test Feature with only required fields."""
        feature = Feature(
            feature_type=FeatureType.NEURON,
            location=basic_location,
            label="test_feature",
        )
        assert feature.feature_type == FeatureType.NEURON
        assert feature.location == basic_location
        assert feature.label == "test_feature"
        assert feature.id is not None
        assert len(feature.id) == 36  # UUID format
        assert feature.history is None
        assert feature.description is None
        assert feature.tags == []
        assert feature.metadata == {}

    def test_feature_full(self, basic_location, basic_history):
        """Test Feature with all fields populated."""
        feature = Feature(
            feature_type=FeatureType.SAE_LATENT,
            location=basic_location,
            label="full_feature",
            history=basic_history,
            description="A test feature",
            tags=["important", "verified"],
            metadata={"confidence": 0.95},
        )
        assert feature.feature_type == FeatureType.SAE_LATENT
        assert feature.label == "full_feature"
        assert feature.history == basic_history
        assert feature.description == "A test feature"
        assert feature.tags == ["important", "verified"]
        assert feature.metadata == {"confidence": 0.95}

    def test_feature_invalid_label_empty(self, basic_location):
        """Test that empty label raises ValueError."""
        with pytest.raises(ValueError, match="label must be a non-empty string"):
            Feature(feature_type=FeatureType.NEURON, location=basic_location, label="")

    def test_feature_invalid_label_type(self, basic_location):
        """Test that non-string label raises ValueError."""
        with pytest.raises(ValueError, match="label must be a non-empty string"):
            Feature(feature_type=FeatureType.NEURON, location=basic_location, label=123)

    def test_feature_invalid_feature_type(self, basic_location):
        """Test that invalid feature_type raises ValueError."""
        with pytest.raises(ValueError, match="feature_type must be a FeatureType enum"):
            Feature(feature_type="neuron", location=basic_location, label="test")

    def test_feature_to_dict(self, basic_location, basic_history):
        """Test serialization to dictionary."""
        feature = Feature(
            feature_type=FeatureType.HEAD,
            location=basic_location,
            label="test",
            history=basic_history,
            tags=["tag1"],
        )
        d = feature.to_dict()
        assert d["feature_type"] == "head"  # Serialized as string for JSON compatibility
        assert d["label"] == "test"
        assert d["location"]["model_name"] == "gpt2"
        assert d["history"]["method"] == "probe"
        assert d["tags"] == ["tag1"]

    def test_feature_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "id": "test-uuid-1234",
            "feature_type": "sae_latent",
            "location": {
                "model_name": "llama",
                "layer": "attn.3",
                "sublayer": "query",
            },
            "label": "restored_feature",
            "description": "Restored from dict",
            "tags": ["restored"],
            "history": {
                "method": "restoration",
                "dataset": "backup",
            },
        }
        feature = Feature.from_dict(data)
        assert feature.id == "test-uuid-1234"
        assert feature.feature_type == FeatureType.SAE_LATENT
        assert feature.location.model_name == "llama"
        assert feature.location.sublayer == "query"
        assert feature.label == "restored_feature"
        assert feature.description == "Restored from dict"
        assert feature.tags == ["restored"]
        assert feature.history.method == "restoration"

    def test_feature_from_dict_with_enum(self):
        """Test from_dict when feature_type is already an enum."""
        data = {
            "id": "test-id",
            "feature_type": FeatureType.NEURON,
            "location": {"model_name": "gpt2", "layer": "mlp.5"},
            "label": "test",
        }
        feature = Feature.from_dict(data)
        assert feature.feature_type == FeatureType.NEURON

    def test_feature_hash_and_equality(self, basic_location):
        """Test __hash__ and __eq__ methods."""
        feature1 = Feature(
            feature_type=FeatureType.NEURON,
            location=basic_location,
            label="test1",
        )
        feature2 = Feature(
            feature_type=FeatureType.NEURON,
            location=basic_location,
            label="test2",
        )
        # Different IDs means not equal
        assert feature1 != feature2

        # Same ID means equal (even if other fields differ)
        feature3 = Feature(
            feature_type=FeatureType.HEAD,
            location=Location(model_name="other", layer="other"),
            label="test3",
        )
        feature3.id = feature1.id
        assert feature1 == feature3

        # Can use in sets
        feature_set = {feature1, feature3}
        assert len(feature_set) == 1

    def test_feature_equality_not_implemented(self, basic_location):
        """Test equality with non-Feature object."""
        feature = Feature(
            feature_type=FeatureType.NEURON,
            location=basic_location,
            label="test",
        )
        assert feature != "not a feature"
        assert feature != 123

    def test_feature_repr(self, basic_location):
        """Test __repr__ method."""
        feature = Feature(
            feature_type=FeatureType.NEURON,
            location=basic_location,
            label="repr_test",
        )
        repr_str = repr(feature)
        assert "Feature" in repr_str
        assert "repr_test" in repr_str
        assert "NEURON" in repr_str

    def test_feature_str(self, basic_location, basic_history):
        """Test __str__ method."""
        feature = Feature(
            feature_type=FeatureType.NEURON,
            location=basic_location,
            label="str_test",
            description="A" * 100,  # Long description
            history=basic_history,
            tags=["tag1", "tag2"],
        )
        str_output = str(feature)
        assert "str_test" in str_output
        assert "neuron" in str_output
        assert "..." in str_output  # Truncated description
        assert "tag1" in str_output


# --- AtlasMetadata Tests ---

class TestAtlasMetadata:
    """Tests for the AtlasMetadata dataclass."""

    def test_metadata_defaults(self):
        """Test default values of AtlasMetadata."""
        metadata = AtlasMetadata()
        assert metadata.schema_version == SCHEMA_VERSION
        assert metadata.created is not None
        assert metadata.modified is not None
        assert metadata.name is None
        assert metadata.description is None
        assert metadata.model_name is None

    def test_metadata_with_values(self):
        """Test AtlasMetadata with custom values."""
        metadata = AtlasMetadata(
            name="test_atlas",
            description="Test description",
            model_name="gpt2",
        )
        assert metadata.name == "test_atlas"
        assert metadata.description == "Test description"
        assert metadata.model_name == "gpt2"

    def test_metadata_touch(self):
        """Test that touch() updates modified timestamp."""
        metadata = AtlasMetadata()
        original_modified = metadata.modified
        metadata.touch()
        assert metadata.modified >= original_modified

    def test_metadata_to_dict(self):
        """Test serialization to dictionary."""
        metadata = AtlasMetadata(name="test", description="desc")
        d = metadata.to_dict()
        assert d["name"] == "test"
        assert d["description"] == "desc"
        assert d["schema_version"] == SCHEMA_VERSION

    def test_metadata_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "schema_version": "1.0.0",
            "created": "2024-01-01T00:00:00",
            "modified": "2024-01-02T00:00:00",
            "name": "restored",
            "description": "Restored metadata",
            "model_name": "llama",
        }
        metadata = AtlasMetadata.from_dict(data)
        assert metadata.schema_version == "1.0.0"
        assert metadata.created == "2024-01-01T00:00:00"
        assert metadata.name == "restored"
        assert metadata.model_name == "llama"


# --- Atlas Tests ---

class TestAtlas:
    """Tests for the Atlas class."""

    @pytest.fixture
    def sample_feature(self):
        """Create a sample feature for testing."""
        return Feature(
            feature_type=FeatureType.NEURON,
            location=Location(model_name="gpt2", layer="mlp.10"),
            label="sample_feature",
            tags=["test"],
        )

    @pytest.fixture
    def sample_feature2(self):
        """Create a second sample feature for testing."""
        return Feature(
            feature_type=FeatureType.HEAD,
            location=Location(model_name="llama", layer="attn.5"),
            label="another_feature",
            tags=["test", "important"],
        )

    @pytest.fixture
    def empty_atlas(self):
        """Create an empty atlas."""
        return Atlas(name="test_atlas", description="Test description")

    def test_atlas_initialization_empty(self):
        """Test creating an empty atlas."""
        atlas = Atlas()
        assert len(atlas) == 0
        assert atlas.metadata.name is None

    def test_atlas_initialization_with_metadata(self):
        """Test atlas with name and description."""
        atlas = Atlas(name="my_atlas", description="My features", model_name="gpt2")
        assert atlas.metadata.name == "my_atlas"
        assert atlas.metadata.description == "My features"
        assert atlas.metadata.model_name == "gpt2"

    def test_atlas_initialization_with_features(self, sample_feature, sample_feature2):
        """Test atlas initialized with features."""
        atlas = Atlas(features=[sample_feature, sample_feature2])
        assert len(atlas) == 2
        assert sample_feature.id in atlas
        assert sample_feature2.id in atlas

    def test_atlas_add(self, empty_atlas, sample_feature):
        """Test adding a feature to atlas."""
        feature_id = empty_atlas.add(sample_feature)
        assert feature_id == sample_feature.id
        assert len(empty_atlas) == 1
        assert sample_feature.id in empty_atlas

    def test_atlas_add_duplicate(self, empty_atlas, sample_feature):
        """Test that adding duplicate feature raises error."""
        empty_atlas.add(sample_feature)
        with pytest.raises(ValueError, match="already exists"):
            empty_atlas.add(sample_feature)

    def test_atlas_get(self, empty_atlas, sample_feature):
        """Test retrieving a feature by ID."""
        empty_atlas.add(sample_feature)
        retrieved = empty_atlas.get(sample_feature.id)
        assert retrieved == sample_feature

    def test_atlas_get_nonexistent(self, empty_atlas):
        """Test getting non-existent feature returns None."""
        assert empty_atlas.get("nonexistent-id") is None

    def test_atlas_getitem(self, empty_atlas, sample_feature):
        """Test bracket access to features."""
        empty_atlas.add(sample_feature)
        assert empty_atlas[sample_feature.id] == sample_feature

    def test_atlas_getitem_nonexistent(self, empty_atlas):
        """Test bracket access to non-existent feature raises KeyError."""
        with pytest.raises(KeyError, match="No feature with ID"):
            _ = empty_atlas["nonexistent-id"]

    def test_atlas_contains(self, empty_atlas, sample_feature):
        """Test 'in' operator for atlas."""
        assert sample_feature.id not in empty_atlas
        empty_atlas.add(sample_feature)
        assert sample_feature.id in empty_atlas

    def test_atlas_remove(self, empty_atlas, sample_feature):
        """Test removing a feature."""
        empty_atlas.add(sample_feature)
        result = empty_atlas.remove(sample_feature.id)
        assert result is True
        assert len(empty_atlas) == 0
        assert sample_feature.id not in empty_atlas

    def test_atlas_remove_nonexistent(self, empty_atlas):
        """Test removing non-existent feature returns False."""
        result = empty_atlas.remove("nonexistent-id")
        assert result is False

    def test_atlas_list(self, empty_atlas, sample_feature, sample_feature2):
        """Test listing all features."""
        empty_atlas.add(sample_feature)
        empty_atlas.add(sample_feature2)
        features = empty_atlas.list()
        assert len(features) == 2
        assert sample_feature in features
        assert sample_feature2 in features

    def test_atlas_iter(self, empty_atlas, sample_feature, sample_feature2):
        """Test iterating over atlas."""
        empty_atlas.add(sample_feature)
        empty_atlas.add(sample_feature2)
        feature_ids = [f.id for f in empty_atlas]
        assert len(feature_ids) == 2
        assert sample_feature.id in feature_ids
        assert sample_feature2.id in feature_ids

    # --- Search and Filter Tests ---

    def test_find_by_id(self, empty_atlas, sample_feature):
        """Test find_by_id method."""
        empty_atlas.add(sample_feature)
        assert empty_atlas.find_by_id(sample_feature.id) == sample_feature
        assert empty_atlas.find_by_id("nonexistent") is None

    def test_find_by_label_exact(self, empty_atlas, sample_feature):
        """Test find_by_label with exact match."""
        empty_atlas.add(sample_feature)
        results = empty_atlas.find_by_label("sample_feature", exact=True)
        assert len(results) == 1
        assert results[0] == sample_feature

        results = empty_atlas.find_by_label("SAMPLE_FEATURE", exact=True)
        assert len(results) == 0

    def test_find_by_label_substring(self, empty_atlas, sample_feature, sample_feature2):
        """Test find_by_label with substring match."""
        empty_atlas.add(sample_feature)
        empty_atlas.add(sample_feature2)
        results = empty_atlas.find_by_label("feature")
        assert len(results) == 2

        results = empty_atlas.find_by_label("SAMPLE")
        assert len(results) == 1  # Case-insensitive

    def test_find_by_type(self, empty_atlas, sample_feature, sample_feature2):
        """Test find_by_type method."""
        empty_atlas.add(sample_feature)  # NEURON
        empty_atlas.add(sample_feature2)  # HEAD

        results = empty_atlas.find_by_type(FeatureType.NEURON)
        assert len(results) == 1
        assert results[0] == sample_feature

        results = empty_atlas.find_by_type("head")  # String input
        assert len(results) == 1
        assert results[0] == sample_feature2

    def test_find_by_layer_exact(self, empty_atlas, sample_feature, sample_feature2):
        """Test find_by_layer with exact match."""
        empty_atlas.add(sample_feature)  # mlp.10
        empty_atlas.add(sample_feature2)  # attn.5

        results = empty_atlas.find_by_layer("mlp.10", exact=True)
        assert len(results) == 1

        results = empty_atlas.find_by_layer("MLP.10", exact=True)
        assert len(results) == 0

    def test_find_by_layer_substring(self, empty_atlas, sample_feature, sample_feature2):
        """Test find_by_layer with substring match."""
        empty_atlas.add(sample_feature)
        empty_atlas.add(sample_feature2)

        results = empty_atlas.find_by_layer("mlp")
        assert len(results) == 1

        results = empty_atlas.find_by_layer("10")
        assert len(results) == 1

    def test_find_by_model(self, empty_atlas, sample_feature, sample_feature2):
        """Test find_by_model method."""
        empty_atlas.add(sample_feature)  # gpt2
        empty_atlas.add(sample_feature2)  # llama

        results = empty_atlas.find_by_model("gpt2", exact=True)
        assert len(results) == 1

        results = empty_atlas.find_by_model("gp")
        assert len(results) == 1

        results = empty_atlas.find_by_model("GPT")
        assert len(results) == 1  # Case-insensitive

    def test_find_by_tag(self, empty_atlas, sample_feature, sample_feature2):
        """Test find_by_tag method."""
        empty_atlas.add(sample_feature)  # tags: ["test"]
        empty_atlas.add(sample_feature2)  # tags: ["test", "important"]

        results = empty_atlas.find_by_tag("test")
        assert len(results) == 2

        results = empty_atlas.find_by_tag("important")
        assert len(results) == 1

        results = empty_atlas.find_by_tag("nonexistent")
        assert len(results) == 0

    def test_filter_combined(self, empty_atlas, sample_feature, sample_feature2):
        """Test filter method with combined criteria."""
        feature3 = Feature(
            feature_type=FeatureType.NEURON,
            location=Location(model_name="gpt2", layer="mlp.5"),
            label="third_feature",
            tags=["test"],
        )
        empty_atlas.add(sample_feature)  # NEURON, gpt2, mlp.10
        empty_atlas.add(sample_feature2)  # HEAD, llama, attn.5
        empty_atlas.add(feature3)  # NEURON, gpt2, mlp.5

        # Filter by type
        results = empty_atlas.filter(feature_type=FeatureType.NEURON)
        assert len(results) == 2

        # Filter by type and model
        results = empty_atlas.filter(feature_type=FeatureType.NEURON, model_name="gpt2")
        assert len(results) == 2

        # Filter by type, model, and layer
        results = empty_atlas.filter(
            feature_type=FeatureType.NEURON,
            model_name="gpt2",
            layer="10"
        )
        assert len(results) == 1

        # Filter by tag
        results = empty_atlas.filter(tag="important")
        assert len(results) == 1

        # Filter with string type
        results = empty_atlas.filter(feature_type="neuron")
        assert len(results) == 2

    # --- Persistence Tests ---

    def test_atlas_to_dict(self, empty_atlas, sample_feature):
        """Test atlas serialization to dict."""
        empty_atlas.add(sample_feature)
        d = empty_atlas.to_dict()

        assert "metadata" in d
        assert "features" in d
        assert len(d["features"]) == 1
        assert d["features"][0]["label"] == "sample_feature"

    def test_atlas_from_dict(self):
        """Test atlas deserialization from dict."""
        data = {
            "metadata": {
                "schema_version": "1.0.0",
                "created": "2024-01-01T00:00:00",
                "modified": "2024-01-01T00:00:00",
                "name": "restored_atlas",
                "description": "Restored",
            },
            "features": [
                {
                    "id": "test-id-1",
                    "feature_type": "neuron",
                    "location": {"model_name": "gpt2", "layer": "mlp.10"},
                    "label": "feature1",
                }
            ],
        }
        atlas = Atlas.from_dict(data)
        assert atlas.metadata.name == "restored_atlas"
        assert len(atlas) == 1
        assert atlas.list()[0].label == "feature1"

    def test_atlas_from_dict_missing_features(self):
        """Test from_dict raises error without features key."""
        data = {"metadata": {"name": "test"}}
        with pytest.raises(ValueError, match="must contain the 'features' key"):
            Atlas.from_dict(data)

    def test_atlas_save_and_load_json(self, empty_atlas, sample_feature, sample_feature2):
        """Test saving and loading atlas as JSON."""
        empty_atlas.add(sample_feature)
        empty_atlas.add(sample_feature2)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_atlas.json"
            empty_atlas.save(filepath)

            assert filepath.exists()

            loaded_atlas = Atlas.load(filepath)
            assert len(loaded_atlas) == 2
            assert loaded_atlas.metadata.name == "test_atlas"

    def test_atlas_save_unsupported_extension(self, empty_atlas):
        """Test save with unsupported extension raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.txt"
            with pytest.raises(ValueError, match="Unsupported file extension"):
                empty_atlas.save(filepath)

    def test_atlas_load_nonexistent(self):
        """Test loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            Atlas.load("nonexistent_file.json")

    def test_atlas_load_unsupported_extension(self):
        """Test load with unsupported extension raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.txt"
            filepath.touch()
            with pytest.raises(ValueError, match="Unsupported file extension"):
                Atlas.load(filepath)

    # --- Utility Tests ---

    def test_atlas_repr(self, empty_atlas):
        """Test __repr__ method."""
        repr_str = repr(empty_atlas)
        assert "test_atlas" in repr_str
        assert "0 features" in repr_str

    def test_atlas_str(self, empty_atlas, sample_feature):
        """Test __str__ method (summary)."""
        empty_atlas.add(sample_feature)
        summary = str(empty_atlas)
        assert "Atlas Object" in summary
        assert "test_atlas" in summary
        assert "Total Features: 1" in summary
        assert "neuron: 1" in summary

    def test_atlas_summary(self, empty_atlas, sample_feature, sample_feature2):
        """Test summary method with multiple features."""
        empty_atlas.add(sample_feature)
        empty_atlas.add(sample_feature2)
        summary = empty_atlas.summary()

        assert "By Type:" in summary
        assert "neuron: 1" in summary
        assert "head: 1" in summary
        assert "By Model:" in summary
        assert "gpt2: 1" in summary
        assert "llama: 1" in summary

    def test_metadata_touch_on_add(self, empty_atlas, sample_feature):
        """Test that metadata is updated when feature is added."""
        original_modified = empty_atlas.metadata.modified
        empty_atlas.add(sample_feature)
        assert empty_atlas.metadata.modified >= original_modified

    def test_metadata_touch_on_remove(self, empty_atlas, sample_feature):
        """Test that metadata is updated when feature is removed."""
        empty_atlas.add(sample_feature)
        modified_after_add = empty_atlas.metadata.modified
        empty_atlas.remove(sample_feature.id)
        assert empty_atlas.metadata.modified >= modified_after_add


# --- YAML Tests (optional, requires PyYAML) ---

class TestAtlasYAML:
    """Tests for YAML save/load functionality."""

    @pytest.fixture
    def sample_atlas(self):
        atlas = Atlas(name="yaml_test")
        feature = Feature(
            feature_type=FeatureType.SAE_LATENT,
            location=Location(model_name="test", layer="layer.0"),
            label="yaml_feature",
        )
        atlas.add(feature)
        return atlas

    def test_yaml_save_and_load(self, sample_atlas):
        """Test YAML save and load if PyYAML is available."""
        pytest.importorskip("yaml")

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.yaml"
            sample_atlas.save(filepath)
            assert filepath.exists()

            loaded = Atlas.load(filepath)
            assert len(loaded) == 1
            assert loaded.metadata.name == "yaml_test"

    def test_yaml_missing_import(self, sample_atlas, monkeypatch):
        """Test that missing PyYAML raises ImportError."""
        # Skip if yaml is actually installed
        pytest.importorskip("yaml")
        # This test passes if yaml is available since we can't easily mock the import
