"""
Adapter functions that convert real glassbox data objects into
Manim-renderable data dictionaries.

Each adapter takes a glassbox object and returns a plain dict
that the corresponding Manim scene can consume without any
dependency on the glassbox library at render time.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data containers returned by adapters
# ---------------------------------------------------------------------------

@dataclass
class CircuitSceneData:
    """Data for CircuitDiscoveryScene."""

    model_name: str
    circuit_name: Optional[str]
    nodes: List[Dict[str, Any]]  # [{id, type, layer, index, label, ...}]
    edges: List[Dict[str, Any]]  # [{source, target, weight, edge_type}]
    layers: List[int]            # sorted unique layer indices
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProbeSceneData:
    """Data for ProbingHyperplaneScene."""

    layer: str
    direction_name: str
    coefficients: np.ndarray          # (n_features,) – learned weight vector
    accuracy: float
    f1: Optional[float]
    points_2d: Optional[np.ndarray]   # (n_samples, 2) projected activations
    labels: Optional[np.ndarray]      # (n_samples,) class labels
    class_names: Optional[List[str]]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SAESceneData:
    """Data for SAEFeatureDiscoveryScene."""

    model_name: str
    layer: int
    features: List[Dict[str, Any]]  # [{id, sparsity, max_activation, top_tokens, decoder_2d}]
    activation_grid: Optional[np.ndarray]  # (n_features, n_samples) activation matrix
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SteeringSceneData:
    """Data for SteeringVectorScene."""

    direction_name: str
    layer: str
    steering_vector_2d: np.ndarray     # (2,) projected steering direction
    points_before_2d: np.ndarray       # (n_samples, 2) activations before steering
    points_after_2d: np.ndarray        # (n_samples, 2) activations after steering
    labels: Optional[np.ndarray]
    strength: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineSceneData:
    """Data for FullPipelineScene."""

    model_name: str
    stages: List[Dict[str, Any]]  # ordered pipeline stages with names & summary stats
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Adapter functions
# ---------------------------------------------------------------------------

def circuit_graph_to_scene_data(graph) -> CircuitSceneData:
    """
    Convert a CircuitGraph into data for CircuitDiscoveryScene.

    Args:
        graph: A ``CircuitGraph`` instance.

    Returns:
        CircuitSceneData ready for rendering.
    """
    nodes = []
    for node in graph.nodes:
        label = node.metadata.get("label", node.id)
        nodes.append({
            "id": node.id,
            "type": node.node_type.value,
            "layer": node.layer,
            "index": node.index,
            "label": label,
            "metadata": node.metadata,
        })

    edges = []
    for edge in graph.edges:
        edges.append({
            "source": edge.source,
            "target": edge.target,
            "weight": edge.weight,
            "edge_type": edge.edge_type.value,
        })

    layers_set = {n.layer for n in graph.nodes if n.layer is not None}
    layers = sorted(layers_set)

    return CircuitSceneData(
        model_name=graph.model,
        circuit_name=graph.name,
        nodes=nodes,
        edges=edges,
        layers=layers,
        metadata=graph.summary(),
    )


def probe_result_to_scene_data(
    probe_result,
    activations: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    layer: str = "",
    direction_name: str = "",
    max_samples: int = 500,
) -> ProbeSceneData:
    """
    Convert a ProbeResult (and optional activations) into data for
    ProbingHyperplaneScene.

    Args:
        probe_result: A ``ProbeResult`` instance.
        activations: Optional (n_samples, n_features) array for scatter plot.
            Will be projected to 2D via PCA internally.
        labels: Optional (n_samples,) class labels for coloring.
        class_names: Optional human-readable class names.
        layer: Layer identifier string.
        direction_name: Name of the probed concept.
        max_samples: Maximum number of samples to include (for performance).

    Returns:
        ProbeSceneData ready for rendering.
    """
    coefficients = probe_result.coefficients
    if coefficients is None:
        coefficients = np.zeros(1)

    points_2d = None
    if activations is not None:
        from sklearn.decomposition import PCA

        acts = np.asarray(activations)
        if acts.ndim == 3:
            acts = acts.mean(axis=1)

        # Subsample if too many
        if acts.shape[0] > max_samples:
            idx = np.random.default_rng(42).choice(acts.shape[0], max_samples, replace=False)
            acts = acts[idx]
            if labels is not None:
                labels = np.asarray(labels)[idx]

        pca = PCA(n_components=2, random_state=42)
        points_2d = pca.fit_transform(acts)

    return ProbeSceneData(
        layer=layer,
        direction_name=direction_name,
        coefficients=coefficients,
        accuracy=probe_result.accuracy,
        f1=probe_result.f1,
        points_2d=points_2d,
        labels=labels,
        class_names=class_names,
        metadata=probe_result.metadata or {},
    )


def sae_features_to_scene_data(
    features,
    activation_matrix: Optional[np.ndarray] = None,
    model_name: str = "",
    layer: int = 0,
    top_k: int = 10,
) -> SAESceneData:
    """
    Convert a list of SAEFeature objects into data for SAEFeatureDiscoveryScene.

    Args:
        features: List of ``SAEFeature`` instances (or a ``FeatureSet``).
        activation_matrix: Optional (n_features, n_samples) activation values.
        model_name: Model identifier.
        layer: Layer index.
        top_k: Number of top features to include (sorted by max activation).

    Returns:
        SAESceneData ready for rendering.
    """
    feature_list = list(features)

    # Sort by max_activation descending, take top_k
    def _sort_key(f):
        ma = f.activation_stats.get("max_activation", 0.0)
        return ma if ma is not None else 0.0

    feature_list.sort(key=_sort_key, reverse=True)
    feature_list = feature_list[:top_k]

    # Project decoder vectors to 2D for scatter
    import torch
    decoder_vecs = []
    for f in feature_list:
        vec = f.decoder_vector
        if isinstance(vec, torch.Tensor):
            vec = vec.detach().cpu().numpy()
        decoder_vecs.append(vec)

    decoder_2d_coords = None
    if decoder_vecs:
        stacked = np.stack(decoder_vecs, axis=0)  # (top_k, input_dim)
        if stacked.shape[0] >= 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=42)
            decoder_2d_coords = pca.fit_transform(stacked)
        else:
            decoder_2d_coords = np.zeros((stacked.shape[0], 2))

    scene_features = []
    for i, f in enumerate(feature_list):
        entry = {
            "id": f.id,
            "sparsity": f.sparsity,
            "max_activation": f.max_activation,
            "mean_activation": f.mean_activation,
        }
        if decoder_2d_coords is not None:
            entry["decoder_2d"] = decoder_2d_coords[i].tolist()
        scene_features.append(entry)

    act_grid = None
    if activation_matrix is not None:
        act_grid = np.asarray(activation_matrix)

    return SAESceneData(
        model_name=model_name or (feature_list[0].model_name if feature_list else ""),
        layer=layer or (feature_list[0].layer if feature_list else 0),
        features=scene_features,
        activation_grid=act_grid,
        metadata={"top_k": top_k, "total_features": len(list(features))},
    )


def steering_result_to_scene_data(
    steering_vector: np.ndarray,
    activations_before: np.ndarray,
    activations_after: np.ndarray,
    direction_name: str = "steering",
    layer: str = "",
    labels: Optional[np.ndarray] = None,
    strength: float = 1.0,
    max_samples: int = 500,
) -> SteeringSceneData:
    """
    Convert steering experiment results into data for SteeringVectorScene.

    Args:
        steering_vector: (n_features,) the steering direction.
        activations_before: (n_samples, n_features) activations before steering.
        activations_after: (n_samples, n_features) activations after steering.
        direction_name: Human-readable name for the steering direction.
        layer: Layer identifier string.
        labels: Optional (n_samples,) labels for coloring.
        strength: Steering coefficient used.
        max_samples: Maximum samples for rendering.

    Returns:
        SteeringSceneData ready for rendering.
    """
    from sklearn.decomposition import PCA

    before = np.asarray(activations_before)
    after = np.asarray(activations_after)
    vec = np.asarray(steering_vector)

    # Subsample
    if before.shape[0] > max_samples:
        idx = np.random.default_rng(42).choice(before.shape[0], max_samples, replace=False)
        before = before[idx]
        after = after[idx]
        if labels is not None:
            labels = np.asarray(labels)[idx]

    # Joint PCA so before/after share the same 2D space
    combined = np.vstack([before, after])
    pca = PCA(n_components=2, random_state=42)
    combined_2d = pca.fit_transform(combined)
    n = before.shape[0]
    before_2d = combined_2d[:n]
    after_2d = combined_2d[n:]

    vec_2d = pca.transform(vec.reshape(1, -1))[0]

    return SteeringSceneData(
        direction_name=direction_name,
        layer=layer,
        steering_vector_2d=vec_2d,
        points_before_2d=before_2d,
        points_after_2d=after_2d,
        labels=labels,
        strength=strength,
    )


def pipeline_to_scene_data(
    model_name: str,
    circuit_graph=None,
    probe_results: Optional[List] = None,
    sae_features: Optional[List] = None,
    steering_results: Optional[Dict] = None,
) -> PipelineSceneData:
    """
    Combine multiple analysis results into data for FullPipelineScene.

    Args:
        model_name: Model identifier.
        circuit_graph: Optional CircuitGraph.
        probe_results: Optional list of (layer, direction, ProbeResult) tuples.
        sae_features: Optional list of SAEFeature objects.
        steering_results: Optional dict with steering experiment info.

    Returns:
        PipelineSceneData ready for rendering.
    """
    stages = []

    # Stage 1: SAE Feature Discovery
    if sae_features is not None:
        feature_list = list(sae_features)
        stages.append({
            "name": "SAE Feature Discovery",
            "type": "sae",
            "summary": f"{len(feature_list)} features extracted",
            "detail": {
                "n_features": len(feature_list),
            },
        })

    # Stage 2: Probing
    if probe_results is not None:
        stages.append({
            "name": "Linear Probing",
            "type": "probe",
            "summary": f"{len(probe_results)} probes trained",
            "detail": {
                "probes": [
                    {
                        "layer": layer,
                        "direction": direction,
                        "accuracy": result.accuracy,
                        "f1": result.f1,
                    }
                    for layer, direction, result in probe_results
                ],
            },
        })

    # Stage 3: Circuit Discovery
    if circuit_graph is not None:
        summary = circuit_graph.summary()
        stages.append({
            "name": "Circuit Discovery",
            "type": "circuit",
            "summary": f"{summary['num_nodes']} nodes, {summary['num_edges']} edges",
            "detail": summary,
        })

    # Stage 4: Steering
    if steering_results is not None:
        stages.append({
            "name": "Activation Steering",
            "type": "steering",
            "summary": f"direction: {steering_results.get('direction', 'unknown')}",
            "detail": steering_results,
        })

    return PipelineSceneData(
        model_name=model_name,
        stages=stages,
    )
