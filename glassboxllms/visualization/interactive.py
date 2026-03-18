"""
Interactive Plotly-based visualizations for interpretability experiments.

All functions return ``plotly.graph_objects.Figure`` instances that can be
displayed in notebooks, exported to standalone HTML files, or embedded in
dashboards.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------

def _require_plotly():
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        return go, px
    except ImportError:
        raise ImportError(
            "plotly is required for interactive visualizations. "
            "Install it with: pip install plotly"
        )


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _save_or_return(fig, save_path: Optional[Union[str, Path]]):
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(path))
    return fig


# ---------------------------------------------------------------------------
# 1. SAE Feature browser
# ---------------------------------------------------------------------------

def feature_browser(
    feature_activations: np.ndarray,
    feature_labels: Optional[Sequence[str]] = None,
    *,
    token_labels: Optional[Sequence[str]] = None,
    top_k: int = 10,
    title: str = "SAE Feature Browser",
    save_path: Optional[Union[str, Path]] = None,
) -> Any:
    """Interactive heatmap for browsing SAE feature activations.

    Rows are features, columns are token positions. Hover shows activation
    strength, feature id, and token.

    Args:
        feature_activations: 2-D array of shape (n_features, seq_len).
            Typically the *features* output from
            :meth:`~glassboxllms.features.SparseAutoencoder.forward`,
            transposed so features are rows.
        feature_labels: Optional labels for each feature (length n_features).
        token_labels: Optional token strings for the x-axis.
        top_k: Show only the top-k most active features (by max activation).
        title: Figure title.
        save_path: If given, write an HTML file.

    Returns:
        plotly Figure.
    """
    go, _ = _require_plotly()
    feature_activations = _to_numpy(feature_activations)

    n_features, seq_len = feature_activations.shape

    # Select top-k features by max activation
    max_acts = feature_activations.max(axis=1)
    top_indices = np.argsort(max_acts)[-top_k:][::-1]
    subset = feature_activations[top_indices]

    y_labels = (
        [feature_labels[i] for i in top_indices]
        if feature_labels is not None
        else [f"Feature {i}" for i in top_indices]
    )
    x_labels = (
        list(token_labels) if token_labels is not None
        else [str(i) for i in range(seq_len)]
    )

    fig = go.Figure(data=go.Heatmap(
        z=subset,
        x=x_labels,
        y=y_labels,
        colorscale="Viridis",
        hovertemplate=(
            "Feature: %{y}<br>"
            "Token: %{x}<br>"
            "Activation: %{z:.4f}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Token position",
        yaxis_title="Feature",
        height=max(400, top_k * 35),
    )

    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# 2. Interactive circuit graph explorer
# ---------------------------------------------------------------------------

def interactive_circuit_explorer(
    circuit_graph,
    *,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> Any:
    """Interactive circuit graph visualization using Plotly.

    Nodes are coloured by type, sized by degree, and positioned by layer.
    Hover reveals node metadata; edges show weight on hover.

    Args:
        circuit_graph: A :class:`~glassboxllms.analysis.circuits.CircuitGraph`.
        title: Figure title (defaults to circuit name/model).
        save_path: If given, write an HTML file.

    Returns:
        plotly Figure.
    """
    go, _ = _require_plotly()

    nodes = circuit_graph.nodes
    edges = circuit_graph.edges

    # Build positions: x = layer, y = index within layer
    layer_buckets: Dict[Optional[int], list] = {}
    for node in nodes:
        layer_buckets.setdefault(node.layer, []).append(node)

    sorted_layers = sorted((k for k in layer_buckets if k is not None))
    if None in layer_buckets:
        sorted_layers = [None] + sorted_layers

    positions: Dict[str, Tuple[float, float]] = {}
    for col, layer_key in enumerate(sorted_layers):
        bucket = layer_buckets[layer_key]
        n = len(bucket)
        for row, node in enumerate(bucket):
            positions[node.id] = (col, -(row - (n - 1) / 2))

    # Node colours
    _COLORS = {
        "neuron": "#4C72B0",
        "attention_head": "#DD8452",
        "feature": "#55A868",
        "mlp_layer": "#C44E52",
        "residual_stream": "#8172B3",
        "embedding": "#937860",
        "unembedding": "#DA8BC3",
    }

    # Degree for sizing
    degree: Dict[str, int] = {n.id: 0 for n in nodes}
    for e in edges:
        degree[e.source] = degree.get(e.source, 0) + 1
        degree[e.target] = degree.get(e.target, 0) + 1

    # Edge traces
    edge_x, edge_y = [], []
    edge_hover = []
    for e in edges:
        x0, y0 = positions.get(e.source, (0, 0))
        x1, y1 = positions.get(e.target, (0, 0))
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        w = f"{e.weight:.3f}" if e.weight is not None else "n/a"
        edge_hover.append(f"{e.source} → {e.target} (weight: {w})")

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1, color="#888"),
        hoverinfo="none",
    )

    # Edge midpoint hover
    mid_x, mid_y, mid_text = [], [], []
    for e in edges:
        x0, y0 = positions.get(e.source, (0, 0))
        x1, y1 = positions.get(e.target, (0, 0))
        mid_x.append((x0 + x1) / 2)
        mid_y.append((y0 + y1) / 2)
        w = f"{e.weight:.3f}" if e.weight is not None else "n/a"
        mid_text.append(f"{e.source} → {e.target}<br>weight: {w}")

    edge_hover_trace = go.Scatter(
        x=mid_x, y=mid_y,
        mode="markers",
        marker=dict(size=4, color="rgba(0,0,0,0)"),
        hovertext=mid_text,
        hoverinfo="text",
    )

    # Node traces
    node_x = [positions[n.id][0] for n in nodes]
    node_y = [positions[n.id][1] for n in nodes]
    node_colors = [_COLORS.get(n.node_type.value, "#999999") for n in nodes]
    node_sizes = [max(15, 10 + degree.get(n.id, 0) * 5) for n in nodes]
    node_text = [
        f"<b>{n.id}</b><br>"
        f"Type: {n.node_type.value}<br>"
        f"Layer: {n.layer}<br>"
        f"Degree: {degree.get(n.id, 0)}"
        for n in nodes
    ]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=[n.id.split(".")[-1] for n in nodes],
        textposition="top center",
        textfont=dict(size=9),
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=1, color="black"),
        ),
    )

    fig = go.Figure(
        data=[edge_trace, edge_hover_trace, node_trace],
        layout=go.Layout(
            title=title or f"Circuit: {circuit_graph.name or circuit_graph.model}",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            hovermode="closest",
            plot_bgcolor="white",
            height=600,
        ),
    )

    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# 3. Embedding space 3D scatter (UMAP / t-SNE / PCA)
# ---------------------------------------------------------------------------

def embedding_scatter_3d(
    embeddings: np.ndarray,
    *,
    labels: Optional[Sequence[str]] = None,
    color_values: Optional[np.ndarray] = None,
    method: str = "pca",
    perplexity: float = 30.0,
    n_neighbors: int = 15,
    title: str = "Embedding Space (3D)",
    save_path: Optional[Union[str, Path]] = None,
) -> Any:
    """Interactive 3D scatter of high-dimensional embeddings.

    Reduces to 3 dimensions using PCA, t-SNE, or UMAP, then renders an
    interactive Plotly scatter.

    Args:
        embeddings: 2-D array of shape (n_samples, hidden_dim).
        labels: Optional per-point labels (for hover / colour).
        color_values: Optional numeric array for continuous colouring.
        method: Dimensionality reduction method — ``"pca"``, ``"tsne"``, or
            ``"umap"``.
        perplexity: t-SNE perplexity (only for ``method="tsne"``).
        n_neighbors: UMAP n_neighbors (only for ``method="umap"``).
        title: Figure title.
        save_path: If given, write an HTML file.

    Returns:
        plotly Figure.
    """
    go, _ = _require_plotly()
    embeddings = _to_numpy(embeddings)

    coords = _reduce_dimensions(embeddings, method, perplexity, n_neighbors)

    hover = labels if labels is not None else [f"sample {i}" for i in range(len(coords))]
    color = _to_numpy(color_values) if color_values is not None else None

    # Use discrete coloring for labels, continuous for color_values
    if color is not None:
        marker = dict(
            size=4,
            color=color,
            colorscale="Viridis",
            colorbar=dict(title="Value"),
            opacity=0.8,
        )
    elif labels is not None:
        # Map labels to integers for colouring
        unique = sorted(set(labels))
        label_to_int = {l: i for i, l in enumerate(unique)}
        marker = dict(
            size=4,
            color=[label_to_int[l] for l in labels],
            colorscale="Viridis",
            opacity=0.8,
        )
    else:
        marker = dict(size=4, color="#4C72B0", opacity=0.8)

    fig = go.Figure(data=[go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode="markers",
        marker=marker,
        hovertext=hover,
        hoverinfo="text",
    )])

    method_label = method.upper()
    fig.update_layout(
        title=f"{title} ({method_label})",
        scene=dict(
            xaxis_title=f"{method_label} 1",
            yaxis_title=f"{method_label} 2",
            zaxis_title=f"{method_label} 3",
        ),
        height=700,
    )

    return _save_or_return(fig, save_path)


def _reduce_dimensions(
    data: np.ndarray,
    method: str,
    perplexity: float,
    n_neighbors: int,
) -> np.ndarray:
    """Reduce to 3 dimensions using the specified method."""
    if data.shape[1] <= 3:
        # Already low-dim; just pad
        if data.shape[1] < 3:
            pad = np.zeros((data.shape[0], 3 - data.shape[1]))
            return np.hstack([data, pad])
        return data

    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=3).fit_transform(data)

    elif method == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(
            n_components=3, perplexity=perplexity, random_state=42,
        ).fit_transform(data)

    elif method == "umap":
        try:
            import umap
        except ImportError:
            raise ImportError(
                "umap-learn is required for UMAP reduction. "
                "Install it with: pip install umap-learn"
            )
        reducer = umap.UMAP(
            n_components=3, n_neighbors=n_neighbors, random_state=42,
        )
        return reducer.fit_transform(data)

    else:
        raise ValueError(f"Unknown reduction method: {method!r}. Use 'pca', 'tsne', or 'umap'.")
