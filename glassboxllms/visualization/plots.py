"""
Static plotting functions for interpretability experiments.

All functions return matplotlib Figure objects so callers can further
customise, save, or display them.  Heavy dependencies (matplotlib, seaborn,
networkx) are imported lazily so the rest of the package stays lightweight.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------

def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        return matplotlib, plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for static plots. "
            "Install it with: pip install matplotlib"
        )


def _require_seaborn():
    try:
        import seaborn as sns
        return sns
    except ImportError:
        raise ImportError(
            "seaborn is required for heatmap plots. "
            "Install it with: pip install seaborn"
        )


def _require_networkx():
    try:
        import networkx as nx
        return nx
    except ImportError:
        raise ImportError(
            "networkx is required for circuit graph rendering. "
            "Install it with: pip install networkx"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy(x) -> np.ndarray:
    """Convert torch tensors or lists to numpy arrays."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _save_or_return(fig, save_path: Optional[Union[str, Path]]):
    """Save figure to disk if *save_path* is given, then return it."""
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=150)
    return fig


# ---------------------------------------------------------------------------
# 1. Attention heatmap
# ---------------------------------------------------------------------------

def plot_attention_heatmap(
    attention: np.ndarray,
    tokens: Optional[Sequence[str]] = None,
    *,
    layer: Optional[int] = None,
    head: Optional[int] = None,
    figsize: Tuple[float, float] = (10, 8),
    cmap: str = "viridis",
    save_path: Optional[Union[str, Path]] = None,
) -> Any:
    """Plot an attention matrix as a heatmap.

    Args:
        attention: 2-D array of shape (seq_len, seq_len).
        tokens: Optional token labels for axes.
        layer: Optional layer index (used in title).
        head: Optional head index (used in title).
        figsize: Figure size.
        cmap: Matplotlib colourmap name.
        save_path: If given, save figure to this path.

    Returns:
        matplotlib Figure.
    """
    _, plt = _require_matplotlib()
    sns = _require_seaborn()
    attention = _to_numpy(attention)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        attention,
        ax=ax,
        cmap=cmap,
        xticklabels=tokens if tokens is not None else "auto",
        yticklabels=tokens if tokens is not None else "auto",
        square=True,
        vmin=0,
        vmax=attention.max(),
    )

    title_parts = ["Attention"]
    if layer is not None:
        title_parts.append(f"Layer {layer}")
    if head is not None:
        title_parts.append(f"Head {head}")
    ax.set_title(" — ".join(title_parts))
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")

    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# 2. Logit lens — layer-by-layer token predictions
# ---------------------------------------------------------------------------

def plot_logit_lens(
    logit_lens_data: np.ndarray,
    tokens: Sequence[str],
    top_k_tokens: Optional[List[List[str]]] = None,
    *,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: str = "RdBu_r",
    save_path: Optional[Union[str, Path]] = None,
) -> Any:
    """Visualise logit-lens layer-by-layer predictions.

    Args:
        logit_lens_data: 2-D array of shape (n_layers, seq_len) containing the
            probability (or log-prob) assigned to the *correct* next token at
            each layer.
        tokens: Token strings for the x-axis (length == seq_len).
        top_k_tokens: Optional nested list of top predicted tokens per
            (layer, position) for annotation.
        figsize: Figure size (auto-computed if *None*).
        cmap: Colourmap.
        save_path: Optional save path.

    Returns:
        matplotlib Figure.
    """
    _, plt = _require_matplotlib()
    sns = _require_seaborn()
    logit_lens_data = _to_numpy(logit_lens_data)

    n_layers, seq_len = logit_lens_data.shape
    if figsize is None:
        figsize = (max(8, seq_len * 0.8), max(6, n_layers * 0.5))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        logit_lens_data,
        ax=ax,
        cmap=cmap,
        xticklabels=list(tokens),
        yticklabels=[f"Layer {i}" for i in range(n_layers)],
        annot=top_k_tokens if top_k_tokens is not None else False,
        fmt="" if top_k_tokens is not None else ".2f",
    )
    ax.set_title("Logit Lens — Correct-Token Probability by Layer")
    ax.set_xlabel("Token position")
    ax.set_ylabel("Layer")

    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# 3. SAE training curves (loss / explained variance / sparsity over time)
# ---------------------------------------------------------------------------

def plot_sae_training_curves(
    training_history: Dict[str, List[float]],
    *,
    metrics: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (12, 8),
    save_path: Optional[Union[str, Path]] = None,
) -> Any:
    """Plot SAE training metrics over time.

    Designed to accept the ``training_history`` dict from
    :class:`~glassboxllms.features.TrainerState`.

    Args:
        training_history: Mapping of metric name to list of per-step values.
            Expected keys include ``loss``, ``mse_loss``, ``explained_variance``,
            ``mean_l0``, ``dead_feature_count``.
        metrics: Subset of keys to plot (default: all non-empty).
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        matplotlib Figure.
    """
    _, plt = _require_matplotlib()

    if metrics is None:
        metrics = [k for k, v in training_history.items() if v]

    n = len(metrics)
    if n == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No training metrics recorded", ha="center", va="center")
        return _save_or_return(fig, save_path)

    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)
    if n == 1:
        axes = [axes]

    for ax, key in zip(axes, metrics):
        values = training_history[key]
        ax.plot(values, linewidth=0.8)
        ax.set_ylabel(key.replace("_", " ").title())
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Training step")
    fig.suptitle("SAE Training Curves", fontsize=14)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# 4. SAE sparsity / reconstruction quality
# ---------------------------------------------------------------------------

def plot_sae_sparsity(
    feature_sparsities: np.ndarray,
    *,
    reconstruction_mse: Optional[np.ndarray] = None,
    figsize: Tuple[float, float] = (12, 5),
    save_path: Optional[Union[str, Path]] = None,
) -> Any:
    """Plot per-feature sparsity histogram and optional reconstruction error.

    Args:
        feature_sparsities: 1-D array of per-feature activation frequencies.
        reconstruction_mse: Optional 1-D array of per-sample MSE values.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        matplotlib Figure.
    """
    _, plt = _require_matplotlib()
    feature_sparsities = _to_numpy(feature_sparsities)

    ncols = 2 if reconstruction_mse is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    if ncols == 1:
        axes = [axes]

    # Sparsity histogram
    axes[0].hist(feature_sparsities, bins=50, edgecolor="black", linewidth=0.5)
    axes[0].set_xlabel("Activation frequency")
    axes[0].set_ylabel("Number of features")
    axes[0].set_title("Feature Sparsity Distribution")
    axes[0].axvline(
        feature_sparsities.mean(), color="red", linestyle="--",
        label=f"Mean = {feature_sparsities.mean():.4f}",
    )
    axes[0].legend()

    # Reconstruction error
    if reconstruction_mse is not None:
        reconstruction_mse = _to_numpy(reconstruction_mse)
        axes[1].hist(reconstruction_mse, bins=50, edgecolor="black", linewidth=0.5)
        axes[1].set_xlabel("Reconstruction MSE")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Reconstruction Error Distribution")
        axes[1].axvline(
            reconstruction_mse.mean(), color="red", linestyle="--",
            label=f"Mean = {reconstruction_mse.mean():.4f}",
        )
        axes[1].legend()

    fig.suptitle("SAE Sparsity & Reconstruction", fontsize=14)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# 5. Probe accuracy curves
# ---------------------------------------------------------------------------

def plot_probe_accuracy(
    results: Dict[str, Any],
    *,
    metric: str = "accuracy",
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[Union[str, Path]] = None,
) -> Any:
    """Plot probe performance across layers.

    Args:
        results: Mapping of layer name/index to a dict (or
            :class:`~glassboxllms.primitives.probes.ProbeResult`) containing
            at least the chosen *metric*.  Alternatively a dict with
            ``"layers"`` and ``"scores"`` keys.
        metric: Which metric to plot (``accuracy``, ``f1``, ``precision``, etc.).
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        matplotlib Figure.
    """
    _, plt = _require_matplotlib()

    # Accept both {layer: ProbeResult/dict} and {"layers": [...], "scores": [...]}
    if "layers" in results and "scores" in results:
        layers = results["layers"]
        scores = _to_numpy(results["scores"])
    else:
        layers = []
        scores = []
        for layer_key, value in results.items():
            layers.append(layer_key)
            if hasattr(value, metric):
                scores.append(getattr(value, metric))
            elif isinstance(value, dict):
                scores.append(value.get(metric, 0))
            else:
                scores.append(float(value))
        scores = np.array(scores)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(layers))
    ax.bar(x, scores, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers], rotation=45, ha="right")
    ax.set_xlabel("Layer")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Probe {metric.replace('_', ' ').title()} by Layer")
    ax.set_ylim(0, max(1.0, scores.max() * 1.1))
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# 6. Circuit graph rendering (networkx + matplotlib)
# ---------------------------------------------------------------------------

# Colour palette for node types
_NODE_COLORS: Dict[str, str] = {
    "neuron": "#4C72B0",
    "attention_head": "#DD8452",
    "feature": "#55A868",
    "mlp_layer": "#C44E52",
    "residual_stream": "#8172B3",
    "embedding": "#937860",
    "unembedding": "#DA8BC3",
}


def plot_circuit_graph(
    circuit_graph,
    *,
    layout: str = "layer",
    figsize: Tuple[float, float] = (14, 10),
    node_size: int = 800,
    font_size: int = 7,
    save_path: Optional[Union[str, Path]] = None,
) -> Any:
    """Render a :class:`~glassboxllms.analysis.circuits.CircuitGraph` as a
    static matplotlib figure.

    Args:
        circuit_graph: A ``CircuitGraph`` instance.
        layout: Layout algorithm — ``"layer"`` (default, groups by layer),
            ``"spring"``, or ``"circular"``.
        figsize: Figure size.
        node_size: Size of drawn nodes.
        font_size: Label font size.
        save_path: Optional save path.

    Returns:
        matplotlib Figure.
    """
    _, plt = _require_matplotlib()
    nx = _require_networkx()

    G = nx.DiGraph()

    for node in circuit_graph.nodes:
        G.add_node(
            node.id,
            node_type=node.node_type.value,
            layer=node.layer,
        )

    for edge in circuit_graph.edges:
        G.add_edge(
            edge.source,
            edge.target,
            weight=edge.weight if edge.weight is not None else 0.5,
        )

    # Layout
    if layout == "layer":
        pos = _layer_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42, k=2.0)

    fig, ax = plt.subplots(figsize=figsize)

    # Node colours
    colors = [
        _NODE_COLORS.get(G.nodes[n].get("node_type", ""), "#999999")
        for n in G.nodes()
    ]

    # Edge widths proportional to weight
    weights = [G.edges[e].get("weight", 0.5) for e in G.edges()]
    max_w = max(weights) if weights else 1.0
    widths = [1.0 + 3.0 * (w / max_w) for w in weights]

    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=colors,
        node_size=node_size, edgecolors="black", linewidths=0.5,
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax, font_size=font_size,
    )
    nx.draw_networkx_edges(
        G, pos, ax=ax, width=widths,
        edge_color="#555555", arrows=True,
        arrowsize=15, connectionstyle="arc3,rad=0.1",
    )

    # Legend
    from matplotlib.lines import Line2D
    legend_handles = []
    seen = set()
    for n in G.nodes():
        nt = G.nodes[n].get("node_type", "")
        if nt not in seen:
            seen.add(nt)
            legend_handles.append(
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=_NODE_COLORS.get(nt, "#999999"),
                       markersize=10, label=nt.replace("_", " ").title())
            )
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper left", framealpha=0.9)

    title = f"Circuit: {circuit_graph.name or circuit_graph.model}"
    ax.set_title(title, fontsize=13)
    ax.axis("off")

    fig.tight_layout()
    return _save_or_return(fig, save_path)


def _layer_layout(G) -> Dict[str, Tuple[float, float]]:
    """Position nodes in columns by their layer attribute."""
    nx = _require_networkx()

    layer_groups: Dict[Optional[int], list] = {}
    for node_id, data in G.nodes(data=True):
        layer = data.get("layer")
        layer_groups.setdefault(layer, []).append(node_id)

    sorted_layers = sorted(
        (k for k in layer_groups if k is not None),
    )
    # Put None-layer nodes on the far left
    if None in layer_groups:
        sorted_layers = [None] + sorted_layers

    pos = {}
    for col_idx, layer_key in enumerate(sorted_layers):
        nodes = layer_groups[layer_key]
        n = len(nodes)
        for row_idx, nid in enumerate(nodes):
            y = -(row_idx - (n - 1) / 2)
            pos[nid] = (col_idx, y)

    return pos


# ---------------------------------------------------------------------------
# 7. Steering effect comparison charts
# ---------------------------------------------------------------------------

def plot_steering_effects(
    effects: Dict[str, Dict[str, float]],
    *,
    metric: str = "score",
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[Union[str, Path]] = None,
) -> Any:
    """Compare the effect of different steering vectors / interventions.

    Args:
        effects: Mapping of intervention label to a dict of metrics.
            Example::

                {
                    "baseline": {"score": 0.2, "perplexity": 15.3},
                    "+love":    {"score": 0.8, "perplexity": 18.1},
                    "+hate":    {"score": -0.4, "perplexity": 20.5},
                }
        metric: Which metric key to plot as the primary bar chart.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        matplotlib Figure.
    """
    _, plt = _require_matplotlib()

    labels = list(effects.keys())
    values = [effects[l].get(metric, 0) for l in labels]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(labels))
    colors = ["#4C72B0" if v >= 0 else "#C44E52" for v in values]

    ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title("Steering Effect Comparison")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return _save_or_return(fig, save_path)
