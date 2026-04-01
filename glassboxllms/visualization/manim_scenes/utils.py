"""
Shared utilities, colour palette, and synthetic data generators for all
Manim scenes in glassbox-llms.

Render any scene with::

    manim -qh <file>.py <SceneName>
"""

import numpy as np
from manim import *

# ── Colour Palette ──────────────────────────────────────────────────
# Inspired by 3Blue1Brown's aesthetic: deep blue background, warm
# highlights, clean accent colours.

GLASS_BG       = "#1a1a2e"      # deep navy background
GLASS_PRIMARY  = "#e94560"      # warm red / coral
GLASS_ACCENT   = "#0f3460"      # dark blue accent
GLASS_GOLD     = "#f5a623"      # warm gold
GLASS_TEAL     = "#16c79a"      # mint / teal
GLASS_PURPLE   = "#9b59b6"      # purple
GLASS_LIGHT    = "#eaf2f8"      # off-white text
GLASS_DIM      = "#4a4a6a"      # muted grey
GLASS_GREEN    = "#2ecc71"      # success green
GLASS_ORANGE   = "#e67e22"      # warning orange

LAYER_COLORS = [
    "#3498db",  # layer 0 — blue
    "#2ecc71",  # layer 1 — green
    "#e74c3c",  # layer 2 — red
    "#f39c12",  # layer 3 — gold
    "#9b59b6",  # layer 4 — purple
    "#1abc9c",  # layer 5 — teal
    "#e67e22",  # layer 6 — orange
    "#e84393",  # layer 7 — pink
    "#00cec9",  # layer 8 — cyan
    "#6c5ce7",  # layer 9 — indigo
    "#fd79a8",  # layer 10 — rose
    "#00b894",  # layer 11 — emerald
]


def layer_color(i: int) -> str:
    """Return a colour for layer index *i* (wraps around)."""
    return LAYER_COLORS[i % len(LAYER_COLORS)]


# ── Text Helpers ────────────────────────────────────────────────────

def title_text(text: str, **kwargs) -> Text:
    """Large, bold title."""
    return Text(text, font_size=42, color=GLASS_LIGHT, weight=BOLD, **kwargs)


def subtitle_text(text: str, **kwargs) -> Text:
    """Smaller subtitle."""
    return Text(text, font_size=28, color=GLASS_GOLD, **kwargs)


def label_text(text: str, **kwargs) -> Text:
    """Small label for annotations."""
    return Text(text, font_size=20, color=GLASS_LIGHT, **kwargs)


def code_text(text: str, **kwargs) -> Text:
    """Monospace-style text for code snippets."""
    return Text(text, font_size=18, color=GLASS_TEAL, font="Menlo", **kwargs)


# ── Shape Helpers ───────────────────────────────────────────────────

def neuron_circle(
    radius: float = 0.2,
    color: str = GLASS_ACCENT,
    fill_opacity: float = 0.8,
) -> Circle:
    """Create a single neuron circle."""
    return Circle(radius=radius, color=color, fill_opacity=fill_opacity, stroke_width=1.5)


def create_layer_block(
    width: float = 2.0,
    height: float = 0.5,
    color: str = GLASS_ACCENT,
    label: str = "",
    opacity: float = 0.7,
) -> VGroup:
    """Rounded rectangle representing a transformer layer."""
    rect = RoundedRectangle(
        width=width, height=height,
        corner_radius=0.1,
        color=color, fill_opacity=opacity,
        stroke_width=1.5,
    )
    grp = VGroup(rect)
    if label:
        txt = Text(label, font_size=16, color=GLASS_LIGHT)
        txt.move_to(rect.get_center())
        grp.add(txt)
    return grp


def create_layer_stack(
    n_layers: int = 6,
    layer_width: float = 3.0,
    layer_height: float = 0.45,
    spacing: float = 0.15,
    labels: list | None = None,
) -> VGroup:
    """
    Vertical stack of transformer layer blocks.
    Returns a VGroup; layer 0 is at the bottom.
    """
    stack = VGroup()
    for i in range(n_layers):
        lbl = labels[i] if labels else f"Layer {i}"
        block = create_layer_block(
            width=layer_width,
            height=layer_height,
            color=layer_color(i),
            label=lbl,
            opacity=0.6,
        )
        block.shift(UP * i * (layer_height + spacing))
        stack.add(block)
    stack.center()
    return stack


# ── Data Generators ─────────────────────────────────────────────────

def generate_attention_matrix(n_tokens: int = 6, seed: int = 42) -> np.ndarray:
    """Synthetic softmax-normalised attention matrix."""
    rng = np.random.default_rng(seed)
    raw = rng.exponential(1.0, (n_tokens, n_tokens))
    # make it causal (lower triangular)
    mask = np.tril(np.ones((n_tokens, n_tokens)))
    raw = raw * mask
    row_sums = raw.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return raw / row_sums


def generate_activation_clusters(
    n_points: int = 200,
    n_clusters: int = 2,
    dim: int = 3,
    separation: float = 3.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthetic activation-space clusters for probing visualisations.
    Returns (points, labels).
    """
    rng = np.random.default_rng(seed)
    points, labels = [], []
    for c in range(n_clusters):
        center = rng.uniform(-separation, separation, dim)
        cluster = rng.normal(center, 0.6, (n_points // n_clusters, dim))
        points.append(cluster)
        labels.append(np.full(n_points // n_clusters, c))
    return np.vstack(points), np.concatenate(labels)


def generate_sae_activations(
    n_features: int = 64,
    n_active: int = 5,
    n_samples: int = 10,
    seed: int = 42,
) -> np.ndarray:
    """
    Synthetic sparse autoencoder activations.
    Returns (n_samples, n_features) with only *n_active* non-zero per row.
    """
    rng = np.random.default_rng(seed)
    acts = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        active_idx = rng.choice(n_features, n_active, replace=False)
        acts[i, active_idx] = rng.exponential(2.0, n_active)
    return acts


def generate_circuit_graph(n_layers: int = 4, n_nodes_per_layer: int = 3, seed: int = 42):
    """
    Synthetic circuit graph data.
    Returns (nodes, edges) where nodes = list of (layer, idx, importance)
    and edges = list of (src_node_idx, dst_node_idx, weight).
    """
    rng = np.random.default_rng(seed)
    nodes = []
    for layer in range(n_layers):
        for idx in range(n_nodes_per_layer):
            importance = rng.uniform(0.1, 1.0)
            nodes.append((layer, idx, importance))

    edges = []
    for i, (l1, _, _) in enumerate(nodes):
        for j, (l2, _, _) in enumerate(nodes):
            if l2 == l1 + 1 and rng.random() > 0.4:
                weight = rng.uniform(0.1, 1.0)
                edges.append((i, j, weight))
    return nodes, edges
