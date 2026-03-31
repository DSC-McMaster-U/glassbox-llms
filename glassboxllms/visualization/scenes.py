"""
Manim visualization scenes for glassbox interpretability data.

Each scene accepts a data object (from adapters.py) and renders an
animated visualization. Scenes are designed to work with ``manim``
CLI or Jupyter magic (``%%manim``).

Usage from CLI::

    manim -ql scenes.py CircuitDiscoveryScene

Usage from Python::

    from glassboxllms.visualization.scenes import CircuitDiscoveryScene
    scene = CircuitDiscoveryScene()
    scene.scene_data = scene_data  # from adapters
    scene.render()
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

from manim import (
    BLUE,
    BLUE_A,
    BLUE_D,
    BLUE_E,
    BOLD,
    BLACK,
    DOWN,
    DEGREES,
    GREEN,
    GRAY,
    GRAY_A,
    LEFT,
    ORANGE,
    ORIGIN,
    RED,
    RED_A,
    RED_D,
    RIGHT,
    TAU,
    TEAL,
    UP,
    WHITE,
    YELLOW,
    Arrow,
    Circle,
    Create,
    CurvedArrow,
    DashedLine,
    Dot,
    FadeIn,
    FadeOut,
    GrowArrow,
    Line,
    MathTex,
    Rectangle,
    ReplacementTransform,
    RoundedRectangle,
    Scene,
    Square,
    Text,
    VGroup,
    VMobject,
    Write,
    interpolate_color,
)

from .adapters import (
    CircuitSceneData,
    PipelineSceneData,
    ProbeSceneData,
    SAESceneData,
    SteeringSceneData,
)


# ======================================================================
# GLASS color palette (consistent with manim_scenes/utils.py)
# ======================================================================

GLASS_BG = "#1a1a2e"
GLASS_PRIMARY = "#e94560"
GLASS_ACCENT = "#0f3460"
GLASS_GOLD = "#f5a623"
GLASS_TEAL = "#16c79a"
GLASS_PURPLE = "#9b59b6"
GLASS_LIGHT = "#eaf2f8"
GLASS_DIM = "#4a4a6a"
GLASS_GREEN = "#2ecc71"
GLASS_ORANGE = "#e67e22"

_NODE_COLORS = {
    "neuron": GLASS_ACCENT,
    "attention_head": GLASS_ORANGE,
    "feature": GLASS_GREEN,
    "mlp_layer": GLASS_TEAL,
    "residual_stream": GLASS_DIM,
    "embedding": GLASS_GOLD,
    "unembedding": GLASS_PRIMARY,
}

_CLASS_COLORS = [GLASS_ACCENT, GLASS_PRIMARY, GLASS_GREEN, GLASS_ORANGE, GLASS_TEAL, GLASS_GOLD]

# Watermark text shown in every demo scene
_WATERMARK_TEXT = "glassbox-llms"


def _color_for_node_type(node_type: str):
    return _NODE_COLORS.get(node_type, GLASS_DIM)


def _add_watermark(scene: Scene) -> Text:
    """Add a subtle branding watermark to the bottom-right corner."""
    wm = Text(_WATERMARK_TEXT, font_size=14, color=GLASS_DIM, opacity=0.5)
    wm.to_corner(DOWN + RIGHT, buff=0.25)
    scene.add(wm)
    return wm


def _set_cinematic_bg(scene: Scene):
    """Set the dark navy cinematic background."""
    scene.camera.background_color = GLASS_BG


def _title_card(scene: Scene, heading: str, subheading: str = ""):
    """Show a brief title card at the start of a scene, then fade out."""
    title = Text(heading, font_size=42, color=GLASS_LIGHT, weight=BOLD)
    title.move_to(ORIGIN)
    elements = [title]
    if subheading:
        sub = Text(subheading, font_size=24, color=GLASS_GOLD)
        sub.next_to(title, DOWN, buff=0.3)
        elements.append(sub)
    group = VGroup(*elements)
    scene.play(FadeIn(group), run_time=0.8)
    scene.wait(1.0)
    scene.play(FadeOut(group), run_time=0.6)
    return group


# ======================================================================
# 1. CircuitDiscoveryScene
# ======================================================================


class CircuitDiscoveryScene(Scene):
    """
    Visualize a CircuitGraph as a layered directed graph.

    Nodes are grouped by layer and colored by type.  Edges are drawn as
    curved arrows with thickness proportional to weight.

    Set ``self.scene_data`` to a :class:`CircuitSceneData` before
    calling ``construct()``, or override ``get_scene_data()`` to provide
    data from a file/object.
    """

    scene_data: Optional[CircuitSceneData] = None

    def get_scene_data(self) -> CircuitSceneData:
        if self.scene_data is not None:
            return self.scene_data
        raise ValueError(
            "No scene_data set. Assign a CircuitSceneData to scene.scene_data "
            "or override get_scene_data()."
        )

    def construct(self):
        _set_cinematic_bg(self)
        data = self.get_scene_data()

        # --- Title card ---
        circuit_label = data.circuit_name or "Circuit Graph"
        _title_card(self, "Circuit Discovery", f"{circuit_label} -- {data.model_name}")
        _add_watermark(self)

        # --- Handle empty data ---
        if not data.nodes:
            empty = Text("No circuit nodes to display.", font_size=24, color=GLASS_LIGHT)
            empty.move_to(ORIGIN)
            self.play(Write(empty), run_time=0.8)
            self.wait(1.5)
            return

        # --- Persistent header ---
        title = Text(
            f"{circuit_label}  ({data.model_name})",
            font_size=30,
            color=GLASS_LIGHT,
        )
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=0.8)

        # --- Layout: group nodes by layer ---
        layers = data.layers if data.layers else [0]
        layer_to_nodes: Dict[int, list] = defaultdict(list)
        no_layer_nodes = []
        for nd in data.nodes:
            if nd["layer"] is not None:
                layer_to_nodes[nd["layer"]].append(nd)
            else:
                no_layer_nodes.append(nd)

        # Assign no-layer nodes to a virtual layer at the end
        if no_layer_nodes:
            virtual_layer = (max(layers) + 1) if layers else 0
            layers = list(layers) + [virtual_layer]
            layer_to_nodes[virtual_layer] = no_layer_nodes

        n_layers = len(layers)
        usable_width = 11.0
        usable_height = 5.5
        layer_spacing = usable_width / max(n_layers, 1)

        # Compute positions
        node_positions: Dict[str, np.ndarray] = {}
        node_mobjects: Dict[str, VGroup] = {}

        for col_idx, layer_val in enumerate(layers):
            col_nodes = layer_to_nodes[layer_val]
            n_in_col = len(col_nodes)
            if n_in_col == 0:
                continue
            x = -usable_width / 2 + (col_idx + 0.5) * layer_spacing
            row_spacing = usable_height / max(n_in_col, 1)

            for row_idx, nd in enumerate(col_nodes):
                y = usable_height / 2 - (row_idx + 0.5) * row_spacing
                pos = np.array([x, y - 0.3, 0])
                node_positions[nd["id"]] = pos

                # Draw node
                color = _color_for_node_type(nd["type"])
                circle = Circle(radius=0.22, color=color, fill_opacity=0.7)
                circle.set_stroke(color=color, width=2)
                circle.move_to(pos)

                label_text = nd.get("label", nd["id"])
                # Truncate long labels
                if len(label_text) > 12:
                    label_text = label_text[:10] + ".."
                label = Text(label_text, font_size=14, color=GLASS_LIGHT)
                label.next_to(circle, DOWN, buff=0.08)

                group = VGroup(circle, label)
                node_mobjects[nd["id"]] = group

        # Layer labels
        layer_labels = VGroup()
        for col_idx, layer_val in enumerate(layers):
            if not layer_to_nodes.get(layer_val):
                continue
            x = -usable_width / 2 + (col_idx + 0.5) * layer_spacing
            lbl = Text(f"L{layer_val}", font_size=20, weight=BOLD, color=GLASS_GOLD)
            lbl.move_to(np.array([x, usable_height / 2 + 0.2, 0]))
            layer_labels.add(lbl)

        self.play(FadeIn(layer_labels), run_time=0.7)

        # Animate nodes
        if node_mobjects:
            all_node_groups = VGroup(*node_mobjects.values())
            self.play(FadeIn(all_node_groups), run_time=1.2)

        # --- Edges ---
        if not data.edges:
            self.wait(2)
            return

        edge_mobs = []
        max_weight = max(
            (abs(e["weight"]) for e in data.edges if e["weight"] is not None),
            default=1.0,
        )
        if max_weight == 0:
            max_weight = 1.0

        # Sort edges weakest-first for visual layering
        sorted_edges = sorted(
            data.edges,
            key=lambda e: abs(e["weight"]) if e["weight"] is not None else 0.0,
        )

        for edge in sorted_edges:
            src_pos = node_positions.get(edge["source"])
            tgt_pos = node_positions.get(edge["target"])
            if src_pos is None or tgt_pos is None:
                continue

            weight = edge["weight"] if edge["weight"] is not None else 0.5
            alpha = min(abs(weight) / max_weight, 1.0)

            arrow = CurvedArrow(
                src_pos,
                tgt_pos,
                angle=0.3,
                stroke_width=1.5 + 4 * alpha,
                color=interpolate_color(GLASS_DIM, GLASS_TEAL, alpha),
                tip_length=0.15,
            )
            edge_mobs.append(arrow)

        if edge_mobs:
            self.play(*[Create(a) for a in edge_mobs], run_time=1.8)

        # Summary text
        summary = data.metadata
        if summary:
            info = Text(
                f"Nodes: {summary.get('num_nodes', '?')}  "
                f"Edges: {summary.get('num_edges', '?')}",
                font_size=20,
                color=GLASS_LIGHT,
            )
            info.to_edge(DOWN, buff=0.3)
            self.play(FadeIn(info), run_time=0.6)

        self.wait(2.5)


# ======================================================================
# 2. ProbingHyperplaneScene
# ======================================================================


class ProbingHyperplaneScene(Scene):
    """
    Visualize a linear probe's decision boundary in projected activation space.

    Points are colored by class label.  The learned hyperplane is drawn as
    a dashed line through the PCA-projected space.

    Set ``self.scene_data`` to a :class:`ProbeSceneData`.
    """

    scene_data: Optional[ProbeSceneData] = None

    def get_scene_data(self) -> ProbeSceneData:
        if self.scene_data is not None:
            return self.scene_data
        raise ValueError("No scene_data set.")

    def construct(self):
        _set_cinematic_bg(self)
        data = self.get_scene_data()

        # --- Title card ---
        direction = data.direction_name or "concept"
        _title_card(
            self,
            "Linear Probing",
            f"Sentiment Decision Boundary -- {direction}",
        )
        _add_watermark(self)

        # --- Persistent header ---
        title = Text(
            f"Probing: '{direction}' at {data.layer}",
            font_size=30,
            color=GLASS_LIGHT,
        )
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.8)

        # Accuracy subtitle
        acc_text = f"Accuracy: {data.accuracy:.1%}"
        if data.f1 is not None:
            acc_text += f"   F1: {data.f1:.3f}"
        subtitle = Text(acc_text, font_size=22, color=GLASS_GOLD)
        subtitle.next_to(title, DOWN, buff=0.25)
        self.play(Write(subtitle), run_time=0.6)

        if data.points_2d is None or len(data.points_2d) == 0:
            # No scatter data — show coefficients bar chart instead
            self._show_coefficient_bars(data)
            return

        points = data.points_2d
        labels = data.labels

        # Normalize points for Manim coordinate space
        xs, ys = points[:, 0], points[:, 1]
        max_range = max(xs.max() - xs.min(), ys.max() - ys.min(), 1e-6)
        norm_x = (xs - xs.mean()) / max_range * 8
        norm_y = (ys - ys.mean()) / max_range * 4

        from manim import Axes

        axes = Axes(
            x_range=[-5, 5, 1],
            y_range=[-3, 3, 1],
            x_length=10,
            y_length=6,
            tips=False,
            axis_config={"color": GLASS_DIM},
        )
        axes.move_to(ORIGIN)
        self.play(Create(axes), run_time=0.7)

        # Draw points colored by label
        dots = VGroup()
        unique_labels = sorted(set(labels)) if labels is not None else [0]

        # Edge case: single-class data still renders fine
        for i in range(len(norm_x)):
            lbl = int(labels[i]) if labels is not None else 0
            color_idx = unique_labels.index(lbl) % len(_CLASS_COLORS)
            dot = Dot(
                axes.c2p(norm_x[i], norm_y[i]),
                radius=0.04,
                color=_CLASS_COLORS[color_idx],
                fill_opacity=0.75,
            )
            dots.add(dot)

        self.play(FadeIn(dots), run_time=1.0)

        # Draw decision boundary as dashed line in PCA-projected space.
        coef = data.coefficients
        if coef is not None and coef.ndim >= 1 and len(coef) >= 2:
            normal_2d_raw = data.metadata.get("normal_2d")
            if normal_2d_raw is None:
                normal_2d_raw = coef[:2]
            normal_2d_raw = np.asarray(normal_2d_raw, dtype=float)
            norm_val = np.linalg.norm(normal_2d_raw)
            # Guard against zero-weight coefficients
            if norm_val > 1e-8:
                normal_2d = normal_2d_raw / norm_val
                # The boundary is perpendicular to the normal
                perp = np.array([-normal_2d[1], normal_2d[0]])
                start = axes.c2p(*(perp * -4.5))
                end = axes.c2p(*(perp * 4.5))
                boundary = DashedLine(
                    start, end, color=GLASS_LIGHT, stroke_width=2.5,
                )
                self.play(Create(boundary), run_time=0.8)

        # Legend
        if data.class_names:
            legend_items = VGroup()
            for i, name in enumerate(data.class_names):
                color = _CLASS_COLORS[i % len(_CLASS_COLORS)]
                dot = Dot(radius=0.06, color=color)
                lbl = Text(name, font_size=16, color=GLASS_LIGHT)
                lbl.next_to(dot, RIGHT, buff=0.1)
                row = VGroup(dot, lbl)
                legend_items.add(row)
            legend_items.arrange(DOWN, buff=0.15, aligned_edge=LEFT)
            legend_items.to_corner(DOWN + RIGHT, buff=0.5)
            self.play(FadeIn(legend_items), run_time=0.5)

        self.wait(2.5)

    def _show_coefficient_bars(self, data: ProbeSceneData):
        """Fallback: show top coefficient magnitudes as bars."""
        coef = data.coefficients
        if coef is None or len(coef) == 0:
            note = Text(
                "No coefficients available.",
                font_size=24,
                color=GLASS_LIGHT,
            )
            note.move_to(ORIGIN)
            self.play(Write(note), run_time=0.8)
            self.wait(1.5)
            return

        if coef.ndim > 1:
            coef = coef[0]

        abs_coef = np.abs(coef)
        top_k = min(15, len(abs_coef))
        top_idx = np.argsort(abs_coef)[-top_k:][::-1]
        top_vals = coef[top_idx]

        max_val = max(abs(top_vals.max()), abs(top_vals.min()), 1e-8)
        bar_width = 8.0
        bar_height = 0.3
        start_y = 2.0

        bars = VGroup()
        for rank, (idx, val) in enumerate(zip(top_idx, top_vals)):
            y = start_y - rank * (bar_height + 0.1)
            width = (val / max_val) * (bar_width / 2)
            color = GLASS_TEAL if val >= 0 else GLASS_PRIMARY
            bar = Rectangle(
                width=abs(width),
                height=bar_height,
                fill_color=color,
                fill_opacity=0.7,
                stroke_width=0.5,
            )
            if val >= 0:
                bar.move_to(np.array([abs(width) / 2, y, 0]))
            else:
                bar.move_to(np.array([-abs(width) / 2, y, 0]))

            label = Text(f"dim {idx}", font_size=14, color=GLASS_LIGHT)
            label.move_to(np.array([-bar_width / 2 - 0.8, y, 0]))
            bars.add(VGroup(bar, label))

        self.play(FadeIn(bars), run_time=1.2)
        self.wait(2.5)


# ======================================================================
# 3. SAEFeatureDiscoveryScene
# ======================================================================


class SAEFeatureDiscoveryScene(Scene):
    """
    Visualize SAE features: top features with their activation statistics
    and 2D-projected decoder directions.

    Set ``self.scene_data`` to a :class:`SAESceneData`.
    """

    scene_data: Optional[SAESceneData] = None

    def get_scene_data(self) -> SAESceneData:
        if self.scene_data is not None:
            return self.scene_data
        raise ValueError("No scene_data set.")

    def construct(self):
        _set_cinematic_bg(self)
        data = self.get_scene_data()
        _add_watermark(self)

        # --- Title ---
        title = Text(
            f"SAE Features -- {data.model_name} Layer {data.layer}",
            font_size=30,
            color=GLASS_LIGHT,
        )
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.8)

        features = data.features
        if not features:
            empty = Text("No features to display.", font_size=24, color=GLASS_LIGHT)
            empty.move_to(ORIGIN)
            self.play(Write(empty), run_time=0.8)
            self.wait(1.5)
            return

        # Split screen: left = bar chart of activations, right = 2D decoder scatter
        # --- Left: top features bar chart ---
        left_anchor = np.array([-3.5, 0, 0])
        bar_header = Text(
            "Top Features (max activation)",
            font_size=20, weight=BOLD, color=GLASS_GOLD,
        )
        bar_header.move_to(left_anchor + np.array([0, 2.5, 0]))
        self.play(Write(bar_header), run_time=0.5)

        max_act_vals = [
            (f.get("max_activation") or 0.0) for f in features
        ]
        max_act = max(max_act_vals) if max_act_vals else 1.0
        if max_act == 0:
            max_act = 1.0

        bars = VGroup()
        bar_width_max = 3.0
        bar_h = 0.3
        for i, feat in enumerate(features):
            y = 2.0 - i * (bar_h + 0.12)
            val = feat.get("max_activation") or 0.0
            alpha = val / max_act
            width = alpha * bar_width_max

            bar = Rectangle(
                width=max(width, 0.05),
                height=bar_h,
                fill_color=interpolate_color(GLASS_ACCENT, GLASS_TEAL, alpha),
                fill_opacity=0.8,
                stroke_width=0.5,
            )
            bar.move_to(left_anchor + np.array([width / 2 - bar_width_max / 2, y, 0]))

            label = Text(f"F{feat['id']}", font_size=14, color=GLASS_LIGHT)
            label.next_to(bar, LEFT, buff=0.1)

            val_label = Text(f"{val:.2f}", font_size=12, color=GLASS_LIGHT)
            val_label.next_to(bar, RIGHT, buff=0.1)

            bars.add(VGroup(bar, label, val_label))

        self.play(FadeIn(bars), run_time=1.2)

        # --- Right: 2D decoder direction scatter ---
        has_2d = any("decoder_2d" in f for f in features)
        if has_2d:
            right_anchor = np.array([3.0, 0, 0])
            scatter_header = Text(
                "Decoder Directions (PCA)",
                font_size=20, weight=BOLD, color=GLASS_GOLD,
            )
            scatter_header.move_to(right_anchor + np.array([0, 2.5, 0]))
            self.play(Write(scatter_header), run_time=0.5)

            coords = np.array([f["decoder_2d"] for f in features if "decoder_2d" in f])
            ids = [f["id"] for f in features if "decoder_2d" in f]

            # Normalize
            if coords.shape[0] > 0:
                c_range = max(
                    coords[:, 0].max() - coords[:, 0].min(),
                    coords[:, 1].max() - coords[:, 1].min(),
                    1e-6,
                )
                norm_coords = (coords - coords.mean(axis=0)) / c_range * 3.5

                dots = VGroup()
                for j, (x, y) in enumerate(norm_coords):
                    pos = right_anchor + np.array([x, y, 0])
                    dot = Dot(pos, radius=0.1, color=GLASS_PURPLE)
                    lbl = Text(f"F{ids[j]}", font_size=12, color=GLASS_LIGHT)
                    lbl.next_to(dot, UP, buff=0.05)
                    dots.add(VGroup(dot, lbl))

                self.play(FadeIn(dots), run_time=1.0)

        # --- Optional: activation grid heatmap ---
        if data.activation_grid is not None:
            self._animate_activation_grid(data)

        self.wait(2.5)

    def _animate_activation_grid(self, data: SAESceneData):
        """Show a small heatmap of feature activations across samples."""
        grid = data.activation_grid
        n_feat, n_samp = grid.shape
        # Limit display
        n_feat = min(n_feat, 10)
        n_samp = min(n_samp, 20)
        grid = grid[:n_feat, :n_samp]

        max_val = grid.max() if grid.max() > 0 else 1.0
        cell_size = 0.2

        grid_group = VGroup()
        for i in range(n_feat):
            for j in range(n_samp):
                v = float(grid[i, j]) / max_val
                sq = Square(side_length=cell_size)
                sq.set_fill(
                    interpolate_color(GLASS_ACCENT, GLASS_TEAL, min(v, 1.0)),
                    opacity=0.9,
                )
                sq.set_stroke(width=0)
                sq.move_to(np.array([
                    j * (cell_size + 0.02) - n_samp * cell_size / 2,
                    -3.2 + i * (cell_size + 0.02),
                    0,
                ]))
                grid_group.add(sq)

        grid_label = Text(
            "Feature x Sample activations",
            font_size=16, color=GLASS_LIGHT,
        )
        grid_label.next_to(grid_group, UP, buff=0.15)

        self.play(FadeIn(grid_group), FadeIn(grid_label), run_time=1.0)


# ======================================================================
# 4. SteeringVectorScene
# ======================================================================


class SteeringVectorScene(Scene):
    """
    Visualize the effect of a steering vector on model representations.

    Shows before/after scatter plots with an arrow indicating the
    steering direction.

    Set ``self.scene_data`` to a :class:`SteeringSceneData`.
    """

    scene_data: Optional[SteeringSceneData] = None

    def get_scene_data(self) -> SteeringSceneData:
        if self.scene_data is not None:
            return self.scene_data
        raise ValueError("No scene_data set.")

    def construct(self):
        _set_cinematic_bg(self)
        data = self.get_scene_data()

        # --- Title card ---
        direction = data.direction_name or "direction"
        _title_card(
            self,
            "Steering Vectors",
            f"Shifting representations along '{direction}'",
        )
        _add_watermark(self)

        # --- Persistent header ---
        title = Text(
            f"Steering: '{direction}' at {data.layer}  "
            f"(strength={data.strength:.2f})",
            font_size=28,
            color=GLASS_LIGHT,
        )
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.8)

        from manim import Axes

        axes = Axes(
            x_range=[-5, 5, 1],
            y_range=[-3, 3, 1],
            x_length=10,
            y_length=6,
            tips=False,
            axis_config={"color": GLASS_DIM},
        )
        axes.move_to(ORIGIN)
        self.play(Create(axes), run_time=0.7)

        before = data.points_before_2d
        after = data.points_after_2d

        # --- Handle edge cases ---
        if before is None or after is None or len(before) == 0 or len(after) == 0:
            empty = Text(
                "Insufficient data to visualize steering.",
                font_size=24,
                color=GLASS_LIGHT,
            )
            empty.move_to(ORIGIN)
            self.play(Write(empty), run_time=0.8)
            self.wait(1.5)
            return

        # Normalize jointly
        all_pts = np.vstack([before, after])
        xs, ys = all_pts[:, 0], all_pts[:, 1]
        max_range = max(xs.max() - xs.min(), ys.max() - ys.min(), 1e-6)
        center_x, center_y = xs.mean(), ys.mean()

        def norm_pt(x, y):
            return (x - center_x) / max_range * 8, (y - center_y) / max_range * 4

        # Draw "before" points
        before_color = GLASS_TEAL
        after_color = GLASS_PRIMARY
        before_dots = VGroup()
        for x, y in before:
            nx, ny = norm_pt(x, y)
            dot = Dot(
                axes.c2p(nx, ny), radius=0.04,
                color=before_color, fill_opacity=0.6,
            )
            before_dots.add(dot)

        before_label = Text("Before", font_size=20, color=before_color)
        before_label.to_corner(DOWN + LEFT, buff=0.5)

        self.play(FadeIn(before_dots), FadeIn(before_label), run_time=1.0)

        # Animate transition to "after"
        after_dots = VGroup()
        for x, y in after:
            nx, ny = norm_pt(x, y)
            dot = Dot(
                axes.c2p(nx, ny), radius=0.04,
                color=after_color, fill_opacity=0.6,
            )
            after_dots.add(dot)

        after_label = Text("After", font_size=20, color=after_color)
        after_label.next_to(before_label, RIGHT, buff=1.0)

        # Animate dots moving from before to after positions
        move_anims = []
        n = min(len(before_dots), len(after_dots))
        for i in range(n):
            move_anims.append(
                before_dots[i]
                .animate.move_to(after_dots[i].get_center())
                .set_color(after_color)
            )

        if move_anims:
            self.play(*move_anims, FadeIn(after_label), run_time=2.0)
        else:
            self.play(FadeIn(after_label), run_time=0.5)

        # Draw steering direction arrow
        vec = data.steering_vector_2d
        vec_mag = np.linalg.norm(vec)
        if vec_mag > 1e-8:
            vec_norm = vec / vec_mag
            arrow_start = axes.c2p(0, 0)
            nx, ny = vec_norm[0] * 2.5, vec_norm[1] * 2.5
            arrow_end = axes.c2p(nx, ny)
            steering_arrow = Arrow(
                arrow_start, arrow_end,
                color=GLASS_GOLD,
                stroke_width=5,
                max_tip_length_to_length_ratio=0.2,
            )
            arrow_label = Text("steering", font_size=18, color=GLASS_GOLD)
            arrow_label.next_to(steering_arrow, UP, buff=0.1)

            self.play(GrowArrow(steering_arrow), FadeIn(arrow_label), run_time=1.0)

        self.wait(2.5)


# ======================================================================
# 5. FullPipelineScene
# ======================================================================


class FullPipelineScene(Scene):
    """
    Overview visualization of the full interpretability pipeline.

    Shows sequential stages (SAE, probing, circuits, steering) as
    connected cards with summary stats.

    Set ``self.scene_data`` to a :class:`PipelineSceneData`.
    """

    scene_data: Optional[PipelineSceneData] = None

    def get_scene_data(self) -> PipelineSceneData:
        if self.scene_data is not None:
            return self.scene_data
        raise ValueError("No scene_data set.")

    def construct(self):
        _set_cinematic_bg(self)
        data = self.get_scene_data()
        _add_watermark(self)

        # --- Title ---
        title = Text(
            f"Interpretability Pipeline -- {data.model_name}",
            font_size=32,
            color=GLASS_LIGHT,
        )
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.8)

        stages = data.stages
        if not stages:
            empty = Text(
                "No pipeline stages configured.",
                font_size=24, color=GLASS_LIGHT,
            )
            empty.move_to(ORIGIN)
            self.play(Write(empty), run_time=0.8)
            self.wait(1.5)
            return

        # Layout: cards in a horizontal row
        n = len(stages)
        card_width = min(2.8, 10.0 / max(n, 1))
        card_height = 2.5
        total_width = n * card_width + (n - 1) * 0.5
        start_x = -total_width / 2 + card_width / 2

        _stage_colors = {
            "sae": GLASS_ACCENT,
            "probe": GLASS_GREEN,
            "circuit": GLASS_ORANGE,
            "steering": GLASS_PRIMARY,
        }

        cards = []
        for i, stage in enumerate(stages):
            x = start_x + i * (card_width + 0.5)
            color = _stage_colors.get(stage.get("type", ""), GLASS_DIM)

            card = RoundedRectangle(
                width=card_width,
                height=card_height,
                corner_radius=0.2,
                fill_color=color,
                fill_opacity=0.15,
                stroke_color=color,
                stroke_width=2,
            )
            card.move_to(np.array([x, -0.3, 0]))

            stage_title = Text(
                stage["name"], font_size=18, weight=BOLD, color=GLASS_LIGHT,
            )
            stage_title.move_to(card.get_top() + DOWN * 0.4)

            summary_text = Text(
                stage.get("summary", ""), font_size=14, color=GLASS_LIGHT,
            )
            summary_text.move_to(card.get_center())

            group = VGroup(card, stage_title, summary_text)
            cards.append(group)

            self.play(FadeIn(group), run_time=0.7)

            # Draw connecting arrow to next stage
            if i < n - 1:
                next_x = start_x + (i + 1) * (card_width + 0.5)
                arrow = Arrow(
                    np.array([x + card_width / 2 + 0.05, -0.3, 0]),
                    np.array([next_x - card_width / 2 - 0.05, -0.3, 0]),
                    color=GLASS_GOLD,
                    stroke_width=2,
                    max_tip_length_to_length_ratio=0.3,
                )
                self.play(GrowArrow(arrow), run_time=0.4)

        self.wait(2.5)
