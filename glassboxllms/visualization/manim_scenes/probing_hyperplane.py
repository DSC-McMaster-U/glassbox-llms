"""
ProbingHyperplaneScene — Visualize linear and nonlinear probing.

Shows how a linear probe finds a separating hyperplane in activation space,
and how a nonlinear probe handles more complex decision boundaries.

Render:
    manim -qh probing_hyperplane.py ProbingHyperplaneScene
"""

from manim import *
import numpy as np

from glassboxllms.visualization.manim_scenes.utils import (
    GLASS_BG, GLASS_PRIMARY, GLASS_GOLD,
    GLASS_TEAL, GLASS_PURPLE, GLASS_LIGHT, GLASS_DIM,
    GLASS_GREEN, GLASS_ORANGE,
    title_text, subtitle_text, label_text,
    generate_activation_clusters,
)


class ProbingHyperplaneScene(ThreeDScene):
    """3D visualisation of probing in activation space."""

    def construct(self):
        self.camera.background_color = GLASS_BG

        # ── Title ───────────────────────────────────────────────
        title = title_text("Probing Activation Space")
        sub = subtitle_text("Finding Concept Directions with Linear Probes")
        sub.next_to(title, DOWN, buff=0.3)

        self.add_fixed_in_frame_mobjects(title, sub)
        self.play(Write(title), run_time=0.8)
        self.play(FadeIn(sub, shift=UP * 0.2))
        self.wait(1.0)
        self.play(FadeOut(title), FadeOut(sub))
        self.remove(title, sub)

        # ── 3D Axes ─────────────────────────────────────────────
        axes = ThreeDAxes(
            x_range=[-4, 4, 2], y_range=[-4, 4, 2], z_range=[-4, 4, 2],
            x_length=6, y_length=6, z_length=5,
            axis_config={"color": GLASS_DIM, "stroke_width": 1},
        )

        axis_labels = VGroup(
            Text("dim 1", font_size=14, color=GLASS_DIM),
            Text("dim 2", font_size=14, color=GLASS_DIM),
            Text("dim 3", font_size=14, color=GLASS_DIM),
        )

        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)
        self.play(Create(axes), run_time=0.8)

        # ── Generate clusters ───────────────────────────────────
        points, labels = generate_activation_clusters(
            n_points=160, n_clusters=2, dim=3, separation=2.0, seed=42,
        )

        pos_dots = VGroup()
        neg_dots = VGroup()
        for pt, lbl in zip(points, labels):
            color = GLASS_TEAL if lbl == 0 else GLASS_PRIMARY
            dot = Dot3D(
                point=axes.c2p(*pt),
                radius=0.05,
                color=color,
            )
            if lbl == 0:
                pos_dots.add(dot)
            else:
                neg_dots.add(dot)

        # Labels
        legend_title = subtitle_text("Sentiment Probe").to_edge(UP, buff=0.5)
        legend_pos = VGroup(
            Dot(radius=0.06, color=GLASS_TEAL),
            Text("positive", font_size=14, color=GLASS_LIGHT),
        ).arrange(RIGHT, buff=0.1)
        legend_neg = VGroup(
            Dot(radius=0.06, color=GLASS_PRIMARY),
            Text("negative", font_size=14, color=GLASS_LIGHT),
        ).arrange(RIGHT, buff=0.1)
        legend = VGroup(legend_pos, legend_neg).arrange(DOWN, buff=0.1).to_corner(UR, buff=0.5)

        self.add_fixed_in_frame_mobjects(legend_title, legend)
        self.play(Write(legend_title), FadeIn(legend))

        self.play(
            LaggedStart(*[FadeIn(d, scale=0.3) for d in pos_dots], lag_ratio=0.01),
            LaggedStart(*[FadeIn(d, scale=0.3) for d in neg_dots], lag_ratio=0.01),
            run_time=1.5,
        )

        # Rotate camera slowly
        self.begin_ambient_camera_rotation(rate=0.08)
        self.wait(2.0)

        # ── Separating Plane ────────────────────────────────────
        # Find rough separating plane normal (direction between cluster centers)
        center_0 = points[labels == 0].mean(axis=0)
        center_1 = points[labels == 1].mean(axis=0)
        normal = center_1 - center_0
        normal = normal / np.linalg.norm(normal)
        midpoint = (center_0 + center_1) / 2

        # Create a surface for the plane
        plane = Surface(
            lambda u, v: axes.c2p(
                midpoint[0] + u * 3 + v * 0,
                midpoint[1] + u * normal[2] - v * normal[0],
                midpoint[2] - u * normal[1] + v * normal[2],
            ),
            u_range=[-1, 1], v_range=[-1, 1],
            resolution=(4, 4),
            fill_opacity=0.25,
            fill_color=GLASS_GOLD,
            stroke_width=0.5,
            stroke_color=GLASS_GOLD,
        )

        plane_label = label_text("Separating Hyperplane")
        plane_label.to_edge(DOWN, buff=0.8)
        self.add_fixed_in_frame_mobjects(plane_label)

        self.play(Create(plane), Write(plane_label), run_time=1.5)

        # Accuracy indicator
        acc = label_text("Accuracy: 92.5%")
        acc.set_color(GLASS_GREEN)
        acc.next_to(plane_label, DOWN, buff=0.2)
        self.add_fixed_in_frame_mobjects(acc)
        self.play(Write(acc))
        self.wait(2.0)

        # ── Direction Arrow ─────────────────────────────────────
        arrow_start = axes.c2p(*midpoint)
        arrow_end = axes.c2p(*(midpoint + normal * 2.5))
        direction_arrow = Arrow3D(
            start=arrow_start, end=arrow_end,
            color=GLASS_GOLD,
        )
        dir_label = label_text("concept\ndirection")
        dir_label.to_edge(LEFT, buff=1.0).shift(DOWN * 1.5)
        self.add_fixed_in_frame_mobjects(dir_label)

        self.play(Create(direction_arrow), Write(dir_label), run_time=1.0)
        self.wait(2.0)

        # ── Transition: show this direction feeds into steering ─
        transition_text = subtitle_text("This direction → DirectionalSteering")
        transition_text.to_edge(DOWN, buff=0.3)
        self.add_fixed_in_frame_mobjects(transition_text)
        self.play(Write(transition_text))
        self.wait(2.0)

        self.stop_ambient_camera_rotation()
        self.play(*[FadeOut(m) for m in self.mobjects])
