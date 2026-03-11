"""
SteeringVectorScene — Visualize directional steering of model behaviour.

Shows a direction vector being added to activations and the resulting
shift in output distribution.

Render:
    manim -qh steering_vector.py SteeringVectorScene
"""

from manim import *
import numpy as np

from glassboxllms.visualization.manim_scenes.utils import (
    GLASS_BG, GLASS_PRIMARY, GLASS_GOLD,
    GLASS_TEAL, GLASS_PURPLE, GLASS_LIGHT, GLASS_DIM,
    GLASS_GREEN, GLASS_ORANGE,
    title_text, subtitle_text, label_text, code_text,
    layer_color,
)


class SteeringVectorScene(Scene):
    """Animate steering: direction + strength → behaviour shift."""

    def construct(self):
        self.camera.background_color = GLASS_BG

        # ── Title ───────────────────────────────────────────────
        title = title_text("Directional Steering")
        sub = subtitle_text("Shifting Model Behaviour with a Vector")
        sub.next_to(title, DOWN, buff=0.3)
        self.play(Write(title), FadeIn(sub, shift=UP * 0.2))
        self.wait(1.0)
        self.play(FadeOut(title), FadeOut(sub))

        # ── Part 1: Activation Space ────────────────────────────
        part_label = subtitle_text("Activations in Representation Space").to_edge(UP, buff=0.5)
        self.play(Write(part_label))

        axes = Axes(
            x_range=[-5, 5, 1], y_range=[-5, 5, 1],
            x_length=7, y_length=5.5,
            axis_config={"color": GLASS_DIM, "stroke_width": 1,
                         "include_ticks": False},
        ).shift(DOWN * 0.3)

        ax_label_x = Text("dim 1", font_size=12, color=GLASS_DIM)
        ax_label_x.next_to(axes.x_axis, DOWN, buff=0.1)
        ax_label_y = Text("dim 2", font_size=12, color=GLASS_DIM)
        ax_label_y.next_to(axes.y_axis, LEFT, buff=0.1)

        self.play(Create(axes), FadeIn(ax_label_x), FadeIn(ax_label_y), run_time=0.5)

        # Cluster of activation points
        np.random.seed(123)
        pts = np.random.randn(50, 2) * 0.8 + np.array([-0.5, -0.3])
        dots = VGroup(*[
            Dot(axes.c2p(x, y), radius=0.05, color=GLASS_TEAL, fill_opacity=0.7)
            for x, y in pts
        ])

        self.play(
            LaggedStart(*[FadeIn(d, scale=0.5) for d in dots], lag_ratio=0.01),
            run_time=0.8,
        )
        self.wait(0.5)

        # ── Part 2: Direction Vector ────────────────────────────
        self.play(FadeOut(part_label))
        part_label = subtitle_text("Adding a Steering Direction").to_edge(UP, buff=0.5)
        self.play(Write(part_label))

        # Direction arrow
        direction = np.array([2.0, 1.5])
        direction_norm = direction / np.linalg.norm(direction)

        dir_arrow = Arrow(
            axes.c2p(0, 0), axes.c2p(*direction_norm * 3),
            color=GLASS_GOLD, stroke_width=3, buff=0,
            max_tip_length_to_length_ratio=0.12,
        )
        dir_label = Text("sentiment direction", font_size=16, color=GLASS_GOLD)
        dir_label.next_to(dir_arrow.get_end(), UR, buff=0.15)

        self.play(GrowArrow(dir_arrow), Write(dir_label), run_time=1.0)
        self.wait(0.8)

        # ── Part 3: Strength Slider ─────────────────────────────
        self.play(FadeOut(part_label))
        part_label = subtitle_text("Strength Controls the Shift").to_edge(UP, buff=0.5)
        self.play(Write(part_label))

        # Slider
        slider_line = Line(LEFT * 2.5, RIGHT * 2.5, color=GLASS_DIM, stroke_width=2)
        slider_line.to_edge(DOWN, buff=1.2)
        slider_labels = VGroup(
            Text("0", font_size=14, color=GLASS_DIM).next_to(slider_line.get_left(), DOWN, buff=0.15),
            Text("strength", font_size=14, color=GLASS_LIGHT).next_to(slider_line, DOWN, buff=0.35),
            Text("5", font_size=14, color=GLASS_DIM).next_to(slider_line.get_right(), DOWN, buff=0.15),
        )

        strength = ValueTracker(0)
        slider_dot = always_redraw(lambda: Dot(
            slider_line.point_from_proportion(strength.get_value() / 5),
            radius=0.1, color=GLASS_GOLD,
        ))
        strength_display = always_redraw(lambda: Text(
            f"strength = {strength.get_value():.1f}",
            font_size=18, color=GLASS_GOLD,
        ).next_to(slider_line, UP, buff=0.2))

        self.play(Create(slider_line), FadeIn(slider_labels), FadeIn(slider_dot), Write(strength_display))

        # Animate dots moving with strength
        original_pts = pts.copy()

        def update_dots(dots_group):
            s = strength.get_value()
            for dot, (ox, oy) in zip(dots_group, original_pts):
                nx = ox + direction_norm[0] * s
                ny = oy + direction_norm[1] * s
                dot.move_to(axes.c2p(nx, ny))

        dots.add_updater(update_dots)

        # Animate strength from 0 to 3
        self.play(strength.animate.set_value(3), run_time=3, rate_func=smooth)
        self.wait(0.5)

        # Back to 0
        self.play(strength.animate.set_value(0), run_time=1.5, rate_func=smooth)

        # Pump to 2 for the "ideal" demonstration
        self.play(strength.animate.set_value(2), run_time=2, rate_func=smooth)
        dots.remove_updater(update_dots)
        self.wait(1.0)

        self.play(
            FadeOut(axes), FadeOut(dots), FadeOut(dir_arrow), FadeOut(dir_label),
            FadeOut(slider_line), FadeOut(slider_labels), FadeOut(slider_dot),
            FadeOut(strength_display), FadeOut(ax_label_x), FadeOut(ax_label_y),
            FadeOut(part_label),
        )

        # ── Part 4: Before / After Output ───────────────────────
        part_label = subtitle_text("Effect on Model Output").to_edge(UP, buff=0.5)
        self.play(Write(part_label))

        # Token probability bars — before
        tokens = ["great", "okay", "bad", "the", "a"]
        probs_before = [0.08, 0.25, 0.15, 0.30, 0.22]
        probs_after  = [0.45, 0.18, 0.02, 0.20, 0.15]

        def make_bar_chart(probs, title_str, color, x_offset):
            grp = VGroup()
            title = Text(title_str, font_size=18, color=GLASS_LIGHT, weight=BOLD)
            bars = VGroup()
            for i, (tok, prob) in enumerate(zip(tokens, probs)):
                bar = Rectangle(
                    width=0.6, height=prob * 5,
                    color=color, fill_opacity=0.6, stroke_width=1,
                )
                bar.move_to(ORIGIN)
                bar.align_to(ORIGIN, DOWN)
                bar.shift(RIGHT * i * 0.9)
                tok_label = Text(tok, font_size=12, color=GLASS_LIGHT)
                tok_label.next_to(bar, DOWN, buff=0.1)
                prob_label = Text(f"{prob:.0%}", font_size=10, color=color)
                prob_label.next_to(bar, UP, buff=0.05)
                bars.add(VGroup(bar, tok_label, prob_label))
            bars.center()
            title.next_to(bars, UP, buff=0.3)
            grp.add(title, bars)
            grp.shift(RIGHT * x_offset)
            return grp

        before_chart = make_bar_chart(probs_before, "Before Steering", GLASS_DIM, -3.2)
        after_chart = make_bar_chart(probs_after, "After Steering", GLASS_TEAL, 3.2)
        before_chart.shift(DOWN * 0.5)
        after_chart.shift(DOWN * 0.5)

        arrow = Arrow(
            before_chart.get_right() + LEFT * 0.3,
            after_chart.get_left() + RIGHT * 0.3,
            color=GLASS_GOLD, stroke_width=2,
        )
        arrow_label = Text("+ sentiment\n  (strength=2)", font_size=12, color=GLASS_GOLD)
        arrow_label.next_to(arrow, UP, buff=0.1)

        self.play(FadeIn(before_chart, shift=LEFT * 0.3), run_time=0.8)
        self.play(GrowArrow(arrow), Write(arrow_label), run_time=0.6)
        self.play(FadeIn(after_chart, shift=RIGHT * 0.3), run_time=0.8)

        # Highlight the shift
        highlight = SurroundingRectangle(
            after_chart[1][0],  # "great" bar
            color=GLASS_GREEN, stroke_width=2, buff=0.05,
        )
        highlight_text = Text('"great" probability ↑ 5.6×', font_size=14, color=GLASS_GREEN)
        highlight_text.next_to(highlight, DOWN, buff=0.3)

        self.play(Create(highlight), Write(highlight_text))
        self.wait(2.0)

        # Code snippet
        self.play(
            FadeOut(before_chart), FadeOut(after_chart), FadeOut(arrow),
            FadeOut(arrow_label), FadeOut(highlight), FadeOut(highlight_text),
            FadeOut(part_label),
        )

        code_lines = [
            "steering = DirectionalSteering(",
            '    layer="transformer.h.8",',
            "    direction=probe.get_direction(),",
            "    strength=2.0,",
            ")",
            "",
            "with steering:",
            "    steering.register(model)",
            "    output = model(input_ids)",
        ]
        code_block = Code(
            code_string="\n".join(code_lines),
            language="python",
            background="rectangle",
            add_line_numbers=False,
            formatter_style="monokai",
        ).scale(0.75)

        api_label = subtitle_text("Clean Python API").to_edge(UP, buff=0.5)
        self.play(Write(api_label))
        self.play(FadeIn(code_block, shift=UP * 0.3), run_time=1.0)
        self.wait(3.0)
        self.play(*[FadeOut(m) for m in self.mobjects])
