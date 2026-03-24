"""
FullPipelineScene — The "money shot" for the showcase.

A continuous animation showing the entire glassbox-llms pipeline:
    Input text → Model → Hook activations → SAE features →
    Probe direction → Circuit analysis → Steering → New output

Render:
    manim -qh full_pipeline.py FullPipelineScene
"""

from manim import *
import numpy as np

from glassboxllms.visualization.manim_scenes.utils import (
    GLASS_BG, GLASS_PRIMARY, GLASS_ACCENT, GLASS_GOLD,
    GLASS_TEAL, GLASS_PURPLE, GLASS_LIGHT, GLASS_DIM,
    GLASS_GREEN, GLASS_ORANGE,
    layer_color, title_text, subtitle_text, label_text, code_text,
    neuron_circle, create_layer_block, create_layer_stack,
    generate_sae_activations,
)


class FullPipelineScene(Scene):
    """End-to-end pipeline animation for the glassbox-llms showcase."""

    def construct(self):
        self.camera.background_color = GLASS_BG

        # ── Title ───────────────────────────────────────────────
        title = title_text("glassbox-llms")
        tagline = subtitle_text("Turning Black-Box LLMs into Glass Boxes")
        tagline.next_to(title, DOWN, buff=0.3)
        self.play(Write(title), run_time=1.0)
        self.play(FadeIn(tagline, shift=UP * 0.2), run_time=0.8)
        self.wait(1.5)
        self.play(FadeOut(title), FadeOut(tagline))

        # ── Stage 1: Input Text ─────────────────────────────────
        stage_label = subtitle_text("1. Input Text").to_edge(UP, buff=0.4)
        input_text = Text(
            '"The cat sat on the mat"',
            font_size=30, color=GLASS_LIGHT,
        ).shift(UP * 0.5)

        tokens_str = ["The", "cat", "sat", "on", "the", "mat"]
        token_boxes = VGroup()
        for i, tok in enumerate(tokens_str):
            box = RoundedRectangle(
                width=1.0, height=0.5, corner_radius=0.08,
                color=GLASS_TEAL, fill_opacity=0.3, stroke_width=1.5,
            )
            txt = Text(tok, font_size=18, color=GLASS_LIGHT)
            txt.move_to(box.get_center())
            grp = VGroup(box, txt)
            token_boxes.add(grp)
        token_boxes.arrange(RIGHT, buff=0.15).shift(DOWN * 0.5)

        self.play(Write(stage_label))
        self.play(FadeIn(input_text))
        self.wait(0.5)
        self.play(
            ReplacementTransform(
                input_text.copy(), token_boxes,
            ),
            run_time=1.2,
        )
        self.wait(1.0)
        self.play(FadeOut(stage_label), FadeOut(input_text))

        # ── Stage 2: Model + Hook Manager ────────────────────────
        stage_label = subtitle_text("2. Model + HookManager").to_edge(UP, buff=0.4)
        self.play(Write(stage_label))

        # Shrink tokens and move up
        self.play(token_boxes.animate.scale(0.7).to_edge(UP, buff=1.2))

        # Build layer stack
        stack = create_layer_stack(
            n_layers=6, layer_width=3.5, layer_height=0.4,
            labels=["Embed", "Attn 0", "MLP 0", "Attn 1", "MLP 1", "Unembed"],
        ).shift(DOWN * 0.3)
        self.play(FadeIn(stack, shift=UP * 0.3), run_time=1.0)

        # Animate "hooks" attaching — small glowing dots on each layer
        hooks = VGroup()
        for i, layer_block in enumerate(stack):
            hook_dot = Dot(
                point=layer_block.get_right() + RIGHT * 0.3,
                radius=0.08, color=GLASS_GOLD,
            )
            hook_label = Text("hook", font_size=10, color=GLASS_GOLD)
            hook_label.next_to(hook_dot, RIGHT, buff=0.1)
            hooks.add(VGroup(hook_dot, hook_label))

        self.play(
            LaggedStart(*[FadeIn(h, shift=LEFT * 0.2) for h in hooks], lag_ratio=0.15),
            run_time=1.5,
        )

        # Activation dots flowing down through layers
        for i in range(len(stack) - 1, -1, -1):
            layer_block = stack[i]
            flash = layer_block[0].copy().set_fill(GLASS_GOLD, opacity=0.5)
            self.play(FadeIn(flash), run_time=0.12)
            self.play(FadeOut(flash), run_time=0.12)

        self.wait(0.8)

        # Capture activations — show them flying to the side
        act_label = label_text("ActivationStore").shift(RIGHT * 4.5 + UP * 0.5)
        act_box = RoundedRectangle(
            width=2.5, height=1.2, corner_radius=0.1,
            color=GLASS_TEAL, fill_opacity=0.2, stroke_width=1.5,
        ).move_to(act_label.get_center() + DOWN * 0.1)

        act_rows = VGroup()
        for j in range(4):
            row = VGroup(*[
                Square(side_length=0.12, color=layer_color(j), fill_opacity=0.7, stroke_width=0)
                for _ in range(12)
            ]).arrange(RIGHT, buff=0.03)
            act_rows.add(row)
        act_rows.arrange(DOWN, buff=0.06).move_to(act_box.get_center())

        self.play(FadeIn(act_box), Write(act_label))
        self.play(FadeIn(act_rows, shift=LEFT * 0.3), run_time=0.8)
        self.wait(0.8)

        # Clean up for next stage
        self.play(
            FadeOut(stack), FadeOut(hooks), FadeOut(token_boxes),
            FadeOut(stage_label),
            VGroup(act_box, act_label, act_rows).animate.scale(0.6).to_corner(UL, buff=0.5),
        )

        # ── Stage 3: SAE Feature Discovery ──────────────────────
        stage_label = subtitle_text("3. Sparse Autoencoder → Features").to_edge(UP, buff=0.4)
        self.play(Write(stage_label))

        # SAE architecture: encoder → sparse → decoder
        enc_neurons = VGroup(*[neuron_circle(0.15, GLASS_ACCENT) for _ in range(4)])
        enc_neurons.arrange(DOWN, buff=0.25).shift(LEFT * 3)
        enc_label = label_text("Input").next_to(enc_neurons, DOWN, buff=0.3)

        sparse_neurons = VGroup(*[neuron_circle(0.12, GLASS_DIM) for _ in range(12)])
        sparse_neurons.arrange(DOWN, buff=0.1)
        sparse_label = label_text("Sparse\nLatents").next_to(sparse_neurons, DOWN, buff=0.3)

        dec_neurons = VGroup(*[neuron_circle(0.15, GLASS_ACCENT) for _ in range(4)])
        dec_neurons.arrange(DOWN, buff=0.25).shift(RIGHT * 3)
        dec_label = label_text("Recon").next_to(dec_neurons, DOWN, buff=0.3)

        # Draw connections
        enc_lines = VGroup()
        for e in enc_neurons:
            for s in sparse_neurons:
                enc_lines.add(Line(
                    e.get_right(), s.get_left(),
                    stroke_width=0.5, color=GLASS_DIM, stroke_opacity=0.3,
                ))
        dec_lines = VGroup()
        for s in sparse_neurons:
            for d in dec_neurons:
                dec_lines.add(Line(
                    s.get_right(), d.get_left(),
                    stroke_width=0.5, color=GLASS_DIM, stroke_opacity=0.3,
                ))

        sae_group = VGroup(
            enc_lines, dec_lines,
            enc_neurons, sparse_neurons, dec_neurons,
            enc_label, sparse_label, dec_label,
        )

        self.play(
            FadeIn(enc_neurons), FadeIn(dec_neurons),
            FadeIn(enc_label), FadeIn(dec_label),
            Create(enc_lines), Create(dec_lines),
            FadeIn(sparse_neurons), FadeIn(sparse_label),
            run_time=1.5,
        )

        # Light up sparse features — only a few activate
        active_indices = [1, 4, 7, 10]
        active_anims = []
        for idx in active_indices:
            active_anims.append(
                sparse_neurons[idx].animate.set_fill(GLASS_PRIMARY, opacity=0.9)
            )
        self.play(*active_anims, run_time=0.8)
        self.wait(0.5)

        # Feature labels appear
        feature_labels = VGroup()
        feature_names = ["syntax", "entity", "negation", "sentiment"]
        for i, idx in enumerate(active_indices):
            fl = Text(feature_names[i], font_size=12, color=GLASS_GOLD)
            fl.next_to(sparse_neurons[idx], RIGHT, buff=0.15)
            feature_labels.add(fl)
        self.play(FadeIn(feature_labels, shift=RIGHT * 0.1), run_time=0.6)
        self.wait(1.0)

        self.play(FadeOut(sae_group), FadeOut(feature_labels), FadeOut(stage_label))

        # ── Stage 4: Probing ────────────────────────────────────
        stage_label = subtitle_text("4. Linear Probe → Concept Direction").to_edge(UP, buff=0.4)
        self.play(Write(stage_label))

        # 2D scatter plot
        axes = Axes(
            x_range=[-4, 4, 1], y_range=[-4, 4, 1],
            x_length=5, y_length=4,
            axis_config={"color": GLASS_DIM, "stroke_width": 1},
        ).shift(DOWN * 0.3)

        np.random.seed(42)
        pos_pts = np.random.randn(30, 2) * 0.8 + np.array([1.5, 1.0])
        neg_pts = np.random.randn(30, 2) * 0.8 + np.array([-1.5, -1.0])

        pos_dots = VGroup(*[
            Dot(axes.c2p(x, y), radius=0.06, color=GLASS_TEAL, fill_opacity=0.8)
            for x, y in pos_pts
        ])
        neg_dots = VGroup(*[
            Dot(axes.c2p(x, y), radius=0.06, color=GLASS_PRIMARY, fill_opacity=0.8)
            for x, y in neg_pts
        ])

        legend_pos = VGroup(
            Dot(radius=0.06, color=GLASS_TEAL),
            Text("positive", font_size=14, color=GLASS_LIGHT),
        ).arrange(RIGHT, buff=0.1).to_corner(UR, buff=0.8)
        legend_neg = VGroup(
            Dot(radius=0.06, color=GLASS_PRIMARY),
            Text("negative", font_size=14, color=GLASS_LIGHT),
        ).arrange(RIGHT, buff=0.1).next_to(legend_pos, DOWN, buff=0.15)

        self.play(Create(axes), run_time=0.5)
        self.play(
            LaggedStart(*[FadeIn(d, scale=0.5) for d in pos_dots], lag_ratio=0.02),
            LaggedStart(*[FadeIn(d, scale=0.5) for d in neg_dots], lag_ratio=0.02),
            FadeIn(legend_pos), FadeIn(legend_neg),
            run_time=1.0,
        )

        # Separating line sweeps in
        sep_line = axes.plot(lambda x: x * 0.6, x_range=[-3.5, 3.5], color=GLASS_GOLD, stroke_width=2)
        acc_text = Text("Accuracy: 94.2%", font_size=20, color=GLASS_GREEN)
        acc_text.next_to(axes, DOWN, buff=0.4)

        self.play(Create(sep_line), run_time=1.0)
        self.play(Write(acc_text), run_time=0.5)

        # Direction arrow
        direction_arrow = Arrow(
            axes.c2p(-1, -0.6), axes.c2p(1, 0.6),
            color=GLASS_GOLD, stroke_width=3, buff=0,
        )
        dir_label = Text("concept direction", font_size=14, color=GLASS_GOLD)
        dir_label.next_to(direction_arrow, RIGHT, buff=0.15)

        self.play(GrowArrow(direction_arrow), Write(dir_label), run_time=0.8)
        self.wait(1.0)

        self.play(
            FadeOut(axes), FadeOut(pos_dots), FadeOut(neg_dots),
            FadeOut(sep_line), FadeOut(acc_text), FadeOut(direction_arrow),
            FadeOut(dir_label), FadeOut(legend_pos), FadeOut(legend_neg),
            FadeOut(stage_label),
        )

        # ── Stage 5: Steering ───────────────────────────────────
        stage_label = subtitle_text("5. DirectionalSteering → Modified Output").to_edge(UP, buff=0.4)
        self.play(Write(stage_label))

        # Before / After comparison
        before_title = Text("Before Steering", font_size=22, color=GLASS_LIGHT)
        after_title = Text("After Steering", font_size=22, color=GLASS_LIGHT)

        before_text = Text(
            '"I feel okay about\nthis product."',
            font_size=18, color=GLASS_DIM, line_spacing=1.3,
        )
        after_text = Text(
            '"I absolutely love\nthis product!"',
            font_size=18, color=GLASS_TEAL, line_spacing=1.3,
        )

        before_box = RoundedRectangle(
            width=3.5, height=2.0, corner_radius=0.1,
            color=GLASS_DIM, fill_opacity=0.1, stroke_width=1.5,
        )
        after_box = RoundedRectangle(
            width=3.5, height=2.0, corner_radius=0.1,
            color=GLASS_TEAL, fill_opacity=0.1, stroke_width=1.5,
        )

        before_grp = VGroup(before_box, before_title, before_text).arrange(DOWN, buff=0.2).shift(LEFT * 3)
        after_grp = VGroup(after_box, after_title, after_text).arrange(DOWN, buff=0.2).shift(RIGHT * 3)

        # Steering arrow between them
        steer_arrow = Arrow(
            before_grp.get_right() + LEFT * 0.2,
            after_grp.get_left() + RIGHT * 0.2,
            color=GLASS_GOLD, stroke_width=3,
        )
        steer_label = Text(
            "+ sentiment direction\n  (strength = 2.0)",
            font_size=14, color=GLASS_GOLD,
        ).next_to(steer_arrow, UP, buff=0.15)

        self.play(FadeIn(before_grp, shift=LEFT * 0.3), run_time=0.8)
        self.play(GrowArrow(steer_arrow), Write(steer_label), run_time=0.8)
        self.play(FadeIn(after_grp, shift=RIGHT * 0.3), run_time=0.8)
        self.wait(1.0)

        self.play(
            FadeOut(before_grp), FadeOut(after_grp),
            FadeOut(steer_arrow), FadeOut(steer_label),
            FadeOut(stage_label),
        )

        # ── Stage 6: Code Interface ─────────────────────────────
        stage_label = subtitle_text("6. One Unified API").to_edge(UP, buff=0.4)
        self.play(Write(stage_label))

        code_lines = [
            'from glassboxllms.experiments import run_experiment',
            '',
            'result = run_experiment("probing", {',
            '    "model_name": "gpt2",',
            '    "probe_type": "logistic",',
            '    "layer": "transformer.h.5",',
            '})',
            '',
            'print(result.summary())',
        ]
        code_block = Code(
            code_string="\n".join(code_lines),
            language="python",
            background="rectangle",
            add_line_numbers=False,
            formatter_style="monokai",
        ).scale(0.85)

        self.play(FadeIn(code_block, shift=UP * 0.3), run_time=1.0)
        self.wait(2.0)
        self.play(FadeOut(code_block), FadeOut(stage_label))

        # Clean up activation store from corner
        self.play(FadeOut(VGroup(act_box, act_label, act_rows)))

        # ── Finale ──────────────────────────────────────────────
        final_title = title_text("glassbox-llms")
        final_sub = subtitle_text("Open-Source LLM Interpretability")
        final_sub.next_to(final_title, DOWN, buff=0.3)

        modules = VGroup()
        module_names = [
            "ModelWrapper", "HookManager", "ActivationStore",
            "SparseAutoencoder", "LinearProbe", "FeatureAtlas",
            "CausalScrubber", "CircuitGraph", "DirectionalSteering",
        ]
        for i, name in enumerate(module_names):
            pill = RoundedRectangle(
                width=2.2, height=0.35, corner_radius=0.08,
                color=layer_color(i), fill_opacity=0.4, stroke_width=1,
            )
            txt = Text(name, font_size=13, color=GLASS_LIGHT)
            txt.move_to(pill.get_center())
            modules.add(VGroup(pill, txt))
        modules.arrange_in_grid(rows=3, cols=3, buff=0.2).shift(DOWN * 1.2)

        self.play(Write(final_title), run_time=0.8)
        self.play(FadeIn(final_sub, shift=UP * 0.2), run_time=0.6)
        self.play(
            LaggedStart(*[FadeIn(m, scale=0.8) for m in modules], lag_ratio=0.1),
            run_time=1.5,
        )
        self.wait(3.0)
        self.play(*[FadeOut(mob) for mob in self.mobjects])
