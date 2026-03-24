"""
SAEFeatureDiscoveryScene — Animate sparse autoencoder feature discovery.

Shows how a dense activation vector passes through an SAE to reveal
sparse, monosemantic features that get registered in a FeatureAtlas.

Render:
    manim -qh sae_feature_discovery.py SAEFeatureDiscoveryScene
"""

from manim import *
import numpy as np

from glassboxllms.visualization.manim_scenes.utils import (
    GLASS_BG, GLASS_PRIMARY, GLASS_ACCENT, GLASS_GOLD,
    GLASS_TEAL, GLASS_PURPLE, GLASS_LIGHT, GLASS_DIM,
    GLASS_GREEN, GLASS_ORANGE,
    layer_color, title_text, subtitle_text, label_text,
    neuron_circle, generate_sae_activations,
)


class SAEFeatureDiscoveryScene(Scene):
    """Sparse Autoencoder: from polysemantic neurons to monosemantic features."""

    def construct(self):
        self.camera.background_color = GLASS_BG

        # ── Title ───────────────────────────────────────────────
        title = title_text("Sparse Autoencoder")
        sub = subtitle_text("Finding Monosemantic Features")
        sub.next_to(title, DOWN, buff=0.3)
        self.play(Write(title), run_time=0.8)
        self.play(FadeIn(sub, shift=UP * 0.2))
        self.wait(1.0)
        self.play(FadeOut(title), FadeOut(sub))

        # ── Part 1: Polysemantic Neurons ────────────────────────
        part_label = subtitle_text("The Problem: Polysemantic Neurons").to_edge(UP, buff=0.5)
        self.play(Write(part_label))

        # A single neuron that responds to many things
        big_neuron = neuron_circle(0.6, GLASS_ACCENT, fill_opacity=0.5)
        big_neuron.shift(DOWN * 0.5)
        neuron_label = label_text("Neuron #42").next_to(big_neuron, DOWN, buff=0.3)

        self.play(FadeIn(big_neuron), Write(neuron_label))

        # Multiple concepts light it up
        concepts = ["cat", "legal", "happy", "3.14", "French"]
        concept_colors = [GLASS_TEAL, GLASS_PRIMARY, GLASS_GOLD, GLASS_PURPLE, GLASS_GREEN]

        concept_labels = VGroup()
        for i, (concept, color) in enumerate(zip(concepts, concept_colors)):
            angle = (i / len(concepts)) * TAU + PI / 2
            pos = big_neuron.get_center() + 2.2 * np.array([np.cos(angle), np.sin(angle), 0])
            cl = Text(concept, font_size=18, color=color)
            cl.move_to(pos)
            concept_labels.add(cl)

        for i, cl in enumerate(concept_labels):
            arrow = Arrow(
                cl.get_center(),
                big_neuron.get_center() + 0.7 * normalize(cl.get_center() - big_neuron.get_center()),
                color=concept_colors[i], stroke_width=2, buff=0.1,
                max_tip_length_to_length_ratio=0.15,
            )
            self.play(
                FadeIn(cl),
                GrowArrow(arrow),
                big_neuron.animate.set_fill(concept_colors[i], opacity=0.6),
                run_time=0.4,
            )

        problem_text = Text(
            "One neuron responds to MANY unrelated concepts",
            font_size=18, color=GLASS_ORANGE,
        ).to_edge(DOWN, buff=0.5)
        self.play(Write(problem_text))
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in self.mobjects])

        # ── Part 2: SAE Architecture ───────────────────────────
        part_label = subtitle_text("The Solution: Sparse Autoencoder").to_edge(UP, buff=0.5)
        self.play(Write(part_label))

        # Input layer (d_model = 4)
        n_input = 5
        n_hidden = 16
        n_output = 5

        input_neurons = VGroup(*[neuron_circle(0.18, GLASS_ACCENT) for _ in range(n_input)])
        input_neurons.arrange(DOWN, buff=0.3).shift(LEFT * 4.5)
        input_label = label_text("Activations\n(d_model)").next_to(input_neurons, DOWN, buff=0.4)

        # Wide hidden layer (d_sae >> d_model)
        hidden_neurons = VGroup(*[neuron_circle(0.1, GLASS_DIM, fill_opacity=0.3) for _ in range(n_hidden)])
        hidden_neurons.arrange(DOWN, buff=0.08)
        hidden_label = label_text("Sparse Latents\n(d_sae ≫ d_model)").next_to(hidden_neurons, DOWN, buff=0.4)

        # Output layer (reconstruction)
        output_neurons = VGroup(*[neuron_circle(0.18, GLASS_ACCENT) for _ in range(n_output)])
        output_neurons.arrange(DOWN, buff=0.3).shift(RIGHT * 4.5)
        output_label = label_text("Reconstruction").next_to(output_neurons, DOWN, buff=0.4)

        self.play(
            FadeIn(input_neurons), Write(input_label),
            FadeIn(hidden_neurons), Write(hidden_label),
            FadeIn(output_neurons), Write(output_label),
            run_time=1.0,
        )

        # Draw faint connections
        enc_lines = VGroup()
        for inp in input_neurons:
            for hid in hidden_neurons:
                enc_lines.add(Line(
                    inp.get_right(), hid.get_left(),
                    stroke_width=0.3, color=GLASS_DIM, stroke_opacity=0.15,
                ))
        dec_lines = VGroup()
        for hid in hidden_neurons:
            for out in output_neurons:
                dec_lines.add(Line(
                    hid.get_right(), out.get_left(),
                    stroke_width=0.3, color=GLASS_DIM, stroke_opacity=0.15,
                ))

        self.play(Create(enc_lines), Create(dec_lines), run_time=0.8)

        # ── Part 3: Training Animation ──────────────────────────
        # Show input activating, then only a few hidden units light up
        # Simulate an input
        for inp_n in input_neurons:
            inp_n.generate_target()
            inp_n.target.set_fill(GLASS_TEAL, opacity=0.8)
        self.play(*[MoveToTarget(n) for n in input_neurons], run_time=0.5)

        # Only a few sparse features activate
        active_indices = [2, 6, 11, 14]
        feature_names = ["syntax", "entity", "negation", "number"]
        feature_colors = [GLASS_TEAL, GLASS_PRIMARY, GLASS_PURPLE, GLASS_GOLD]

        anims = []
        feature_tags = VGroup()
        for i, (idx, name, color) in enumerate(zip(active_indices, feature_names, feature_colors)):
            anims.append(hidden_neurons[idx].animate.set_fill(color, opacity=0.9))
            tag = Text(name, font_size=11, color=color)
            tag.next_to(hidden_neurons[idx], RIGHT, buff=0.15)
            feature_tags.add(tag)

        self.play(*anims, run_time=0.8)
        self.play(FadeIn(feature_tags, shift=RIGHT * 0.1), run_time=0.5)

        # Sparsity annotation
        sparsity_text = Text(
            f"Only {len(active_indices)}/{n_hidden} features active (Top-K sparsity)",
            font_size=16, color=GLASS_GREEN,
        ).to_edge(DOWN, buff=0.5)
        self.play(Write(sparsity_text))
        self.wait(1.5)

        # Reconstruction lights up
        for out_n in output_neurons:
            out_n.generate_target()
            out_n.target.set_fill(GLASS_TEAL, opacity=0.7)
        self.play(*[MoveToTarget(n) for n in output_neurons], run_time=0.5)

        recon_text = Text("R² = 0.94", font_size=18, color=GLASS_GREEN)
        recon_text.next_to(output_neurons, RIGHT, buff=0.3)
        self.play(Write(recon_text))
        self.wait(1.0)
        self.play(*[FadeOut(m) for m in self.mobjects])

        # ── Part 4: Feature Atlas Registration ──────────────────
        part_label = subtitle_text("Registering Features in the Atlas").to_edge(UP, buff=0.5)
        self.play(Write(part_label))

        # Atlas as a grid/catalog
        atlas_title = label_text("FeatureAtlas").shift(UP * 1.5)
        atlas_box = RoundedRectangle(
            width=8, height=4, corner_radius=0.15,
            color=GLASS_GOLD, fill_opacity=0.05, stroke_width=1.5,
        )
        self.play(Write(atlas_title), Create(atlas_box))

        # Feature cards fly in
        cards = VGroup()
        card_data = [
            ("SAE_LATENT #2", "syntax", "layer.5.mlp", GLASS_TEAL),
            ("SAE_LATENT #6", "entity", "layer.5.mlp", GLASS_PRIMARY),
            ("SAE_LATENT #11", "negation", "layer.5.mlp", GLASS_PURPLE),
            ("SAE_LATENT #14", "number", "layer.5.mlp", GLASS_GOLD),
            ("PROBE_DIR", "sentiment", "layer.8", GLASS_GREEN),
            ("CIRCUIT", "IOI", "layers 0-11", GLASS_ORANGE),
        ]
        for i, (ftype, fname, floc, fcolor) in enumerate(card_data):
            card = RoundedRectangle(
                width=2.2, height=1.0, corner_radius=0.08,
                color=fcolor, fill_opacity=0.15, stroke_width=1,
            )
            type_txt = Text(ftype, font_size=11, color=fcolor)
            name_txt = Text(fname, font_size=16, color=GLASS_LIGHT, weight=BOLD)
            loc_txt = Text(floc, font_size=10, color=GLASS_DIM)
            inner = VGroup(type_txt, name_txt, loc_txt).arrange(DOWN, buff=0.06)
            inner.move_to(card.get_center())
            cards.add(VGroup(card, inner))

        cards.arrange_in_grid(rows=2, cols=3, buff=0.3)
        cards.move_to(atlas_box.get_center())

        self.play(
            LaggedStart(*[FadeIn(c, shift=UP * 0.5, scale=0.8) for c in cards], lag_ratio=0.2),
            run_time=2.0,
        )
        self.wait(1.5)

        # Final count
        count_text = Text(
            f"{len(card_data)} features discovered and cataloged",
            font_size=18, color=GLASS_GREEN,
        ).to_edge(DOWN, buff=0.5)
        self.play(Write(count_text))
        self.wait(2.0)
        self.play(*[FadeOut(m) for m in self.mobjects])
