"""
CircuitDiscoveryScene — Visualize causal scrubbing and circuit discovery.

Shows a model as a layered graph, ablates components, and reveals
the minimal circuit that explains a behaviour.

Render:
    manim -qh circuit_discovery.py CircuitDiscoveryScene
"""

from manim import *
import numpy as np

from glassboxllms.visualization.manim_scenes.utils import (
    GLASS_BG, GLASS_PRIMARY, GLASS_GOLD,
    GLASS_TEAL, GLASS_PURPLE, GLASS_LIGHT, GLASS_DIM,
    GLASS_GREEN, GLASS_ORANGE,
    layer_color, title_text, subtitle_text, label_text,
    generate_circuit_graph,
)


class CircuitDiscoveryScene(Scene):
    """Causal scrubbing reveals the circuit that matters."""

    def construct(self):
        self.camera.background_color = GLASS_BG

        # ── Title ───────────────────────────────────────────────
        title = title_text("Circuit Discovery")
        sub = subtitle_text("Finding the Computation That Matters")
        sub.next_to(title, DOWN, buff=0.3)
        self.play(Write(title), FadeIn(sub, shift=UP * 0.2))
        self.wait(1.0)
        self.play(FadeOut(title), FadeOut(sub))

        # ── Build the model graph ───────────────────────────────
        part_label = subtitle_text("Model as a Computational Graph").to_edge(UP, buff=0.5)
        self.play(Write(part_label))

        n_layers = 5
        n_per_layer = 4
        node_labels = {
            0: ["Embed"] * n_per_layer,
            1: ["A0.H0", "A0.H1", "A0.H2", "A0.H3"],
            2: ["MLP0", "A1.H0", "A1.H1", "MLP1"],
            3: ["A2.H0", "A2.H1", "MLP2", "A2.H3"],
            4: ["Unembed"] * n_per_layer,
        }

        # Create nodes
        nodes = {}
        node_circles = {}
        for layer in range(n_layers):
            for idx in range(n_per_layer):
                x = (idx - (n_per_layer - 1) / 2) * 1.8
                y = (layer - (n_layers - 1) / 2) * 1.3
                pos = np.array([x, y, 0])

                circle = Circle(
                    radius=0.3, color=layer_color(layer),
                    fill_opacity=0.5, stroke_width=1.5,
                )
                circle.move_to(pos)
                lbl = Text(
                    node_labels[layer][idx],
                    font_size=10, color=GLASS_LIGHT,
                )
                lbl.move_to(pos)
                nodes[(layer, idx)] = VGroup(circle, lbl)
                node_circles[(layer, idx)] = circle

        all_nodes = VGroup(*nodes.values())
        self.play(
            LaggedStart(*[FadeIn(n, scale=0.7) for n in all_nodes], lag_ratio=0.03),
            run_time=1.5,
        )

        # Create edges (between adjacent layers)
        np.random.seed(42)
        edges = {}
        edge_lines = VGroup()
        for layer in range(n_layers - 1):
            for src_idx in range(n_per_layer):
                for dst_idx in range(n_per_layer):
                    if np.random.random() > 0.5:
                        start = node_circles[(layer, src_idx)].get_top()
                        end = node_circles[(layer + 1, dst_idx)].get_bottom()
                        line = Line(
                            start, end,
                            color=GLASS_DIM, stroke_width=0.8, stroke_opacity=0.3,
                        )
                        edges[((layer, src_idx), (layer + 1, dst_idx))] = line
                        edge_lines.add(line)

        self.play(Create(edge_lines), run_time=1.0)
        self.wait(0.5)

        # ── Activation flow ─────────────────────────────────────
        self.play(FadeOut(part_label))
        part_label = subtitle_text("Forward Pass: Activations Flow Upward").to_edge(UP, buff=0.5)
        self.play(Write(part_label))

        for layer in range(n_layers):
            anims = []
            for idx in range(n_per_layer):
                flash = node_circles[(layer, idx)].copy()
                flash.set_fill(GLASS_GOLD, opacity=0.7)
                flash.set_stroke(GLASS_GOLD, width=2)
                anims.append(FadeIn(flash))
                anims.append(FadeOut(flash))
            self.play(*anims, run_time=0.35)

        self.wait(0.5)

        # ── Logit diff metric ───────────────────────────────────
        metric_box = RoundedRectangle(
            width=4, height=0.6, corner_radius=0.08,
            color=GLASS_GREEN, fill_opacity=0.1, stroke_width=1,
        ).to_edge(RIGHT, buff=0.5).shift(UP * 2)
        metric_text = Text("Logit Diff: 3.42", font_size=16, color=GLASS_GREEN)
        metric_text.move_to(metric_box.get_center())
        self.play(Create(metric_box), Write(metric_text))

        # ── Causal Scrubbing: Ablate nodes ──────────────────────
        self.play(FadeOut(part_label))
        part_label = subtitle_text("Causal Scrubbing: Which Nodes Matter?").to_edge(UP, buff=0.5)
        self.play(Write(part_label))

        # Ablate an unimportant node (visually: turn red, cross out)
        ablate_targets = [(1, 2), (2, 3), (3, 2)]  # nodes to ablate
        for layer, idx in ablate_targets:
            circle = node_circles[(layer, idx)]

            # Red flash
            self.play(
                circle.animate.set_fill(GLASS_PRIMARY, opacity=0.8).set_stroke(GLASS_PRIMARY, width=2),
                run_time=0.4,
            )

            # X mark
            x_mark = VGroup(
                Line(circle.get_center() + UL * 0.15, circle.get_center() + DR * 0.15,
                     color=WHITE, stroke_width=2),
                Line(circle.get_center() + UR * 0.15, circle.get_center() + DL * 0.15,
                     color=WHITE, stroke_width=2),
            )
            self.play(Create(x_mark), run_time=0.2)

        # Logit diff barely changes — these nodes don't matter
        new_metric = Text("Logit Diff: 3.38 (Δ = -0.04)", font_size=16, color=GLASS_GREEN)
        new_metric.move_to(metric_box.get_center())
        self.play(
            FadeOut(metric_text),
            FadeIn(new_metric),
            run_time=0.5,
        )

        not_important = Text(
            "Removing these nodes barely changes the output!",
            font_size=16, color=GLASS_ORANGE,
        ).to_edge(DOWN, buff=0.5)
        self.play(Write(not_important))
        self.wait(1.5)
        self.play(FadeOut(not_important))

        # Now ablate an IMPORTANT node
        important_node = (1, 1)
        imp_circle = node_circles[important_node]
        self.play(
            imp_circle.animate.set_fill(GLASS_PRIMARY, opacity=0.9).set_stroke(GLASS_PRIMARY, width=3),
            run_time=0.5,
        )
        imp_x = VGroup(
            Line(imp_circle.get_center() + UL * 0.15, imp_circle.get_center() + DR * 0.15,
                 color=WHITE, stroke_width=2),
            Line(imp_circle.get_center() + UR * 0.15, imp_circle.get_center() + DL * 0.15,
                 color=WHITE, stroke_width=2),
        )
        self.play(Create(imp_x), run_time=0.2)

        # Logit diff drops significantly
        critical_metric = Text("Logit Diff: 0.12 (Δ = -3.30)", font_size=16, color=GLASS_PRIMARY)
        critical_metric.move_to(metric_box.get_center())
        self.play(
            FadeOut(new_metric), FadeIn(critical_metric),
            metric_box.animate.set_stroke(GLASS_PRIMARY, width=2),
            run_time=0.5,
        )

        important_text = Text(
            "A0.H1 is CRITICAL for this behaviour!",
            font_size=18, color=GLASS_PRIMARY, weight=BOLD,
        ).to_edge(DOWN, buff=0.5)
        self.play(Write(important_text))
        self.wait(2.0)
        self.play(*[FadeOut(m) for m in self.mobjects])

        # ── Part 3: The Discovered Circuit ──────────────────────
        part_label = subtitle_text("The Minimal Circuit").to_edge(UP, buff=0.5)
        self.play(Write(part_label))

        # Rebuild graph but only show the important nodes
        important_nodes_set = {(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (4, 1)}
        circuit_nodes = VGroup()
        circuit_circles = {}
        for (layer, idx), node_group in nodes.items():
            if (layer, idx) in important_nodes_set:
                new_circle = Circle(
                    radius=0.35, color=GLASS_GREEN,
                    fill_opacity=0.6, stroke_width=2,
                )
                new_circle.move_to(node_group[0].get_center())
                new_label = node_group[1].copy()
                new_label.set_color(GLASS_LIGHT)
                grp = VGroup(new_circle, new_label)
                circuit_nodes.add(grp)
                circuit_circles[(layer, idx)] = new_circle

        # Circuit edges
        circuit_edges = VGroup()
        circuit_connections = [
            ((0, 0), (1, 0)), ((0, 1), (1, 1)),
            ((1, 0), (2, 0)), ((1, 1), (2, 1)),
            ((2, 0), (3, 0)), ((2, 1), (3, 1)),
            ((3, 0), (4, 0)), ((3, 1), (4, 1)),
            ((1, 1), (2, 0)),  # cross-connection
        ]
        for (l1, i1), (l2, i2) in circuit_connections:
            if (l1, i1) in circuit_circles and (l2, i2) in circuit_circles:
                line = Line(
                    circuit_circles[(l1, i1)].get_top(),
                    circuit_circles[(l2, i2)].get_bottom(),
                    color=GLASS_GREEN, stroke_width=2, stroke_opacity=0.7,
                )
                circuit_edges.add(line)

        self.play(
            LaggedStart(*[FadeIn(n, scale=0.8) for n in circuit_nodes], lag_ratio=0.05),
            run_time=1.0,
        )
        self.play(Create(circuit_edges), run_time=0.8)

        summary = Text(
            "10 out of 20 nodes form the essential circuit",
            font_size=16, color=GLASS_GREEN,
        ).to_edge(DOWN, buff=0.5)
        self.play(Write(summary))
        self.wait(3.0)
        self.play(*[FadeOut(m) for m in self.mobjects])
