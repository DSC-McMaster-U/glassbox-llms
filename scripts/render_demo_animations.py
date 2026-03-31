#!/usr/bin/env python3
"""
Render 3 high-quality Manim demo animations using real GPT-2 data.

Produces:
  media/demo/ProbingHyperplaneScene.mp4
  media/demo/CircuitDiscoveryScene.mp4
  media/demo/SteeringVectorScene.mp4

Usage:
    python scripts/render_demo_animations.py
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = PROJECT_ROOT / "media" / "demo"
DEMO_DIR.mkdir(parents=True, exist_ok=True)

# Ensure project is importable
sys.path.insert(0, str(PROJECT_ROOT))


# ===================================================================
# Helpers
# ===================================================================

def _timer(label: str):
    """Simple context-manager timer that prints elapsed time."""
    class _T:
        def __enter__(self):
            self.t0 = time.time()
            print(f"\n{'='*60}")
            print(f"  {label}")
            print(f"{'='*60}")
            return self
        def __exit__(self, *exc):
            elapsed = time.time() - self.t0
            print(f"  Done in {elapsed:.1f}s")
    return _T()


def _move_video(scene_name: str, media_dir: Path):
    """
    Manim writes videos into a nested directory structure under media_dir.
    Find the rendered .mp4 and copy it to DEMO_DIR/<scene_name>.mp4.
    """
    target = DEMO_DIR / f"{scene_name}.mp4"
    # Walk the media directory tree looking for the mp4
    for root, _dirs, files in os.walk(media_dir):
        for f in files:
            if f.endswith(".mp4") and scene_name in f:
                src = Path(root) / f
                shutil.copy2(src, target)
                print(f"  Output: {target}")
                return
    # Fallback: grab the first mp4 we find
    for root, _dirs, files in os.walk(media_dir):
        for f in files:
            if f.endswith(".mp4"):
                src = Path(root) / f
                shutil.copy2(src, target)
                print(f"  Output: {target}")
                return
    print(f"  WARNING: could not locate rendered mp4 for {scene_name}")


# ===================================================================
# 1. Load GPT-2 model and tokenizer (shared across scenes)
# ===================================================================

def load_gpt2():
    """Load GPT-2 small and its tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading GPT-2 model and tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"  GPT-2 loaded on {device}")
    return model, tokenizer, device


# ===================================================================
# 2. Sentiment dataset (tiny, self-contained)
# ===================================================================

_POSITIVE_TEXTS = [
    "This movie was absolutely wonderful and I loved every moment of it.",
    "What a fantastic experience, truly the best I have ever had.",
    "I am so happy with this product, it exceeded all my expectations.",
    "The food was delicious and the service was outstanding.",
    "A brilliant performance that left the audience cheering.",
    "Everything about this trip was perfect and magical.",
    "I highly recommend this book, it is truly inspirational.",
    "The team did an amazing job delivering the project on time.",
    "Such a beautiful day filled with joy and laughter.",
    "The concert was phenomenal, one of the best nights of my life.",
    "I feel grateful and blessed for all the good things in my life.",
    "This restaurant serves the most incredible pasta I have ever tasted.",
    "An outstanding achievement that deserves the highest praise.",
    "The sunset over the ocean was breathtakingly beautiful.",
    "I am thrilled with how well everything turned out.",
    "What a lovely surprise, this made my entire week.",
    "The customer support was excellent and resolved my issue quickly.",
    "This is hands down the best coffee shop in the city.",
    "A heartwarming story that will make you smile from ear to ear.",
    "The new update is fantastic and makes the app so much better.",
    "I could not be happier with my purchase, it works perfectly.",
    "The garden was gorgeous, full of vibrant colors and sweet scents.",
    "An absolute delight from start to finish.",
    "I love this place, the atmosphere is warm and inviting.",
    "The presentation was engaging and truly informative.",
]

_NEGATIVE_TEXTS = [
    "This movie was terrible and a complete waste of time.",
    "I am very disappointed with the quality of this product.",
    "The food was cold and tasteless, worst meal I have had.",
    "What an awful experience, I would never go back there.",
    "The service was rude and unacceptably slow.",
    "I regret buying this, it broke after just two days.",
    "A boring and uninspired performance that put me to sleep.",
    "The hotel room was dirty and the staff was unhelpful.",
    "I hate how unreliable this software is, constant crashes.",
    "Such a frustrating process, nothing works as advertised.",
    "The book was dull and poorly written from beginning to end.",
    "I am furious about the hidden charges on my bill.",
    "This is the worst customer service I have ever encountered.",
    "The flight was delayed for hours and nobody gave us information.",
    "An embarrassing failure that should never have been released.",
    "The noise from the construction site was unbearable.",
    "I feel cheated, the product looks nothing like the photos.",
    "A horrible waste of money, do not buy this.",
    "The traffic was atrocious and ruined our entire evening.",
    "I am extremely unhappy with how this situation was handled.",
    "The app is buggy and crashes every time I open it.",
    "What a letdown, the sequel is far worse than the original.",
    "The packaging was damaged and the item arrived broken.",
    "I would give zero stars if I could, absolutely dreadful.",
    "The meeting was a disaster, nothing was accomplished.",
]


def build_sentiment_data():
    """Return texts and labels for a simple sentiment dataset."""
    texts = _POSITIVE_TEXTS + _NEGATIVE_TEXTS
    labels = np.array([1] * len(_POSITIVE_TEXTS) + [0] * len(_NEGATIVE_TEXTS))
    return texts, labels


# ===================================================================
# 3. Extract activations from GPT-2 at a given layer
# ===================================================================

def extract_activations(model, tokenizer, texts, layer_name, device):
    """
    Extract mean-pooled activations from a GPT-2 layer.

    Processes one text at a time with hooks to handle variable-length
    tokenization without padding issues.
    """
    import torch
    from operator import attrgetter

    model.to(device)
    model.eval()
    mod = attrgetter(layer_name)(model)
    all_acts = []

    for text in texts:
        cache = {}
        def hook(module, input, output, c=cache):
            tensor = output[0] if isinstance(output, tuple) else output
            c['act'] = tensor.detach().cpu()
            return output
        h = mod.register_forward_hook(hook)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)
        with torch.no_grad():
            model(**inputs)
        h.remove()
        # Mean pool over sequence dimension: (1, seq_len, hidden) -> (hidden,)
        all_acts.append(cache['act'].squeeze(0).mean(dim=0).numpy())

    return np.stack(all_acts)  # (n_samples, hidden_dim)


# ===================================================================
# Scene 1: ProbingHyperplaneScene
# ===================================================================

def render_probing_scene(model, tokenizer, device):
    """Train a sentiment probe on GPT-2 h.10 and render the scatter scene."""
    from manim import config, tempconfig

    from glassboxllms.primitives.probes.linear import LinearProbe
    from glassboxllms.visualization.adapters import probe_result_to_scene_data
    from glassboxllms.visualization.scenes import ProbingHyperplaneScene

    texts, labels = build_sentiment_data()
    layer_name = "transformer.h.10"

    print("  Extracting activations from GPT-2 layer h.10 ...")
    activations = extract_activations(model, tokenizer, texts, layer_name, device)
    print(f"  Activations shape: {activations.shape}")

    # Train/evaluate probe (use full set since it is small)
    print("  Training logistic probe ...")
    probe = LinearProbe(layer="h.10", direction="sentiment", model_type="logistic")
    probe.fit(activations, labels)
    probe_result = probe.evaluate(activations, labels)
    print(f"  Probe result: {probe_result}")

    # Adapt to scene data
    scene_data = probe_result_to_scene_data(
        probe_result,
        activations=activations,
        labels=labels,
        class_names=["negative", "positive"],
        layer="h.10",
        direction_name="sentiment",
    )

    # Render
    media_dir = str(DEMO_DIR / "_manim_probing")
    with tempconfig({
        "quality": "high_quality",
        "pixel_width": 1920,
        "pixel_height": 1080,
        "media_dir": media_dir,
    }):
        scene = ProbingHyperplaneScene()
        scene.scene_data = scene_data
        scene.render()

    _move_video("ProbingHyperplaneScene", Path(media_dir))


# ===================================================================
# Scene 2: CircuitDiscoveryScene
# ===================================================================

def build_ioi_circuit_graph():
    """
    Build a simplified IOI (Indirect Object Identification) circuit graph
    for GPT-2, based on the well-known IOI circuit from the literature.

    Returns a CircuitGraph instance.
    """
    from glassboxllms.analysis.circuits.graph import CircuitGraph

    graph = CircuitGraph(model="gpt2", name="IOI Circuit")

    # -- Embedding layer --
    graph.add_node("embed.tok", node_type="embedding", layer=0, index=0,
                   label="Token Embed")

    # -- Duplicate token heads (layer 0) --
    graph.add_node("attn.0.h1", node_type="attention_head", layer=0, index=1,
                   label="Dup 0.1")
    graph.add_node("attn.0.h10", node_type="attention_head", layer=0, index=10,
                   label="Dup 0.10")

    # -- Previous token heads (layer 2) --
    graph.add_node("attn.2.h2", node_type="attention_head", layer=2, index=2,
                   label="Prev 2.2")

    # -- Induction heads (layer 5-6) --
    graph.add_node("attn.5.h5", node_type="attention_head", layer=5, index=5,
                   label="Ind 5.5")
    graph.add_node("attn.6.h9", node_type="attention_head", layer=6, index=9,
                   label="Ind 6.9")

    # -- S-Inhibition heads (layer 7-8) --
    graph.add_node("attn.7.h3", node_type="attention_head", layer=7, index=3,
                   label="S-Inh 7.3")
    graph.add_node("attn.7.h9", node_type="attention_head", layer=7, index=9,
                   label="S-Inh 7.9")
    graph.add_node("attn.8.h6", node_type="attention_head", layer=8, index=6,
                   label="S-Inh 8.6")
    graph.add_node("attn.8.h10", node_type="attention_head", layer=8, index=10,
                   label="S-Inh 8.10")

    # -- Name Mover heads (layer 9-10) --
    graph.add_node("attn.9.h9", node_type="attention_head", layer=9, index=9,
                   label="NM 9.9")
    graph.add_node("attn.9.h6", node_type="attention_head", layer=9, index=6,
                   label="NM 9.6")
    graph.add_node("attn.10.h0", node_type="attention_head", layer=10, index=0,
                   label="NM 10.0")

    # -- MLP layers --
    graph.add_node("mlp.9", node_type="mlp_layer", layer=9, index=0,
                   label="MLP 9")
    graph.add_node("mlp.10", node_type="mlp_layer", layer=10, index=0,
                   label="MLP 10")

    # -- Output --
    graph.add_node("unembed", node_type="unembedding", layer=11, index=0,
                   label="Logits")

    # ==== Edges (based on the IOI circuit paper) ====
    # Embedding -> duplicate token heads
    graph.add_edge("embed.tok", "attn.0.h1", weight=0.6)
    graph.add_edge("embed.tok", "attn.0.h10", weight=0.5)

    # Duplicate -> previous token
    graph.add_edge("attn.0.h1", "attn.2.h2", weight=0.7)
    graph.add_edge("attn.0.h10", "attn.2.h2", weight=0.55)

    # Previous -> induction heads
    graph.add_edge("attn.2.h2", "attn.5.h5", weight=0.8)
    graph.add_edge("attn.2.h2", "attn.6.h9", weight=0.75)

    # Induction -> S-inhibition
    graph.add_edge("attn.5.h5", "attn.7.h3", weight=0.65)
    graph.add_edge("attn.5.h5", "attn.7.h9", weight=0.6)
    graph.add_edge("attn.6.h9", "attn.8.h6", weight=0.7)
    graph.add_edge("attn.6.h9", "attn.8.h10", weight=0.6)

    # S-inhibition -> name movers
    graph.add_edge("attn.7.h3", "attn.9.h9", weight=0.85)
    graph.add_edge("attn.7.h9", "attn.9.h6", weight=0.8)
    graph.add_edge("attn.8.h6", "attn.10.h0", weight=0.9)
    graph.add_edge("attn.8.h10", "attn.10.h0", weight=0.7)

    # Name movers -> MLP / output
    graph.add_edge("attn.9.h9", "mlp.9", weight=0.5)
    graph.add_edge("attn.9.h6", "mlp.9", weight=0.45)
    graph.add_edge("mlp.9", "mlp.10", weight=0.6)
    graph.add_edge("attn.10.h0", "mlp.10", weight=0.55)

    graph.add_edge("mlp.10", "unembed", weight=0.9)
    graph.add_edge("attn.9.h9", "unembed", weight=0.75)
    graph.add_edge("attn.10.h0", "unembed", weight=0.85)

    return graph


def render_circuit_scene():
    """Render the IOI circuit graph scene."""
    from manim import tempconfig

    from glassboxllms.visualization.adapters import circuit_graph_to_scene_data
    from glassboxllms.visualization.scenes import CircuitDiscoveryScene

    graph = build_ioi_circuit_graph()
    scene_data = circuit_graph_to_scene_data(graph)
    print(f"  Circuit: {graph}")

    media_dir = str(DEMO_DIR / "_manim_circuit")
    with tempconfig({
        "quality": "high_quality",
        "pixel_width": 1920,
        "pixel_height": 1080,
        "media_dir": media_dir,
    }):
        scene = CircuitDiscoveryScene()
        scene.scene_data = scene_data
        scene.render()

    _move_video("CircuitDiscoveryScene", Path(media_dir))


# ===================================================================
# Scene 3: SteeringVectorScene
# ===================================================================

def render_steering_scene(model, tokenizer, device):
    """
    Compute a real sentiment steering vector from GPT-2 h.10 activations
    and render the before/after scatter scene.
    """
    from manim import tempconfig

    from glassboxllms.visualization.adapters import steering_result_to_scene_data
    from glassboxllms.visualization.scenes import SteeringVectorScene

    texts, labels = build_sentiment_data()
    layer_name = "transformer.h.10"

    print("  Extracting activations from GPT-2 layer h.10 ...")
    activations = extract_activations(model, tokenizer, texts, layer_name, device)

    pos_mask = labels == 1
    neg_mask = labels == 0
    pos_acts = activations[pos_mask]
    neg_acts = activations[neg_mask]

    # Steering vector: difference in class means
    steering_vector = pos_acts.mean(axis=0) - neg_acts.mean(axis=0)
    print(f"  Steering vector norm: {np.linalg.norm(steering_vector):.4f}")

    # Simulate "after steering": shift all activations along the steering direction
    strength = 1.5
    steered_activations = activations + strength * steering_vector

    # Adapt to scene data
    scene_data = steering_result_to_scene_data(
        steering_vector=steering_vector,
        activations_before=activations,
        activations_after=steered_activations,
        direction_name="sentiment",
        layer="h.10",
        labels=labels,
        strength=strength,
    )

    media_dir = str(DEMO_DIR / "_manim_steering")
    with tempconfig({
        "quality": "high_quality",
        "pixel_width": 1920,
        "pixel_height": 1080,
        "media_dir": media_dir,
    }):
        scene = SteeringVectorScene()
        scene.scene_data = scene_data
        scene.render()

    _move_video("SteeringVectorScene", Path(media_dir))


# ===================================================================
# Main
# ===================================================================

def main():
    overall_start = time.time()
    print("=" * 60)
    print("  Rendering 3 demo animations with real GPT-2 data")
    print(f"  Output directory: {DEMO_DIR}")
    print("=" * 60)

    # Load model once (shared by scenes 1 and 3)
    with _timer("Loading GPT-2"):
        model, tokenizer, device = load_gpt2()

    # Scene 1: Probing
    with _timer("Scene 1/3: ProbingHyperplaneScene"):
        render_probing_scene(model, tokenizer, device)

    # Scene 2: Circuit (does not need GPT-2 inference)
    with _timer("Scene 2/3: CircuitDiscoveryScene"):
        render_circuit_scene()

    # Scene 3: Steering
    with _timer("Scene 3/3: SteeringVectorScene"):
        render_steering_scene(model, tokenizer, device)

    # Summary
    elapsed = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"  All 3 scenes rendered in {elapsed:.1f}s")
    print(f"{'='*60}")
    for name in ["ProbingHyperplaneScene", "CircuitDiscoveryScene", "SteeringVectorScene"]:
        mp4 = DEMO_DIR / f"{name}.mp4"
        status = "OK" if mp4.exists() else "MISSING"
        print(f"  [{status}] {mp4}")
    print()


if __name__ == "__main__":
    main()
