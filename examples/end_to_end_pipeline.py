#!/usr/bin/env python3
"""
End-to-end GlassBox LLMs Pipeline
==================================

Demonstrates the full interpretability workflow on GPT-2:

1. Load model via TransformersModelWrapper
2. Extract activations with ActivationExtractor
3. Run a logit lens analysis
4. Train a linear probe on a concept
5. Apply directional steering
6. Run causal ablation scan

Requirements:
    pip install glassboxllms torch transformers scikit-learn

Usage:
    python examples/end_to_end_pipeline.py
"""

from __future__ import annotations

import sys

import numpy as np
import torch


def main():
    print("=" * 60)
    print("  GlassBox LLMs — End-to-End Pipeline Demo")
    print("=" * 60)

    # ── Step 1: Load Model ──────────────────────────────────────
    print("\n[1/6] Loading GPT-2...")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Model: {model_name}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Step 2: Discover Layers ─────────────────────────────────
    print("\n[2/6] Discovering model layers...")

    from glassboxllms.instrumentation import get_layer_names

    all_layers = get_layer_names(model, layer_type="all")
    attn_layers = get_layer_names(model, layer_type="attention")
    mlp_layers = get_layer_names(model, layer_type="mlp")

    print(f"  Total modules: {len(all_layers)}")
    print(f"  Attention modules: {len(attn_layers)}")
    print(f"  MLP modules: {len(mlp_layers)}")

    # Find transformer blocks
    blocks = [
        name for name in all_layers
        if name.startswith("transformer.h.") and name.count(".") == 2
    ]
    print(f"  Transformer blocks: {len(blocks)}")
    for b in blocks[:3]:
        print(f"    {b}")
    if len(blocks) > 3:
        print(f"    ... and {len(blocks) - 3} more")

    # ── Step 3: Extract Activations ─────────────────────────────
    print("\n[3/6] Extracting activations...")

    from glassboxllms.instrumentation import ActivationExtractor

    extractor = ActivationExtractor(model.transformer)

    texts = [
        "I love this movie, it was fantastic!",
        "This film was terrible and boring.",
        "The weather is nice today.",
        "I hate waiting in long lines.",
        "What a wonderful surprise!",
        "This is the worst experience ever.",
    ]

    # Extract from a middle layer
    target_layer = "h.6"
    activations = extractor.extract(
        texts=texts,
        tokenizer=tokenizer,
        layers=[target_layer],
        pooling="mean",
        return_type="numpy",
    )

    acts = activations[target_layer]
    print(f"  Extracted shape: {acts.shape}")
    print(f"  Layer: {target_layer}")
    print(f"  Texts processed: {len(texts)}")

    # ── Step 4: Train a Linear Probe ────────────────────────────
    print("\n[4/6] Training sentiment probe...")

    from glassboxllms.primitives.probes import LinearProbe

    # Simple sentiment labels: 1 = positive, 0 = neutral/negative
    labels = np.array([1, 0, 1, 0, 1, 0])

    probe = LinearProbe(probe_type="classification")
    result = probe.train(acts, labels)

    print(f"  Probe type: classification (logistic regression)")
    print(f"  Accuracy: {result.accuracy:.1%}")
    print(f"  Direction shape: {probe.get_direction().shape}")

    # ── Step 5: Directional Steering ────────────────────────────
    print("\n[5/6] Steering model output...")

    from glassboxllms.interventions import DirectionalSteering

    # Use the probe direction for steering
    direction = torch.tensor(probe.get_direction(), dtype=torch.float32)

    steering = DirectionalSteering(
        layer="transformer.h.6",
        direction=direction,
        strength=3.0,
    )

    test_text = "The movie was"
    inputs = tokenizer(test_text, return_tensors="pt")

    # Clean output
    with torch.no_grad():
        clean_out = model(**inputs)
    clean_probs = torch.softmax(clean_out.logits[0, -1], dim=-1)
    clean_top5 = clean_probs.topk(5)

    print(f"  Input: '{test_text}'")
    print(f"  Clean top-5 predictions:")
    for tok_id, prob in zip(clean_top5.indices, clean_top5.values):
        print(f"    {tokenizer.decode([tok_id]):>15}  {prob:.1%}")

    # Steered output
    steering.register(model)
    with torch.no_grad():
        steered_out = model(**inputs)
    steering.remove(model)

    steered_probs = torch.softmax(steered_out.logits[0, -1], dim=-1)
    steered_top5 = steered_probs.topk(5)

    print(f"  Steered top-5 predictions (strength=3.0):")
    for tok_id, prob in zip(steered_top5.indices, steered_top5.values):
        print(f"    {tokenizer.decode([tok_id]):>15}  {prob:.1%}")

    # ── Step 6: Logit Lens ──────────────────────────────────────
    print("\n[6/6] Running logit lens...")

    from glassboxllms.experiments import run_experiment

    lens_result = run_experiment("logit_lens", {
        "model_name": model_name,
        "text": "The capital of France is",
        "top_k": 3,
    })

    print(f"  Input: 'The capital of France is'")
    print(f"  Final prediction: {lens_result.metrics['final_top_token']!r}")
    print(f"  Convergence: layer {lens_result.metrics['convergence_layer']}"
          f" / {lens_result.metrics['n_layers_analyzed']}")

    predictions = lens_result.artifacts.get("layer_predictions", [])
    print(f"\n  {'Layer':<30} {'Top-1':>10} {'Prob':>8}")
    print("  " + "-" * 50)
    for lp in predictions:
        tok = lp["top_tokens"][0]
        prob = lp["top_probs"][0]
        marker = " <--" if tok.strip() == lens_result.metrics.get("final_top_token", "").strip() else ""
        print(f"  {lp['layer']:<30} {tok:>10} {prob:>7.1%}{marker}")

    # ── Done ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
