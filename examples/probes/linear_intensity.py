"""
Linear Probe Example: Sentiment Intensity Regression

This example trains a linear regression probe to predict continuous
sentiment intensity scores (0.0 = very negative → 1.0 = very positive).

Key question: "Can we predict sentiment INTENSITY (not just class) from activations?"

Unlike logistic probes (binary/multiclass), this probe predicts a continuous value,
useful for graded concepts like toxicity scores, confidence levels, or intensity.

Usage:
    python -m probes.examples.linear_intensity
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from transformers import AutoModel, AutoTokenizer

from probes import LinearProbe, ActivationStore


def main():
    print("=" * 60)
    print("Linear Probe: Sentiment Intensity Regression")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. Load Model
    # -------------------------------------------------------------------------
    model_name = "distilbert-base-uncased"
    print(f"\n[1] Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    store = ActivationStore(model)

    # -------------------------------------------------------------------------
    # 2. Prepare Graded Intensity Data
    # -------------------------------------------------------------------------
    print("\n[2] Preparing intensity-graded dataset")
    
    # Sentences with graded intensity scores (0.0 = very negative, 1.0 = very positive)
    # This is more nuanced than binary classification
    train_data = [
        # Very negative (0.0 - 0.2)
        ("I absolutely hate this, it's the worst thing ever!", 0.05),
        ("This is terrible, a complete disaster.", 0.10),
        ("I'm furious, this ruined everything.", 0.15),
        
        # Somewhat negative (0.2 - 0.4)
        ("I'm disappointed, it didn't meet expectations.", 0.25),
        ("Not great, I expected better.", 0.30),
        ("It was okay but had several problems.", 0.35),
        
        # Neutral (0.4 - 0.6)
        ("It's fine, nothing special.", 0.45),
        ("Average experience, neither good nor bad.", 0.50),
        ("It was acceptable, met basic requirements.", 0.55),
        
        # Somewhat positive (0.6 - 0.8)
        ("Pretty good, I enjoyed it.", 0.65),
        ("Nice experience, would consider again.", 0.70),
        ("I liked it, solid performance.", 0.75),
        
        # Very positive (0.8 - 1.0)
        ("Excellent! Highly recommended.", 0.85),
        ("I loved it, absolutely wonderful!", 0.90),
        ("This is amazing, the best I've ever seen!", 0.95),
    ]
    
    train_texts = [t[0] for t in train_data]
    train_scores = np.array([t[1] for t in train_data])
    
    # Test data (held-out)
    test_data = [
        ("Terrible quality, very disappointing.", 0.15),
        ("It was just okay, nothing memorable.", 0.50),
        ("Great job, I'm impressed!", 0.85),
        ("Meh, could be better.", 0.40),
    ]
    test_texts = [t[0] for t in test_data]
    test_scores = np.array([t[1] for t in test_data])

    print(f"    Training samples: {len(train_texts)}")
    print(f"    Score range: [{train_scores.min():.2f}, {train_scores.max():.2f}]")
    print(f"    Test samples: {len(test_texts)}")

    # -------------------------------------------------------------------------
    # 3. Probe Each Layer
    # -------------------------------------------------------------------------
    print("\n[3] Probing intensity encoding across layers")
    
    layers = [f"transformer.layer.{i}" for i in range(6)]
    
    results = {}
    best_layer, best_r2 = None, -float("inf")
    
    for layer in layers:
        # Extract activations
        train_acts = store.extract(train_texts, tokenizer, layers=[layer], pooling="mean")
        test_acts = store.extract(test_texts, tokenizer, layers=[layer], pooling="mean")
        
        # Train linear regression probe
        probe = LinearProbe(
            layer=layer,
            direction="intensity",
            model_type="linear",  # ← Regression, not classification
            normalize=True,
            regularization=1.0,
        )
        probe.fit(train_acts[layer], train_scores)
        
        # Evaluate
        result = probe.evaluate(test_acts[layer], test_scores)
        results[layer] = result
        
        # R² score (coefficient of determination)
        r2 = result.accuracy  # For linear probe, accuracy = R²
        mse = result.metadata.get("mse", 0)
        
        if r2 > best_r2:
            best_layer, best_r2 = layer, r2
        
        print(f"    {layer}: R²={r2:.3f}, MSE={mse:.4f}")

    # -------------------------------------------------------------------------
    # 4. Detailed Results on Best Layer
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DETAILED RESULTS (Best Layer)")
    print("=" * 60)
    
    # Retrain on best layer for predictions
    best_train_acts = store.extract(train_texts, tokenizer, layers=[best_layer], pooling="mean")
    best_test_acts = store.extract(test_texts, tokenizer, layers=[best_layer], pooling="mean")
    
    best_probe = LinearProbe(layer=best_layer, direction="intensity", model_type="linear")
    best_probe.fit(best_train_acts[best_layer], train_scores)
    
    predictions = best_probe.predict(best_test_acts[best_layer])
    
    print(f"\nBest layer: {best_layer} (R² = {best_r2:.3f})")
    print("\nTest predictions vs actual:")
    print("-" * 60)
    for i, (text, actual) in enumerate(test_data):
        pred = predictions[i]
        error = abs(pred - actual)
        print(f"  Text: \"{text[:40]}...\"")
        print(f"  Actual: {actual:.2f} | Predicted: {pred:.2f} | Error: {error:.2f}")
        print()

    # -------------------------------------------------------------------------
    # 5. Interpretation
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    
    if best_r2 >= 0.7:
        print("\n  ✓ Intensity is STRONGLY linearly encoded.")
        print("    The model captures nuanced sentiment gradations.")
    elif best_r2 >= 0.4:
        print("\n  ~ Intensity is MODERATELY linearly encoded.")
        print("    The model captures some intensity differences.")
    else:
        print("\n  ✗ Intensity is WEAKLY encoded.")
        print("    The model may only encode binary sentiment, not gradations.")

    print("\nLayer progression (R² scores):")
    for i, layer in enumerate(layers):
        r2 = results[layer].accuracy
        bar = "█" * int(max(0, r2) * 20)
        marker = " ← best" if layer == best_layer else ""
        print(f"  Layer {i}: {bar} {r2:.2f}{marker}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
