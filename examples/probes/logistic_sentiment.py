"""
Logistic Probe Example: Sentiment Classification

This example trains a logistic probe to measure how well sentiment
(positive vs negative) is linearly encoded across different layers
of a DistilBERT model.

Key question: "At which layer is sentiment most linearly separable?"

Usage:
    python -m probes.examples.logistic_sentiment
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from transformers import AutoModel, AutoTokenizer

from probes import LinearProbe, ActivationStore


def main():
    print("=" * 60)
    print("Logistic Probe: Sentiment Classification")
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
    print(f"    Available layers: {len(store.list_layers())} modules")

    # -------------------------------------------------------------------------
    # 2. Prepare Labeled Data
    # -------------------------------------------------------------------------
    print("\n[2] Preparing sentiment dataset")
    
    # Training sentences with clear sentiment
    train_texts = [
        # Positive (label = 1)
        "I absolutely love this movie, it was fantastic!",
        "This is the best day of my life, I'm so happy!",
        "What a wonderful experience, highly recommended!",
        "The food was delicious and the service was excellent.",
        "I'm thrilled with the results, exceeded expectations!",
        "This book changed my life in the best way possible.",
        "Amazing performance, the actors were brilliant!",
        "I can't stop smiling, this made my day!",
        # Negative (label = 0)
        "I hate this movie, it was terrible and boring.",
        "This is the worst day ever, everything went wrong.",
        "What a horrible experience, never going back!",
        "The food was disgusting and the service was awful.",
        "I'm so disappointed with the results, total failure.",
        "This book was a waste of time, completely useless.",
        "Terrible performance, the actors were embarrassing.",
        "I'm so frustrated, this ruined my entire day!",
    ]
    train_labels = [1] * 8 + [0] * 8  # 8 positive, 8 negative

    # Test sentences (held-out)
    test_texts = [
        "I really enjoyed the concert, it was magical!",       # Positive
        "The vacation was perfect, we had so much fun!",       # Positive
        "I regret buying this product, complete waste.",       # Negative
        "The movie was so boring I fell asleep.",              # Negative
    ]
    test_labels = [1, 1, 0, 0]

    print(f"    Training samples: {len(train_texts)} ({sum(train_labels)} pos, {len(train_labels) - sum(train_labels)} neg)")
    print(f"    Test samples: {len(test_texts)}")

    # -------------------------------------------------------------------------
    # 3. Probe Each Layer
    # -------------------------------------------------------------------------
    print("\n[3] Probing sentiment encoding across layers")
    
    # DistilBERT has 6 transformer layers
    layers = [f"transformer.layer.{i}" for i in range(6)]
    
    results = {}
    best_layer, best_acc = None, 0
    
    for layer in layers:
        # Extract activations
        train_acts = store.extract(train_texts, tokenizer, layers=[layer], pooling="mean")
        test_acts = store.extract(test_texts, tokenizer, layers=[layer], pooling="mean")
        
        # Train probe
        probe = LinearProbe(
            layer=layer,
            direction="sentiment",
            model_type="logistic",
            normalize=True,
        )
        probe.fit(train_acts[layer], np.array(train_labels))
        
        # Evaluate
        result = probe.evaluate(test_acts[layer], np.array(test_labels))
        results[layer] = result
        
        # Track best
        if result.accuracy > best_acc:
            best_layer, best_acc = layer, result.accuracy
        
        print(f"    {layer}: accuracy={result.accuracy:.3f}, f1={result.f1:.3f}")

    # -------------------------------------------------------------------------
    # 4. Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nBest layer: {best_layer}")
    print(f"Best accuracy: {best_acc:.3f}")
    
    # Interpretation
    print("\nInterpretation:")
    if best_acc >= 0.9:
        print("  ✓ Sentiment is STRONGLY linearly encoded at the best layer.")
    elif best_acc >= 0.7:
        print("  ~ Sentiment is MODERATELY linearly encoded.")
    else:
        print("  ✗ Sentiment is WEAKLY encoded (may need more data or non-linear probe).")

    # Show layer progression
    print("\nLayer progression (early → late):")
    for i, layer in enumerate(layers):
        acc = results[layer].accuracy
        bar = "█" * int(acc * 20)
        marker = " ← best" if layer == best_layer else ""
        print(f"  Layer {i}: {bar} {acc:.2f}{marker}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
