"""
CAV Probe Example: Gender Concept Direction

This example finds the "gender direction" in activation space using
Concept Activation Vectors (CAVs). Once found, we can:
- Measure how strongly any sentence expresses "male" vs "female"
- Test if the model has gender bias in neutral contexts
- Understand what concepts the model has learned to represent

Key question: "What direction in activation space encodes gender?"

Usage:
    python -m probes.examples.cav_gender
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from transformers import AutoModel, AutoTokenizer

from probes import LinearProbe, ActivationStore


def main():
    print("=" * 60)
    print("CAV Probe: Finding the Gender Direction")
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
    # 2. Prepare Concept Examples (Male vs Female)
    # -------------------------------------------------------------------------
    print("\n[2] Preparing gender concept examples")
    
    # Training examples for learning the gender direction
    # Key: sentences should differ ONLY in gender, not in other ways
    male_sentences = [
        "The king ruled the kingdom wisely.",
        "He went to work in the morning.",
        "The father played with his son.",
        "My brother is a doctor.",
        "The waiter served the food.",
        "His majesty addressed the crowd.",
        "The boy ran across the field.",
        "The gentleman opened the door.",
    ]
    
    female_sentences = [
        "The queen ruled the kingdom wisely.",
        "She went to work in the morning.",
        "The mother played with her daughter.",
        "My sister is a doctor.",
        "The waitress served the food.",
        "Her majesty addressed the crowd.",
        "The girl ran across the field.",
        "The lady opened the door.",
    ]
    
    # Labels: 0 = male, 1 = female
    train_texts = male_sentences + female_sentences
    train_labels = np.array([0] * len(male_sentences) + [1] * len(female_sentences))
    
    print(f"    Male examples: {len(male_sentences)}")
    print(f"    Female examples: {len(female_sentences)}")

    # -------------------------------------------------------------------------
    # 3. Train CAV on Best Layer
    # -------------------------------------------------------------------------
    print("\n[3] Training CAV to find gender direction")
    
    # First, find which layer best encodes gender
    layers = [f"transformer.layer.{i}" for i in range(6)]
    
    print("\n    Probing gender encoding across layers:")
    best_layer, best_acc = None, 0
    
    for layer in layers:
        acts = store.extract(train_texts, tokenizer, layers=[layer], pooling="mean")
        
        probe = LinearProbe(
            layer=layer,
            direction="gender",
            model_type="cav",
            normalize=True,
        )
        probe.fit(acts[layer], train_labels)
        
        # Simple train accuracy (for illustration)
        preds = probe.predict(acts[layer])
        acc = (preds == train_labels).mean()
        
        if acc > best_acc:
            best_layer, best_acc = layer, acc
        
        print(f"    {layer}: train_accuracy = {acc:.3f}")
    
    print(f"\n    Best layer: {best_layer} (accuracy = {best_acc:.3f})")

    # -------------------------------------------------------------------------
    # 4. Extract the Gender Direction Vector
    # -------------------------------------------------------------------------
    print("\n[4] Extracting gender direction vector")
    
    # Train CAV on best layer
    best_acts = store.extract(train_texts, tokenizer, layers=[best_layer], pooling="mean")
    
    cav = LinearProbe(
        layer=best_layer,
        direction="gender",
        model_type="cav",
        normalize=True,
    )
    cav.fit(best_acts[best_layer], train_labels)
    
    # Get the direction vector
    gender_direction = cav.get_direction()
    
    print(f"    Direction vector shape: {gender_direction.shape}")
    print(f"    Direction norm: {np.linalg.norm(gender_direction):.3f}")
    print(f"    Top 5 components: {gender_direction[:5]}")

    # -------------------------------------------------------------------------
    # 5. Score New Sentences Along Gender Direction
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SCORING NEW SENTENCES")
    print("=" * 60)
    
    # Test sentences: some gendered, some neutral, some ambiguous
    test_sentences = [
        # Clearly male
        ("The prince attended the royal ball.", "male"),
        ("He is a strong man.", "male"),
        
        # Clearly female
        ("The princess attended the royal ball.", "female"),
        ("She is a strong woman.", "female"),
        
        # Neutral / Professional (testing for bias)
        ("The doctor performed the surgery.", "neutral"),
        ("The nurse helped the patient.", "neutral"),
        ("The engineer designed the bridge.", "neutral"),
        ("The teacher explained the lesson.", "neutral"),
        
        # Ambiguous
        ("They went to the store.", "ambiguous"),
        ("The person walked down the street.", "ambiguous"),
    ]
    
    print("\nScoring sentences along gender direction:")
    print("(Negative = male-like, Positive = female-like)")
    print("-" * 60)
    
    for sentence, expected in test_sentences:
        acts = store.extract([sentence], tokenizer, layers=[best_layer], pooling="mean")
        score = cav.score_activation(acts[best_layer][0])
        
        # Interpret score
        if score > 0.5:
            gender_pred = "female"
        elif score < -0.5:
            gender_pred = "male"
        else:
            gender_pred = "neutral"
        
        bar_pos = int(max(0, score) * 10)
        bar_neg = int(max(0, -score) * 10)
        bar = "◀" + "█" * bar_neg + "│" + "█" * bar_pos + "▶"
        
        print(f"\n  \"{sentence[:45]}...\"" if len(sentence) > 45 else f"\n  \"{sentence}\"")
        print(f"    Expected: {expected:10s} | Predicted: {gender_pred:8s} | Score: {score:+.3f}")
        print(f"    {bar}")

    # -------------------------------------------------------------------------
    # 6. Bias Analysis on Professions
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PROFESSION BIAS ANALYSIS")
    print("=" * 60)
    
    professions = [
        "The doctor examined the patient.",
        "The nurse cared for the patient.",
        "The engineer built the machine.",
        "The secretary scheduled the meeting.",
        "The CEO made the decision.",
        "The receptionist answered the phone.",
        "The pilot flew the airplane.",
        "The flight attendant served drinks.",
        "The programmer wrote the code.",
        "The scientist conducted experiments.",
    ]
    
    print("\nGender direction scores for profession sentences:")
    print("(If neutral sentences show gender bias, the model may have learned stereotypes)")
    print("-" * 60)
    
    scores = []
    for sentence in professions:
        acts = store.extract([sentence], tokenizer, layers=[best_layer], pooling="mean")
        score = cav.score_activation(acts[best_layer][0])
        scores.append((sentence, score))
    
    # Sort by score
    scores.sort(key=lambda x: x[1])
    
    print("\nRanked from most 'male-coded' to most 'female-coded':")
    for sentence, score in scores:
        direction = "←M" if score < -0.1 else ("F→" if score > 0.1 else "  ")
        bar_len = int(abs(score) * 20)
        bar = "█" * bar_len
        print(f"  {score:+.3f} {direction} {bar:20s} {sentence[:50]}")

    # -------------------------------------------------------------------------
    # 7. Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"""
    ✓ Found gender direction at {best_layer}
    ✓ Direction separates male/female examples with {best_acc*100:.1f}% accuracy
    ✓ Can now score ANY sentence along this direction
    
    Applications:
    - Detect gender bias in neutral contexts (professions, etc.)
    - Measure how strongly a sentence expresses gender
    - Compare gender encoding across models/layers
    - Potentially "steer" model outputs by modifying activations
    """)
    
    print("=" * 60)


if __name__ == "__main__":
    main()
