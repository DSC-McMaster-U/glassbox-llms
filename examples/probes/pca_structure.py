"""
PCA Probe Example: Activation Space Structure

This example uses PCA to analyze the structure of activation space:
- What are the principal directions of variance?
- How much variance is captured by top components?
- Do activations cluster by semantic category?

Key question: "What is the geometric structure of the activation space?"

Unlike classification/regression probes, PCA is UNSUPERVISED — no labels needed.
It reveals the intrinsic structure of how the model organizes representations.

Usage:
    python -m probes.examples.pca_structure
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from transformers import AutoModel, AutoTokenizer

from probes import LinearProbe, ActivationStore


def main():
    print("=" * 60)
    print("PCA Probe: Activation Space Structure Analysis")
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
    # 2. Prepare Diverse Sentences (Multiple Categories)
    # -------------------------------------------------------------------------
    print("\n[2] Preparing diverse sentence categories")
    
    # Sentences from different semantic categories
    # We'll see if PCA reveals this structure WITHOUT using labels
    sentences_by_category = {
        "animals": [
            "The cat slept on the warm windowsill.",
            "Dogs love to play fetch in the park.",
            "The elephant walked slowly across the savanna.",
            "Birds sing beautiful songs at dawn.",
        ],
        "food": [
            "The pizza was hot and delicious.",
            "I love eating fresh strawberries in summer.",
            "The chef prepared an amazing pasta dish.",
            "Coffee tastes better with a little cream.",
        ],
        "technology": [
            "The new smartphone has an amazing camera.",
            "Computers have become incredibly powerful.",
            "Software engineers write code every day.",
            "The internet connects people worldwide.",
        ],
        "nature": [
            "The mountains were covered in snow.",
            "Ocean waves crashed against the rocks.",
            "The forest was quiet and peaceful.",
            "Flowers bloom beautifully in spring.",
        ],
    }
    
    # Flatten for processing
    all_texts = []
    all_categories = []
    category_names = list(sentences_by_category.keys())
    
    for cat, texts in sentences_by_category.items():
        all_texts.extend(texts)
        all_categories.extend([cat] * len(texts))
    
    # Convert to numeric labels for visualization
    category_to_idx = {cat: i for i, cat in enumerate(category_names)}
    all_labels = np.array([category_to_idx[c] for c in all_categories])

    print(f"    Categories: {category_names}")
    print(f"    Total sentences: {len(all_texts)}")

    # -------------------------------------------------------------------------
    # 3. PCA Analysis Across Layers
    # -------------------------------------------------------------------------
    print("\n[3] Analyzing activation structure with PCA")
    
    layers = [f"transformer.layer.{i}" for i in range(6)]
    n_components = 2  # For visualization
    
    print(f"\n    Extracting top {n_components} principal components per layer:")
    print("-" * 60)
    
    layer_results = {}
    
    for layer in layers:
        # Extract activations
        acts = store.extract(all_texts, tokenizer, layers=[layer], pooling="mean")
        X = acts[layer]
        
        # Fit PCA probe (UNSUPERVISED - no labels!)
        pca_probe = LinearProbe(
            layer=layer,
            direction="structure",
            model_type="pca",
            n_components=n_components,
            normalize=True,
        )
        pca_probe.fit(X)  # No labels needed!
        
        # Get explained variance
        components = pca_probe.get_direction()
        result = pca_probe.evaluate(X, np.zeros(len(X)))  # Dummy labels for API
        explained_var = result.explained_variance
        
        # Project activations to 2D
        projected = pca_probe.predict(X)  # Shape: (n_samples, 2)
        
        layer_results[layer] = {
            "explained_variance": explained_var,
            "projected": projected,
            "components": components,
        }
        
        # Print variance explained
        print(f"    {layer}: explained variance = {explained_var:.3f} ({explained_var*100:.1f}%)")

    # -------------------------------------------------------------------------
    # 4. Analyze Clustering
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CLUSTERING ANALYSIS (Do categories cluster in PC space?)")
    print("=" * 60)
    
    # Use the last layer (typically most semantic)
    final_layer = layers[-1]
    projected = layer_results[final_layer]["projected"]
    
    print(f"\nLayer: {final_layer}")
    print("\nCategory centroids in PC1-PC2 space:")
    print("-" * 40)
    
    centroids = {}
    for cat in category_names:
        mask = np.array(all_categories) == cat
        cat_points = projected[mask]
        centroid = cat_points.mean(axis=0)
        centroids[cat] = centroid
        print(f"    {cat:12s}: PC1={centroid[0]:+.3f}, PC2={centroid[1]:+.3f}")
    
    # Compute inter-category distances
    print("\nInter-category distances:")
    print("-" * 40)
    for i, cat1 in enumerate(category_names):
        for cat2 in category_names[i+1:]:
            dist = np.linalg.norm(centroids[cat1] - centroids[cat2])
            print(f"    {cat1} ↔ {cat2}: {dist:.3f}")

    # -------------------------------------------------------------------------
    # 5. ASCII Scatter Plot
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ASCII SCATTER PLOT (PC1 vs PC2)")
    print("=" * 60)
    
    # Simple ASCII visualization
    width, height = 50, 20
    grid = [[" " for _ in range(width)] for _ in range(height)]
    
    # Normalize to grid coordinates
    pc1 = projected[:, 0]
    pc2 = projected[:, 1]
    pc1_norm = (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-8)
    pc2_norm = (pc2 - pc2.min()) / (pc2.max() - pc2.min() + 1e-8)
    
    symbols = {"animals": "A", "food": "F", "technology": "T", "nature": "N"}
    
    for i, (x, y) in enumerate(zip(pc1_norm, pc2_norm)):
        col = int(x * (width - 1))
        row = int((1 - y) * (height - 1))  # Flip y-axis
        cat = all_categories[i]
        grid[row][col] = symbols[cat]
    
    print(f"\nLegend: A=animals, F=food, T=technology, N=nature")
    print("+" + "-" * width + "+")
    for row in grid:
        print("|" + "".join(row) + "|")
    print("+" + "-" * width + "+")
    print(f"  PC1 →")

    # -------------------------------------------------------------------------
    # 6. Interpretation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    
    final_var = layer_results[final_layer]["explained_variance"]
    print(f"\nTop 2 PCs explain {final_var*100:.1f}% of variance at {final_layer}.")
    
    if final_var > 0.5:
        print("→ High concentration: activations lie mostly in a low-dimensional subspace.")
    elif final_var > 0.2:
        print("→ Moderate concentration: some structure exists but space is complex.")
    else:
        print("→ Low concentration: activations are spread across many dimensions.")
    
    print("\nVariance explained progression (layer 0 → 5):")
    for i, layer in enumerate(layers):
        var = layer_results[layer]["explained_variance"]
        bar = "█" * int(var * 40)
        print(f"  Layer {i}: {bar} {var*100:.1f}%")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
