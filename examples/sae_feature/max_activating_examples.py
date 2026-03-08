"""
Max Activating Examples: Finding What Turns Features On

This example demonstrates Method 1 for understanding SAE features:
Run many diverse texts through the model and find which inputs
maximally activate each feature.

Key Question: "What inputs cause Feature #X to activate most strongly?"

Usage:
    python -m examples.sae_feature.max_activating_examples
"""

import random
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer, GPT2LMHeadModel
from torch.utils.data import TensorDataset, DataLoader

from glassboxllms.instrumentation.activations import ActivationStore
from glassboxllms.features import SparseAutoencoder, SAETrainer

# Support both direct execution and module execution
try:
    from .dataset import create_diverse_dataset
except ImportError:
    from dataset import create_diverse_dataset


def truncate_for_display(text, max_len=100):
    """
    Truncate text for readable display.
    
    Tries to cut at sentence boundary if possible.
    """
    if len(text) <= max_len:
        return text
    
    # Try to cut at sentence boundary
    truncated = text[:max_len]
    
    # Look for last period, exclamation, or question mark
    for punct in ['. ', '! ', '? ']:
        last_punct = truncated.rfind(punct)
        if last_punct > max_len * 0.5:  # At least halfway through
            return truncated[:last_punct + 1]
    
    # No good break point, just truncate with ellipsis
    return truncated.rstrip() + "..."


def select_diverse_features(sae, candidate_indices, n_select=3, max_similarity=0.7):
    """
    Select features that are far apart in decoder space (low pairwise similarity).

    Uses greedy selection: pick first candidate, then repeatedly add the candidate
    that has cosine similarity < max_similarity with all already-selected features.

    Args:
        sae: Trained SparseAutoencoder (has W_dec).
        candidate_indices: 1D tensor or list of feature indices to choose from.
        n_select: Number of features to select.
        max_similarity: Maximum allowed cosine similarity between any two selected (default 0.7).

    Returns:
        List of n_select feature indices (or fewer if not enough diverse candidates).
    """
    import torch.nn.functional as F

    W_dec = sae.W_dec.data  # [feature_dim, input_dim]
    candidates = list(candidate_indices)

    if len(candidates) < n_select:
        return candidates

    # Normalize decoder rows for cosine similarity
    W_dec_norm = F.normalize(W_dec.float(), p=2, dim=1)

    selected = [candidates[0]]
    remaining = candidates[1:]

    while len(selected) < n_select and remaining:
        best_idx = None
        best_max_sim = 2.0  # want candidate with smallest max_sim to selected

        for i, cand in enumerate(remaining):
            cand_vec = W_dec_norm[cand].unsqueeze(0)  # [1, input_dim]
            selected_vecs = W_dec_norm[torch.tensor(selected)]  # [len(selected), input_dim]
            sims = F.cosine_similarity(cand_vec, selected_vecs, dim=1)  # [len(selected)]
            max_sim_to_selected = sims.max().item()

            if max_sim_to_selected < max_similarity:
                # This candidate is diverse enough; take first valid one
                best_idx = i
                break

            # Fallback: track candidate with smallest max similarity to selected
            if max_sim_to_selected < best_max_sim:
                best_max_sim = max_sim_to_selected
                best_idx = i

        if best_idx is None:
            break

        chosen = remaining.pop(best_idx)
        selected.append(chosen)

    return selected[:n_select]


def main():
    print("=" * 70)
    print("METHOD 1: MAX ACTIVATING EXAMPLES")
    print("Finding What Inputs Turn Features On")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # 1. Load Model
    # -------------------------------------------------------------------------
    print("\n[1] Loading GPT-2 model")
    model_name = "gpt2"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    
    print(f"    Model: {model_name}")
    print(f"    Hidden size: {model.config.n_embd}")
    print(f"    Layers: {model.config.n_layer}")
    
    # -------------------------------------------------------------------------
    # 2. Prepare Dataset
    # -------------------------------------------------------------------------
    print("\n[2] Loading diverse dataset")
    texts = create_diverse_dataset()
    
    # -------------------------------------------------------------------------
    # 3. Capture Activations
    # -------------------------------------------------------------------------
    print("\n[3] Capturing activations from target layer")
    layer_name = "transformer.h.6"  # Middle layer for semantic features
    print(f"    Target layer: {layer_name}")
    
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ActivationStore(device="cpu", storage_dir=tmpdir)
        
        print(f"    Processing {len(texts)} texts...")
        for i, text in enumerate(texts):
            if (i + 1) % 100 == 0:
                print(f"      Processed {i + 1}/{len(texts)} texts...")
            
            # Tokenize and get activations
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                # Get layer 6 activations, average over sequence
                layer_acts = outputs.hidden_states[7].mean(dim=1)  # [1, hidden_size]
                store.save(layer_name, layer_acts)
        
        # Load all activations
        activations = store.get_all(layer_name).squeeze(1)  # [n_texts, hidden_size]
    
    print(f"    ✓ Captured activations: {activations.shape}")
    
    # -------------------------------------------------------------------------
    # 4. Train SAE
    # -------------------------------------------------------------------------
    print("\n[4] Training Sparse Autoencoder")
    
    input_dim = activations.shape[1]
    feature_dim = 4096  # 4x expansion
    k = 64  # Top-k sparsity
    
    sae = SparseAutoencoder(
        input_dim=input_dim,
        feature_dim=feature_dim,
        k=k,
        sparsity_mode="topk"
    )
    
    print(f"    SAE config: {input_dim}d → {feature_dim}d (k={k})")
    
    # Initialize with geometric median
    sae.initialize_geometric_median(activations[:50])
    
    # Create dataloader
    dataset = TensorDataset(activations)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Train
    trainer = SAETrainer(
        sae,
        dataloader,
        aux_coef=1e-3,
        log_every=20,
    )
    
    print("    Training for 3 epochs...")
    stats = trainer.train(n_epochs=9)
    
    print(f"    ✓ Training complete")
    print(f"      Final explained variance: {stats['final_explained_variance']:.3f}")
    print(f"      Mean L0: {stats['mean_l0']:.1f}")
    
    # -------------------------------------------------------------------------
    # 5. Find Max Activating Examples
    # -------------------------------------------------------------------------
    print("\n[5] Finding max activating examples for features")
    
    # Run SAE on all activations
    print("    Running SAE on all texts...")
    with torch.no_grad():
        # Get sparse features for all activations
        feature_activations = []
        for i in range(0, len(activations), 16):
            batch = activations[i:i+16]
            _, features, _ = sae(batch)
            feature_activations.append(features)
        
        feature_activations = torch.cat(feature_activations, dim=0)  # [n_texts, feature_dim]
    
    print(f"    Feature activations shape: {feature_activations.shape}")
    
    # Select interesting features to analyze
    # Find features with moderate sparsity (not dead, not too common)
    feature_frequencies = (feature_activations > 0).float().mean(dim=0)
    interesting_features = torch.where(
        (feature_frequencies > 0.05) & (feature_frequencies < 0.5)
    )[0]
    
    if len(interesting_features) < 3:
        # Fallback: just use first few features
        selected_features = [5, 42, 100] if feature_dim > 100 else [0, 1, 2]
    else:
        # Select 3 diverse features (decoder cosine similarity < 0.7)
        selected_features = select_diverse_features(
            sae, interesting_features, n_select=3, max_similarity=0.7
        )
        # If we got fewer than 3, fill with random from remaining interesting
        if len(selected_features) < 3:
            remaining = [i for i in interesting_features.tolist() if i not in selected_features]
            k = min(3 - len(selected_features), len(remaining))
            if k and remaining:
                selected_features = selected_features + random.sample(remaining, k)
            if len(selected_features) < 3:
                fallback = [5, 42, 100] if feature_dim > 100 else [0, 1, 2]
                for x in fallback:
                    if len(selected_features) >= 3:
                        break
                    if x not in selected_features:
                        selected_features.append(x)

    print(f"    Analyzing features: {selected_features}")
    
    # -------------------------------------------------------------------------
    # 6. Display Results
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS: MAX ACTIVATING EXAMPLES")
    print("=" * 70)
    
    for feature_idx in selected_features:
        # Get activations for this feature across all texts
        activations_for_feature = feature_activations[:, feature_idx]
        
        # Find top 10 activating texts
        top_k = 10
        top_values, top_indices = torch.topk(activations_for_feature, k=top_k)
        
        # Get corresponding texts
        top_texts = [texts[idx.item()] for idx in top_indices]
        top_activations = top_values.tolist()
        
        # Calculate statistics
        max_act = activations_for_feature.max().item()
        mean_act = activations_for_feature[activations_for_feature > 0].mean().item() if (activations_for_feature > 0).any() else 0
        sparsity = (activations_for_feature > 0).float().mean().item()
        
        # Combine texts and activations
        top_10 = list(zip(top_texts, top_activations))
        
        print(f"\n{'─' * 70}")
        print(f"FEATURE #{feature_idx}")
        print(f"{'─' * 70}")
        print(f"Statistics:")
        print(f"  Max activation: {max_act:.2f}")
        print(f"  Mean (when active): {mean_act:.2f}")
        print(f"  Sparsity: {sparsity:.2%} of inputs activate this feature")
        print(f"\nTop 10 Activating Texts (truncated for readability):")
        
        for i, (text, activation) in enumerate(top_10, 1):
            display_text = truncate_for_display(text, max_len=100)
            print(f"  {i}. [{activation:6.2f}] {display_text}")
    
    # -------------------------------------------------------------------------
    # 7. Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nWhat We Did:")
    print("  1. Loaded GPT-2 and captured activations from layer 6")
    print("  2. Trained an SAE to find sparse features")
    print(f"  3. Found {feature_dim} features (with k={k} sparsity)")
    print("  4. Identified max activating examples for selected features")
    print("\nHow to Interpret:")
    print("  - Higher activation = stronger presence of the concept")
    print("  - Look for patterns across top activating texts")
    print("  - Features should be monosemantic (one concept)")
    print("\nNext Steps:")
    print("  - Try with your own dataset (edit create_diverse_dataset())")
    print("  - Analyze more features to find interesting patterns")
    print("  - Compare with Method 2: decoder_analysis.py")
    print("  - Use these features for steering or interpretation")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
