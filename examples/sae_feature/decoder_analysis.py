"""
Decoder Analysis: Understanding Features Through Vocabulary

This example demonstrates Method 2 for understanding SAE features:
Analyze the decoder weights (W_dec) and project them to vocabulary
using the "Logit Lens" technique.

Key Question: "What vocabulary does Feature #X predict?"

Usage:
    python -m examples.sae_feature.decoder_analysis
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from transformers import AutoTokenizer, GPT2LMHeadModel
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from glassboxllms.instrumentation.activations import ActivationStore
from glassboxllms.features import SparseAutoencoder, SAETrainer

# Support both direct execution and module execution
try:
    from .dataset import create_diverse_dataset
except ImportError:
    from dataset import create_diverse_dataset


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
    print("METHOD 2: DECODER WEIGHT ANALYSIS")
    print("Understanding Features Through Vocabulary (Logit Lens)")
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
    print(f"    Vocabulary size: {model.config.vocab_size}")
    
    # -------------------------------------------------------------------------
    # 2. SAE Training (Improved)
    # -------------------------------------------------------------------------
    print("\n[2] Training SAE with increased epochs for better convergence")
    
    texts = create_diverse_dataset()
    layer_name = "transformer.h.6"
    
    print(f"    Dataset: {len(texts)} texts")
    print(f"    Target layer: {layer_name}")
    
    # Capture activations
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ActivationStore(device="cpu", storage_dir=tmpdir)
        
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                layer_acts = outputs.hidden_states[7].mean(dim=1)
                store.save(layer_name, layer_acts)
        
        activations = store.get_all(layer_name).squeeze(1)
    
    print(f"    Activations shape: {activations.shape}")
    
    # Train SAE with more epochs for better convergence
    input_dim = activations.shape[1]
    feature_dim = 4096  # Smaller for faster training
    k = 64
    
    sae = SparseAutoencoder(
        input_dim=input_dim,
        feature_dim=feature_dim,
        k=k,
        sparsity_mode="topk"
    )
    
    sae.initialize_geometric_median(activations[:20])
    
    dataset = TensorDataset(activations)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)  # Smaller batch = more steps
    
    trainer = SAETrainer(sae, dataloader, aux_coef=1e-3, log_every=50)
    
    print("    Training SAE (15 epochs for better convergence)...")
    stats = trainer.train(n_epochs=10)
    print(f"    ✓ Training complete")
    print(f"      Final explained variance: {stats['final_explained_variance']:.3f}")
    print(f"      Mean L0: {stats.get('mean_l0', 'N/A')}")
    
    # -------------------------------------------------------------------------
    # 3. Extract Decoder Weights
    # -------------------------------------------------------------------------
    print("\n[3] Extracting decoder weights")
    
    W_dec = sae.W_dec.data  # Shape: [feature_dim, input_dim]
    print(f"    W_dec shape: {W_dec.shape}")
    print(f"    Each row = one feature's decoder vector")
    
    # Get LM head (final layer that projects to vocabulary)
    lm_head = model.lm_head.weight  # Shape: [vocab_size, hidden_dim]
    print(f"    LM head shape: {lm_head.shape}")
    
    # -------------------------------------------------------------------------
    # 4. Logit Lens Analysis with Diverse Feature Selection
    # -------------------------------------------------------------------------
    print("\n[4] Performing Logit Lens analysis")
    print("    Projecting feature vectors to vocabulary...")
    
    # Select diverse features to analyze
    # First, find features with moderate activity
    with torch.no_grad():
        _, sample_features, _ = sae(activations)
        feature_activity = (sample_features > 0).float().mean(dim=0)
        active_features = torch.where((feature_activity > 0.05) & (feature_activity < 0.5))[0]
    
    if len(active_features) < 3:
        # Fallback: use any active features
        active_features = torch.where(feature_activity > 0)[0]
        if len(active_features) < 3:
            selected_features = list(range(min(3, feature_dim)))
        else:
            selected_features = select_diverse_features(sae, active_features, n_select=3, max_similarity=0.7)
    else:
        # Select 3 diverse features (decoder cosine similarity < 0.7)
        selected_features = select_diverse_features(sae, active_features, n_select=3, max_similarity=0.7)
    
    print(f"    Analyzing {len(selected_features)} diverse features: {selected_features}")
    if len(selected_features) == 3:
        # Show their pairwise similarities
        W_dec_norm = F.normalize(sae.W_dec.data.float(), p=2, dim=1)
        sims = []
        for i in range(len(selected_features)):
            for j in range(i+1, len(selected_features)):
                sim = F.cosine_similarity(
                    W_dec_norm[selected_features[i]].unsqueeze(0),
                    W_dec_norm[selected_features[j]].unsqueeze(0)
                ).item()
                sims.append(sim)
        print(f"    Pairwise similarities: {[f'{s:.3f}' for s in sims]}")
    
    # -------------------------------------------------------------------------
    # 5. Display Results
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS: DECODER VOCABULARY ANALYSIS")
    print("=" * 70)
    
    for feature_idx in selected_features:
        # Get feature's decoder vector
        feature_vec = W_dec[feature_idx]  # Shape: [input_dim]
        
        # Project to vocabulary using Logit Lens
        # This shows what tokens this feature "predicts"
        logits = feature_vec @ lm_head.T  # Shape: [vocab_size]
        
        # Get top-k tokens
        top_k = 15
        top_values, top_indices = torch.topk(logits, k=top_k)
        
        # Decode tokens
        top_tokens = [tokenizer.decode([idx.item()]) for idx in top_indices]
        
        # Calculate some statistics
        feature_norm = torch.norm(feature_vec, p=2).item()
        
        print(f"\n{'─' * 70}")
        print(f"FEATURE #{feature_idx}")
        print(f"{'─' * 70}")
        print(f"Decoder Vector Stats:")
        print(f"  L2 norm: {feature_norm:.4f}")
        print(f"  Non-zero elements: {(feature_vec != 0).sum().item()}/{len(feature_vec)}")
        
        print(f"\nTop {top_k} Predicted Tokens (via Logit Lens):")
        for i, (token, value) in enumerate(zip(top_tokens, top_values), 1):
            # Clean up token display
            token_display = repr(token)[1:-1]  # Remove quotes
            if len(token_display) > 20:
                token_display = token_display[:20] + "..."
            print(f"  {i:2d}. [{value:7.2f}] '{token_display}'")
        
        print(f"\nInterpretation:")
        print(f"  This feature's decoder vector, when projected to vocabulary,")
        print(f"  most strongly predicts the tokens shown above.")
        print(f"  Look for semantic patterns in the token list.")
    
    # -------------------------------------------------------------------------
    # 6. Feature Similarity Analysis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BONUS: FEATURE SIMILARITY ANALYSIS")
    print("=" * 70)
    
    print("\nComputing pairwise similarities between analyzed features...")
    
    similarity_matrix = torch.zeros((len(selected_features), len(selected_features)))
    
    for i, feat_i in enumerate(selected_features):
        for j, feat_j in enumerate(selected_features):
            vec_i = W_dec[feat_i]
            vec_j = W_dec[feat_j]
            
            # Cosine similarity
            similarity = F.cosine_similarity(vec_i.unsqueeze(0), vec_j.unsqueeze(0))
            similarity_matrix[i, j] = similarity.item()
    
    print("\nSimilarity Matrix:")
    print("    ", end="")
    for feat in selected_features:
        print(f"F#{feat:3d}  ", end="")
    print()
    
    for i, feat_i in enumerate(selected_features):
        print(f"F#{feat_i:3d}", end=" ")
        for j in range(len(selected_features)):
            sim = similarity_matrix[i, j]
            print(f"{sim:6.3f} ", end="")
        print()
    
    # Find similar feature pairs
    print("\nFeature Relationships:")
    for i, feat_i in enumerate(selected_features):
        for j, feat_j in enumerate(selected_features):
            if i < j:  # Only upper triangle
                sim = similarity_matrix[i, j].item()
                if sim > 0.5:
                    print(f"  Feature #{feat_i} ↔ Feature #{feat_j}: {sim:.3f} similarity (HIGH)")
                elif sim > 0.3:
                    print(f"  Feature #{feat_i} ↔ Feature #{feat_j}: {sim:.3f} similarity (moderate)")
    
    # -------------------------------------------------------------------------
    # 7. Comparison with Random Baseline
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON")
    print("=" * 70)
    
    print("\nComparing learned features vs random directions...")
    
    # Create a random direction
    random_vec = torch.randn_like(W_dec[0])
    random_vec = F.normalize(random_vec, p=2, dim=0)
    
    # Project to vocabulary
    random_logits = random_vec @ lm_head.T
    random_top_values, random_top_indices = torch.topk(random_logits, k=10)
    random_tokens = [tokenizer.decode([idx.item()]) for idx in random_top_indices]
    
    print("\nRandom Direction - Top 10 Tokens:")
    for i, (token, value) in enumerate(zip(random_tokens, random_top_values), 1):
        token_display = repr(token)[1:-1]
        if len(token_display) > 20:
            token_display = token_display[:20] + "..."
        print(f"  {i:2d}. [{value:7.2f}] '{token_display}'")
    
    print("\nObservation:")
    print("  Random directions typically predict common/generic tokens.")
    print("  Learned features should predict more specific/meaningful tokens.")
    print("  Compare the coherence of learned vs random token lists.")
    
    # -------------------------------------------------------------------------
    # 8. Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\nWhat We Did:")
    print("  1. Trained an SAE on GPT-2 activations (15 epochs, diverse HF datasets)")
    print("  2. Extracted decoder weight matrix (W_dec)")
    print("  3. Selected diverse features (decoder cosine < 0.7)")
    print("  4. Projected feature vectors to vocabulary (Logit Lens)")
    print("  5. Analyzed feature similarities")
    print("  6. Compared with random baseline")
    
    print("\nKey Insights:")
    print("  - Decoder weights encode what each feature 'means'")
    print("  - Logit Lens reveals vocabulary associations")
    print("  - Feature similarity shows redundancy or clustering")
    print("  - Learned features should be more coherent than random")
    
    print("\nAdvantages of This Method:")
    print("  ✓ Fast: No need to run inference on many texts")
    print("  ✓ Direct: Shows vocabulary predictions explicitly")
    print("  ✓ Quantitative: Provides similarity metrics")
    
    print("\nLimitations:")
    print("  ✗ Doesn't show contextual usage")
    print("  ✗ May miss polysemantic behavior")
    print("  ✗ Vocabulary-centric (not concept-centric)")
    
    print("\nNext Steps:")
    print("  - Compare with Method 1: max_activating_examples.py")
    print("  - Analyze more features to find patterns")
    print("  - Use feature similarity for clustering")
    print("  - Try feature steering based on decoder vectors")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
