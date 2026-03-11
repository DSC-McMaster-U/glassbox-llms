"""
Full SAE Experiment Pipeline: GPT-2 Layer Analysis

Demonstrates the complete workflow for discovering monosemantic features:
1. Load model and dataset
2. Collect activations from target layer
3. Train Sparse Autoencoder
4. Validate training quality
5. Extract and register features to Atlas

This example uses GPT-2 and analyzes layer 11's MLP output.

Expected Results:
- Explained variance > 0.7 (high reconstruction quality)
- Mean L0 < 50 (sparse feature activations)
- Thousands of interpretable SAE features registered to Atlas

Usage:
    python -m glassboxllms.experiments.sae.experiment
    or
    python glassboxllms/experiments/sae/experiment.py
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset

import sys
from pathlib import Path

# Handle imports for both execution methods
try:
    from glassboxllms.experiments.sae import SAEExperiment
except ImportError:
    # Add project root to path when running directly
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    from glassboxllms.experiments.sae import SAEExperiment


def create_dataloader(tokenizer, num_texts=500, batch_size=8, max_length=128):
    """
    Create a DataLoader from diverse text datasets.
    
    Args:
        tokenizer: HuggingFace tokenizer
        num_texts: Number of texts to load
        batch_size: Batch size for DataLoader
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (DataLoader, list of raw text strings)
    """
    print(f"Loading {num_texts} diverse texts...")
    
    texts = []
    
    # Load diverse datasets
    try:
        # Wikipedia
        wiki = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
        for example in wiki:
            if len(texts) >= num_texts // 2:
                break
            text = example["text"][:500]  # Truncate long articles
            if len(text) > 50:
                texts.append(text)
        print(f"  Loaded {len(texts)} Wikipedia texts")
    except Exception as e:
        print(f"  Warning: Could not load Wikipedia: {e}")
    
    # OpenWebText (if available)
    try:
        if len(texts) < num_texts:
            owt = load_dataset("openwebtext", split="train", streaming=True)
            for example in owt:
                if len(texts) >= num_texts:
                    break
                text = example["text"][:500]
                if len(text) > 50:
                    texts.append(text)
            print(f"  Loaded {len(texts)} total texts (including OpenWebText)")
    except Exception as e:
        print(f"  Warning: Could not load OpenWebText: {e}")
    
    # Fallback: duplicate if needed
    while len(texts) < num_texts:
        texts.extend(texts[:min(len(texts), num_texts - len(texts))])
    
    texts = texts[:num_texts]
    print(f"  Final dataset: {len(texts)} texts")
    
    # Tokenize
    print("Tokenizing...")
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    
    # Create dataset
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    
    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, input_ids, attention_mask):
            self.input_ids = input_ids
            self.attention_mask = attention_mask
        
        def __len__(self):
            return len(self.input_ids)
        
        def __getitem__(self, idx):
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx]
            }
    
    dataset = TextDataset(input_ids, attention_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    print(f"  Created DataLoader: {len(dataloader)} batches")
    return dataloader, texts


def select_diverse_features(sae, candidate_indices, n_select=5, max_similarity=0.7):
    """
    Select features that are far apart in decoder space (low pairwise cosine similarity).

    Uses greedy selection: pick first candidate, then repeatedly add the candidate
    with cosine similarity < max_similarity to all already-selected features.

    Args:
        sae: Trained SparseAutoencoder (has W_dec).
        candidate_indices: List of feature indices to choose from.
        n_select: Number of features to select.
        max_similarity: Maximum allowed cosine similarity between any two selected features.

    Returns:
        List of selected feature indices.
    """
    import torch.nn.functional as F

    W_dec = sae.W_dec.data  # [feature_dim, input_dim]
    candidates = list(candidate_indices)

    if len(candidates) <= n_select:
        return candidates

    W_dec_norm = F.normalize(W_dec.float(), p=2, dim=1)
    selected = [candidates[0]]
    remaining = candidates[1:]

    while len(selected) < n_select and remaining:
        best_idx = None
        best_max_sim = 2.0

        for i, cand in enumerate(remaining):
            cand_vec = W_dec_norm[cand].unsqueeze(0)
            selected_vecs = W_dec_norm[torch.tensor(selected)]
            sims = F.cosine_similarity(cand_vec, selected_vecs, dim=1)
            max_sim = sims.max().item()

            if max_sim < max_similarity:
                best_idx = i
                break

            if max_sim < best_max_sim:
                best_max_sim = max_sim
                best_idx = i

        if best_idx is None:
            break

        selected.append(remaining.pop(best_idx))

    return selected[:n_select]


def show_dominant_features(sae, activations, texts, n_features=5, top_k_examples=3):
    """
    Find dominant SAE features and display the texts that maximally activate them.

    Args:
        sae: Trained SparseAutoencoder.
        activations: Tensor [n_samples, input_dim] - same order as texts.
        texts: List of original text strings.
        n_features: Number of dominant features to display.
        top_k_examples: Number of top activating texts to show per feature.
    """
    print("\nFinding dominant features and max activating examples...")

    sae.eval()
    feature_acts_list = []
    with torch.no_grad():
        for i in range(0, len(activations), 256):
            batch = activations[i:i + 256]
            _, features, _ = sae(batch)
            feature_acts_list.append(features.cpu())

    feature_acts = torch.cat(feature_acts_list, dim=0)  # [n_samples, d_sae]

    # Features that are selective: active on some inputs but not too common
    frequencies = (feature_acts > 0).float().mean(dim=0)
    interesting = torch.where((frequencies > 0.02) & (frequencies < 0.3))[0]

    if len(interesting) < n_features:
        # Fall back to top features by mean activation
        interesting = torch.argsort(feature_acts.mean(dim=0), descending=True)[:n_features * 3]

    selected_features = select_diverse_features(
        sae, interesting.tolist(), n_select=n_features, max_similarity=0.7
    )
    print(f"  Selected {len(selected_features)} diverse dominant features")

    print("\n" + "=" * 70)
    print("DOMINANT FEATURES & MAX ACTIVATING EXAMPLES")
    print("=" * 70)

    for feature_idx in selected_features:
        acts = feature_acts[:, feature_idx]
        active_mask = acts > 0
        mean_act = acts[active_mask].mean().item() if active_mask.any() else 0.0
        max_act = acts.max().item()
        sparsity = active_mask.float().mean().item()

        top_vals, top_idxs = torch.topk(acts, k=min(top_k_examples, len(acts)))

        print(f"\n{'─' * 70}")
        print(f"Feature #{feature_idx}")
        print(f"  Max: {max_act:.2f} | Mean (active): {mean_act:.2f} | Fires on: {sparsity:.1%} of inputs")
        print(f"  Top {top_k_examples} activating texts:")
        for rank, (val, idx) in enumerate(zip(top_vals.tolist(), top_idxs.tolist()), 1):
            if idx < len(texts):
                text = texts[idx].replace("\n", " ").replace("\r", " ")
                display = text[:120].rstrip() + ("..." if len(text) > 120 else "")
                print(f"    {rank}. [{val:.2f}] {display}")


def main():
    """Run the full SAE experiment pipeline."""
    
    print("=" * 70)
    print("SAE Experiment: Discovering Monosemantic Features in GPT-2")
    print("=" * 70)
    
    # Configuration
    MODEL_NAME = "gpt2"
    TARGET_LAYER = "transformer.h.11.mlp"  # Layer 11 MLP output
    D_SAE = 16384  # 16x expansion (GPT-2 has 768 hidden dim)
    K = 32  # Number of active features for TopK mode (auto-computed if None)
    SPARSITY_ALPHA = 0.15  # Auxiliary loss coefficient
    N_ACTIVATIONS = 10000  # Number of activations to collect
    N_EPOCHS = 30  # Number of epochs to train for
    BATCH_SIZE = 256  # Batch size for training
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Target layer: {TARGET_LAYER}")
    print(f"  SAE dimension: {D_SAE}")
    print(f"  TopK: {K}")
    print(f"  Device: {DEVICE}")
    print()
    
    # Load model and tokenizer
    print("Loading GPT-2 model and tokenizer...")
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model.to(DEVICE)
    model.eval()
    print("  ✓ Model loaded")
    
    # Create experiment
    print(f"\nInitializing SAEExperiment...")
    experiment = SAEExperiment(
        model=model,
        layer=TARGET_LAYER,
        sparsity_alpha=SPARSITY_ALPHA,
        d_sae=D_SAE,
        k=K,
        sparsity_mode="topk",
        model_name=MODEL_NAME,
        device=DEVICE
    )
    print(f"  ✓ Experiment initialized")
    print(f"    Input dimension: {experiment.input_dim}")
    print(f"    Feature dimension: {experiment.d_sae}")
    print(f"    TopK: {experiment.k}")
    
    # Create DataLoader (also returns texts for max-activating examples display)
    dataloader, texts = create_dataloader(
        tokenizer,
        num_texts=2000,
        batch_size=8,
        max_length=128
    )
    
    # Step 1: Collect activations
    print("\n" + "=" * 70)
    print("[1/4] Collecting activations...")
    activations = experiment.collect_activations(
        dataloader,
        num_samples=N_ACTIVATIONS,
        pooling="mean"
    )
    print(f"  Collected activations shape: {activations.shape}")

    # Step 2: Train SAE
    print("\n" + "=" * 70)
    print("[2/4] Training SAE...")
    stats = experiment.train(
        activations,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=3e-4,
        log_every=50
    )

    # Step 3: Validate training quality
    print("\n" + "=" * 70)
    print("[3/4] Validating training quality...")
    criteria = experiment.validate_training()
    
    print("\n  Success Criteria:")
    for criterion, passed in criteria.items():
        status = "✓" if passed else "✗"
        print(f"    {status} {criterion.replace('_', ' ').title()}: {passed}")
    
    all_passed = all(criteria.values())
    if all_passed:
        print("\n  🎉 All success criteria met!")
    else:
        print("\n  ⚠ Some criteria not met. Consider:")
        if not criteria["high_reconstruction"]:
            print("    - Training longer (more epochs)")
            print("    - Increasing SAE dimension (d_sae)")
        if not criteria["sparse_activations"]:
            print("    - Decreasing k (stricter TopK)")
            print("    - Increasing sparsity_alpha")
        if not criteria["low_dead_features"]:
            print("    - Using auxiliary loss (already enabled)")
            print("    - Collecting more diverse activations")
    
    # Step 4: Register features to Atlas
    print("\n" + "=" * 70)
    print("[4/4] Registering features to Atlas...")
    atlas = experiment.register_features(
        atlas_name=f"{MODEL_NAME}_layer11_sae_features",
        dataset_name="wikipedia+openwebtext",
        skip_dead=True
    )
    
    # Display summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"Training Results:")
    print(f"  Explained variance: {stats['final_explained_variance']:.3f}")
    print(f"  Mean L0:            {stats.get('mean_l0', 'N/A')}")
    print(f"  Dead features:      {stats.get('dead_features', 'N/A')}")
    print(f"\nAtlas:")
    print(f"  Name:           {atlas.metadata.name}")
    print(f"  Total features: {len(atlas)}")
    
    # Save Atlas
    output_path = Path("sae_features_atlas.json")
    atlas.save(output_path)
    print(f"\n  ✓ Atlas saved to {output_path}")
    
    # Save SAE checkpoint
    checkpoint_path = Path("sae_checkpoint.pt")
    experiment.save_checkpoint(checkpoint_path)
    print(f"  ✓ SAE checkpoint saved to {checkpoint_path}")
    
    # Show dominant features with max activating examples
    show_dominant_features(
        experiment.sae,
        activations,
        texts,
        n_features=5,
        top_k_examples=3
    )

    print("\n" + "=" * 70)
    print("✓ PIPELINE COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("  - Use decoder_analysis.py to interpret features via logit lens")
    print("  - Load the atlas for further analysis: Atlas.load('sae_features_atlas.json')")


if __name__ == "__main__":
    main()
