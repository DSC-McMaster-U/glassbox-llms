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
    from glassboxllms.analysis.feature_atlas import Atlas
except ImportError:
    # Add project root to path when running directly
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    from glassboxllms.experiments.sae import SAEExperiment
    from glassboxllms.analysis.feature_atlas import Atlas


def create_dataloader(tokenizer, num_texts=500, batch_size=8, max_length=128):
    """
    Create a DataLoader from diverse text datasets.
    
    Args:
        tokenizer: HuggingFace tokenizer
        num_texts: Number of texts to load
        batch_size: Batch size for DataLoader
        max_length: Maximum sequence length
        
    Returns:
        DataLoader with tokenized texts
    """
    print(f"Loading {num_texts} diverse texts...")
    
    texts = []
    
    # Load diverse datasets
    try:
        # Wikipedia
        wiki = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
        for i, example in enumerate(wiki):
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
            for i, example in enumerate(owt):
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
    return dataloader


def main():
    """Run the full SAE experiment pipeline."""
    
    print("=" * 70)
    print("SAE Experiment: Discovering Monosemantic Features in GPT-2")
    print("=" * 70)
    
    # Configuration
    MODEL_NAME = "gpt2"
    TARGET_LAYER = "transformer.h.11.mlp"  # Layer 11 MLP output
    D_SAE = 16384  # 16x expansion (GPT-2 has 768 hidden dim)
    K = 32  # Reduce from 64 to get L0 < 50
    SPARSITY_ALPHA = 0.1  # Auxiliary loss coefficient
    N_ACTIVATIONS = 10000  # Increase from 500 (config says 50k but dataset only has 500)
    N_EPOCHS = 20  # Increase from 10 for better convergence
    BATCH_SIZE = 256
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
    
    # Create DataLoader
    dataloader = create_dataloader(
        tokenizer,
        num_texts=2000,  # Increase from 500 to get more activations
        batch_size=8,
        max_length=128
    )
    
    # Step 1: Collect activations
    print("\n" + "=" * 70)
    activations = experiment.collect_activations(
        dataloader,
        num_samples=N_ACTIVATIONS,
        pooling="mean"  # Average over sequence dimension
    )
    print(f"Collected activations shape: {activations.shape}")
    
    # Step 2: Train SAE
    print("\n" + "=" * 70)
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
    atlas = experiment.register_features(
        atlas_name=f"{MODEL_NAME}_layer11_sae_features",
        dataset_name="wikipedia+openwebtext",
        skip_dead=True
    )
    
    # Display summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Layer: {TARGET_LAYER}")
    print(f"SAE dimension: {D_SAE}")
    print(f"TopK: {K}")
    print(f"\nTraining Results:")
    print(f"  Explained variance: {stats['final_explained_variance']:.3f}")
    print(f"  Mean L0: {stats.get('mean_l0', 'N/A')}")
    print(f"  Dead features: {stats.get('dead_feature_count', 'N/A')}")
    print(f"\nAtlas:")
    print(f"  Name: {atlas.metadata.name}")
    print(f"  Total features: {len(atlas)}")
    
    # Save Atlas
    output_path = Path("sae_features_atlas.json")
    atlas.save(output_path)
    print(f"\n  ✓ Atlas saved to {output_path}")
    
    # Save SAE checkpoint
    checkpoint_path = Path("sae_checkpoint.pt")
    experiment.save_checkpoint(checkpoint_path)
    print(f"  ✓ SAE checkpoint saved to {checkpoint_path}")
    
    # Example: Query features
    print("\n" + "=" * 70)
    print("FEATURE QUERIES")
    print("=" * 70)
    
    layer_features = atlas.find_by_layer(TARGET_LAYER)
    print(f"Features in {TARGET_LAYER}: {len(layer_features)}")
    
    # Show top 5 features by decoder norm (most "important")
    top_features = sorted(
        layer_features,
        key=lambda f: f.metadata.get("decoder_norm", 0),
        reverse=True
    )[:5]
    
    print("\nTop 5 features by decoder norm:")
    for i, feature in enumerate(top_features, 1):
        print(f"  {i}. {feature.label}")
        print(f"     Decoder norm: {feature.metadata['decoder_norm']:.3f}")
        print(f"     Neuron idx: {feature.location.neuron_idx}")
    
    print("\n" + "=" * 70)
    print("✓ PIPELINE COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("  - Use max_activating_examples.py to find texts that activate features")
    print("  - Use decoder_analysis.py to interpret features via logit lens")
    print("  - Load the atlas for further analysis: Atlas.load('sae_features_atlas.json')")


if __name__ == "__main__":
    main()
