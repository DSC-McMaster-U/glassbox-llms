# SAE Experiment Module

Automated pipeline for discovering monosemantic features using Sparse Autoencoders (SAEs).

## Overview

This module provides the `SAEExperiment` class, which orchestrates the complete workflow for training Sparse Autoencoders on LLM activations and extracting interpretable features.

**Key capabilities:**

- Collect activations from any model layer via hooks
- Train SAE with TopK or L1 sparsity
- Validate training quality (R², L0 sparsity, dead features)
- Extract and register features to FeatureAtlas

## Files

| File                | Description                                            |
| ------------------- | ------------------------------------------------------ |
| `sae_experiment.py` | Main `SAEExperiment` class implementation (~500 lines) |
| `experiment.py`     | End-to-end example using GPT-2 (~300 lines)            |
| `readme.md`         | This file                                              |

## Quick Start

### Using the SAEExperiment Class

```python
from glassboxllms.experiments.sae import SAEExperiment
from glassboxllms.analysis.feature_atlas import Atlas

# Initialize experiment
experiment = SAEExperiment(
    model=gpt2_model,
    layer="transformer.h.11.mlp",
    sparsity_alpha=0.1,
    d_sae=16384,
    k=64,
    device="cuda"
)

# Collect activations
activations = experiment.collect_activations(
    dataloader=text_dataloader,
    num_samples=50000
)

# Train SAE
stats = experiment.train(activations, n_epochs=10)

# Validate and register
criteria = experiment.validate_training()
atlas = experiment.register_features(atlas_name="gpt2-features")
atlas.save("features.json")
```

### Running the Full Example

```bash
# Run the complete pipeline example
python -m glassboxllms.experiments.sae.experiment

# Or directly
python glassboxllms/experiments/sae/experiment.py
```

The example demonstrates:

- Loading GPT-2 and targeting layer 11 MLP
- Collecting activations from diverse text
- Training a 16K-dimensional SAE
- Validating quality and registering features

## API Reference

### SAEExperiment

**Constructor:**

```python
SAEExperiment(
    model: nn.Module,              # PyTorch model to analyze
    layer: str,                    # Target layer name
    sparsity_alpha: float = 0.1,   # Sparsity coefficient
    d_sae: int = 32768,            # SAE feature dimension
    k: Optional[int] = None,       # TopK active features
    sparsity_mode: str = "topk",   # "topk" or "l1"
    model_name: Optional[str] = None,
    device: str = "cpu"
)
```

## TopK: “Only k features can be on” → fixed sparsity; auxiliary loss trains the features that are never on by making them explain the leftover error.

## L1: “You can use any features, but we penalize how much you use” → learned sparsity; no auxiliary loss, just reconstruction + L1 penalty.

**Key Methods:**

- **`collect_activations(dataloader, num_samples, pooling="mean")`**  
  Collect activations from the target layer
- **`train(activations, n_epochs=10, batch_size=256, learning_rate=3e-4)`**  
  Train the SAE on collected activations
- **`validate_training() -> Dict[str, bool]`**  
  Check success criteria (R² > 0.7, L0 < 50, dead features < 10%)
- **`extract_features(skip_dead=True) -> List[Feature]`**  
  Extract learned features as Atlas Feature objects
- **`register_features(atlas=None, atlas_name=None) -> Atlas`**  
  Register features to a FeatureAtlas
- **`save_checkpoint(path)`**  
  Save trained SAE checkpoint

## Configuration

### Sparsity Modes

**TopK (recommended):**

```python
experiment = SAEExperiment(
    model=model,
    layer="transformer.h.11.mlp",
    sparsity_mode="topk",
    k=64,                    # Number of active features
    sparsity_alpha=0.1       # Auxiliary loss coefficient
)
```

**L1 Penalty:**

```python
experiment = SAEExperiment(
    model=model,
    layer="transformer.h.11.mlp",
    sparsity_mode="l1",
    sparsity_alpha=1e-3      # L1 penalty weight
)
```

### Typical Hyperparameters

| Parameter     | Small Model | Large Model  | Description                  |
| ------------- | ----------- | ------------ | ---------------------------- |
| `d_sae`       | 8192-16384  | 32768-131072 | SAE dimension (8-32x hidden) |
| `k`           | 32-64       | 64-128       | Active features (TopK)       |
| `n_epochs`    | 10-20       | 20-50        | Training epochs              |
| `num_samples` | 50K-500K    | 500K-5M      | Activation samples           |

## Success Criteria

The `validate_training()` method checks three criteria:

1. **High Reconstruction**: R² > 0.7 (SAE accurately reconstructs inputs)
2. **Sparse Activations**: Mean L0 < 50 (few features active per token)
3. **Low Dead Features**: < 10% dead (most features utilized)

If criteria aren't met, the method suggests improvements:

- Training longer
- Increasing SAE dimension
- Adjusting sparsity parameters
- Collecting more diverse activations

## Output

### Feature Atlas

Features are saved to a `FeatureAtlas` with:

- **Type**: `FeatureType.SAE_LATENT`
- **Location**: Model name, layer, neuron index
- **Metadata**: Decoder norm, explained variance, L0, dead status
- **History**: Method, dataset, hyperparameters

Query features:

```python
atlas = Atlas.load("features.json")

# Find by layer
layer_features = atlas.find_by_layer("transformer.h.11.mlp")

# Find by type
sae_features = atlas.find_by_type(FeatureType.SAE_LATENT)

# Sort by importance
top_features = sorted(
    layer_features,
    key=lambda f: f.metadata["decoder_norm"],
    reverse=True
)
```

### Checkpoint

SAE checkpoint includes:

- Model state dict
- Configuration (dimensions, sparsity settings)
- Training statistics
- Model and layer information

## Troubleshooting

**Low explained variance (< 0.7):**

- Increase `d_sae` (more capacity)
- Train longer (`n_epochs`)
- Collect more activations (`num_samples`)

**Too many active features (L0 > 50):**

- Decrease `k` (stricter TopK)
- Increase `sparsity_alpha` (L1 mode)

**Many dead features (> 10%):**

- Increase `sparsity_alpha` (auxiliary loss)
- Collect more diverse activations
- Check geometric median initialization is working

**Out of memory:**

- Reduce `batch_size` (e.g., 128 instead of 256)
- Reduce `d_sae` (e.g., 8192 instead of 16384)
- Use gradient checkpointing (requires trainer modification)

## Integration

### With Other Interpretability Methods

The registered features can be analyzed using:

**Max Activating Examples:**

```python
# Find texts that maximally activate specific features
from examples.sae_feature import max_activating_examples
# See examples/sae_feature/max_activating_examples.py
```

**Decoder Analysis (Logit Lens):**

```python
# Project decoder weights to vocabulary space
from examples.sae_feature import decoder_analysis
# See examples/sae_feature/decoder_analysis.py
```

## Research Context

Based on modern SAE best practices from:

**Cunningham et al. (2023)**: "Sparse Autoencoders Find Highly Interpretable Features in Language Models" ([arXiv:2309.08600](https://arxiv.org/abs/2309.08600))

Key techniques implemented:

- TopK and L1 sparsity mechanisms
- Auxiliary loss for dead neuron revival
- Geometric median initialization
- Unit norm decoder constraints
- High-dimensional expansion (8-32x)

## Dependencies

Required packages:

```bash
pip install torch transformers datasets
```

All dependencies are standard PyTorch ecosystem packages.

## Architecture

```
SAEExperiment
│
├─ Activation Collection
│  ├─ HookManager (attach hooks to layers)
│  └─ ActivationStore (buffer and persist activations)
│
├─ SAE Training
│  ├─ SparseAutoencoder (TopK/L1 sparsity)
│  ├─ SAETrainer (training loop + metrics)
│  └─ Geometric median initialization
│
├─ Feature Extraction
│  ├─ Extract decoder vectors W_dec[i]
│  ├─ Calculate per-feature stats
│  └─ Convert to Feature objects
│
└─ Feature Registration
   └─ Add to FeatureAtlas with metadata
```
