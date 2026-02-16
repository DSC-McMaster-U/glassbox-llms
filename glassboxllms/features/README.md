# Glassbox Features: Sparse Autoencoders for LLM Interpretability

**Sparse Autoencoders (SAEs)** turn the mixed internal state of an LLM into discrete, interpretable **features**.

- **Before:** The model has a dense activation vector that mixes many concepts (e.g. "Eiffel Tower" + "Python code" + "negative sentiment").
- **After:** The SAE decomposes it into separate features—e.g. Feature A: "Paris landmarks", Feature B: "Programming syntax", Feature C: "Negative emotion".

This module provides the SAE model, training loop, and feature storage needed to go from raw activations to a saved, queryable feature set.

---

## Installation

```bash
pip install glassboxllms
```

Requires: PyTorch, Safetensors.

---

## Quick Start

```python
import torch
from glassboxllms.features import SparseAutoencoder, SAETrainer, FeatureSet

# 1. Define the SAE
# input_dim: LLM layer size (e.g. 768 for GPT-2)
# feature_dim: number of feature directions (often 16x–32x input_dim)
# k: max features active per token (sparsity)
sae = SparseAutoencoder(input_dim=768, feature_dim=16384, k=128)

# 2. Center on your data (recommended)
sample_data = torch.randn(10000, 768)  # or real LLM activations
sae.initialize_geometric_median(sample_data)

# 3. Train
trainer = SAETrainer(sae, dataloader, aux_coef=1e-3, resample_dead_every=500)
trainer.train(n_epochs=10)

# 4. Export to a feature set
feature_set = FeatureSet.from_sae(sae, trainer.get_stats())
feature_set.save("gpt2_layer6_features.safetensors")
```

---

## Module Overview

| File | Role | Main types |
|------|------|------------|
| `sae.py` | SAE model | `SparseAutoencoder`: encoder/decoder, TopK or L1 sparsity, forward pass |
| `trainer.py` | Training loop | `SAETrainer`: batches, optimizer, loss, dead-neuron resampling |
| `utils.py` | Helpers | `geometric_median`, `unit_norm_decoder`, `calc_mse_loss`, `calc_l0_sparsity`, `identify_dead_features` |
| `feature.py` | Single feature | `SAEFeature`: one feature (decoder vector + optional encoder + stats) |
| `feature_set.py` | Feature storage | `FeatureSet`: load/save from SAE to `.safetensors`, iterate as `SAEFeature`s |

### Data flow

1. **`sae.py`** — Defines the SAE (encoder matrix, decoder matrix, sparsity). No training logic.
2. **`utils.py`** — Used before/during training: geometric median for centering; MSE/L0 and dead-feature helpers.
3. **`trainer.py`** — Runs the loop: loads batches from a `DataLoader`, forwards through the SAE, computes loss, backward, optimizer step; tracks and resamples dead features.
4. **`feature_set.py`** — After training, `FeatureSet.from_sae(sae, stats)` packages weights and metadata and saves to `.safetensors`.

---

## How the SAE Works

Unlike a standard autoencoder that compresses, an **SAE expands** the representation so that mixed concepts can be separated.

1. **Encode (expand)**  
   Input: activation vector of size `input_dim` (e.g. 768).  
   Multiply by encoder matrix → vector of size `feature_dim` (e.g. 16,384). Many possible “feature” directions.

2. **Sparsity**  
   Only a few dimensions stay non-zero:
   - **TopK:** keep the top `k` activations, set the rest to zero (then ReLU).
   - **L1:** ReLU + L1 penalty on activations to encourage sparsity.

3. **Decode (reconstruct)**  
   Sparse feature vector × decoder matrix → reconstruction of the original `input_dim` vector. Decoder rows are **unit-norm** so each feature is a direction, not a scale.

4. **Loss**  
   Reconstruction is compared to the input (e.g. MSE). The trainer minimizes this (and auxiliary/L1 terms) so the SAE learns sparse features that reconstruct well.

---

## Using Stored Features

A saved `FeatureSet` is a dictionary of learned directions. Each feature can be interpreted and used for steering or auditing.

- **Interpretability** — Run new text through the SAE and see which feature indices fire. Inspect “max-activating examples” for a feature to name it (e.g. “Python code”).
- **Steering** — Suppress or amplify a feature (e.g. set its activation to 0 or clamp it) to reduce or encourage that concept in the model’s behavior.
- **Auditing** — Search for features that fire on deception, bias, or rare triggers; filter by activation or sparsity to find unusual or safety-relevant features.

---

## Key Concepts

- **Geometric median initialization** — Set the decoder bias to the geometric median of a sample of activations so the SAE is centered on real data; training is faster and more stable.
- **TopK sparsity** — For each token, only the top `k` feature activations are kept; the rest are zero. Gives a fixed, interpretable sparsity level.
- **Dead neurons** — Some feature directions may never win in TopK and never update. The trainer detects them (via activation history) and **resamples** them: reinitialize their decoder (and encoder) toward high-reconstruction-error samples so they get a chance to activate.
- **Unit-norm decoder** — Decoder rows are normalized to length 1 so the model cannot “cheat” sparsity by making decoder weights huge and encoder weights tiny; feature magnitude lives in the activations.

---

## Pipeline Summary

1. **Initialize** — Create the SAE and optionally center it with `initialize_geometric_median(sample_activations)`.
2. **Train** — `SAETrainer` runs the loop (forward, loss, backward, optimizer step, decoder normalization, dead-feature resampling).
3. **Extract** — Build a `FeatureSet` from the trained SAE and training stats.
4. **Save** — `feature_set.save("path.safetensors")` for fast, safe storage.

---

## Glossary

- **Activations** — Vectors of numbers at a layer in the LLM; the SAE’s input.
- **Reconstruction** — The SAE’s output; we train it to match the input activation.
- **Explained variance** — How much of the activation’s variance is explained by the reconstruction (0–1).
- **SafeTensors** — File format for storing tensors safely and with fast loading.
