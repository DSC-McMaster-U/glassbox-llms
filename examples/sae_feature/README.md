# SAE Feature Interpretation Examples

This directory contains examples demonstrating how to extract **interpretable knowledge** from trained Sparse Autoencoders (SAEs).

## What is Feature Interpretation?

When you train an SAE on LLM activations, you get a set of learned features (sparse directions in activation space). But the features are just numbers—how do you figure out what each feature represents?

There are two main methods to understand what knowledge is encoded in SAE features:

### Method 1: Max Activating Examples

**The Question:** "Which inputs cause this feature to activate strongly?"

**The Process:**

1. Run many diverse texts through your LLM
2. Capture activations and run them through the trained SAE
3. For each feature, find the top-K inputs that maximally activate it
4. Look for patterns in those inputs to interpret the feature

**Example:**

```
Feature #42 - Top Activating Texts:
  [9.1] "Silence is golden and precious."
  [8.5] "The Golden Gate Bridge stands tall."
  [7.8] "He won a gold medal at the Olympics."
  [7.2] "Golden retrievers are friendly dogs."
  [6.9] "The golden sunset was beautiful."

→ Interpretation: Feature #42 represents the concept of "Gold/Golden"
```

**Pros:** Shows real-world behavior, captures context
**Cons:** Requires large dataset and inference time

### Method 2: Decoder Weight Analysis (Logit Lens)

**The Question:** "What vocabulary does this feature predict?"

**The Process:**

1. Extract the decoder vector for a feature from `W_dec` matrix
2. Project it through the LLM's final layer (vocabulary projection)
3. See which tokens have the highest scores
4. Interpret based on vocabulary associations

**Example:**

```
Feature #42 - Top Predicted Tokens:
  [12.3] 'gold'
  [11.8] 'golden'
  [10.5] 'Gold'
  [9.2] 'Golden'
  [8.7] 'precious'
  [8.1] 'treasure'

→ Interpretation: Feature #42 directly predicts gold-related vocabulary
```

**Pros:** Fast, direct semantic understanding
**Cons:** Only shows vocabulary, not contextual usage

## Files in This Directory

| File                         | Purpose                                  | Method   |
| ---------------------------- | ---------------------------------------- | -------- |
| `max_activating_examples.py` | Find inputs that activate features       | Method 1 |
| `decoder_analysis.py`        | Analyze decoder weights                  | Method 2 |
| `dataset.py`                 | Shared dataset loading from Hugging Face | Both     |
| `__init__.py`                | Package initialization                   | -        |
| `README.md`                  | This file                                | -        |

Both analysis methods use the same diverse dataset for consistent comparison.

## Installation

Ensure you have the required dependencies:

```bash
source .venv/bin/activate
pip install torch transformers safetensors datasets
```

**Note:** The examples use Hugging Face `datasets` library to load diverse training data. The `trust_remote_code` parameter has been removed as it's deprecated in newer versions of the datasets library.

## Running the Examples

Both scripts can be run in two ways:

**Method A: As a module (recommended)**
```bash
# From project root
python -m examples.sae_feature.max_activating_examples
python -m examples.sae_feature.decoder_analysis
```

**Method B: Direct execution**
```bash
# From examples/sae_feature directory
cd examples/sae_feature
python max_activating_examples.py
python decoder_analysis.py
```

From the project root:

### Method 1: Max Activating Examples

```bash
python -m examples.sae_feature.max_activating_examples
```

This will:

1. Load GPT-2 model
2. Load diverse text dataset from Hugging Face (news, reviews, science, Wikipedia)
3. Capture activations from layer 6
4. Train a SAE with diverse feature selection
5. Find max activating examples for selected features
6. Display interpretations

**Expected runtime:** 10-15 minutes on CPU (includes dataset download)

### Method 2: Decoder Analysis

```bash
python -m examples.sae_feature.decoder_analysis
```

This will:

1. Load GPT-2 model
2. Load diverse text dataset from Hugging Face (same as Method 1)
3. Train SAE with 15 epochs for better convergence
4. Select diverse features (decoder cosine similarity < 0.7)
5. Extract decoder weights and project to vocabulary
6. Display top tokens for each feature
7. Show feature similarities and compare with random baseline

**Expected runtime:** 5-8 minutes on CPU (includes dataset download)

**Expected Results:**

- Explained Variance: ~0.75-0.85 (indicates good SAE convergence)
- 3 diverse features with low pairwise similarity (< 0.1)
- Coherent token clusters for each feature

## Dataset

Both methods use the same diverse dataset for consistent comparison, loaded via `dataset.py`:

**Primary Sources (from Hugging Face):**

- **News articles** (ag_news): 200 texts from diverse news categories
- **Movie reviews** (imdb): 200 texts with varied sentiment and topics
- **Scientific papers** (arxiv abstracts): 200 texts from academic literature ⚠️ May fail with newer datasets library
- **Wikipedia** (20220301.en): 200 texts from encyclopedia articles ⚠️ May fail with newer datasets library

**Note:** Some datasets (scientific_papers, wikipedia) may fail to load due to deprecated dataset scripts. The code gracefully falls back and continues with available datasets (typically 400 texts from news + reviews).

**Fallback:** If Hugging Face datasets fail to load, a hardcoded dataset with 96 diverse texts across 12 categories (programming, emotions, science, food, sports, nature, technology, history, medicine, business, education) is used.

The dataset provides:

- **Domain diversity**: Technical, narrative, scientific, encyclopedic content
- **Style diversity**: Formal, informal, emotional, neutral writing
- **Consistent basis**: Both methods analyze the exact same learned features

## Understanding the Output

### Max Activating Examples Output

```
Feature #5 - Top Activating Texts:
  [12.45] "The Python programming language is versatile."
  [11.32] "She wrote elegant Python code for the project."
  [10.87] "import numpy as np  # Python scientific computing"
  [10.23] "Python's syntax makes it beginner-friendly."
  ...

Interpretation:
  Feature #5 appears to represent "Python programming language"
  - Activates on sentences mentioning Python (language, not snake)
  - High activation when technical/coding context is present
  - Associated with programming-related vocabulary
```

### Decoder Analysis Output

```
Feature #5 - Top Predicted Tokens:
  [15.23] ' Python'
  [14.87] ' python'
  [12.45] ' programming'
  [11.32] ' code'
  [10.98] ' language'
  [10.23] ' syntax'
  ...

Feature Direction Strength: 0.023 (normalized)

Similar Features:
  Feature #127: 0.85 similarity (also codes Python-related)
  Feature #234: 0.72 similarity (programming in general)
```

## Key Concepts

### Sparsity

SAE features are **sparse**—most are zero for any given input. This is crucial:

- Dense representation: `[0.3, -0.5, 0.8, 0.1, ...]` (hard to interpret)
- Sparse representation: `[0, 0, 5.2, 0, 0, 3.1, 0, ...]` (clear which features matter)

### Monosemanticity

The goal is **monosemantic features**—each feature represents one concept:

- ❌ Polysemantic: Feature #10 = "Python (snake)" + "Python (language)"
- ✓ Monosemantic: Feature #5 = "Python (language)", Feature #8 = "Python (snake)"

### Activation Strength

The magnitude of activation matters:

- `0.0`: Feature not present
- `1-5`: Weak presence
- `5-10`: Moderate presence
- `10+`: Strong presence

Higher values indicate stronger/clearer instances of the concept.

## Customization

### Using Your Own Dataset

Edit the dataset in `max_activating_examples.py`:

```python
# Replace with your texts
texts = [
    "Your custom text here...",
    "Another example...",
    # ... more texts ...
]
```

### Analyzing Different Layers

Change the target layer:

```python
# Try different layers (0-11 for GPT-2)
layer_name = "transformer.h.8"  # Default: transformer.h.6
```

### Adjusting SAE Size

Modify training config:

```python
sae = SparseAutoencoder(
    input_dim=768,
    feature_dim=4096,  # More features = finer-grained concepts
    k=64,              # Higher k = less sparse
)
```

## Troubleshooting

### "Out of memory" error

Reduce batch size or feature dimension:

```python
batch_size = 16  # Default: 32
feature_dim = 1024  # Default: 2048
```

### Features not interpretable

Try:

1. More diverse training data (500+ examples)
2. Longer training (more epochs) - the examples use 15 epochs for good convergence
3. Different sparsity (adjust `k` parameter)
4. Different layer (earlier layers = syntax, later = semantics)
5. Check explained variance - should be > 0.5 for interpretable features (examples achieve ~0.8)

### No clear patterns in max activating examples

- Increase dataset size (need 1000+ for clear patterns)
- Try multiple features (some will be clearer than others)
- Check if SAE actually learned (explained variance > 0.5)
- Use diverse feature selection (decoder cosine < 0.7) to avoid redundant features

### Deprecation warnings about `trust_remote_code`

These are harmless warnings from the datasets library. The code handles them gracefully and continues loading available datasets.

## Further Reading

### SAE Research Papers

- [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features) - Anthropic (2023)
- [Sparse Autoencoders Find Highly Interpretable Features in Language Models](https://arxiv.org/abs/2309.08600) - Cunningham et al. (2023)

### Logit Lens

- [Interpreting GPT: The Logit Lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) - nostalgebraist (2020)

### Related Techniques

- **Activation Patching:** Modify features to test causal impact
- **Feature Steering:** Amplify/suppress features to control behavior
- **Circuit Discovery:** Find how features connect in computation graphs
