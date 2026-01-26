# 🔬 Linear Probes Module

A toolkit for probing neural network representations to understand what concepts are encoded in model activations.

## Overview

**Linear probes** are simple linear models trained on frozen activations to test whether specific concepts (e.g., grammatical tense, sentiment, gender) are linearly encoded at a given layer.

**Key principle**: The probe must be simple enough that it cannot learn the task on its own. It can only succeed if the activations _already_ encode the target concept.

### References

- [Alain & Bengio (2016)](https://arxiv.org/abs/1610.01644) – Understanding intermediate layers using linear classifiers
- [Kim et al. (2018)](https://arxiv.org/abs/1711.11279) – Concept Activation Vectors (TCAV)
- [Carpentries Incubator: Linear Probes Tutorial](https://carpentries-incubator.github.io/fair-explainable-ml/5c-probes.html)

---

## 📁 Module Structure

```
probes/
├── __init__.py           # Package exports
├── base.py               # BaseProbe abstract class + ProbeResult dataclass
├── linear.py             # LinearProbe implementation (logistic, linear, PCA, CAV)
├── activation_store.py   # Extract activations from transformer models
└── README.md             # This file
```

---

## 🚀 Quick Start

### Installation

The module requires `scikit-learn`, `torch`, `transformers`, and `numpy`.

**Option 2: Using a new virtual environment**

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r probes/requirements.txt
# OR
pip install numpy scikit-learn torch transformers
```

### Running Examples

Once your environment is activated and dependencies are installed:

```bash
# From project root
python -m probes.examples.logistic_sentiment
python -m probes.examples.linear_intensity
python -m probes.examples.pca_structure
python -m probes.examples.cav_gender
```

### Basic Usage

```python
from transformers import AutoModel, AutoTokenizer
from probes import LinearProbe, ActivationStore

# 1. Load a model
model = AutoModel.from_pretrained("distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# 2. Extract activations
store = ActivationStore(model)
activations = store.extract(
    texts=["I love this movie", "I hate this movie", "Great film!", "Terrible film."],
    tokenizer=tokenizer,
    layers=["transformer.layer.5"],
    pooling="mean"
)

# 3. Train a probe
labels = [1, 0, 1, 0]  # sentiment: 1=positive, 0=negative
probe = LinearProbe(layer="transformer.layer.5", direction="sentiment")
probe.fit(activations["transformer.layer.5"], labels)

# 4. Evaluate
results = probe.evaluate(test_activations, test_labels)
print(results)
# ProbeResult(accuracy=0.85, f1=0.83)
```

---

## 📖 API Reference

### `LinearProbe`

Unified interface for fitting linear models to frozen activations.

```python
LinearProbe(
    layer: str,              # Target layer (e.g., "mlp.10")
    direction: str,          # Concept name (e.g., "tense")
    model_type: str = "logistic",  # 'logistic', 'linear', 'pca', 'cav'
    normalize: bool = True,  # Standardize activations
    n_components: int = 1,   # For PCA
    regularization: float = 1.0,
    max_iter: int = 1000,
    incremental: bool = False,  # For large datasets
)
```

#### Model Types

| Type       | Use Case                                      | Output                          |
| ---------- | --------------------------------------------- | ------------------------------- |
| `logistic` | Binary/multiclass classification              | Class labels, probabilities     |
| `linear`   | Continuous target prediction (regression)     | Scalar values                   |
| `pca`      | Find principal directions in activation space | Transformed coordinates         |
| `cav`      | Concept Activation Vector direction           | Class labels + direction vector |

#### Methods

| Method                             | Description                                      |
| ---------------------------------- | ------------------------------------------------ |
| `fit(activations, labels)`         | Train probe on frozen activations                |
| `partial_fit(activations, labels)` | Incremental training for large datasets          |
| `predict(activations)`             | Predict labels from activations                  |
| `predict_proba(activations)`       | Class probabilities (classification only)        |
| `evaluate(activations, labels)`    | Get metrics (accuracy, F1, etc.)                 |
| `get_direction()`                  | Return learned linear direction (weight vector)  |
| `score_activation(activation)`     | Project single activation onto concept direction |

---

### `ActivationStore`

Extract and cache activations from transformer models using forward hooks.

```python
store = ActivationStore(model)

activations = store.extract(
    texts: List[str],         # Input texts
    tokenizer: Any,           # HuggingFace tokenizer
    layers: List[str],        # Layer names to extract
    batch_size: int = 32,
    max_length: int = 512,
    pooling: str = "mean",    # 'mean', 'cls', 'last', 'none'
    device: str = None,       # 'cuda', 'cpu', or auto
)
# Returns: Dict[layer_name -> np.ndarray]
```

#### Pooling Strategies

| Strategy | Description                           |
| -------- | ------------------------------------- |
| `mean`   | Average over all tokens (most common) |
| `cls`    | Take [CLS] token (position 0)         |
| `last`   | Take last token                       |
| `none`   | Keep full sequence (returns 3D array) |

#### Finding Layer Names

```python
# List all available layers
store.list_layers()

# Filter by pattern
store.list_layers(pattern="layer.6")

# Or use the utility function
from probes.activation_store import get_layer_names
get_layer_names(model, layer_type="attention")  # attention layers only
get_layer_names(model, layer_type="mlp")        # MLP/FFN layers only
```

---

### `ProbeResult`

Container for evaluation results.

```python
@dataclass
class ProbeResult:
    accuracy: float
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    explained_variance: Optional[float]  # For PCA/regression
    coefficients: np.ndarray             # Learned direction
    metadata: dict                       # Additional info
```

---

## 🧪 Example: Layer-wise Probing

Find which layer best encodes sentiment:

```python
from probes import LinearProbe, ActivationStore

store = ActivationStore(model)
layers = [f"transformer.layer.{i}" for i in range(6)]

results = {}
for layer in layers:
    acts = store.extract(texts, tokenizer, layers=[layer])
    probe = LinearProbe(layer=layer, direction="sentiment")
    probe.fit(acts[layer], train_labels)
    results[layer] = probe.evaluate(test_acts[layer], test_labels)

# Find best layer
best = max(results, key=lambda k: results[k].accuracy)
print(f"Best layer: {best} with accuracy {results[best].accuracy:.3f}")
```

---

## Example: sentiment probe

Test if "sentiment" is linearly encoded:

```python
# TRAINING SET
train_texts = [
    "I love this movie",      # label: 1 (positive)
    "I hate this movie",      # label: 0 (negative)
    "Great film!",            # label: 1
    "Terrible film.",         # label: 0
]
train_acts = store.extract(train_texts, tokenizer, layers=["layer.6"])
train_labels = [1, 0, 1, 0]

# Train probe
probe = LinearProbe(layer="layer.6", direction="sentiment", model_type="logistic")
probe.fit(train_acts["layer.6"], train_labels)

# TEST SET (NEW sentences)
test_texts = [
    "This is amazing!",       # label: 1 (positive)
    "This is awful",          # label: 0 (negative)
    "I enjoyed it",           # label: 1
    "I disliked it",          # label: 0
]
test_acts = store.extract(test_texts, tokenizer, layers=["layer.6"])
test_labels = [1, 0, 1, 0]

# Evaluate
results = probe.evaluate(test_acts["layer.6"], test_labels)
print(f"Accuracy: {results.accuracy:.3f}")  # e.g., 0.875 = 87.5%
print(f"F1: {results.f1:.3f}")              # e.g., 0.860

# Interpretation:
# - If accuracy ≈ 0.90 → Layer 6 **strongly encodes sentiment** linearly
# - If accuracy ≈ 0.55 → Layer 6 **weakly encodes sentiment** (barely above random)
# - If accuracy ≈ 0.50 → Layer 6 **does NOT encode sentiment linearly** at all
```

---

---

## 🎯 Example: Concept Activation Vectors (CAV)

Test if "gender" is linearly encoded:

```python
# Prepare concept examples
male_texts = ["The king ruled wisely", "He went to work"]
female_texts = ["The queen ruled wisely", "She went to work"]
labels = [0] * len(male_texts) + [1] * len(female_texts)

# Extract activations
acts = store.extract(male_texts + female_texts, tokenizer, layers=["layer.8"])

# Train CAV
cav = LinearProbe(layer="layer.8", direction="gender", model_type="cav")
cav.fit(acts["layer.8"], labels)

# Get the gender direction vector
gender_direction = cav.get_direction()

# Score new activations along this direction
score = cav.score_activation(new_activation)  # positive = female, negative = male
```

---

## 🔄 Incremental Training (Large Datasets)

For datasets too large to fit in memory:

```python
probe = LinearProbe(
    layer="mlp.10",
    direction="sentiment",
    model_type="logistic",
    incremental=True  # Use SGDClassifier
)

# First batch (must provide all class labels)
probe.partial_fit(batch1_acts, batch1_labels, classes=[0, 1])

# Subsequent batches
for batch_acts, batch_labels in data_loader:
    probe.partial_fit(batch_acts, batch_labels)
```

---

## 📊 Interpreting Results

### What does a high probe accuracy mean?

- The concept is **linearly encoded** at this layer
- A simple linear classifier can separate the classes
- This suggests the model has learned to represent this concept

### What does a low probe accuracy mean?

- The concept may not be encoded at this layer
- OR it's encoded **non-linearly** (a linear probe can't find it)
- Try probing different layers or using non-linear probes

### Limitations

1. **Correlation ≠ Causation**: A probe shows the information _exists_ in activations, not that the model _uses_ it for predictions. Consider **causal tracing** for that.

2. **Probe Complexity**: Too simple → can't find the concept. Too complex → learns it from scratch. L2 regularization helps.

3. **Dataset Bias**: Probes can pick up spurious correlations. Use balanced, diverse probe datasets.

---

## 🗂️ Common Layer Name Patterns

| Model           | Attention Layers              | MLP Layers                       |
| --------------- | ----------------------------- | -------------------------------- |
| BERT/DistilBERT | `encoder.layer.{i}.attention` | `encoder.layer.{i}.intermediate` |
| GPT-2           | `transformer.h.{i}.attn`      | `transformer.h.{i}.mlp`          |
| LLaMA           | `model.layers.{i}.self_attn`  | `model.layers.{i}.mlp`           |
| Mistral         | `model.layers.{i}.self_attn`  | `model.layers.{i}.mlp`           |

Use `store.list_layers()` to discover the exact names for your model.

---

## 📚 Further Reading

- [Anthropic](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) – On the Biology of a Large Language Mode
- [Alain & Bengio (2016)](https://arxiv.org/abs/1610.01644) – Understanding intermediate layers using linear classifier probes
- [Explainability method](https://carpentries-incubator.github.io/fair-explainable-ml/5c-probes.html) – Linear Probes

---
