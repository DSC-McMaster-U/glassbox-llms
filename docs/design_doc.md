
# 1. Overview

The Glassbox Library provides a modular, lightweight, and framework-agnostic foundation for exploring LLM interpretability. Its architecture prioritizes:

## Independence between modules

- Extensibility for future research directions

- Testability — all modules can be unit tested in isolation

- Minimal external dependencies, relying only on standard Python and NumPy/matplotlib where necessary

- The code entry point (e.g., main.py or cli.py) can integrate with models, datasets, or APIs, but the core modules must remain agnostic to them.

# 2. Core Modules
## GlassModel

### Purpose:

Provides a generic interface for model abstraction, activation extraction, and causal experimentation.

### Responsibilities:

- Manage hooks to extract activations from arbitrary model layers.

- Provide a consistent API for forward passes, activation capture, and patching.

- Stay independent of any specific framework (e.g., PyTorch, TensorFlow).

- Optional adapters (e.g., HuggingFace) should live in glassbox.adapters.

### Design Principles:

- Define only abstract data flow: encode() → forward() → get_activations().

- Accept generic tensor types (np.ndarray, torch.Tensor, etc.) via typing.Union.

- Support lightweight dependency injection for model backends.

### Testability:

- Use mock models (simple linear or dictionary-based) for unit tests.

- Verify logic of hooks, activation caching, and patching without real networks.

## GlassProbe

### Purpose:
- Implements probe interfaces for testing hypotheses about what information is represented in activations.

### Responsibilities:

- Provide a clean, dependency-light probe API: fit(), predict(), evaluate().

- Support any backend (NumPy, scikit-learn, PyTorch) via adapter registration.

- Remain independent of both GlassModel and external datasets.

### Design Principles:

- Treat probes as pluggable components — linear classifiers by default.

- Expose metrics as return values (accuracy, F1, mutual information).

- Avoid coupling to specific feature shapes or tokenizers.

### Testability:

- Use synthetic activations (NumPy arrays) and labels to verify training logic.

- Ensure reproducible results without relying on model weights.

## GlassViz

### Purpose:
Provide visualization utilities for attention, attribution, and activation distributions.

### Responsibilities:

- Offer standardized plotting functions for interpretability outputs:

- attention_map()

- attribution_heatmap()

- activation_hist()

- Stay independent of modeling frameworks and only depend on matplotlib.

### Design Principles:

- Pure visualization — no model calls.

- Accept simple Python lists or arrays; return figure handles for testability.

- All visuals should be reproducible and usable in notebooks or dashboards.

### Testability:

- Validate that plotting functions run without exceptions and return a matplotlib.Figure.

- Avoid rendering-heavy tests (mock plt.show()).

<pre>
+-------------------+       +----------------+       +----------------+
|   GlassModel      |-----> |   GlassProbe   |-----> |   GlassViz     |
|  (model wrapper)  |       | (representation tests) | (visual tools) |
+-------------------+       +----------------+       +----------------+
           ↑                        ↑                        ↑
           |                        |                        |
           |     (interfaces only)  |     (interfaces only)  |
           +------------------------+------------------------+
</pre>




Each module is standalone: they can be imported independently.

Shared types (e.g., TensorLike, ActivationDict) reside in glassbox/types.py.

Optional integrations live in glassbox/adapters/ (for HF, OpenAI, etc.).

Tests live in tests/ and mock only internal logic.

# 4. Development & Testing Best Practices
### Principle	Implementation

***Loose Coupling:*** Avoid direct imports between modules (e.g., GlassProbe never calls GlassModel).

***Dependency Injection:*** Allow backend objects (e.g., models, datasets) to be passed at runtime.

***Unit Testing:*** Mock external dependencies; test pure logic and data flow.

***Linter/Formatter:***	Use ruff for linting and black/PEP8 conventions.

***Type Checking:*** Enforce mypy strictness for clean API boundaries.

***Documentation:*** Each module includes docstrings and usage examples.

# 5. Example Usage (Conceptual)
from glassbox import GlassModel, GlassProbe, GlassViz

## 1. Load model & capture activations
gm = GlassModel.from_hf("bert-base-uncased")
acts = gm.get_activations(inputs=["The cat sat on the mat"], layers=[1, 3, 5])

## 2. Train a probe on the activations
probe = GlassProbe(task_name="syntax")
probe.fit(X=acts["layer_3"], labels=[0, 1])
acc = probe.evaluate(X=acts["layer_3"], labels=[0, 1])

## 3. Visualize attention or attribution
viz = GlassViz()
viz.attention_map(attn=[[0.2, 0.8], [0.5, 0.5]], tokens=["The", "cat"])

# 6. Future Extensions

glassbox.adapters → interfaces for frameworks (HuggingFace, OpenAI, Anthropic).

glassbox.metrics → standardized interpretability evaluation metrics.

glassbox.data → synthetic datasets for probing experiments.

glassbox.cli → optional command-line integration.

glassbox.web → optional visualization dashboard.