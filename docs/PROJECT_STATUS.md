# Glassbox LLMs: Project Status

A quick guide to what exists today and where the project is headed.

---

## What This Project Is

Glassbox LLMs is an open-source library for **LLM interpretability** — understanding *why* models produce certain outputs. The goal is to make it easy to add interpretability to any codebase that uses LLMs (HuggingFace, LangChain, etc.).

---

## Current Features (What Exists Today)

### 1. Models (`glassboxllms/models/`)

**ModelWrapper (base.py)** — Abstract interface for model backends.

- Defines: `forward()`, `get_activations()`, `get_layer_module()`, `get_layer_shape()`, `layer_names`, `device`, `model_config`
- Designed for HuggingFace, GGUF, or custom architectures

**TransformersModelWrapper (huggingface.py)** — Concrete HuggingFace implementation.

- Wraps any AutoModel-compatible model
- Supports `model_class="auto"` (default) and `model_class="causal_lm"` (adds `lm_head` access)
- Forward hooks for activation extraction
- **Status**: Working, tested

### 2. Instrumentation (`glassboxllms/instrumentation/`)

**HookManager (hook_manager.py)** — Manages PyTorch forward hooks.
**ActivationStore (activations.py)** — Buffer + disk storage for activations (pt/safetensors).
**ActivationExtractor (extractor.py)** — High-level extraction with pooling modes.
**Activation Patching (activation_patching.py)** — Temporary activation replacement for causal experiments.

- **Status**: All working, tested

### 3. Features (`glassboxllms/features/`)

**SparseAutoencoder (sae.py)** — TopK or L1 sparse autoencoder for feature decomposition.
**SAETrainer (trainer.py)** — Training loop with dead neuron handling, resampling, mixed precision.
**FeatureSet (feature_set.py)** — Serializable container for SAE features (SafeTensors format).
**SAEFeature (feature.py)** — Individual feature with decoder vector and activation stats.

- **Status**: Working, extensive test coverage (37+ SAE tests, 14+ integration tests)

### 4. Primitives — Probes (`glassboxllms/primitives/probes/`)

**LinearProbe (linear.py)** — Logistic, ridge, PCA, CAV probes on frozen activations.
**NonLinearProbe (nonlinear.py)** — MLP-based probing for complex concepts.
**ProbeResult** — Dataclass with accuracy, f1, coefficients (direction vector).

- **Status**: Working, tested

### 5. Analysis — Circuits & Feature Atlas (`glassboxllms/analysis/`)

**CircuitGraph (circuits/graph.py)** — Directed graph with typed nodes/edges, JSON serialization.
**CircuitDiscoveryExperiment (circuits/discovery.py)** — Attribution + connectivity + pruning pipeline.
**CausalScrubber (circuits/causal_scrubbing.py)** — Path patching with multiple strategies.
**Feature Atlas (feature_atlas/)** — Registry for discovered features with search/filter.

- **Status**: Working, tested (58+ circuit graph tests, 5+ discovery tests)

### 6. Interventions (`glassboxllms/interventions/`)

**DirectionalSteering (steering.py)** — Add a direction vector to layer activations.
**BaseIntervention (base.py)** — Abstract hook-based intervention with context manager support.

- **Status**: Working, tested

### 7. Pipeline (`glassboxllms/pipeline.py`)

High-level glue layer for end-to-end workflows:
- `extract_activations()` — model → layer activations
- `train_sae_on_model()` — model → SAE + FeatureSet
- `train_probe_on_model()` — model → probe + direction vector
- `discover_circuit()` — model → CircuitGraph (with inter-layer edges)
- `run_logit_lens()` — model → layer-by-layer predictions
- `steer_on_model()` — model → before/after steered activations

- **Status**: Working, tested (22+ tests)

### 8. Visualization (`glassboxllms/visualization/`)

Three-tier visualization stack:

**Static Plots (plots.py)** — 7 matplotlib functions: attention heatmaps, logit lens, SAE training curves, sparsity, probe accuracy, circuit graphs, steering effects.

**Interactive Plots (interactive.py)** — 3 Plotly functions: feature browser, circuit explorer, 3D embedding scatter (PCA/t-SNE/UMAP).

**Manim Scenes (scenes.py + manim_scenes/)** — 5 adapter-connected scenes + 5 cinematic scenes for showcase videos.

**Adapters (adapters.py)** — 5 functions converting real analysis objects to scene-renderable data (CircuitGraph → CircuitSceneData, ProbeResult → ProbeSceneData, etc.).

- **Status**: Working, tested (33+ plot tests, 22+ adapter tests)

### 9. Experiments (`glassboxllms/experiments/`)

**BaseExperiment** — Standard interface with ExperimentResult dataclass.
**LogitLensExperiment** — Layer-by-layer prediction convergence analysis.
**ProbingExperiment** — Automated probe sweep across layers.
**CoT Faithfulness** — Chain-of-thought faithfulness evaluation (truncation + error injection tests).

- **Status**: Working

### 10. CLI (`glassboxllms/cli/`)

Command-line interface for running experiments, extracting activations, and exploring models.

- **Status**: Basic implementation, tested

### 11. Demo (`examples/`)

**generate_demo_artifacts.py** — Runs full pipeline on GPT-2 and produces artifact bundle:
- logit_lens.png, probe.png, circuit.png, steering.png (real data)
- features.html (interactive Plotly)
- metrics.json, demo_summary.md

**full_pipeline_demo.py** — Demonstrates all pipeline functions end-to-end.

---

## Test Coverage

- **302+ tests passing** across 16 test files
- Key coverage: SAE (37), circuit graph (58), pipeline (22+), visualization (33), adapters (22), instrumentation (22)

---

## File Structure

```
glassboxllms/
├── models/           # ModelWrapper (abstract) + TransformersModelWrapper (HuggingFace)
├── instrumentation/  # hook_manager, activations, extractor, activation_patching
├── primitives/       # probes (linear, nonlinear), attribution (integrated gradients)
├── features/         # SparseAutoencoder, SAETrainer, FeatureSet, SAEFeature
├── analysis/         # circuits (graph, discovery, causal_scrubbing), feature_atlas
├── interventions/    # DirectionalSteering, BaseIntervention
├── experiments/      # base, logit_lens, probing, cot_faithfulness
├── visualization/    # plots (static), interactive (Plotly), scenes + manim_scenes, adapters
├── cli/              # Command-line interface
├── pipeline.py       # High-level glue (6 functions)
└── utils/            # Logging utilities
```

---

## Where the Project Is Headed

### Near-term
1. **Demo polish** — Generate artifact bundle, cinematic showcase video
2. **Fold open PRs** — #69 (SAE experiment), #72 (probing), #79 (patching) into experiment framework

### Medium-term
3. **Larger model support** — Caching, streaming activations for models that don't fit in memory
4. **Cloud integration** — Log to W&B, S3; Streamlit/Gradio dashboards
5. **Evaluation** — Bias detection, hallucination tracking

### Longer-term
6. **PyPI release** — Package, docs, tutorials, v0.1
7. **Real circuit discovery** — Attribution-based node selection with proper connectivity functions

---

*Last updated: March 2026*
