# Glassbox LLMs — Product Vision & Roadmap

## The Honest Current State

### What Actually Works End-to-End (tested with real models)

| Module | Real Models? | Status |
|--------|-------------|--------|
| `TransformersModelWrapper` | ✅ Any HuggingFace model | Works — loads, tokenizes, runs inference, extracts activations |
| `SparseAutoencoder` + `SAETrainer` | ✅ Trains on real activations | Works — TopK/L1 sparsity, dead neuron resampling, checkpointing |
| `LinearProbe` (logistic, ridge, PCA, CAV) | ✅ On frozen activations | Works — trains, evaluates, returns direction vectors |
| `NonLinearProbe` (MLP-based) | ✅ On frozen activations | Works — standard API path |
| `DirectionalSteering` | ✅ Any PyTorch model | Works — hook-based activation addition, context manager |
| `CoTFaithfulnessEvaluator` | ✅ Any LLM with generate_fn | Works — truncation + error injection tests |
| `ExperimentResult` / `run_experiment()` | ✅ Registry works | Works — unified interface for probing + CoT |

### What's Broken or Scaffolding-Only

| Module | Problem |
|--------|---------|
| `CausalScrubber` | **BROKEN** — hook API doesn't match HookManager, store API is wrong |
| `activation_patching.py` | **BROKEN** — assumes `model.hook_manager` which doesn't exist |
| `CircuitGraph` / `CircuitNode` | Data structures only — no discovery algorithms |
| `FeatureAtlas` | Data container only — no integration with SAE outputs |
| Two `ActivationStore` classes | Incompatible APIs — `instrumentation/` vs `primitives/probes/` |
| `NonLinearProbe` ticket API | Calls methods that don't exist on ActivationStore |
| No end-to-end examples | Examples import from non-existent modules |

### Bottom Line
We have ~60% of a real tool. The individual pieces (model loading, SAE training, probing, steering) work. But they don't connect into a pipeline you can actually run. There's no CLI. The circuit analysis is broken. There's no way for a user to go from "I have a model" to "here are its interpretable features" without writing significant glue code.

---

## The Competitive Landscape (What We're Up Against)

| Tool | Approach | Strength | Weakness |
|------|----------|----------|----------|
| **TransformerLens** | Reimplements models | Clean API, great for learning | 2x memory, limited models, numerical divergence |
| **NNsight** | Wraps existing models | Any PyTorch model, exact behavior | Learning curve, HF naming fragmentation |
| **SAELens** | SAE training framework | Best SAE tooling, pre-trained zoo | SAE-only, not general interpretability |
| **pyvene** | Declarative interventions | Architecture-agnostic, composable | Less community, intervention-focused only |
| **nnterp** | Standardized NNsight wrapper | 50+ models, write-once code | Very new, small community |

### The Gap Nobody Fills
**No tool provides a unified CLI-driven pipeline**: load model → extract activations → train SAE → discover features → probe for concepts → discover circuits → steer behavior → generate report. Every existing tool is a point solution. Researchers glue 3-4 tools together manually.

---

## What Would Make Glassbox Actually Useful

### Target Users

1. **ML Safety Researchers** — Need to audit models before deployment. Want: "Run this suite of interpretability checks and tell me what you find."
2. **AI Students/Educators** — Learning mechanistic interpretability. Want: "Show me what's happening inside GPT-2 with one command."
3. **Applied ML Engineers** — Have a fine-tuned model, want to understand it. Want: "What concepts did my model learn? Is it using spurious features?"

### The Product: A CLI-Driven Interpretability Pipeline

```bash
# The dream command
glassbox analyze gpt2 \
  --layer transformer.h.5 \
  --experiments sae,probing,steering \
  --dataset "The cat sat on the mat" \
  --output report.html

# Or step by step
glassbox extract gpt2 --layers all --dataset pile-subset --output activations/
glassbox train-sae activations/ --d-sae 32768 --sparsity topk --k 32
glassbox probe activations/ --concept sentiment --type logistic
glassbox steer gpt2 --direction sentiment --strength 2.0 --prompt "I think this movie is"
glassbox report --format html --include sae,probes,steering
```

### Why This Wins
- **One install, full pipeline** — not 4 separate libraries
- **Works on any HuggingFace model** — no reimplementation needed
- **CLI-first** — researchers can script it, automate it, CI/CD it
- **Produces actionable reports** — not just numbers, but visualizations + natural language summaries
- **Modular** — use the CLI or import individual modules in Python

---

## Proposed Architecture (Future State)

```
glassbox analyze "gpt2" --experiments all
         │
         ▼
┌─────────────────────────────────────────────────────┐
│                    CLI Layer                          │
│  glassbox extract | train-sae | probe | steer | ...  │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              Experiment Runner                       │
│  run_experiment(name, config) → ExperimentResult     │
│  Registered: sae, probing, cot_faithfulness,         │
│              circuit_discovery, steering_eval         │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              Analysis Layer                          │
│  SAETrainer → FeatureSet → Atlas                     │
│  LinearProbe / NonLinearProbe → ProbeResult           │
│  CausalScrubber → CircuitGraph                       │
│  DirectionalSteering → SteeringResult                │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              Instrumentation Layer                    │
│  HookManager ─ ActivationStore ─ ActivationCache     │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              Model Layer                             │
│  TransformersModelWrapper (any HuggingFace model)    │
│  Future: GGUFModelWrapper, vLLMModelWrapper          │
└─────────────────────────────────────────────────────┘

         ┌─────────────────────────────────┐
         │        Output Layer             │
         │  HTML Report │ JSON │ Manim     │
         │  Feature Dashboards             │
         └─────────────────────────────────┘
```

---

## Roadmap: What to Build

### Phase 1: Fix the Foundation (Week 1-2) ← WE ARE HERE

**Already done ✅:**
- [x] Fix TransformersModelWrapper
- [x] Extract DirectionalSteering into interventions/
- [x] Create unified experiment interface (BaseExperiment, registry)
- [x] Build 5 Manim visualization scenes
- [x] Add pyproject.toml

**Still needed:**
- [ ] Fix CausalScrubber to use correct hook API
- [ ] Consolidate the two ActivationStore classes into one
- [ ] Fix activation_patching.py to work with TransformersModelWrapper
- [ ] Create one real end-to-end example (GPT-2 → activations → probe → steer)

### Phase 2: CLI + Real Pipeline (Week 3-4)

**New module: `glassboxllms/cli.py`**
- [ ] `glassbox extract` — Extract activations from any HuggingFace model
- [ ] `glassbox train-sae` — Train SAE on cached activations
- [ ] `glassbox probe` — Train probes on cached activations
- [ ] `glassbox steer` — Run steering experiments
- [ ] `glassbox run` — Run any registered experiment by name

**New module: `glassboxllms/pipeline.py`**
- [ ] `InterpretabilityPipeline` class that chains: model → extract → analyze → report
- [ ] Config-driven: YAML/JSON config files for reproducible runs
- [ ] Progress bars, logging, checkpointing

**Fix the broken modules:**
- [ ] Rewrite `CausalScrubber` with correct PyTorch hook API
- [ ] Unify `ActivationStore` (merge both into one with RAM + disk + extract-from-model)
- [ ] Wire `CircuitGraph` to actually receive data from `CausalScrubber`

### Phase 3: New Experiments & Features (Week 5-8)

**New experiment: Logit Lens**
- [ ] `glassboxllms/experiments/logit_lens.py`
- Project intermediate layer representations through the unembedding matrix
- Shows how the model's "guess" evolves through layers
- Dead simple to implement, extremely popular in MI, we don't have it

**New experiment: Attention Pattern Analysis**
- [ ] `glassboxllms/experiments/attention_analysis.py`
- Extract and analyze attention patterns (induction heads, positional heads, etc.)
- Detect known circuit motifs automatically
- Generate attention heatmaps

**New experiment: Feature Steering Evaluation**
- [ ] `glassboxllms/experiments/steering_eval.py`
- Systematically test steering vectors across prompts
- Measure steering fidelity vs. coherence tradeoff
- Find optimal strength for each direction

**New experiment: Activation Patching (proper implementation)**
- [ ] `glassboxllms/experiments/activation_patching.py`
- Clean/corrupted input comparison
- Layer-by-layer and component-by-component importance
- Produces importance heatmaps compatible with CircuitGraph

**New module: Auto-Interpretation**
- [ ] `glassboxllms/interpretation/auto_interp.py`
- Feed max-activating examples to an LLM and ask "what concept does this feature represent?"
- Uses the model's own tokenizer to find max-activating text spans
- Stores interpretations in FeatureAtlas

### Phase 4: Reports & Dashboards (Week 9-10)

**New module: `glassboxllms/reports/`**
- [ ] HTML report generator (Jinja2 templates)
- [ ] Feature dashboard pages (like Neuronpedia but local)
- [ ] Probe result visualizations
- [ ] Circuit graph interactive viewer
- [ ] Steering experiment comparison tables

**Manim integration:**
- [ ] Auto-generate Manim scenes from experiment results
- [ ] Render attention patterns, activation spaces, circuits from real data

### Phase 5: Scale & Polish (Week 11-12)

- [ ] Multi-GPU support for large models
- [ ] Batch activation extraction with progress bars
- [ ] Pre-trained SAE zoo (download community SAEs)
- [ ] Integration tests on GPT-2, Llama-3.2-1B, Gemma-2-2B
- [ ] Documentation site (MkDocs or Sphinx)
- [ ] PyPI release: `pip install glassboxllms`

---

## Concrete Use Cases (Real Scenarios)

### Use Case 1: "Is my fine-tuned model safe?"
```bash
glassbox analyze my-finetuned-model \
  --experiments probing,steering,cot_faithfulness \
  --safety-concepts toxicity,deception,bias \
  --output safety_report.html
```
The tool trains probes for safety-relevant concepts, tests if the model can be steered toward harmful outputs, and evaluates CoT faithfulness. Outputs a pass/fail safety report.

### Use Case 2: "What did my model learn?"
```bash
glassbox analyze gpt2 \
  --experiments sae,probing \
  --layers transformer.h.0,transformer.h.5,transformer.h.11 \
  --dataset wikitext \
  --output features/
```
Discovers monosemantic features via SAE, probes for known concepts, catalogs everything in a FeatureAtlas. Outputs interactive feature dashboards.

### Use Case 3: "Why does my model get this wrong?"
```bash
glassbox trace gpt2 \
  --clean "The Eiffel Tower is in Paris" \
  --corrupted "The Eiffel Tower is in London" \
  --method activation_patching \
  --output circuit.html
```
Runs activation patching to find which layers/heads are responsible for the factual recall. Outputs a circuit graph showing the critical path.

### Use Case 4: "Explain this for my class"
```bash
glassbox visualize gpt2 \
  --scene full_pipeline \
  --quality high \
  --output showcase.mp4
```
Generates 3Blue1Brown-style Manim animations showing interpretability concepts with real model data.

---

## Priority Matrix

| Feature | Impact | Effort | Priority |
|---------|--------|--------|----------|
| Fix CausalScrubber | High | Medium | P0 |
| Unify ActivationStore | High | Medium | P0 |
| End-to-end GPT-2 example | Very High | Low | P0 |
| CLI (`glassbox` command) | Very High | Medium | P1 |
| Logit Lens experiment | High | Low | P1 |
| Attention Analysis experiment | High | Low | P1 |
| HTML report generator | High | Medium | P1 |
| Steering evaluation experiment | Medium | Medium | P2 |
| Auto-interpretation | High | High | P2 |
| Pre-trained SAE zoo | Medium | Medium | P2 |
| Multi-GPU support | Medium | High | P3 |
| PyPI release | Medium | Low | P3 |

---

## What to Delegate vs Own

### Aaron should own:
- **Manim visualizations** (already done, iterate on quality)
- **End-to-end example/demo** (you understand the full system)
- **CLI interface** (you're the integration person)
- **HTML report generation** (ties into viz work)

### Delegate to team:
- **CausalScrubber rewrite** → Ankita (she wrote the original)
- **ActivationStore unification** → Jason (he owns both versions)
- **Logit Lens experiment** → good first issue for a new contributor
- **Attention Analysis** → could be Stella's next task
- **SAE experiment CLI wrapper** → Uday (he owns SAEExperiment)
- **Steering evaluation** → Spencer (he owns DirectionalSteering)
- **Auto-interpretation** → Lukhsaan or Peter

---

## The One-Liner Pitch

> **glassbox-llms**: The first unified CLI for LLM interpretability. Load any HuggingFace model, discover features with SAEs, probe for concepts, trace circuits, steer behavior, and generate reports — all from one `pip install`.
