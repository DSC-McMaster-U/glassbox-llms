# Glassbox Demo Finalization Plan (March 23, 2026)

## 1) Branch Snapshot

Current branch: `fix/transformers-wrapper`  
Status after integration and test hygiene:

- `300 passed, 4 skipped` (`pytest -q`)
- Visualization stack now includes:
  - static plots (`glassboxllms/visualization/plots.py`)
  - interactive views (`glassboxllms/visualization/interactive.py`)
  - adapter layer for real objects (`glassboxllms/visualization/adapters.py`)
  - data-driven scenes (`glassboxllms/visualization/scenes.py`)
  - cinematic scenes (`glassboxllms/visualization/manim_scenes/`)
- Pipeline glue now exists (`glassboxllms/pipeline.py`) for:
  - activation extraction
  - probe training
  - SAE training
  - circuit discovery scan

## 2) What Was Integrated In This Pass

Integrated commits:

- `fc7e62e` — visualization static/interactive module (from issue-74 track)
- `94b86a5` — adapter + data-driven scene bridge (from issue-73 track)
- `143bc7b` — end-to-end pipeline glue + demo script (from issue-77 track)
- `df744cf`, `2a9783f` — open PR #71 auto circuit discovery + tests

Additional stabilization:

- Fixed stale tests to match current APIs:
  - `tests/test_activation_patching.py`
  - `tests/test_activation_store.py`
- Updated optional-dependency handling in tests:
  - skip safetensors-dependent tests when `safetensors` is unavailable
- Exported `NonLinearProbe` in `glassboxllms/primitives/probes/__init__.py`

## 3) Open PR Triage (Current)

### Merge-ready / already integrated

- `#71` CircuitDiscoveryExperiment: integrated into this branch.

### Investigated, **do not merge directly** without rework

- `#72` probing experiment:
  - uses unresolved imports (`from utils import ...`)
  - does not fit current experiment registry/base interfaces
  - should be rewritten as `BaseExperiment` implementation
- `#79` patching experiment:
  - useful concept, but duplicates existing activation patching pathway
  - not integrated with registry/CLI or shared wrapper abstractions
- `#69` SAE experiment:
  - large vertical script, heavy dataset/runtime assumptions
  - partial overlap with existing pipeline + SAE trainer
  - should be refactored into registry-compatible experiment entrypoints
- `#70` runner:
  - broad framework addition (`runner/`, tracking deps, config system)
  - high integration risk for tonight’s demo timeline
  - treat as separate track after demo lock

## 4) Demo Objective (Twitter-Ready)

Ship a short, compelling artifact that proves:

1. We can inspect internals of a real model.
2. We can discover concepts/features/circuits.
3. We can causally intervene and change behavior.
4. The workflow is reproducible via one pipeline.

## 5) Tonight Execution Plan (Parallel-Friendly)

## P0 (Must Ship Tonight)

1. **Single hero narrative**
   - Prompt pair: clean/corrupted factual prompt + sentiment steering pair.
   - Fix one model target (`gpt2` for speed/reliability).

2. **Generate real result artifacts**
   - Run pipeline functions on a small curated text set.
   - Save:
     - logit-lens heatmap
     - probe separation + metrics
     - SAE top-feature browser HTML
     - circuit graph figure + summary JSON
     - before/after steering token distribution plot

3. **One short cinematic video (20-35s)**
   - Use `FullPipelineScene` or data-driven scene sequence.
   - Overlay only metrics actually produced from run artifacts.

4. **One “proof” markdown summary**
   - Inputs, model, layers, key quantitative outputs, links to artifacts.

## P1 (High Value If Time Remains)

1. Add `glassbox demo` CLI command:
   - run small end-to-end pipeline
   - emit files into `outputs/demo_<timestamp>/`
2. Add HTML report scaffold:
   - key plots + metrics + model metadata + run config
3. Add reproducibility config file:
   - model, layers, prompts, probe labels, SAE params, seed

## P2 (Post-Demo)

1. Fold #69/#79 into unified experiment framework.
2. Decide on runner track (#70) architecture direction.
3. Add larger-model support and caching strategy.

## 6) Suggested Agent Split (For Parallel Work Tonight)

1. **Agent A: Artifact generation**
   - Own `examples/full_pipeline_demo.py` and output writer helpers.
   - Goal: deterministic result bundle in `outputs/demo_run/`.

2. **Agent B: Visual packaging**
   - Own `glassboxllms/visualization/*` and scene parameter wiring.
   - Goal: consistent visual language + one polished video render path.

3. **Agent C: CLI/report UX**
   - Own `glassboxllms/cli/main.py` and report template module.
   - Goal: single command for “demo build”.

4. **Agent D: Evaluation guardrails**
   - Own tests for demo command + regression checks for artifact schema.
   - Goal: fail-fast if demo output contract breaks.

## 7) Proposed Output Contract

For each demo run, write:

- `outputs/demo_run/metrics.json`
- `outputs/demo_run/logit_lens.png`
- `outputs/demo_run/probe.png`
- `outputs/demo_run/steering.png`
- `outputs/demo_run/circuit.png`
- `outputs/demo_run/features.html`
- `outputs/demo_run/demo_summary.md`
- `outputs/demo_run/showcase.mp4` (optional if render time allows)

## 8) Critical Risks To Watch

1. **Synthetic vs real-data mismatch** in scene overlays.
2. **Runtime slowness** on full-model passes; use tiny datasets for demo path.
3. **Dependency drift** (`manim`, `plotly`, optional `safetensors`).
4. **Narrative overload**; keep to one coherent storyline.

## 9) Immediate Next Commands

Run this baseline before any additional edits:

```bash
pytest -q
pytest -q tests/test_visualization.py tests/test_visualization_adapters.py tests/test_pipeline.py
```

Then generate first artifact bundle:

```bash
python3 examples/full_pipeline_demo.py
```

Then render showcase:

```bash
bash examples/render_showcase.sh -ql
```

