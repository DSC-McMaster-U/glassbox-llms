# Runner Module: Contributor Guide

Welcome to the runner module. This document serves as a comprehensive guide for contributors who want to extend or modify the runner's functionality.

## Overview

The runner orchestrates and is responsible for calling experiments. It provides a unified interface for:

- Loading and configuring models from various backends
- Preprocessing datasets
- Running experiments with consistent interfaces
- Tracking metrics and artifacts

## Architecture

```
cli.py (entry point)
    ↓
core.py (Runner class - main orchestration logic)
    ↓
├── tracking.py (metric/artifact logging backends)
├── config.py (Pydantic models for configuration)
├── preprocessing/ (dataset transformation functions)
└── models/factory.py (model backend registry)
```

### Component Responsibilities

| File | Purpose |
|------|---------|
| `cli.py` | Command-line interface, argument parsing, entry point |
| `core.py` | Main `Runner` class that orchestrates setup and execution |
| `config.py` | Pydantic models for configuration validation and loading |
| `tracking.py` | Abstract tracker interface and implementations (WandB, MLflow, NoOp) |
| `preprocessing/` | Dataset transformation utilities |
| `models/factory.py` | Model backend registry and factory function |

## Data Flow

1. **CLI Entry**: User runs `python -m glassboxllms.runner.cli --config <path>`
2. **Config Loading**: `config.py` loads and validates YAML/JSON configuration
3. **Runner Setup**: `Runner.setup()` initializes model and dataset
4. **Experiment Execution**: `Runner.run()` imports and calls experiment's `run_experiment`
5. **Tracking**: Metrics logged via `Tracker` interface throughout execution
6. **Finalization**: Resources cleaned up, tracking session finished

## Configuration Structure

The configuration is defined in `config.py` and supports YAML or JSON formats:

```json
{
  "model": {
    "name": "model-name",
    "checkpoint": "path/to/checkpoint",
    "wrapper_type": "transformers",
    "device": "cuda",
    "dtype": "float16"
  },
  "dataset": {
    "path": "dataset/path",
    "split": "train",
    "preprocess": {
      "columns": ["text"],
      "clean_text": {
        "text_column": "text",
        "lowercase": true
      }
    }
  },
  "experiment": {
    "type": "experiment_name",
    "parameters": {},
    "seed": 67
  },
  "tracking": {
    "enabled": true,
    "type": "wandb",
    "project": "my-project",
    "entity": "my-entity",
    "tags": ["tag1", "tag2"],
    "config": {}
  },
  "output": {
    "base_dir": "runs/",
    "name": "experiment-run"
  }
}
```

## Extending the Runner

### Adding New Model Backends

To add support for a new model backend (e.g., GGUF, custom PyTorch):

1. **Create a new wrapper class** that inherits from `ModelWrapper` in `models/base.py`:

```python
from .base import ModelWrapper

class MyCustomWrapper(ModelWrapper):
    def __init__(self, checkpoint: str):
        super().__init__()
        # Initialize your model here
    
    def forward(self, inputs: Any, **kwargs) -> Any:
        # Implement forward pass
        pass
    
    def get_activations(
        self, inputs: Any, layers: List[str], return_type: str = "numpy"
    ) -> Dict[str, Any]:
        # Implement activation extraction
        pass
    
    def get_layer_module(self, layer: str) -> Any:
        # Return the actual module for hooking
        pass
    
    def get_layer_shape(self, layer: str) -> Tuple[int, ...]:
        # Return layer output shape
        pass
    
    @property
    def layer_names(self) -> List[str]:
        # Return list of available layer identifiers
        pass
    
    @property
    def device(self) -> str:
        # Return device string
        pass
    
    @property
    def model_config(self) -> Dict[str, Any]:
        # Return model metadata
        pass
```

2. **Register the wrapper** in `models/factory.py`:

```python
from .base import ModelWrapper
from .huggingface import TransformersModelWrapper
from .my_custom import MyCustomWrapper  # Import your new wrapper

MODEL_REGISTRY = {
    "transformers": TransformersModelWrapper,
    "my_custom": MyCustomWrapper,  # Add your new entry
}

def create_model_wrapper(
    wrapper_type: str, checkpoint: str, device: str = "cuda", dtype: str = "float16"
) -> ModelWrapper:
    if wrapper_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown wrapper type: {wrapper_type}! Available: {list(MODEL_REGISTRY.keys())}"
        )

    wrapper_class = MODEL_REGISTRY[wrapper_type]

    # Add your wrapper-specific initialization here
    if wrapper_type == "transformers":
        wrapper = wrapper_class(checkpoint)
        wrapper.model.to(device)
        if dtype == "float16":
            wrapper.model.half()
        elif dtype == "float32":
            wrapper.model.float()
        return wrapper
    
    # Add similar initialization for your new wrapper
    elif wrapper_type == "my_custom":
        wrapper = wrapper_class(checkpoint)
        # Custom initialization logic
        return wrapper

    return wrapper_class(checkpoint)
```

**Key Points:**
- The `ModelWrapper` abstract base class defines a stable interface used by other modules
- All wrappers must implement the same methods for compatibility with `instrumentation`, `activation_patching`, and `probes`
- Layer naming should follow consistent patterns: `layer.{i}.attention`, `layer.{i}.mlp`, `embedding`

### Adding New Experiment Types

To add a new experiment:

1. **Create an experiment module** in `glassboxllms/experiments/`:

```python
# glassboxllms/experiments/my_experiment.py

import logging
from typing import Any

from glassboxllms.runner.config import Config
from glassboxllms.runner.tracking import Tracker
from glassboxllms.models.base import ModelWrapper

def run_experiment(
    cfg: Config, 
    model: ModelWrapper, 
    dataset: Any, 
    tracker: Tracker
):
    """
    Main experiment entry point.
    
    Required signature for all experiments:
    - cfg: Configuration object
    - model: ModelWrapper instance
    - dataset: HuggingFace dataset
    - tracker: Tracker instance for logging
    
    Returns:
        None (experiments should log metrics via tracker)
    """
    logging.info("Starting my experiment...")
    
    # Your experiment logic here
    # Example:
    for i, sample in enumerate(dataset):
        # Process sample
        outputs = model.forward(sample["text"])
        
        # Log metrics
        tracker.log({
            "step": i,
            "loss": 0.5,
            "accuracy": 0.9
        })
    
    logging.info("Experiment completed.")
```

2. **Ensure the function is exposed** as `run_experiment` in your module

3. **Reference it in the config** by setting `experiment.type` to your module name:

```json
{
  "experiment": {
    "type": "my_experiment"  // Matches glassboxllms/experiments/my_experiment.py
  }
}
```

**Key Points:**
- The runner imports experiments dynamically using `importlib`
- Every experiment must expose a `run_experiment(cfg, model, dataset, tracker)` function
- Use the `tracker` for all metric logging to ensure consistency across backends
- Experiments can use preprocessing functions from `glassboxllms.runner.preprocessing`

### Extending Preprocessing

The preprocessing module provides modular transformations that can be configured in your config file.

**Available transformations:**

| Transformation | Config Key | Description |
|---------------|------------|-------------|
| Column selection | `columns` | Keep only specified columns |
| Column renaming | `rename` | Rename columns via mapping |
| Sampling | `num_samples` | Sample N rows from dataset |
| Text cleaning | `clean_text` | Lowercase, remove special chars, normalize whitespace |
| Text normalization | `normalize_text` | Strip whitespace, normalize spaces, remove accents |
| Length filtering | `max_length`, `min_length` | Character-based length limits |
| Token limits | `max_tokens`, `min_tokens` | Token-based limits (auto-tokenizes first) |
| Custom transforms | `apply_transform` | Apply custom Python functions |

To use them, import them through the module like `from glassboxllms.runner.preprocessing import ______`.

**Adding new preprocessing functions:**

1. **Implement the function** in `preprocessing/dataset.py` or `preprocessing/tokenizers.py`:

```python
def my_custom_transform(dataset, param1, param2):
    """Apply custom transformation to dataset."""
    def transform_function(examples):
        # Your transformation logic
        examples["new_column"] = [process(x) for x in examples["text"]]
        return examples
    
    return dataset.map(transform_function)
```

2. **Export it** in `preprocessing/__init__.py`:

```python
from .dataset import (
    my_custom_transform,
    # ... other functions
)

__all__ = [
    "my_custom_transform",
    # ... other exports
]
```

3. **Add it to the preprocessing pipeline** in `preprocessing/start.py` (only if you want it to be accessible as a key in the config!):

```python
def start_preprocess(dataset, cfg) -> Any:
    preprocess_config = cfg.dataset.preprocess
    
    # ... existing transforms ...
    
    if "my_custom" in preprocess_config:
        custom_config = preprocess_config.get("my_custom")
        dataset = my_custom_transform(
            dataset,
            param1=custom_config.get("param1"),
            param2=custom_config.get("param2")
        )
        logging.info(f"Applied custom transform")
    
    return dataset
```

However, this is not recommended to keep code clean. A preprocessor function `apply_transforms` is provided that runs custom functions. Custom functions can be supplied to `apply_transforms`'s scope via the `custom.py` function. Linkage has been set up for all the functions in `custom.py` to be accessible in `dataset.py`. [custom.py](preprocessing/custom.py).

### Adding New Tracking Backends

To add support for a new tracking backend (e.g., TensorBoard, Comet):

1. **Create a new tracker class** in `tracking.py`:

```python
class MyTrackingTracker(Tracker):
    def __init__(self, project: str, **kwargs):
        # Initialize your tracking client
        pass
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        # Implement metric logging
        pass
    
    def log_artifact(self, path: str):
        # Implement artifact logging
        pass
    
    def log_figure(self, figure: Any, name: str):
        # Implement figure logging
        pass
    
    def finish(self):
        # Cleanup and finalize tracking session
        pass
```

2. **Register it** in the `get_tracker` factory function:

```python
def get_tracker(cfg: Any) -> Tracker:
    if not cfg.tracking.enabled:
        return NoOpTracker()

    if cfg.tracking.type == "wandb":
        return WandBTracker(...)
    elif cfg.tracking.type == "mlflow":
        return MLflowTracker(...)
    elif cfg.tracking.type == "my_tracking":
        return MyTrackingTracker(
            project=cfg.tracking.project,
            # ... other config
        )
    else:
        raise ValueError(f"Unknown tracker type: {cfg.tracking.type}")
```

## Best Practices

### 1. Logging
- Use `logging.info`, `logging.warning`, `logging.error` for all output
- Include context about what operation is being performed
- Log configuration values during setup for reproducibility

### 2. Error Handling
- Wrap experiment execution in try/finally to ensure cleanup
- The `Runner.finalize()` method is called in a `finally` block
- Use descriptive error messages that include context

### 3. Configuration Validation
- Pydantic models automatically validate config structure
- Provide sensible defaults where appropriate (eg. `.get("your_bool_key", True)`)
- Document required vs optional configuration fields via typing

### 4. Interface Stability
- The `ModelWrapper` interface is stable and used by other modules
- Don't break existing methods when extending functionality
- Follow the established conventions for layers and attributes

### 5. Experiment Design
- Keep experiments focused on a single thing. Just make another if you need to
- Use the `tracker` for all metrics, don't log to a file directly
- Make experiments configurable via the `cfg` parameter
- Set seeds for reproducibility

## Example Patterns

### Running Multiple Samples

```python
def run_experiment(cfg, model, dataset, tracker):
    for i, sample in enumerate(dataset):
        outputs = model.forward(sample["text"])
        metrics = compute_metrics(outputs)
        tracker.log(metrics, step=i)
```

### Using Preprocessing Functions

```python
from glassboxllms.runner.preprocessing import clean_text

def run_experiment(cfg, model, dataset, tracker):
    # Apply custom preprocessing
    dataset = clean_text(dataset, text_column="text", lowercase=True)
    
    # Continue with experiment
```

### Extracting Layer Activations

```python
def run_experiment(cfg, model, dataset, tracker):
    activations = model.get_activations(
        inputs="sample text",
        layers=["layer.0.attention", "layer.1.mlp"],
        return_type="numpy"
    )
    # Use activations for analysis
```

## Helpful Tips

1. **Use dry-run mode** to validate configuration without running experiments:
   ```bash
   python -m glassboxllms.runner.cli --config config.yaml --dry-run
   ```

2. **Check logs** for preprocessing steps and model loading (example):
   ```bash
   python -m glassboxllms.runner.cli --config config.yaml 2>&1 | grep -i "preprocess\|model"
   ```

3. **Verify experiment import** by checking that `run_experiment` exists:
   ```python
   import importlib
   module = importlib.import_module("glassboxllms.experiments.my_experiment")
   assert hasattr(module, "run_experiment")
   ```

## Troubleshooting

### "Unknown wrapper type"
- Check that your wrapper is registered in `MODEL_REGISTRY`
- Verify the `wrapper_type` in your config matches the registry key

### "Experiment does not have run_experiment function"
- Ensure your experiment module exposes `run_experiment` at the module level
- Check the experiment name in config matches the module filename

### "Layer not found"
- Verify layer names match the naming convention
- Check that `layer_names` property returns the correct identifiers
- If you're using a custom backend, check that first

### Preprocessing not applying
- Check that preprocessing keys match exactly (case-sensitive)
- Verify preprocessing config is nested under `dataset.preprocess`
