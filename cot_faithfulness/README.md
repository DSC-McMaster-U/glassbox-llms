# CoT Faithfulness Evaluation

A module for evaluating Chain-of-Thought (CoT) faithfulness in language models.

## What is CoT Faithfulness?

When LLMs use step-by-step reasoning, we want to know:
- **Is the reasoning actually driving the answer?** (Faithful)
- **Or is it just post-hoc rationalization?** (Unfaithful)

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from cot_faithfulness import CoTFaithfulnessEvaluator

# Define how to call your model
def my_model(prompt: str) -> str:
    # Your API call here
    return response_text

# Run evaluation
evaluator = CoTFaithfulnessEvaluator()
results = evaluator.evaluate(
    generate_fn=my_model,
    model_name="MyModel",
    dataset="arc",      # Options: 'arc', 'aqua', 'mmlu'
    n_samples=20,
)

# Compare to baselines
evaluator.compare_to_baseline(results, baseline="Llama-3.1-70B")
```

See `demo.ipynb` for a full walkthrough.

## How It Works

Two tests probe whether CoT reasoning is faithful:

1. **Truncation Test**: Cut off reasoning halfway. If answer changes → faithful
2. **Error Injection Test**: Inject wrong reasoning. If model follows → faithful

## Available Baselines

| Model | Size | Avg Faithfulness (ARC) |
|-------|------|------------------------|
| Llama-3.1-70B | 70B | 67.5% |
| Gemma-3n-E4B | 4B | 42.5% |
| Qwen-3-235B | 235B | 32.5% |

## References

Based on methodology from:
- Lanham et al. (2023) "Measuring Faithfulness in Chain-of-Thought Reasoning"
- Turpin et al. (2024) "Language Models Don't Always Say What They Think"
