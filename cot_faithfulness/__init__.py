"""
CoT Faithfulness Evaluation Module

Tests whether Chain-of-Thought reasoning in LLMs is faithful (actually drives answers)
or post-hoc (rationalization after the fact).

Example usage:
    >>> from cot_faithfulness import CoTFaithfulnessEvaluator
    >>> 
    >>> def my_model(prompt: str) -> str:
    ...     # Your model's generate function
    ...     return response_text
    >>> 
    >>> evaluator = CoTFaithfulnessEvaluator()
    >>> results = evaluator.evaluate(my_model, dataset="arc", n_samples=20)
    >>> evaluator.compare_to_baseline(results, baseline="Llama-3.1-70B")
"""

from cot_faithfulness.evaluator import CoTFaithfulnessEvaluator
from cot_faithfulness.tests import truncation_test, error_injection_test
from cot_faithfulness.baselines import BASELINES, get_baseline

__all__ = [
    "CoTFaithfulnessEvaluator",
    "truncation_test",
    "error_injection_test", 
    "BASELINES",
    "get_baseline",
]
