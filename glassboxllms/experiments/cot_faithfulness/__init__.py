"""
CoT Faithfulness Evaluation

Tests whether Chain-of-Thought reasoning in LLMs is faithful
(actually drives answers) or post-hoc rationalization.

Usage:
    from glassboxllms.experiments.cot_faithfulness import CoTFaithfulnessEvaluator

    evaluator = CoTFaithfulnessEvaluator()
    results = evaluator.evaluate(generate_fn, model_name="MyModel", dataset="arc")
    evaluator.compare_to_baseline(results, baseline="Llama-3.1-70B")
"""

from glassboxllms.experiments.cot_faithfulness.evaluator import CoTFaithfulnessEvaluator
from glassboxllms.experiments.cot_faithfulness.tests import truncation_test, error_injection_test
from glassboxllms.experiments.cot_faithfulness.baselines import BASELINES, get_baseline

__all__ = [
    "CoTFaithfulnessEvaluator",
    "truncation_test",
    "error_injection_test",
    "BASELINES",
    "get_baseline",
]
