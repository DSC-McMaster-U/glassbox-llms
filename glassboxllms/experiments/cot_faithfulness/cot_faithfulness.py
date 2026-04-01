import logging
from typing import Any
import os
from datetime import datetime
import torch

from glassboxllms.runner.config import Config
from glassboxllms.runner.tracking import Tracker
from glassboxllms.models.base import ModelWrapper
from glassboxllms.experiments.cot_faithfulness.evaluator import CoTFaithfulnessEvaluator


def run_experiment(
    cfg: Config,
    model: ModelWrapper,
    dataset: Any,
    tracker: Tracker
):
    """
    Wrapper for cot_faithfulness.
    Wrapper instead of refactor because it's being grandfathered in from an older experiment format
    """
    logging.info("Starting CoT faithfulness experiment...")

    # evaluator from the old format
    evaluator = CoTFaithfulnessEvaluator(seed=cfg.experiment.seed)

    def generate_fn(prompt: str) -> str:
        # Use model.generate() which returns decoded text string
        return model.generate(prompt, max_new_tokens=256)

    # === RUNNING EVAL ===
    logging.info("Running evaluation...")
    dataset_name = cfg.dataset.name if cfg.dataset.name else ""
    results = evaluator.evaluate(
        generate_fn=generate_fn,
        model_name=cfg.model.name,
        dataset=dataset_name,
        n_samples=cfg.experiment.parameters.get("n_samples", 20),
        verbose=True
    )

    output_path = os.path.join(cfg.output.base_dir, cfg.output.name)
    os.makedirs(output_path, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(os.path.join(output_path, f"cot_faithfulness_{timestamp}.txt"), "w") as f:
        f.write("== COT FAITHFULNESS EXPERIMENT RUN ==\n")
        f.write(str(datetime.now()) + "\n\n")
        f.write(f"truncation_faithfulness: {results.truncation_faithfulness}\n")
        f.write(f"error_following: {results.error_following}\n")
        f.write(f"avg_faithfulness: {results.avg_faithfulness}\n")
        f.write(f"n_samples: {results.n_samples}\n============================")

    tracker.log({
        "truncation_faithfulness": results.truncation_faithfulness,
        "error_following": results.error_following,
        "avg_faithfulness": results.avg_faithfulness,
        "n_samples": results.n_samples
    })

    logging.info("CoT faithfulness experiment completed. Saved to file.")
