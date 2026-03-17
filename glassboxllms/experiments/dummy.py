import logging
from typing import Any
import os
from datetime import datetime

from glassboxllms.runner.config import Config
from glassboxllms.runner.tracking import Tracker


def run_experiment(cfg: Config, model: Any, dataset: Any, tracker: Tracker):
    """
    A simple dummy experiment to test the runner.
    """
    logging.info("DUMMY || DUMMY || DUMMY || DUMMY")
    logging.info(
        "This is a dummy experiment to demonstrate the runner/config functionality."
    )

    # here's accessing information about the model
    logging.info(f"Model: {cfg.model.name} (Checkpoint: {cfg.model.checkpoint})")
    logging.info(f"Device: {cfg.model.device}, Dtype: {cfg.model.dtype}")

    # here's accessing information about the dataset
    logging.info(f"Dataset: {cfg.dataset.path} (Split: {cfg.dataset.split})")
    if cfg.dataset.preprocess:
        logging.info(f"Preprocessing: {cfg.dataset.preprocess}")

    # accessing tracker information
    logging.info(f"Tracking Enabled: {cfg.tracking.enabled}")
    if cfg.tracking.enabled:
        logging.info(f"Tracker Type: {cfg.tracking.type}, Project: {cfg.tracking.project}")

    # some random work
    metrics = {"total_samples": 67, "accuracy": 0.95, "loss": 0.05}

    # here's accessing arbitrary parameters provided
    # in the config's experiment's parameters key
    lr = cfg.experiment.parameters.get("lr")
    bsz = cfg.experiment.parameters.get("bsz")
    time = cfg.experiment.parameters.get("time")
    logging.info(f"Experiment parameters: lr={lr}, bsz={bsz}, time={time}")

    # log it using tracker interface
    tracker.log(metrics)

    logging.info("This is Information log text the runner will emit.")
    logging.warning("This is Warning log text the runner may emit.")
    logging.error("This is Error log text the runner will (hopefuly not) emit.")
    logging.info("Dummy experiment completed.")

    # here's how you can make it output things
    output_path = os.path.join(cfg.output.base_dir, cfg.output.name)
    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(output_path, "results.txt"), "w") as f:
        f.write("== DUMMY EXPERIMENT RUN ==\n")
        f.write(str(datetime.now()) + "\n\n")
        f.write(f"Some metrics you supplied in the config: lr={lr}, bsz={bsz}, and time={time}\n")
        f.write(f"Some useful data we just got from this experiment: {metrics}")

    logging.info(f"Experiment summary saved to {output_path}")
    logging.info("--- Dummy experiment completed! ---")
