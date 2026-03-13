import logging
from typing import Any

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

    # some random work
    metrics = {"accuracy": 0.95, "loss": 0.05}

    # log it using tracker interface
    tracker.log(metrics)

    logging.info("This is Information log text the runner will emit.")
    logging.warning("This is Warning log text the runner may emit.")
    logging.error("This is Error log text the runner will (hopefuly not) emit.")
    logging.info("Dummy experiment completed.")
