import importlib
import logging
from typing import Any

from .config import Config
from .tracking import get_tracker


class Runner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tracker = get_tracker(cfg)
        self.model = None
        self.dataset = None

    def setup(self):
        """Loads model and dataset based on config."""
        logging.info("Setting up model and dataset...")
        # TODO: Set this up???
        # Should load datasets/models with whatever api we settle on
        pass

    def run(self):
        logging.info(f"Running experiment: {self.cfg.experiment.type}")

        module_path = None
        try:
            # TODO: Make sure this is actually the right thing if we change how experiments work
            # Currently, this assumes there is an exposed interface run_experiment for each experiment
            # Also assumes that there will acc be experiemnts in glassbox.experiments (does it get imported lke that?)
            module_path = f"glassboxllms.experiments.{self.cfg.experiment.type}"
            experiment_module = importlib.import_module(module_path)

            if not hasattr(experiment_module, "run_experiment"):
                raise AttributeError(
                    f"Experiment {self.cfg.experiment.type} does not have a 'run_experiment' function."
                )

            experiment_module.run_experiment(
                cfg=self.cfg,
                model=self.model,
                dataset=self.dataset,
                tracker=self.tracker,
            )

        except ImportError as e:
            logging.error(
                f"Could not import experiment module: {module_path if module_path else '(Path not found!)'}"
            )
            raise e
        finally:
            self.finalize()

    def finalize(self):
        logging.info("Finalizing experiment...")
        self.tracker.finish()
