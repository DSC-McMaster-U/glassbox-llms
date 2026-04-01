import importlib
import logging
import time
from pathlib import Path
from typing import Any

# careful of relative imports
from ..models.base import ModelWrapper
from ..models.factory import create_model_wrapper
from .config import Config
from .preprocessing import start_preprocess
from .tracking import get_tracker

def format_duration(seconds: float) -> str:
    # TODO: theres definitely a library that does this already
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs:.2f}s")

    return " ".join(parts)

class Runner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tracker = get_tracker(cfg)
        self.model: ModelWrapper
        self.dataset = None

    def setup(self):
        logging.info("Setting up model and dataset...")

        # load model
        # This is mainly handled by /models/factory.py
        logging.info(
            f"Attempting model load {self.cfg.model.name} ({self.cfg.model.checkpoint})"
        )

        # make sure the classvalidator that handles auto is called
        device = self.cfg.model.device

        self.model = create_model_wrapper(
            wrapper_type="transformers",
            checkpoint=self.cfg.model.checkpoint,
            device=device,
            dtype=self.cfg.model.dtype,
        )
        logging.info(f"Model successfully loaded on {self.model.device}")

        # load dataset
        logging.info(f"Attempting dataset load {self.cfg.dataset.path}")
        self.dataset = self._load_dataset()
        logging.info(f"Dataset loaded: {len(self.dataset)} samples")

    def _load_dataset(self):
        from datasets import load_dataset

        dataset_path = self.cfg.dataset.path
        dataset_name = self.cfg.dataset.name
        data_dir = self.cfg.dataset.data_dir
        data_files = self.cfg.dataset.data_files
        split = self.cfg.dataset.split
        dataset: Any = None

        dataset = load_dataset(
            path=dataset_path,
            name=dataset_name,
            data_dir=data_dir,
            data_files=data_files,
            split=split
        )

        # apply preprocessing based on config
        if self.cfg.dataset.preprocess:
            dataset = start_preprocess(dataset, self.cfg)

        return dataset

    def run(self):
        logging.info(f"Running experiment: {self.cfg.experiment.type}")

        module_path = None
        start_time = time.time()
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

            # TODO: see if we need a kwargs or if this is all
            # the data any experiment will possibly need
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
            elapsed_time = time.time() - start_time
            formatted_time = format_duration(elapsed_time)
            logging.info(f"Experiment completed in {formatted_time}")
            self.finalize()

    def finalize(self):
        logging.info("Finalizing experiment...")
        self.tracker.finish()
