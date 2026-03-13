import importlib
import logging
from pathlib import Path
from typing import Any, Union

# careful of relative imports
from ..models.base import ModelWrapper
from ..models.factory import create_model_wrapper
from .config import Config
from .tracking import get_tracker


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
        self.model = create_model_wrapper(
            model_name=self.cfg.model.name,
            checkpoint=self.cfg.model.checkpoint,
            device=self.cfg.model.device,
            dtype=self.cfg.model.dtype,
        )
        logging.info(f"Model successfully loaded on {self.model.device}")

        # load dataset
        logging.info(f"Attempting dataset load {self.cfg.dataset.path}")
        self.dataset = self._load_dataset()
        logging.info(f"Dataset loaded: {len(self.dataset)} samples")

    def _load_dataset(self):
        from datasets import load_dataset
        from torch.utils.data import DataLoader, Dataset

        dataset_path = self.cfg.dataset.path
        split = self.cfg.dataset.split
        preprocess_config = self.cfg.dataset.preprocess
        dataset: Any = None

        try:
            # the package datasets is by huggingface
            dataset = load_dataset(dataset_path, split=split)
        except Exception as e:
            # fallback for local/custom datasets
            logging.warning(
                f"Could not load as HF dataset: {e}. Attempting local load..."
            )

            dataset_type = Path(dataset_path).suffix
            if dataset_type in ["csv", "json", "parquet", "arrow", "hdf5"]:
                dataset = load_dataset(dataset_type, dataset_path)

        # if preprocessing is required by the dataset
        if preprocess_config and self.model and hasattr(self.model, "tokenizer"):
            max_length = preprocess_config.get("max_length", 512)
            truncation = preprocess_config.get("truncation", True)
            padding = preprocess_config.get("padding", "max_length")

            def tokenize_function(examples):
                # not 100% sure this is accurate
                text_column = "text" if "text" in examples else "input"
                return self.model.tokenizer(
                    examples[text_column],
                    max_length=max_length,
                    truncation=truncation,
                    padding=padding,
                    return_tensors="pt",
                )

            dataset = dataset.map(tokenize_function, batched=True)
        elif preprocess_config and self.model and not hasattr(self.model, "tokenizer"):
            logging.warning(
                "Preprocessing requested but model has no tokenizer... "
                "Skipping tokenization. Experiment may need to handle preprocessing!"
            )

        return dataset

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
