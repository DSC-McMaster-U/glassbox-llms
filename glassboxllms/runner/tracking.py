"""
provides a unified interface for logging experiment metrics,
artifacts, and plots.
supports multiple backends like WandB and MLflow
through Tracker, or NoOpTracker for disabled tracking.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


# make this abstract
class Tracker(ABC):
    @abstractmethod
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        pass

    @abstractmethod
    def log_artifact(self, path: str):
        pass

    @abstractmethod
    def log_figure(self, figure: Any, name: str):
        pass

    @abstractmethod
    def finish(self):
        pass


class NoOpTracker(Tracker):
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        pass

    def log_artifact(self, path: str):
        pass

    def log_figure(self, figure: Any, name: str):
        pass

    def finish(self):
        pass


class WandBTracker(Tracker):
    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        entity: Optional[str] = None,
        tags: Optional[list] = None,
        config: Optional[Dict] = None,
    ):
        import wandb

        self.run = wandb.init(
            project=project, name=name, entity=entity, tags=tags, config=config
        )

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        self.run.log(metrics, step=step)

    def log_artifact(self, path: str):
        self.run.save(path)

    def log_figure(self, figure: Any, name: str):
        import wandb

        self.run.log({name: wandb.Image(figure)})

    def finish(self):
        self.run.finish()


class MLflowTracker(Tracker):
    def __init__(self, project: str, tags: Optional[list] = None):
        import mlflow

        mlflow.set_experiment(project)
        self.run = mlflow.start_run()
        if tags:
            for tag in tags:
                mlflow.set_tag(tag, True)

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        import mlflow

        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str):
        import mlflow

        mlflow.log_artifact(path)

    def log_figure(self, figure: Any, name: str):
        import mlflow

        # Assuming figure is a matplotlib figure!!
        figure.savefig(f"{name}.png")
        mlflow.log_artifact(f"{name}.png")
        os.remove(f"{name}.png")

    def finish(self):
        import mlflow

        mlflow.end_run()


def get_tracker(cfg: Any) -> Tracker:
    if not cfg.tracking.enabled:
        return NoOpTracker()

    if cfg.tracking.type == "wandb":
        return WandBTracker(
            project=cfg.tracking.project,
            name=cfg.tracking.name,
            entity=cfg.tracking.entity,
            tags=cfg.tracking.tags,
            config=cfg.tracking.config,
        )
    elif cfg.tracking.type == "mlflow":
        return MLflowTracker(project=cfg.tracking.project, tags=cfg.tracking.tags)
    else:
        raise ValueError(f"Unknown tracker type: {cfg.tracking.type}")
