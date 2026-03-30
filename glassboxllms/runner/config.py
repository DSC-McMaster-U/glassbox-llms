"""
this module defines the Pydantic models used to take in exp configs.
processes experiment parameters, model settings, dataset configurations, tracking.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
import yaml
from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    name: str
    checkpoint: str
    wrapper_type: str = "transformers"
    device: str = "auto"
    dtype: str = "float16"

    @field_validator("device")
    @classmethod
    def detect_device(cls, v: str) -> str:
        # for if the user does not specify a device
        if v == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                # TODO: is mps even supported in glassbox...?
                return "mps"
            else:
                return "cpu"
        return v


class DatasetConfig(BaseModel):
    path: str
    name: Optional[str] = None
    data_dir: Optional[str] = None
    data_files: Optional[Any] = None
    split: str = "train"
    preprocess: Dict[str, Any] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    type: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    seed: int = 67  # :)


class TrackingConfig(BaseModel):
    enabled: bool = False
    type: Optional[str] = None  # this will be "wandb" or "mlflow"
    project: Optional[str] = None
    entity: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    config: Dict[str, Any] = Field(default_factory=dict)


class OutputConfig(BaseModel):
    base_dir: str = "runs/"
    name: str


class Config(BaseModel):
    model: ModelConfig
    dataset: DatasetConfig
    experiment: ExperimentConfig
    tracking: TrackingConfig
    output: OutputConfig


def load_config(path: str) -> Config:
    path_obj = Path(path)
    with open(path_obj, "r") as f:
        if path_obj.suffix in [".yaml", ".yml"]:
            data = yaml.safe_load(f)
        elif path_obj.suffix == ".json":
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {path_obj.suffix}")

    # TODO: error check if this unpack fails
    return Config(**data)
