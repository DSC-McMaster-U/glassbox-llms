from typing import Type

from .base import ModelWrapper
from .huggingface import TransformersModelWrapper

MODEL_REGISTRY = {
    "transformers": TransformersModelWrapper,
    # TODO: Add more entries here
    # "name": "name of the wrapper type in the definition file"
}


def create_model_wrapper(
    model_name: str, checkpoint: str, device: str = "cuda", dtype: str = "float16"
) -> ModelWrapper:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_name}! Available: {list(MODEL_REGISTRY.keys())}"
        )

    wrapper_class = MODEL_REGISTRY[model_name]

    # this must be expanded EVERY TIME you expand the registry
    if model_name == "transformers":
        wrapper = wrapper_class(checkpoint)
        wrapper.model.to(device)
        if dtype == "float16":
            wrapper.model.half()
        elif dtype == "float32":
            wrapper.model.float()
        return wrapper

    return wrapper_class(checkpoint)
