from typing import Type

from .base import ModelWrapper
from .huggingface import TransformersModelWrapper

MODEL_REGISTRY = {
    "transformers": TransformersModelWrapper,
    # TODO: Add more entries here
    # "name": "name of the wrapper type in the definition file"
}


def create_model_wrapper(
    wrapper_type: str, checkpoint: str, device: str = "cuda", dtype: str = "float16"
) -> ModelWrapper:
    if wrapper_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown wrapper type: {wrapper_type}! Available: {list(MODEL_REGISTRY.keys())}"
        )

    wrapper_class = MODEL_REGISTRY[wrapper_type]

    # this must be expanded EVERY TIME you expand the registry
    if wrapper_type == "transformers":
        wrapper = wrapper_class(checkpoint)
        wrapper.model.to(device)
        if dtype == "float16":
            wrapper.model.half()
        elif dtype == "float32":
            wrapper.model.float()
        return wrapper

    return wrapper_class(checkpoint)
