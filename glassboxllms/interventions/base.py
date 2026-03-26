from abc import ABC, abstractmethod


class BaseIntervention(ABC):
    """
    Abstract base class for all model interventions.

    Provides a standard interface for attaching/removing forward hooks
    that modify layer activations. Supports context-manager usage::

        with DirectionalSteering(layer, direction, strength) as s:
            s.register(model)
            output = model(input)
    """

    def __init__(self, layer: str):
        self.layer = layer
        self.handle = None

    @abstractmethod
    def hook_fn(self, module, input, output):
        """The activation-modification logic. Implemented by subclasses."""
        ...

    def register(self, model):
        """Attach the hook to *layer* inside *model* (a ``torch.nn.Module``)."""
        target_module = dict(model.named_modules()).get(self.layer)
        if target_module is None:
            raise ValueError(f"Layer '{self.layer}' not found in model.")
        self.handle = target_module.register_forward_hook(self.hook_fn)

    def remove(self):
        """Remove the hook, restoring original model behaviour."""
        if self.handle:
            self.handle.remove()
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()
