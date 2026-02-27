from abc import ABC, abstractmethod
import torch

class BaseIntervention(ABC):
    """
    Abstract base class for all model interventions.
    Ensures that every intervention has a standard way to be applied and removed.
    """
    def __init__(self, layer: str):
        self.layer = layer
        self.handle = None # Stores the hook handle so we can remove it later

    @abstractmethod
    def hook_fn(self, module, input, output):
        """
        The actual logic that modifies the activations.
        Must be implemented by child classes.
        """
        pass

    def register(self, model):
        """
        Attaches the hook to the specified layer in the model.
        Assumes 'model' is a PyTorch module with named modules.
        """
        # Find the correct layer by name
        target_module = dict(model.named_modules()).get(self.layer)
        if target_module is None:
            raise ValueError(f"Layer '{self.layer}' not found in model.")
        
        # Register the forward hook
        self.handle = target_module.register_forward_hook(self.hook_fn)

    def remove(self):
        """Removes the hook from the model."""
        if self.handle:
            self.handle.remove()
            self.handle = None

    def __enter__(self):
        """Allows usage as a context manager (with statement)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Automatically removes the hook when exiting the context."""
        self.remove()