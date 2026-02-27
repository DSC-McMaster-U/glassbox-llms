import torch
from .base import BaseIntervention

class DirectionalSteering(BaseIntervention):
    """
    Adds a constant vector to the activations of a specific layer 
    to steer model behavior.
    """
    def __init__(self, layer: str, direction: torch.Tensor, strength: float = 1.0):
        super().__init__(layer)
        self.direction = direction
        self.strength = strength

    def hook_fn(self, module, input, output):
        """
        PyTorch forward hook function.
        'output' is the activation tensor at this layer.
        """
        # Ensure direction is on the same device (CPU/GPU) and datatype as the output
        if self.direction.device != output.device:
            self.direction = self.direction.to(output.device)
        
        if self.direction.dtype != output.dtype:
            self.direction = self.direction.to(output.dtype)

        # Apply the steering: New Act = Old Act + (Strength * Vector)
        # We generally add to the residual stream or output of the layer.
        # Note: We rely on broadcasting for batch size and sequence length.
        steered_output = output + (self.direction * self.strength)
        
        return steered_output

    def run(self, model, *args, **kwargs):
        """
        Convenience method to register, run a forward pass, and cleanup.
        """
        self.register(model)
        try:
            return model(*args, **kwargs)
        finally:
            self.remove()