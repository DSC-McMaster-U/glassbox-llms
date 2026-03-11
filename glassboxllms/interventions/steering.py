import torch

from .base import BaseIntervention


class DirectionalSteering(BaseIntervention):
    """
    Add a constant direction vector to a layer's activations to steer behaviour.

    The intervention computes::

        new_activation = activation + (direction * strength)

    Args:
        layer: Named module path (e.g., ``"transformer.h.11.mlp"``).
        direction: Direction vector (e.g., from a probe's ``get_direction()``).
        strength: Scalar multiplier controlling the steering magnitude.
    """

    def __init__(self, layer: str, direction: torch.Tensor, strength: float = 1.0):
        super().__init__(layer)
        self.direction = direction
        self.strength = strength

    def hook_fn(self, module, input, output):
        """Forward hook that adds the scaled direction to the layer output."""
        # Handle tuple outputs (common for transformer layers)
        if isinstance(output, tuple):
            activation = output[0]
        else:
            activation = output

        # Move direction to match device and dtype
        direction = self.direction.to(device=activation.device, dtype=activation.dtype)

        steered = activation + (direction * self.strength)

        if isinstance(output, tuple):
            return (steered,) + output[1:]
        return steered

    def run(self, model, *args, **kwargs):
        """Register hook, run forward pass, clean up. Returns model output."""
        self.register(model)
        try:
            return model(*args, **kwargs)
        finally:
            self.remove()
