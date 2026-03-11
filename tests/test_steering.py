import torch
import torch.nn as nn
import pytest

from glassboxllms.interventions import DirectionalSteering, BaseIntervention


class TestDirectionalSteering:
    """Tests for the DirectionalSteering intervention."""

    def test_steering_changes_output(self):
        """Steering should modify the layer's output."""
        model = nn.Linear(4, 2)
        input_tensor = torch.randn(1, 4)
        direction = torch.ones(2)

        original = model(input_tensor).clone()

        steering = DirectionalSteering(layer="", direction=direction, strength=5.0)
        handle = model.register_forward_hook(steering.hook_fn)
        steered = model(input_tensor)
        handle.remove()

        assert not torch.allclose(original, steered), "Steering should change output"

    def test_hook_removal_restores_output(self):
        """After removing the hook, output should match the original."""
        model = nn.Linear(4, 2)
        input_tensor = torch.randn(1, 4)
        direction = torch.ones(2)

        original = model(input_tensor).clone()

        steering = DirectionalSteering(layer="", direction=direction, strength=5.0)
        handle = model.register_forward_hook(steering.hook_fn)
        _ = model(input_tensor)
        handle.remove()

        restored = model(input_tensor)
        assert torch.allclose(original, restored), "Output should restore after hook removal"

    def test_context_manager(self):
        """Context manager should auto-remove hooks."""
        model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
        model.eval()
        input_tensor = torch.randn(1, 4)
        direction = torch.ones(4)

        original = model(input_tensor).clone()

        steering = DirectionalSteering(layer="0", direction=direction, strength=1.0)
        with steering:
            steering.register(model)
            steered = model(input_tensor)
            assert not torch.allclose(original, steered)

        # After context exit, hook is removed
        restored = model(input_tensor)
        assert torch.allclose(original, restored)

    def test_register_invalid_layer_raises(self):
        """Registering on a nonexistent layer should raise ValueError."""
        model = nn.Linear(4, 2)
        direction = torch.ones(2)
        steering = DirectionalSteering(layer="nonexistent.layer", direction=direction)

        with pytest.raises(ValueError, match="not found"):
            steering.register(model)

    def test_strength_zero_no_change(self):
        """With strength=0, output should be unchanged."""
        model = nn.Linear(4, 2)
        input_tensor = torch.randn(1, 4)
        direction = torch.ones(2)

        original = model(input_tensor).clone()

        steering = DirectionalSteering(layer="", direction=direction, strength=0.0)
        handle = model.register_forward_hook(steering.hook_fn)
        steered = model(input_tensor)
        handle.remove()

        assert torch.allclose(original, steered, atol=1e-6)
