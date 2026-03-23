import pytest
import torch
import torch.nn as nn

from glassboxllms.instrumentation.activation_patching import patch_activation


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.layer.weight.copy_(torch.eye(4))

    def forward(self, x):
        return self.layer(x)


def test_patch_activation_replaces_layer_output():
    model = _ToyModel()
    inputs = torch.ones(1, 4)
    patched_value = torch.full((1, 4), 3.0)

    output = patch_activation(
        model=model,
        layer="layer",
        new_value=patched_value,
        inputs=inputs,
    )

    assert torch.equal(output, patched_value)


def test_patch_activation_raises_for_missing_layer():
    model = _ToyModel()
    with pytest.raises(ValueError):
        patch_activation(
            model=model,
            layer="does_not_exist",
            new_value=torch.zeros(1, 4),
            inputs=torch.ones(1, 4),
        )

