import torch
import torch.nn as nn
from glassboxllms.interventions.steering import DirectionalSteering

def test_steering_changes_output():
    # 1. Setup a dummy model
    model = nn.Linear(4, 2)
    input_tensor = torch.randn(1, 4)
    direction = torch.ones(4)

    # 2. Get original output
    original_output = model(input_tensor)

    # 3. Apply steering
    steering = DirectionalSteering(layer="place_in_network", direction=direction, strength=5.0)
    
    # We manually register/remove here since nn.Linear isn't a complex model dict
    # But for the test, we can just attach to the module directly
    handle = model.register_forward_hook(steering.hook_fn)
    
    steered_output = model(input_tensor)
    handle.remove()

    # 4. Assertions
    # Output should change
    assert not torch.allclose(original_output, steered_output), "Steering did not change output!"
    
    # Output should return to normal after hook removal
    post_output = model(input_tensor)
    assert torch.allclose(original_output, post_output), "Model did not return to original state!"