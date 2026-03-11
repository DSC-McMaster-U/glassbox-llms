"""
Activation patching — replace a layer's output for one forward pass.

Originally by Ankita Sharma. Rewritten to work directly with any
``nn.Module`` or ``TransformersModelWrapper`` via standard PyTorch hooks
(no ``hook_manager`` attribute required on the model).

Usage::

    from glassboxllms.instrumentation.activation_patching import patch_activation

    patched = patch_activation(
        model.model,                 # raw nn.Module
        layer="transformer.h.5",
        new_value=replacement_tensor,
        inputs=tokenized_input,
    )
"""

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn


def patch_activation(
    model: nn.Module,
    layer: str,
    new_value: torch.Tensor,
    inputs: Union[Dict[str, torch.Tensor], str, Any],
    *,
    wrapper: Optional[Any] = None,
) -> Any:
    """
    Run one forward pass with *layer*'s output replaced by *new_value*.

    Args:
        model: Raw ``nn.Module`` **or** the inner ``.model`` from a wrapper.
        layer: Named module path (e.g. ``"transformer.h.5"``).
        new_value: Tensor to substitute.  Must be broadcastable to the
            layer's output shape.
        inputs: Tokenized input dict (``input_ids``, ``attention_mask``, …)
            **or** raw text if *wrapper* is provided.
        wrapper: Optional ``TransformersModelWrapper`` — if provided,
            *inputs* can be a text string and the wrapper handles
            tokenization.

    Returns:
        Model output with the patch applied.
    """
    # Resolve the target module
    modules_dict = dict(model.named_modules())
    if layer not in modules_dict:
        raise ValueError(f"Layer '{layer}' not found in model.")
    target_module = modules_dict[layer]

    # Build the patching hook
    def patch_hook(module, inp, output):
        device = output.device if isinstance(output, torch.Tensor) else output[0].device
        patched = new_value.to(device=device)
        if isinstance(output, tuple):
            return (patched,) + output[1:]
        return patched

    # Attach, forward, clean up
    handle = target_module.register_forward_hook(patch_hook)
    try:
        if wrapper is not None and isinstance(inputs, str):
            patched_output = wrapper.forward(inputs)
        elif isinstance(inputs, dict):
            with torch.no_grad():
                patched_output = model(**inputs)
        else:
            with torch.no_grad():
                patched_output = model(inputs)
    finally:
        handle.remove()

    return patched_output
