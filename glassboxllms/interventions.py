# glassboxllms/interventions.py
# Implemented by Ankita Sharma (GitHub: sharmaankita3387)
# Part of GDSC Research Project
# Date: January 12, 2026

def patch_activation(model, layer, new_value, text):
    """
    Temporarily replace a layer's output with new_value for one forward pass.
    
    Args:
        model: A ModelWrapper instance (with hook_manager and get_layer_module)
        layer: String identifier for the layer (e.g., "mlp.10")
        new_value: Tensor to use as the layer's output (must match shape)
        text: Input text to process
    
    Returns:
        Model output with the patch applied
    
    Example:
        >>> from glassboxllms.interventions import patch_activation
        >>> patched_output = patch_activation(
        ...     model,
        ...     layer="mlp.10",
        ...     new_value=tensor,
        ...     text="Hello world!"
        ... )
    """
    # 1) Get the target module to patch
    target_module = model.get_layer_module(layer)
    
    # 2) Define the patching hook function
    def patch_hook(module, input, output):
        """Hook that replaces module output with new_value."""
        # Ensure new_value is on correct device
        return new_value.to(output.device)
    
    # 3) Attach the hook using model's HookManager
    hook_id = model.hook_manager.add_hook(
        module=target_module,
        hook_fn=patch_hook
    )
    
    # 4) Run forward pass with hook active
    try:
        patched_output = model.forward(text)
    finally:
        # 5) Always clean up the hook
        model.hook_manager.remove_hook(hook_id)
    
    return patched_output

