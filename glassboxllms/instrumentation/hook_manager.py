import torch.nn as nn

class HookManager:
    # Helper class to manage hooks to be used on a model
    # Attaches hooks to specified layers, captures the activation of those layers, and removes hooks when done

    def __init__(self, model):
        self.model = model
        self.hooks = [] # List to store hook handles
        self.activations = {} # Outputs of captured hook layers

    def attach_hook(self, layer_name, hook_fn=None):
        # Attach a forward hook to a specified layer
        # Default capture_output is used if no hook_fn is provided
        layer = dict(self.model.named_modules())[layer_name]
        if hook_fn is None:
            hook_fn = self.capture_output(layer_name)
        handle = layer.register_forward_hook(hook_fn)
        self.hooks.append(handle)

    def capture_output(self,layer_name):
        # Default hook function to capture layer output
        def hook_fn(module, input, output):
            # Detach output and move to CPU for storage
            self.activations[module] = output.detach().cpu()
        return hook_fn
    
    def remove_hooks(self):
        # Remove attached hooks and clear the hooks list
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def clear_activations(self):
        # Clear stored activations
        self.activations.clear()