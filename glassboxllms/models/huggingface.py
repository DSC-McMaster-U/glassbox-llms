from abc import ABC
from typing import Any, Dict, List, Tuple
import torch
from transformers import AutoModel, AutoTokenizer

class TransformersModelWrapper(ModelWrapper, ABC):
    def __init__(self, model_name: str):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # Set model to evaluation mode

    def forward(self, inputs: Any, **kwargs) -> Any:
        tokens = self.tokenizer(inputs, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**tokens, **kwargs)
        return outputs

    def get_activations(self, inputs: Any, layers: List[str], return_type: str = "numpy") -> Dict[str, Any]:
        tokens = self.tokenizer(inputs, return_tensors='pt', padding=True, truncation=True)
        activations = {}

        def hook_fn(module, input, output):
            layer_name = module.__class__.__name__
            if layer_name in layers:
                activations[layer_name] = output.detach()

        hooks = []
        for layer in layers:
            layer_module = self.get_layer_module(layer)
            hooks.append(layer_module.register_forward_hook(hook_fn))

        with torch.no_grad():
            self.model(**tokens)

        for hook in hooks:
            hook.remove()

        if return_type == "numpy":
            return {k: v.cpu().numpy() for k, v in activations.items()}
        return activations

    def get_layer_module(self, layer: str) -> Any:
        layer_names = self.layer_names
        if layer in layer_names:
            return dict(self.model.named_modules())[layer]
        raise ValueError(f"Layer {layer} not found in model.")

    def get_layer_shape(self, layer: str) -> Tuple[int, ...]:
        layer_module = self.get_layer_module(layer)
        return layer_module.output_shape[1:]  # Exclude batch size

    @property
    def layer_names(self) -> List[str]:
        return list(self.model.named_modules())

    @property
    def device(self) -> str:
        return next(self.model.parameters()).device

    @property
    def model_config(self) -> Dict[str, Any]:
        return {
            "hidden_size": self.model.config.hidden_size,
            "num_layers": self.model.config.num_hidden_layers,
            "vocab_size": self.model.config.vocab_size,
            "model_type": self.model.config.model_type,
        }