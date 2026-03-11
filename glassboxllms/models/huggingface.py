from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from .base import ModelWrapper


class TransformersModelWrapper(ModelWrapper):
    """
    HuggingFace Transformers model wrapper.

    Wraps any AutoModel-compatible model with the standard ModelWrapper
    interface for activation extraction, hooking, and introspection.

    Example:
        >>> wrapper = TransformersModelWrapper("gpt2")
        >>> output = wrapper.forward("Hello, world!")
        >>> activations = wrapper.get_activations("Hello", layers=["transformer.h.0"])
    """

    def __init__(self, model_name: str, device: Optional[str] = None):
        super().__init__()
        self._model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self._device)

        # Cache the module dict for fast lookups
        self._modules_dict: Dict[str, nn.Module] = dict(self.model.named_modules())

    def forward(self, inputs: Any, **kwargs) -> Any:
        """Execute a forward pass. Accepts text strings or pre-tokenized inputs."""
        if isinstance(inputs, str) or (isinstance(inputs, list) and isinstance(inputs[0], str)):
            tokens = self.tokenizer(
                inputs, return_tensors="pt", padding=True, truncation=True
            ).to(self._device)
        else:
            tokens = inputs

        with torch.no_grad():
            outputs = self.model(**tokens, **kwargs)
        return outputs

    def get_activations(
        self,
        inputs: Any,
        layers: List[str],
        return_type: str = "numpy",
    ) -> Dict[str, Any]:
        """
        Extract activations from specified layers using forward hooks.

        Args:
            inputs: Text string(s) or pre-tokenized inputs.
            layers: Layer names matching named_modules() keys
                    (e.g., ["transformer.h.0", "transformer.h.5.mlp"]).
            return_type: "numpy" or "torch".

        Returns:
            Dict mapping each requested layer name to its output tensor/array.
        """
        if isinstance(inputs, str) or (isinstance(inputs, list) and isinstance(inputs[0], str)):
            tokens = self.tokenizer(
                inputs, return_tensors="pt", padding=True, truncation=True
            ).to(self._device)
        else:
            tokens = inputs

        activations: Dict[str, Any] = {}
        hooks = []

        for layer_name in layers:
            module = self.get_layer_module(layer_name)

            def make_hook(name):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        activations[name] = output[0].detach()
                    else:
                        activations[name] = output.detach()
                return hook_fn

            hooks.append(module.register_forward_hook(make_hook(layer_name)))

        with torch.no_grad():
            self.model(**tokens)

        for hook in hooks:
            hook.remove()

        if return_type == "numpy":
            return {k: v.cpu().numpy() for k, v in activations.items()}
        return activations

    def get_layer_module(self, layer: str) -> nn.Module:
        """Get a named module by its dotted path (e.g., 'transformer.h.5.mlp')."""
        if layer in self._modules_dict:
            return self._modules_dict[layer]
        raise ValueError(
            f"Layer '{layer}' not found. Available layers: "
            f"{self.layer_names[:10]}... (use .layer_names for full list)"
        )

    def get_layer_shape(self, layer: str) -> Tuple[int, ...]:
        """
        Infer a layer's output shape by inspecting weight matrices.
        Falls back to (hidden_size,) if no weights are found.
        """
        module = self.get_layer_module(layer)
        # Try to infer from the last weight parameter
        for param in reversed(list(module.parameters())):
            if param.dim() >= 2:
                return (param.shape[0],)
            elif param.dim() == 1:
                return (param.shape[0],)
        # Fallback to model config hidden size
        return (self.model.config.hidden_size,)

    @property
    def layer_names(self) -> List[str]:
        """Return all named module paths in the model."""
        return [name for name, _ in self.model.named_modules() if name]

    @property
    def device(self) -> str:
        """Return the device string (e.g., 'cpu', 'cuda:0')."""
        return str(self._device)

    @property
    def model_config(self) -> Dict[str, Any]:
        """Return model metadata from the HuggingFace config."""
        config = self.model.config
        return {
            "hidden_size": getattr(config, "hidden_size", getattr(config, "n_embd", None)),
            "num_layers": getattr(config, "num_hidden_layers", getattr(config, "n_layer", None)),
            "vocab_size": getattr(config, "vocab_size", None),
            "model_type": getattr(config, "model_type", "unknown"),
            "model_name": self._model_name,
        }

    def set_eval_mode(self):
        """Set model to evaluation mode."""
        self.model.eval()

    def set_train_mode(self):
        """Set model to training mode."""
        self.model.train()

    def __repr__(self) -> str:
        return f"TransformersModelWrapper(model='{self._model_name}', device='{self.device}')"
