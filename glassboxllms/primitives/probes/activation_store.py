"""
Activation Store — Extract and cache activations from transformer models.

Works with Hugging Face Transformers models using forward hooks.
Designed to integrate cleanly with LinearProbe for interpretability analysis.

Usage:
    >>> from transformers import AutoModel, AutoTokenizer
    >>> from glassbox.probes.activation_store import ActivationStore
    >>>
    >>> model = AutoModel.from_pretrained("bert-base-uncased")
    >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    >>> store = ActivationStore(model)
    >>>
    >>> activations = store.extract(
    ...     texts=["Hello world", "Goodbye world"],
    ...     tokenizer=tokenizer,
    ...     layers=["encoder.layer.6", "encoder.layer.11"]
    ... )
"""

from typing import Any, Callable, Dict, List, Optional, Union
from collections import defaultdict
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ActivationStore:
    """
    Extracts and stores activations from transformer model layers.

    Uses PyTorch forward hooks to capture intermediate representations
    without modifying the model's forward pass.

    Attributes:
        model: The transformer model (HuggingFace or any nn.Module).
        cache: Dictionary mapping layer names to captured activations.
    """

    def __init__(self, model: "nn.Module"):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for ActivationStore")

        self.model = model
        self.cache: Dict[str, List[np.ndarray]] = defaultdict(list)
        self._hooks: List = []
        self._target_layers: List[str] = []

    def extract(
        self,
        texts: List[str],
        tokenizer: Any,
        layers: List[str],
        batch_size: int = 32,
        max_length: int = 512,
        pooling: str = "mean",  # 'mean', 'cls', 'last', 'none'
        device: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Extract activations from specified layers for a list of texts.

        Args:
            texts: List of input strings.
            tokenizer: HuggingFace tokenizer.
            layers: List of layer names to extract (e.g., ["encoder.layer.6"]).
            batch_size: Batch size for processing.
            max_length: Maximum sequence length.
            pooling: How to pool sequence dimension:
                - 'mean': Average over tokens
                - 'cls': Take [CLS] token (position 0)
                - 'last': Take last token
                - 'none': Keep full sequence (returns 3D)
            device: Device to run on ('cuda', 'cpu', or None for auto).

        Returns:
            Dict mapping layer names to activation arrays of shape:
                - (n_samples, hidden_dim) if pooling != 'none'
                - (n_samples, seq_len, hidden_dim) if pooling == 'none'
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model.to(device)
        self.model.eval()

        # Clear cache and register hooks
        self.cache.clear()
        self._target_layers = layers
        self._register_hooks(layers)

        try:
            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i : i + batch_size]
                    inputs = tokenizer(
                        batch_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                    ).to(device)

                    # Forward pass triggers hooks
                    _ = self.model(**inputs)

        finally:
            self._remove_hooks()

        # Pool and convert to numpy
        return self._finalize_cache(pooling)

    def extract_from_tensors(
        self,
        input_ids: "torch.Tensor",
        attention_mask: Optional["torch.Tensor"],
        layers: List[str],
        pooling: str = "mean",
    ) -> Dict[str, np.ndarray]:
        """
        Extract activations from pre-tokenized inputs.

        Args:
            input_ids: Token IDs (batch, seq_len).
            attention_mask: Attention mask (batch, seq_len).
            layers: Layer names to extract.
            pooling: Pooling strategy.

        Returns:
            Dict mapping layer names to activation arrays.
        """
        device = next(self.model.parameters()).device
        self.model.eval()

        self.cache.clear()
        self._target_layers = layers
        self._register_hooks(layers)

        try:
            with torch.no_grad():
                inputs = {"input_ids": input_ids.to(device)}
                if attention_mask is not None:
                    inputs["attention_mask"] = attention_mask.to(device)
                _ = self.model(**inputs)
        finally:
            self._remove_hooks()

        return self._finalize_cache(pooling)

    def _register_hooks(self, layers: List[str]):
        """Register forward hooks on target layers."""
        self._hooks = []

        for name, module in self.model.named_modules():
            if name in layers:
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)

        registered = set(name for name, _ in self.model.named_modules() if name in layers)
        missing = set(layers) - registered
        if missing:
            available = [n for n, _ in self.model.named_modules() if n][:20]
            raise ValueError(
                f"Layers not found: {missing}. "
                f"Available layers (first 20): {available}"
            )

    def _make_hook(self, layer_name: str) -> Callable:
        """Create a hook function that captures activations."""
        def hook(module, input, output):
            # Handle different output types
            if isinstance(output, tuple):
                tensor = output[0]
            else:
                tensor = output

            # Move to CPU and store
            self.cache[layer_name].append(tensor.detach().cpu())

        return hook

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def _finalize_cache(self, pooling: str) -> Dict[str, np.ndarray]:
        """Concatenate batches and apply pooling."""
        result = {}

        for layer_name, tensors in self.cache.items():
            # Concatenate all batches: (total_samples, seq_len, hidden_dim)
            combined = torch.cat(tensors, dim=0)

            if pooling == "mean":
                pooled = combined.mean(dim=1)
            elif pooling == "cls":
                pooled = combined[:, 0, :]
            elif pooling == "last":
                pooled = combined[:, -1, :]
            elif pooling == "none":
                pooled = combined
            else:
                raise ValueError(f"Unknown pooling: {pooling}")

            result[layer_name] = pooled.numpy()

        return result

    def list_layers(self, pattern: Optional[str] = None) -> List[str]:
        """
        List available layer names in the model.

        Args:
            pattern: Optional substring to filter layers (e.g., "layer.6").

        Returns:
            List of matching layer names.
        """
        names = [name for name, _ in self.model.named_modules() if name]
        if pattern:
            names = [n for n in names if pattern in n]
        return names


def get_layer_names(model: "nn.Module", layer_type: str = "all") -> List[str]:
    """
    Utility to get common layer names from a HuggingFace model.

    Args:
        model: The transformer model.
        layer_type: Filter by type:
            - 'all': All named modules
            - 'attention': Attention layers only
            - 'mlp': MLP/FFN layers only
            - 'output': Output layers (last hidden states)

    Returns:
        List of layer names.
    """
    names = [name for name, _ in model.named_modules() if name]

    if layer_type == "all":
        return names
    elif layer_type == "attention":
        return [n for n in names if "attention" in n.lower() or "attn" in n.lower()]
    elif layer_type == "mlp":
        return [n for n in names if "mlp" in n.lower() or "ffn" in n.lower() or "intermediate" in n.lower()]
    elif layer_type == "output":
        return [n for n in names if "output" in n.lower() or "pooler" in n.lower()]
    else:
        raise ValueError(f"Unknown layer_type: {layer_type}")
