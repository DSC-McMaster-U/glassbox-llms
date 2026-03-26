"""
ActivationExtractor — Extract activations from any ``nn.Module``.

High-level pipeline that registers PyTorch forward hooks, runs inference,
and returns captured activations as NumPy arrays or PyTorch tensors.

Supports four pooling strategies for sequence data and can optionally
persist results to an :class:`ActivationStore` for large-scale collection.

Usage::

    from glassboxllms.instrumentation import ActivationExtractor

    extractor = ActivationExtractor(model)
    activations = extractor.extract(
        texts=["Hello world", "Goodbye world"],
        tokenizer=tokenizer,
        layers=["encoder.layer.6", "encoder.layer.11"],
    )
    # activations["encoder.layer.6"].shape => (2, hidden_dim)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from glassboxllms.instrumentation.activations import ActivationStore


class ActivationExtractor:
    """
    Extract and pool activations from transformer model layers.

    Uses PyTorch forward hooks to capture intermediate representations
    without modifying the model architecture or forward pass.

    Args:
        model: Any ``torch.nn.Module`` (HuggingFace model, custom model, etc.).
        store: Optional :class:`ActivationStore` for buffered / disk-backed
            storage.  When provided, captured activations are also pushed
            into the store for later retrieval.
    """

    def __init__(
        self,
        model: "nn.Module",
        store: Optional[ActivationStore] = None,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for ActivationExtractor.")

        self.model = model
        self.store = store
        self._cache: Dict[str, List["torch.Tensor"]] = defaultdict(list)
        self._hooks: List = []

    # ── High-level API ───────────────────────────────────────────

    def extract(
        self,
        texts: List[str],
        tokenizer: Any,
        layers: List[str],
        batch_size: int = 32,
        max_length: int = 512,
        pooling: str = "mean",
        device: Optional[str] = None,
        return_type: str = "numpy",
    ) -> Dict[str, Any]:
        """
        Extract activations from specified layers for a list of texts.

        Args:
            texts: Input strings.
            tokenizer: A HuggingFace-compatible tokenizer.
            layers: Layer name paths (e.g. ``["encoder.layer.6"]``).
            batch_size: Batch size for inference.
            max_length: Maximum tokenized sequence length.
            pooling: How to pool the sequence dimension:
                ``"mean"`` | ``"cls"`` | ``"last"`` | ``"none"``.
            device: ``"cuda"``, ``"cpu"``, or ``None`` (auto-detect).
            return_type: ``"numpy"`` (default) or ``"torch"``.

        Returns:
            Dict mapping layer names to activation arrays/tensors.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model.to(device)
        self.model.eval()

        self._cache.clear()
        self._register_hooks(layers)

        try:
            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    batch = texts[i : i + batch_size]
                    inputs = tokenizer(
                        batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                    ).to(device)
                    self.model(**inputs)
        finally:
            self._remove_hooks()

        return self._finalize(pooling, return_type)

    def extract_from_tensors(
        self,
        input_ids: "torch.Tensor",
        layers: List[str],
        attention_mask: Optional["torch.Tensor"] = None,
        pooling: str = "mean",
        return_type: str = "numpy",
    ) -> Dict[str, Any]:
        """
        Extract activations from pre-tokenized inputs.

        Args:
            input_ids: ``(batch, seq_len)`` token IDs.
            layers: Layer name paths.
            attention_mask: Optional ``(batch, seq_len)`` mask.
            pooling: Pooling strategy.
            return_type: ``"numpy"`` or ``"torch"``.

        Returns:
            Dict mapping layer names to activation arrays/tensors.
        """
        device = next(self.model.parameters()).device
        self.model.eval()

        self._cache.clear()
        self._register_hooks(layers)

        try:
            with torch.no_grad():
                kwargs: Dict[str, Any] = {"input_ids": input_ids.to(device)}
                if attention_mask is not None:
                    kwargs["attention_mask"] = attention_mask.to(device)
                self.model(**kwargs)
        finally:
            self._remove_hooks()

        return self._finalize(pooling, return_type)

    # ── Layer introspection ──────────────────────────────────────

    def list_layers(self, pattern: Optional[str] = None) -> List[str]:
        """
        List available layer names in the model.

        Args:
            pattern: Optional substring filter (e.g. ``"layer.6"``).

        Returns:
            Matching layer name strings.
        """
        names = [name for name, _ in self.model.named_modules() if name]
        if pattern:
            names = [n for n in names if pattern in n]
        return names

    # ── Internals ────────────────────────────────────────────────

    def _register_hooks(self, layers: List[str]) -> None:
        self._hooks = []
        modules_dict = dict(self.model.named_modules())

        missing = [l for l in layers if l not in modules_dict]
        if missing:
            available = [k for k in modules_dict if k][:20]
            raise ValueError(
                f"Layers not found: {missing}. "
                f"Available (first 20): {available}"
            )

        for layer_name in layers:
            module = modules_dict[layer_name]
            hook = module.register_forward_hook(self._make_hook(layer_name))
            self._hooks.append(hook)

    def _make_hook(self, layer_name: str) -> Callable:
        def hook_fn(module: Any, input: Any, output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            detached = tensor.detach().cpu()
            self._cache[layer_name].append(detached)

            # If a backing store is configured, push into it too
            if self.store is not None:
                self.store.save(layer_name, detached)

        return hook_fn

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _finalize(
        self,
        pooling: str,
        return_type: str,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {}

        for layer_name, tensors in self._cache.items():
            combined = torch.cat(tensors, dim=0)
            pooled = _apply_pooling(combined, pooling)

            if return_type == "numpy":
                result[layer_name] = pooled.numpy()
            else:
                result[layer_name] = pooled

        return result


# ── Module-level utilities ───────────────────────────────────────


def _apply_pooling(tensor: "torch.Tensor", pooling: str) -> "torch.Tensor":
    """
    Apply pooling over the sequence dimension (dim 1).

    Handles both 3-D ``(batch, seq, hidden)`` and 2-D ``(batch, hidden)``
    inputs gracefully.
    """
    if tensor.ndim == 2:
        return tensor  # already (batch, hidden) — nothing to pool

    if pooling == "mean":
        return tensor.mean(dim=1)
    elif pooling == "cls":
        return tensor[:, 0, :]
    elif pooling == "last":
        return tensor[:, -1, :]
    elif pooling == "none":
        return tensor
    else:
        raise ValueError(
            f"Unknown pooling strategy: {pooling!r}. "
            f"Choose from 'mean', 'cls', 'last', 'none'."
        )


def get_layer_names(
    model: "nn.Module",
    layer_type: str = "all",
) -> List[str]:
    """
    List layer names from a model, optionally filtered by type.

    Args:
        model: Any ``torch.nn.Module``.
        layer_type: ``"all"`` | ``"attention"`` | ``"mlp"`` | ``"output"``.

    Returns:
        List of matching layer name strings.
    """
    names = [name for name, _ in model.named_modules() if name]

    filters = {
        "all": lambda n: True,
        "attention": lambda n: any(
            kw in n.lower() for kw in ("attention", "attn")
        ),
        "mlp": lambda n: any(
            kw in n.lower() for kw in ("mlp", "ffn", "intermediate")
        ),
        "output": lambda n: any(
            kw in n.lower() for kw in ("output", "pooler")
        ),
    }

    if layer_type not in filters:
        raise ValueError(
            f"Unknown layer_type: {layer_type!r}. "
            f"Choose from {list(filters.keys())}."
        )

    return [n for n in names if filters[layer_type](n)]
