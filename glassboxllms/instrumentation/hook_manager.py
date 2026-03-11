"""
HookManager — Centralized hook lifecycle for PyTorch models.

Manages forward hooks across any ``nn.Module``, providing a consistent
interface for activation capture, patching, and intervention.

Usage::

    manager = HookManager(model)
    manager.attach_hook("transformer.h.5")
    output = model(**inputs)
    acts = manager.activations["transformer.h.5"]
    manager.remove_hooks()
"""

from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn


class HookManager:
    """
    Attach, track, and clean up forward hooks on a PyTorch model.

    Activations are keyed by **layer name strings** (not module objects),
    making them safe to serialize and easy to work with.

    Args:
        model: Any ``torch.nn.Module``.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks: List = []
        self.activations: Dict[str, torch.Tensor] = {}
        self._modules_dict: Dict[str, nn.Module] = dict(model.named_modules())

    # ── Core API ────────────────────────────────────────────────

    def attach_hook(
        self,
        layer_name: str,
        hook_fn: Optional[Callable] = None,
    ) -> None:
        """
        Register a forward hook on *layer_name*.

        If *hook_fn* is ``None`` a default capture hook is used that stores
        the layer output in ``self.activations[layer_name]``.

        Args:
            layer_name: Dotted module path (e.g. ``"transformer.h.5"``).
            hook_fn: ``(module, input, output) -> output | None``.
        """
        module = self._resolve_layer(layer_name)
        if hook_fn is None:
            hook_fn = self._default_hook(layer_name)
        handle = module.register_forward_hook(hook_fn)
        self.hooks.append(handle)

    def add_hook(
        self,
        layer_name: str,
        hook_fn: Callable,
    ) -> None:
        """Alias for :meth:`attach_hook` (used by CausalScrubber)."""
        self.attach_hook(layer_name, hook_fn)

    def remove_hooks(self) -> None:
        """Remove **all** registered hooks and clear stored activations."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def remove_all_hooks(self) -> None:
        """Alias for :meth:`remove_hooks` (used by CausalScrubber)."""
        self.remove_hooks()

    def clear_activations(self) -> None:
        """Clear stored activation tensors without removing hooks."""
        self.activations.clear()

    # ── Convenience ─────────────────────────────────────────────

    def capture_output(self, layer_name: str) -> Callable:
        """Return a hook that stores the layer output under *layer_name*."""
        return self._default_hook(layer_name)

    def get(self, layer_name: str) -> Optional[torch.Tensor]:
        """Retrieve a captured activation by layer name, or ``None``."""
        return self.activations.get(layer_name)

    # ── Internals ───────────────────────────────────────────────

    def _resolve_layer(self, layer_name: str) -> nn.Module:
        if layer_name not in self._modules_dict:
            available = [k for k in self._modules_dict if k][:10]
            raise ValueError(
                f"Layer '{layer_name}' not found in model. "
                f"Available (first 10): {available}"
            )
        return self._modules_dict[layer_name]

    def _default_hook(self, layer_name: str) -> Callable:
        def hook_fn(module: nn.Module, input: Any, output: Any) -> None:
            if isinstance(output, tuple):
                tensor = output[0]
            else:
                tensor = output
            self.activations[layer_name] = tensor.detach()
        return hook_fn
