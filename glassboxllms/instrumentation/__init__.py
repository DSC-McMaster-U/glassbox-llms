"""
glassboxllms.instrumentation
=============================

Core instrumentation primitives for capturing, storing, and patching
activations in PyTorch models.

* :class:`HookManager` — Centralized hook lifecycle management.
* :class:`ActivationStore` — Buffered activation storage with disk persistence.
* :class:`ActivationExtractor` — High-level model → activations pipeline.
* :func:`patch_activation` — One-shot activation replacement.
* :func:`get_layer_names` — Discover available layers in a model.
"""

from .activations import ActivationStore
from .activation_patching import patch_activation
from .extractor import ActivationExtractor, get_layer_names
from .hook_manager import HookManager

__all__ = [
    "ActivationStore",
    "ActivationExtractor",
    "HookManager",
    "get_layer_names",
    "patch_activation",
]
