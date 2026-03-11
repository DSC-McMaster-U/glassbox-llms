"""
Backward-compatible re-export of activation extraction utilities.

The canonical implementations now live in :mod:`glassboxllms.instrumentation`:

* :class:`~glassboxllms.instrumentation.extractor.ActivationExtractor`
* :func:`~glassboxllms.instrumentation.extractor.get_layer_names`

This module keeps the old import path working so that existing code like
``from glassboxllms.primitives.probes.activation_store import ActivationStore``
continues to function.  ``ActivationStore`` here is an alias for
``ActivationExtractor``.
"""

from glassboxllms.instrumentation.extractor import (
    ActivationExtractor as ActivationStore,
    get_layer_names,
)

__all__ = ["ActivationStore", "get_layer_names"]
