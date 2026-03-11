"""
ActivationStore — Buffered activation storage with optional disk persistence.

Stores captured activations in RAM with automatic flushing to disk when
the buffer exceeds a configurable threshold.  Supports token-level indexing
and SafeTensors serialization when available.

This is the *storage engine* — it does not know how to extract activations
from a model.  See :class:`ActivationExtractor` for the extraction pipeline.

Usage::

    from glassboxllms.instrumentation import ActivationStore

    store = ActivationStore(buffer_size=500)
    store.save("encoder.layer.6", activation_tensor, token_idx=3)
    all_acts = store.get_all("encoder.layer.6")
"""

from __future__ import annotations

import os
import uuid
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import torch

try:
    from safetensors.torch import load_file, save_file

    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


class ActivationStore:
    """
    Buffer and persist activation tensors by layer name.

    Args:
        device: Device for in-memory tensors (``"cpu"`` by default).
        storage_dir: Directory for disk-flushed activations.
        buffer_size: Per-layer capacity before flushing to disk.
    """

    def __init__(
        self,
        device: str = "cpu",
        storage_dir: str = "./activations",
        buffer_size: int = 1000,
    ):
        self.device = device
        self.storage_dir = storage_dir
        self.buffer_size = buffer_size

        # layer_name -> list of tensors (RAM buffer)
        self._buffer: Dict[str, List[torch.Tensor]] = defaultdict(list)

        # layer_name -> token_idx -> list of buffer indices
        self._token_index: Dict[str, Dict[int, List[int]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # layer_name -> list of file paths (flushed to disk)
        self._disk_manifest: Dict[str, List[str]] = defaultdict(list)

        os.makedirs(self.storage_dir, exist_ok=True)

    # ── Write ────────────────────────────────────────────────────

    def save(
        self,
        layer_name: str,
        activations: torch.Tensor,
        token_idx: Optional[int] = None,
    ) -> None:
        """
        Store an activation tensor for *layer_name*.

        Automatically flushes to disk when the buffer is full.

        Args:
            layer_name: Layer that produced the activation.
            activations: The activation tensor to store.
            token_idx: Optional token position for indexed retrieval.
        """
        acts = activations.detach().to(self.device)

        self._buffer[layer_name].append(acts)
        buf_idx = len(self._buffer[layer_name]) - 1

        if token_idx is not None:
            self._token_index[layer_name][token_idx].append(buf_idx)

        if len(self._buffer[layer_name]) >= self.buffer_size:
            self._flush_layer(layer_name)

    def create_hook(
        self,
        layer_name: str,
        token_idx: Optional[int] = None,
    ) -> Callable:
        """
        Return a PyTorch forward hook that pushes activations into this store.

        Args:
            layer_name: Key under which to store the captured output.
            token_idx: Optional token position for indexed retrieval.
        """

        def hook(module: torch.nn.Module, input: object, output: object) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            self.save(layer_name, tensor, token_idx)

        return hook

    # ── Read ─────────────────────────────────────────────────────

    def get_all(self, layer_name: str) -> torch.Tensor:
        """
        Retrieve all activations for *layer_name* (RAM + disk).

        Returns:
            Concatenated tensor of shape ``(n, ...)``, or ``torch.empty(0)``
            if nothing has been stored.
        """
        parts: List[torch.Tensor] = []

        # Reload from disk
        for filepath in self._disk_manifest.get(layer_name, []):
            if HAS_SAFETENSORS and filepath.endswith(".safetensors"):
                parts.append(load_file(filepath)[layer_name])
            else:
                parts.append(
                    torch.load(filepath, map_location=self.device, weights_only=True)
                )

        # In-memory buffer
        if self._buffer[layer_name]:
            parts.append(torch.stack(self._buffer[layer_name]))

        if not parts:
            return torch.empty(0)
        return torch.cat(parts, dim=0)

    def get_by_token(self, layer_name: str, token_idx: int) -> torch.Tensor:
        """
        Retrieve activations for a specific token position.

        .. note::
            Token-level indexing only covers data still in the RAM buffer.
            Flushed-to-disk data loses token-level granularity.

        Returns:
            Stacked tensor or ``torch.empty(0)`` if not found.
        """
        indices = self._token_index.get(layer_name, {}).get(token_idx, [])
        if not indices:
            return torch.empty(0)

        buf = self._buffer.get(layer_name, [])
        acts = [buf[i] for i in indices if i < len(buf)]
        if not acts:
            return torch.empty(0)
        return torch.stack(acts)

    @property
    def layer_names(self) -> List[str]:
        """All layer names that have stored data."""
        return sorted(set(self._buffer.keys()) | set(self._disk_manifest.keys()))

    # ── Lifecycle ────────────────────────────────────────────────

    def clear(self) -> None:
        """Clear all in-memory data (does not delete disk files)."""
        self._buffer.clear()
        self._token_index.clear()

    def flush(self) -> None:
        """Force-flush all in-memory buffers to disk."""
        for layer in list(self._buffer.keys()):
            self._flush_layer(layer)

    def persist(self, filename: str) -> str:
        """
        Save all in-memory data to a single file and return the path.

        Args:
            filename: Name of the output file (saved inside *storage_dir*).
        """
        filepath = os.path.join(self.storage_dir, filename)
        payload = {
            "data": {k: torch.stack(v) for k, v in self._buffer.items() if v},
        }
        torch.save(payload, filepath)
        return filepath

    # ── Internals ────────────────────────────────────────────────

    def _flush_layer(self, layer_name: str) -> None:
        if not self._buffer[layer_name]:
            return

        tensor_stack = torch.stack(self._buffer[layer_name])

        # Sanitize layer name for filesystem (dots → dashes)
        safe_name = layer_name.replace(".", "-").replace("/", "-")
        filename = f"{safe_name}_{uuid.uuid4().hex[:8]}"

        if HAS_SAFETENSORS:
            filepath = os.path.join(self.storage_dir, f"{filename}.safetensors")
            save_file({layer_name: tensor_stack}, filepath)
        else:
            filepath = os.path.join(self.storage_dir, f"{filename}.pt")
            torch.save(tensor_stack, filepath)

        self._disk_manifest[layer_name].append(filepath)
        self._buffer[layer_name].clear()
        self._token_index[layer_name].clear()

    # ── Display ──────────────────────────────────────────────────

    def __repr__(self) -> str:
        n_layers = len(set(self._buffer.keys()) | set(self._disk_manifest.keys()))
        return (
            f"ActivationStore(device='{self.device}', "
            f"storage_dir='{self.storage_dir}', "
            f"buffer_size={self.buffer_size}, "
            f"tracked_layers={n_layers})"
        )

    def __str__(self) -> str:
        lines = [
            f"ActivationStore(device={self.device}, "
            f"path={self.storage_dir}, "
            f"buffer_limit={self.buffer_size}/layer)",
        ]

        all_layers = sorted(
            set(self._buffer.keys()) | set(self._disk_manifest.keys())
        )
        if not all_layers:
            lines.append("  (empty)")
            return "\n".join(lines)

        lines.append(f"  {'Layer':<30} {'RAM':>8} {'Disk':>8}")
        lines.append("  " + "-" * 48)
        for layer in all_layers:
            ram = len(self._buffer.get(layer, []))
            disk = len(self._disk_manifest.get(layer, []))
            lines.append(f"  {layer:<30} {ram:>8} {disk:>8}")

        return "\n".join(lines)
