import torch
import os
from typing import Dict, List, Optional, Union, Any, Callable
from collections import defaultdict
import uuid

try:
    HAS_SAFETENSORS = True
    from safetensors.torch import save_file, load_file
except ImportError:
    HAS_SAFETENSORS = False

class ActivationStore:
    # this structure can store, index, and retrieve activations.

    def __init__(self, device: str = "cpu", storage_dir: str = "./activations", buffer_size: int = 1000):
        self.device = device
        self.storage_dir = storage_dir
        self.buffer_size = buffer_size

        # layer_name -> list of tensors
        self._buffer: Dict[str, List[torch.Tensor]] = defaultdict(list)

        # layer_name -> token_idx -> list of buffer indices
        self._metadata_index: Dict[str, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))

        # track saved files to reload them later if needed
        self._disk_manifest: Dict[str, List[str]] = defaultdict(list)

        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

    def _flush_layer(self, layer_name: str):
        if not self._buffer[layer_name]:
            return

        # put the tensors in a stack for efficiency
        tensor_stack = torch.stack(self._buffer[layer_name])

        # todo: other way to get unique identifiers?
        filename = f"{layer_name}_{uuid.uuid4().hex}.pt"
        filepath = os.path.join(self.storage_dir, filename)

        if HAS_SAFETENSORS:
            # safetensors wants a tensor dict
            save_file({layer_name: tensor_stack}, filepath.replace(".pt", ".safetensors"))
            self._disk_manifest[layer_name].append(filepath.replace(".pt", ".safetensors"))
        else:
            torch.save(tensor_stack, filepath)
            self._disk_manifest[layer_name].append(filepath)

        # clear memory
        self._buffer[layer_name].clear()
        self._metadata_index[layer_name].clear()

    def create_hook(self, layer_name: str, token_idx: Optional[int] = None) -> Callable:
        # returns a pytorch hook
        # is this even necessary??

        def hook(module, input, output):
            # handle output being a tuple (transformers can do this sometimes)
            if isinstance(output, tuple):
                out_tensor = output[0]
            else:
                out_tensor = output
            self.save(layer_name, out_tensor, token_idx)
        return hook

    def save(self, layer_name: str, activations: torch.Tensor, token_idx: Optional[int] = None):
        # detach
        acts = activations.detach().to(self.device)

        # add to the buffer
        self._buffer[layer_name].append(acts)
        current_idx = len(self._buffer[layer_name]) - 1

        # metadata indexing
        if token_idx is not None:
            self._metadata_index[layer_name][token_idx].append(current_idx)

        if len(self._buffer[layer_name]) >= self.buffer_size:
            self._flush_layer(layer_name)

    def get_all(self, layer_name: str) -> torch.Tensor:
        # simple getter to stitch together from the disk for a layer

        parts = []

        if layer_name in self._disk_manifest:
            for filepath in self._disk_manifest[layer_name]:
                if HAS_SAFETENSORS and filepath.endswith(".safetensors"):
                    # ignore your linter here, it's being unreasonable. ignore the error
                    parts.append(load_file(filepath)[layer_name])
                else:
                    parts.append(torch.load(filepath, map_location=self.device))

        if self._buffer[layer_name]:
            parts.append(torch.stack(self._buffer[layer_name]))

        if not parts:
            # ?
            return torch.empty(0)

        return torch.cat(parts, dim=0)

    def get_by_token(self, layer_name: str, token_idx: int) -> torch.Tensor:
        if layer_name not in self._metadata_index:
            return torch.empty(0)

        indices = self._metadata_index[layer_name].get(token_idx, [])

        if not indices:
            return torch.empty(0)

        # !!! NOTE: THIS ONLY WORKS FOR STUFF IN RAM!!!!! NOT CACHED DATA ON DISK
        acts = [self._buffer[layer_name][i] for i in indices]

        if not acts:
            # ?
            return torch.empty(0)

        return torch.stack(acts)

    def clear(self):
        self._buffer.clear()
        self._metadata_index.clear()
        # maybe clear the blob st files

    def persist_to_disk(self, filename: str):
        # save data to disk

        load = {
            "data": dict(self._buffer),
            "metadata": dict(self._metadata_index)
        }
        torch.save(load, os.path.join(self.storage_dir, filename))

    def __repr__(self) -> str:
        return (f"ActivationStore(device='{self.device}', storage_dir='{self.storage_dir}', buffer_size={self.buffer_size}, tracked_layers={len(self._buffer) + len(self._disk_manifest)})")

    def __str__(self) -> str:
        temp = [f"---", f"ActivationStore(device={self.device}, physical_path={self.storage_dir}, buffer_limit={self.buffer_size}/layer)"]

        all_layers = set(self._buffer.keys()) | set(self._disk_manifest.keys())

        if not all_layers:
            temp.append("\n  (No data stored)")
            return "\n".join(temp)

        # fancy data table display
        temp.append("\n  {:<25} | {:<15} | {:<15}".format("Layer Name", "Buffer (RAM)", "Disk Files"))
        temp.append("  " + "-"*61)

        for layer in sorted(all_layers):
            ram_count = len(self._buffer.get(layer, []))
            disk_count = len(self._disk_manifest.get(layer, []))

            # todo: if buffer is full/near full, maybe flag it?
            temp.append(f"  {layer:<25} | {ram_count:<15} | {disk_count:<15}")

        temp.append("---")
        return "\n".join(temp)

if __name__ == "__main__":
    store = ActivationStore()

    a = torch.randn(1, 512)

    store.save("mlp.0", activations=a, token_idx=5)

    acts = store.get_all("mlp.0")
    print(f"shape: {acts.shape}")
    print(store)
    print(store.create_hook("mlp.0"))
