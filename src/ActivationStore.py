import torch
import os
from typing import Dict, List, Optional, Union, Any
from collections import defaultdict

class ActivationStore:
    # This structure can store, index, and retrieve activations.

    def __init__(self, device: str = "cpu", storage_dir: Optional[str] = None):
        # initializes object
        # todo: evaluate value of putting this on the gpu if available

        self.device = device
        self.storage_dir = storage_dir

        # note that practically speaking you should probably pipe this to a .safetensors at a certain size
        self._data: Dict[str, List[torch.Tensor]] = defaultdict(list)

        self._metadata: Dict[str, List[int]] = defaultdict(list)

        if self.storage_dir and not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

    def save(self, layer_name: str, activations: torch.Tensor, token_idx: Optional[int] = None):

        # we move the graph to the cpu and out of memory
        acts = activations.detach().to(self.device).clone()

        self._data[layer_name].append(acts)
        if token_idx is not None:
            self._metadata[layer_name].append(token_idx)

    def get(self, layer_name: str, concat: bool = True) -> Union[torch.Tensor, List[torch.Tensor]]:
        # gets activations for a layer
        # concat allows you to merge the retrieved tensors into one big tensor

        if layer_name not in self._data:
            raise KeyError(f"No activations found for layer: {layer_name}")

        acts_list = self._data[layer_name]

        if concat:
            return torch.cat(acts_list, dim=0)
        return acts_list

    def get_by_token(self, layer_name: str, token_idx: int) -> torch.Tensor:
        # gets activations for a (token) index

        indices = [i for (i, idx) in enumerate(self._metadata[layer_name]) if (idx == token_idx)]
        if not indices:
            return torch.empty(0)

        acts = [self._data[layer_name][i] for i in indices]
        return torch.cat(acts, dim=0)

    def clear(self):
        self._data.clear()
        self._metadata.clear()

    def persist_to_disk(self, filename: str):
        # save data to disk

        load = {
            "data": dict(self._data),
            "metadata": dict(self._metadata)
        }
        torch.save(load, os.path.join(self.storage_dir or ".", filename))
