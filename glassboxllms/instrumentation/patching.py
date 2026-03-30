import torch
import torch.nn.functional as F
from operator import attrgetter
from typing import Dict, Any

class PatchingExperiment:
    def __init__(
        self, 
        model: torch.nn.Module, 
        unit: str, 
        source_prompt: str, 
        corrupted_prompt: str,
        target_token_str: str = None
    ):
        self.model = model
        self.tokenizer = getattr(model, "tokenizer", None) 
        self.unit = unit
        self.source_prompt = source_prompt
        self.corrupted_prompt = corrupted_prompt
        self.target_token_str = target_token_str

    def _resolve_unit_to_module(self) -> torch.nn.Module:
        try:
            return attrgetter(self.unit)(self.model)
        except AttributeError:
            modules = dict(self.model.named_modules())
            if self.unit in modules:
                return modules[self.unit]
            raise ValueError(f"Could not find unit '{self.unit}' in model. "
                             f"Available units include: {list(modules.keys())[:10]}...")

    def _calculate_logit_difference(self, clean_logits: torch.Tensor, patched_logits: torch.Tensor) -> float:
        clean_probs = F.softmax(clean_logits[0, -1, :], dim=-1)
        patched_probs = F.softmax(patched_logits[0, -1, :], dim=-1)
        
        if self.target_token_str and self.tokenizer:
            target_id = self.tokenizer.encode(self.target_token_str)[-1]
            return (patched_probs[target_id] - clean_probs[target_id]).item()
        
        return torch.dist(patched_probs, clean_probs).item()

    def run(self) -> Dict[str, Any]:
        target_module = self._resolve_unit_to_module()
        device = next(self.model.parameters()).device
        
        clean_tokens = self.tokenizer(self.source_prompt, return_tensors="pt").input_ids.to(device)
        corrupted_tokens = self.tokenizer(self.corrupted_prompt, return_tensors="pt").input_ids.to(device)

        activation_cache = {}

        #Clean Run
        def cache_hook(module, input, output):
            activation_cache['clean_state'] = output[0].detach().clone() if isinstance(output, tuple) else output.detach().clone()
            return output

        handle_clean = target_module.register_forward_hook(cache_hook)
        with torch.no_grad():
            clean_out = self.model(clean_tokens)
        handle_clean.remove()

        #Patched Run
        def patch_hook(module, input, output):
            if isinstance(output, tuple):
                return (activation_cache['clean_state'],) + output[1:]
            return activation_cache['clean_state']

        handle_patch = target_module.register_forward_hook(patch_hook)
        with torch.no_grad():
            patched_out = self.model(corrupted_tokens).logits
        handle_patch.remove()

        # Final Score
        indirect_effect = self._calculate_logit_difference(clean_out.logits, patched_out)

        return {
            "unit": self.unit,
            "Indirect Effect": indirect_effect
        }