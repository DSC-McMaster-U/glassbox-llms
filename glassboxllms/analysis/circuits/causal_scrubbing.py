# Implemented by Ankita Sharma (GitHub: sharmaankita3387)
# Part of GDSC Research Project
# Date: February 07, 2026

import torch
import torch.nn as nn
from typing import Dict, Callable, Optional, Union, List

class CausalScrubber:
    """
    Implements Causal Scrubbing and Path Patching for glassboxllms.
    Enables controlled interventions on model activations to test causal hypotheses.
    """
    def __init__(self, model: nn.Module, hook_manager, activation_store):
        self.model = model
        self.hook_manager = hook_manager
        self.store = activation_store

    def _get_replacement(self, strategy: str, original_act: torch.Tensor, 
                         baseline_act: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Determines the 'scrubbed' activation value based on the chosen strategy."""
        if strategy == "zero":
            return torch.zeros_like(original_act)
        elif strategy == "mean":
            # Assumes baseline_act contains pre-computed mean across a dataset
            return baseline_act if baseline_act is not None else original_act.mean(dim=0, keepdim=True)
        elif strategy == "random":
            return torch.randn_like(original_act)
        elif strategy == "patch":
            if baseline_act is None:
                raise ValueError("Baseline activation required for 'patch' strategy.")
            return baseline_act
        return original_act

    def scrub_path(
        self,
        source: str,
        target: str,
        clean_input: torch.Tensor,
        corrupted_input: Optional[torch.Tensor] = None,
        strategy: str = "patch",
        metric_fn: Optional[Callable] = None
    ) -> torch.Tensor:
        """
        Intervenes on the path between source and target.
        If corrupted_input is provided, it performs a 'resample ablation'.
        """
        # 1. Capture baseline activations if patching
        baseline_activations = None
        if corrupted_input is not None:
            with torch.no_grad():
                _ = self.model(corrupted_input)
                baseline_activations = self.store.get(source)

        # 2. Define the Hook
        def intervention_hook(activation, hook):
            # Only intervene if we are at the target component
            return self._get_replacement(strategy, activation, baseline_activations)

        # 3. Run model with intervention
        self.hook_manager.add_hook(target, intervention_hook)
        try:
            output = self.model(clean_input)
            if metric_fn:
                return metric_fn(output)
            return output
        finally:
            self.hook_manager.remove_all_hooks()

def logit_diff_metric(logits: torch.Tensor, correct_idx: int, incorrect_idx: int):
    """Standard metric for measuring behavioral shift."""
    return logits[0, -1, correct_idx] - logits[0, -1, incorrect_idx]