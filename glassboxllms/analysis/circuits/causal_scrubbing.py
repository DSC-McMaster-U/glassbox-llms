"""
CausalScrubber — Causal interventions for circuit discovery.

Implements causal scrubbing and path patching to test which model
components are causally responsible for a given behaviour.

Originally by Ankita Sharma. Rewritten for correct HookManager integration.

Usage::

    from glassboxllms.instrumentation.hook_manager import HookManager
    from glassboxllms.analysis.circuits import CausalScrubber

    manager = HookManager(model)
    scrubber = CausalScrubber(model, manager)

    diff = scrubber.scrub_node(
        layer="transformer.h.5",
        clean_input=clean_tokens,
        strategy="zero",
        metric_fn=lambda out: logit_diff_metric(out.logits, correct, incorrect),
    )
"""

import torch
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Union


class CausalScrubber:
    """
    Test causal hypotheses by intervening on model activations.

    Supports four replacement strategies:

    * ``"zero"``   — Replace activation with zeros.
    * ``"mean"``   — Replace with dataset-mean activation (if provided).
    * ``"random"`` — Replace with random Gaussian noise.
    * ``"patch"``  — Replace with activations from a corrupted input.

    Args:
        model: Any ``torch.nn.Module`` (or ``TransformersModelWrapper``).
        hook_manager: A :class:`HookManager` bound to *model*.
    """

    def __init__(self, model: nn.Module, hook_manager):
        self.model = model
        self.hook_manager = hook_manager

    # ── Public API ──────────────────────────────────────────────

    def scrub_node(
        self,
        layer: str,
        clean_input: Dict[str, torch.Tensor],
        corrupted_input: Optional[Dict[str, torch.Tensor]] = None,
        strategy: str = "zero",
        mean_activation: Optional[torch.Tensor] = None,
        metric_fn: Optional[Callable] = None,
    ) -> Any:
        """
        Ablate a single node (layer) and measure the behavioural effect.

        Args:
            layer: Named module path to intervene on.
            clean_input: Tokenized input dict (input_ids, attention_mask, …).
            corrupted_input: If strategy is ``"patch"``, the alternative input
                whose activations will replace the clean ones.
            strategy: One of ``"zero"``, ``"mean"``, ``"random"``, ``"patch"``.
            mean_activation: Pre-computed mean activation for ``"mean"`` strategy.
            metric_fn: ``(model_output) -> scalar``.  If provided the metric
                value is returned instead of the raw output.

        Returns:
            Model output (or metric value if *metric_fn* is given).
        """
        # Capture corrupted activations if patching
        baseline_act = None
        if strategy == "patch":
            if corrupted_input is None:
                raise ValueError("corrupted_input required for 'patch' strategy.")
            baseline_act = self._capture_activation(layer, corrupted_input)
        elif strategy == "mean" and mean_activation is not None:
            baseline_act = mean_activation

        # Build the intervention hook
        def intervention_hook(module, input, output):
            if isinstance(output, tuple):
                original = output[0]
            else:
                original = output

            replacement = self._get_replacement(strategy, original, baseline_act)

            if isinstance(output, tuple):
                return (replacement,) + output[1:]
            return replacement

        # Run model with intervention
        self.hook_manager.add_hook(layer, intervention_hook)
        try:
            with torch.no_grad():
                output = self.model(**clean_input)
            return metric_fn(output) if metric_fn else output
        finally:
            self.hook_manager.remove_all_hooks()

    def scrub_path(
        self,
        source: str,
        target: str,
        clean_input: Dict[str, torch.Tensor],
        corrupted_input: Optional[Dict[str, torch.Tensor]] = None,
        strategy: str = "patch",
        metric_fn: Optional[Callable] = None,
    ) -> Any:
        """
        Intervene on the path from *source* to *target*.

        Captures activations at *source* from the corrupted input, then
        patches them into the clean forward pass at *target*.
        """
        baseline_act = None
        if corrupted_input is not None:
            baseline_act = self._capture_activation(source, corrupted_input)

        def intervention_hook(module, input, output):
            if isinstance(output, tuple):
                original = output[0]
            else:
                original = output

            replacement = self._get_replacement(strategy, original, baseline_act)

            if isinstance(output, tuple):
                return (replacement,) + output[1:]
            return replacement

        self.hook_manager.add_hook(target, intervention_hook)
        try:
            with torch.no_grad():
                output = self.model(**clean_input)
            return metric_fn(output) if metric_fn else output
        finally:
            self.hook_manager.remove_all_hooks()

    def scan_layers(
        self,
        layers: List[str],
        clean_input: Dict[str, torch.Tensor],
        metric_fn: Callable,
        strategy: str = "zero",
        corrupted_input: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """
        Ablate each layer one at a time and record the metric change.

        Returns:
            Dict mapping layer name → metric value with that layer ablated.
        """
        results = {}
        for layer in layers:
            value = self.scrub_node(
                layer=layer,
                clean_input=clean_input,
                corrupted_input=corrupted_input,
                strategy=strategy,
                metric_fn=metric_fn,
            )
            if isinstance(value, torch.Tensor):
                value = value.item()
            results[layer] = value
        return results

    # ── Helpers ─────────────────────────────────────────────────

    def _capture_activation(
        self,
        layer: str,
        inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Run a forward pass and capture the activation at *layer*."""
        self.hook_manager.attach_hook(layer)
        try:
            with torch.no_grad():
                _ = self.model(**inputs)
            act = self.hook_manager.get(layer)
            if act is None:
                raise RuntimeError(f"Failed to capture activation at '{layer}'.")
            return act.clone()
        finally:
            self.hook_manager.remove_all_hooks()
            self.hook_manager.clear_activations()

    @staticmethod
    def _get_replacement(
        strategy: str,
        original: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if strategy == "zero":
            return torch.zeros_like(original)
        elif strategy == "mean":
            if baseline is not None:
                return baseline.to(device=original.device, dtype=original.dtype)
            return original.mean(dim=0, keepdim=True).expand_as(original)
        elif strategy == "random":
            return torch.randn_like(original)
        elif strategy == "patch":
            if baseline is None:
                raise ValueError("Baseline activation required for 'patch' strategy.")
            return baseline.to(device=original.device, dtype=original.dtype)
        raise ValueError(f"Unknown strategy: {strategy!r}")


def logit_diff_metric(
    logits: torch.Tensor,
    correct_idx: int,
    incorrect_idx: int,
) -> torch.Tensor:
    """
    Standard metric for measuring behavioural shift.

    Computes ``logit[correct] - logit[incorrect]`` for the **last** token
    position in the sequence.
    """
    return logits[0, -1, correct_idx] - logits[0, -1, incorrect_idx]
