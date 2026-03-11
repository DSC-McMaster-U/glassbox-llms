"""
Logit Lens experiment — Decode intermediate activations through the unembedding.

The "logit lens" technique (introduced by nostalgebraist, 2020) takes the
hidden state at each layer, applies the model's final layer-norm and
unembedding matrix, and reads off the top-predicted tokens.  This reveals
how the model's predictions *refine* through depth.

Usage::

    from glassboxllms.experiments import run_experiment

    result = run_experiment("logit_lens", {
        "model_name": "gpt2",
        "text": "The capital of France is",
        "top_k": 5,
    })
    print(result.summary())

The result ``artifacts["layer_predictions"]`` is a list of dicts, one per
layer, with keys ``layer``, ``top_tokens``, ``top_probs``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from glassboxllms.experiments.base import BaseExperiment, ExperimentResult
from glassboxllms.instrumentation import HookManager


class LogitLensExperiment(BaseExperiment):
    """
    Decode every transformer layer's hidden state into vocabulary space.

    Supports any HuggingFace causal-LM model (GPT-2, Llama, Pythia, …).
    """

    @property
    def name(self) -> str:
        return "logit_lens"

    @property
    def default_config(self) -> Dict[str, Any]:
        return {
            "model_name": "gpt2",
            "text": "The capital of France is",
            "top_k": 5,
            "device": None,  # auto-detect
            "layers": None,  # None = all transformer blocks
            "token_position": -1,  # last token by default
        }

    def validate_config(self, config: Dict[str, Any]) -> bool:
        merged = {**self.default_config, **config}
        return "model_name" in merged and "text" in merged

    def run(self, config: Dict[str, Any]) -> ExperimentResult:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cfg = {**self.default_config, **config}
        model_name = cfg["model_name"]
        text = cfg["text"]
        top_k = cfg["top_k"]
        token_pos = cfg["token_position"]
        device = cfg["device"] or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        model.eval()

        # Tokenize
        inputs = tokenizer(text, return_tensors="pt").to(device)
        input_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # Discover transformer block layers
        inner_model = _unwrap_model(model)
        block_layers = _find_transformer_blocks(inner_model)
        if cfg["layers"] is not None:
            block_layers = [l for l in block_layers if l in cfg["layers"]]

        if not block_layers:
            return ExperimentResult(
                experiment_type=self.name,
                model_name=model_name,
                status="failed",
                metrics={},
                details={"error": "No transformer block layers found."},
            )

        # Capture activations at every block
        hook_manager = HookManager(inner_model)
        for layer in block_layers:
            hook_manager.attach_hook(layer)

        with torch.no_grad():
            output = model(**inputs)

        hook_manager.remove_hooks()

        # Get the unembedding components
        ln_final, unembed = _get_unembedding(model)

        # Decode each layer's hidden state
        layer_predictions: List[Dict[str, Any]] = []
        for layer_name in block_layers:
            hidden = hook_manager.activations.get(layer_name)
            if hidden is None:
                continue

            # Take the specified token position
            h = hidden[0, token_pos, :]  # (hidden_dim,)

            # Apply final LN + unembedding
            if ln_final is not None:
                h = ln_final(h.unsqueeze(0)).squeeze(0)
            logits = h @ unembed.T  # (vocab_size,)

            probs = F.softmax(logits, dim=-1)
            top_vals, top_ids = probs.topk(top_k)

            top_tokens = [tokenizer.decode([tid]) for tid in top_ids.tolist()]
            top_probs = [round(p, 4) for p in top_vals.tolist()]

            layer_predictions.append({
                "layer": layer_name,
                "top_tokens": top_tokens,
                "top_probs": top_probs,
            })

        # Final output predictions for comparison
        final_logits = output.logits[0, token_pos, :]
        final_probs = F.softmax(final_logits, dim=-1)
        final_vals, final_ids = final_probs.topk(top_k)
        final_tokens = [tokenizer.decode([tid]) for tid in final_ids.tolist()]

        # Compute convergence: at which layer does the final top-1 appear?
        final_top1 = final_tokens[0].strip()
        convergence_layer = None
        for i, lp in enumerate(layer_predictions):
            if lp["top_tokens"][0].strip() == final_top1:
                convergence_layer = i
                break

        metrics = {
            "n_layers_analyzed": len(layer_predictions),
            "final_top_token": final_top1,
            "final_top_prob": round(final_vals[0].item(), 4),
            "convergence_layer": convergence_layer,
            "convergence_frac": (
                round(convergence_layer / len(layer_predictions), 3)
                if convergence_layer is not None
                else None
            ),
        }

        return ExperimentResult(
            experiment_type=self.name,
            model_name=model_name,
            status="success",
            metrics=metrics,
            artifacts={
                "layer_predictions": layer_predictions,
                "final_predictions": {
                    "tokens": final_tokens,
                    "probs": [round(p, 4) for p in final_vals.tolist()],
                },
                "input_tokens": input_tokens,
                "text": text,
                "token_position": token_pos,
            },
            config=cfg,
        )


# ── Model introspection helpers ──────────────────────────────────


def _unwrap_model(model: nn.Module) -> nn.Module:
    """
    Get the inner transformer model from a CausalLM wrapper.

    HuggingFace wraps the base model: ``GPT2LMHeadModel.transformer``,
    ``LlamaForCausalLM.model``, etc.
    """
    # Common HF attribute names for the inner model
    for attr in ("transformer", "model", "gpt_neox", "bert"):
        if hasattr(model, attr):
            inner = getattr(model, attr)
            if isinstance(inner, nn.Module) and inner is not model:
                return inner
    return model


def _find_transformer_blocks(model: nn.Module) -> List[str]:
    """
    Auto-detect the layer paths for transformer blocks.

    Looks for the common patterns used by HuggingFace models:
    ``h.0``, ``layers.0``, ``block.0``, ``encoder.layer.0``, etc.
    """
    candidates: List[str] = []
    for name, module in model.named_modules():
        # Skip the top-level model itself
        if not name:
            continue

        parts = name.split(".")

        # Match patterns like "h.0", "layers.0", "block.0",
        # "encoder.layer.0", "decoder.layers.0"
        if len(parts) >= 2 and parts[-1].isdigit():
            parent = parts[-2]
            if parent in ("h", "layers", "layer", "block", "blocks"):
                candidates.append(name)

    return candidates


def _get_unembedding(model: nn.Module):
    """
    Extract the final layer-norm and unembedding (lm_head) weight.

    Returns:
        (ln_final, unembed_weight) — ln_final may be ``None`` if not found.
    """
    # Try to find the LM head weight
    unembed = None
    for attr in ("lm_head",):
        if hasattr(model, attr):
            head = getattr(model, attr)
            if isinstance(head, nn.Linear):
                unembed = head.weight.data
                break

    if unembed is None:
        # Fallback: use input embeddings (weight-tied models)
        for attr in ("transformer", "model", "gpt_neox"):
            inner = getattr(model, attr, None)
            if inner is None:
                continue
            for emb_attr in ("wte", "embed_tokens", "embed_in"):
                emb = getattr(inner, emb_attr, None)
                if emb is not None and hasattr(emb, "weight"):
                    unembed = emb.weight.data
                    break
            if unembed is not None:
                break

    if unembed is None:
        raise RuntimeError("Could not find unembedding matrix in model.")

    # Try to find the final layer norm
    ln_final = None
    inner = _unwrap_model(model)
    for attr in ("ln_f", "norm", "final_layer_norm", "layer_norm"):
        ln = getattr(inner, attr, None)
        if ln is not None and isinstance(ln, nn.Module):
            ln_final = ln
            break

    return ln_final, unembed
