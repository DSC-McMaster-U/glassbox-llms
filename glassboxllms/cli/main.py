"""
glassbox — CLI entry point.

Usage::

    glassbox layers gpt2
    glassbox logit-lens gpt2 "The capital of France is"
    glassbox probe gpt2 --layer transformer.h.6 --dataset sentiment
    glassbox scan gpt2 "The capital of France is" --strategy zero
    glassbox experiments
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="glassbox",
        description="GlassBox LLMs — Interpretability from the command line.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              glassbox layers gpt2
              glassbox logit-lens gpt2 "The capital of France is"
              glassbox scan gpt2 "The capital of France is"
              glassbox experiments
        """),
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ── layers ───────────────────────────────────────────────────
    p_layers = sub.add_parser(
        "layers",
        help="List named layers in a model.",
    )
    p_layers.add_argument("model", help="HuggingFace model name or path")
    p_layers.add_argument(
        "--type",
        choices=["all", "attention", "mlp", "output"],
        default="all",
        help="Filter layer type (default: all)",
    )
    p_layers.add_argument(
        "--pattern", default=None, help="Substring filter on layer names"
    )

    # ── logit-lens ───────────────────────────────────────────────
    p_lens = sub.add_parser(
        "logit-lens",
        help="Run logit lens: decode predictions at every layer.",
    )
    p_lens.add_argument("model", help="HuggingFace causal LM name")
    p_lens.add_argument("text", help="Input text to analyze")
    p_lens.add_argument(
        "--top-k", type=int, default=5, help="Number of top tokens per layer"
    )
    p_lens.add_argument(
        "--position",
        type=int,
        default=-1,
        help="Token position to decode (default: last)",
    )
    p_lens.add_argument(
        "--json", action="store_true", dest="output_json", help="Output as JSON"
    )

    # ── scan ─────────────────────────────────────────────────────
    p_scan = sub.add_parser(
        "scan",
        help="Ablation scan: zero/mean/random ablate each layer and measure impact.",
    )
    p_scan.add_argument("model", help="HuggingFace causal LM name")
    p_scan.add_argument("text", help="Input text")
    p_scan.add_argument(
        "--strategy",
        choices=["zero", "mean", "random"],
        default="zero",
        help="Ablation strategy (default: zero)",
    )
    p_scan.add_argument(
        "--correct", type=int, default=None, help="Correct token ID"
    )
    p_scan.add_argument(
        "--incorrect", type=int, default=None, help="Incorrect token ID"
    )
    p_scan.add_argument(
        "--json", action="store_true", dest="output_json", help="Output as JSON"
    )

    # ── extract ──────────────────────────────────────────────────
    p_extract = sub.add_parser(
        "extract",
        help="Extract activations from a model and save to disk.",
    )
    p_extract.add_argument("model", help="HuggingFace model name")
    p_extract.add_argument(
        "--texts", nargs="+", required=True, help="Input texts"
    )
    p_extract.add_argument(
        "--layers", nargs="+", required=True, help="Layer names to extract"
    )
    p_extract.add_argument(
        "--output", default="./activations.pt", help="Output file path"
    )
    p_extract.add_argument(
        "--pooling",
        choices=["mean", "cls", "last", "none"],
        default="mean",
        help="Pooling strategy (default: mean)",
    )

    # ── experiments ──────────────────────────────────────────────
    p_exp = sub.add_parser(
        "experiments",
        help="List all registered experiments.",
    )

    # ── run ──────────────────────────────────────────────────────
    p_run = sub.add_parser(
        "run",
        help="Run a registered experiment by name.",
    )
    p_run.add_argument("experiment", help="Experiment name")
    p_run.add_argument(
        "--config",
        default=None,
        help="JSON config file or inline JSON string",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    try:
        if args.command == "layers":
            return _cmd_layers(args)
        elif args.command == "logit-lens":
            return _cmd_logit_lens(args)
        elif args.command == "scan":
            return _cmd_scan(args)
        elif args.command == "extract":
            return _cmd_extract(args)
        elif args.command == "experiments":
            return _cmd_experiments(args)
        elif args.command == "run":
            return _cmd_run(args)
        else:
            parser.print_help()
            return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


# ── Command implementations ─────────────────────────────────────


def _cmd_layers(args: argparse.Namespace) -> int:
    from transformers import AutoModel

    from glassboxllms.instrumentation import get_layer_names

    print(f"Loading {args.model}...")
    model = AutoModel.from_pretrained(args.model)
    names = get_layer_names(model, layer_type=args.type)

    if args.pattern:
        names = [n for n in names if args.pattern in n]

    print(f"\n{len(names)} layers found:\n")
    for name in names:
        print(f"  {name}")

    return 0


def _cmd_logit_lens(args: argparse.Namespace) -> int:
    from glassboxllms.experiments import run_experiment

    print(f"Running logit lens on {args.model}...")
    result = run_experiment("logit_lens", {
        "model_name": args.model,
        "text": args.text,
        "top_k": args.top_k,
        "token_position": args.position,
    })

    if args.output_json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
        return 0

    print(result.summary())

    predictions = result.artifacts.get("layer_predictions", [])
    print(f"{'Layer':<35} {'Top-1':>12} {'Prob':>8}")
    print("-" * 57)
    for lp in predictions:
        tok = lp["top_tokens"][0]
        prob = lp["top_probs"][0]
        layer = lp["layer"]
        # Highlight convergence
        marker = ""
        if tok.strip() == result.metrics.get("final_top_token", "").strip():
            marker = " <--"
        print(f"  {layer:<33} {tok:>12} {prob:>7.1%}{marker}")

    final = result.artifacts.get("final_predictions", {})
    if final:
        print(f"\n  {'(final output)':<33} {final['tokens'][0]:>12} {final['probs'][0]:>7.1%}")

    return 0


def _cmd_scan(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from glassboxllms.analysis.circuits.causal_scrubbing import (
        CausalScrubber,
        logit_diff_metric,
    )
    from glassboxllms.instrumentation import HookManager

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()

    inputs = tokenizer(args.text, return_tensors="pt")

    # Determine correct/incorrect token IDs
    if args.correct is not None and args.incorrect is not None:
        correct_id = args.correct
        incorrect_id = args.incorrect
    else:
        # Auto-detect: use top-2 logits from the clean pass
        with torch.no_grad():
            out = model(**inputs)
        top2 = out.logits[0, -1].topk(2)
        correct_id = top2.indices[0].item()
        incorrect_id = top2.indices[1].item()
        print(
            f"Auto-detected: correct={tokenizer.decode([correct_id])!r} "
            f"(id={correct_id}), incorrect={tokenizer.decode([incorrect_id])!r} "
            f"(id={incorrect_id})"
        )

    # Find transformer blocks in the inner model
    inner = model.transformer if hasattr(model, "transformer") else model.model
    hook_manager = HookManager(inner)
    scrubber = CausalScrubber(inner, hook_manager)

    # Discover layers
    layers = [
        name for name, _ in inner.named_modules()
        if name and name.split(".")[-1].isdigit()
        and name.split(".")[-2] in ("h", "layers", "layer", "block")
    ]

    if not layers:
        print("No transformer block layers found.", file=sys.stderr)
        return 1

    def metric(out):
        return logit_diff_metric(out.logits, correct_id, incorrect_id)

    print(f"\nScanning {len(layers)} layers with '{args.strategy}' ablation...\n")

    # Need to pass the right inputs to inner model
    # For CausalScrubber, we pass inputs that the inner model expects
    # The inner model doesn't take labels, just input_ids etc.
    clean_input = {k: v for k, v in inputs.items()}

    # CausalScrubber calls self.model(**clean_input), but inner model
    # needs the right format. For GPT-2, transformer takes input_ids etc.
    # Actually, CausalScrubber works on the inner model, so we need to
    # ensure the metric function works with inner model output.
    # The inner model returns BaseModelOutputWithPast, not CausalLMOutput.
    # We need to handle this differently.

    # Use the full model instead, and just manually scan
    results = {}
    for layer_name in layers:
        # Build absolute path for the full model
        if hasattr(model, "transformer"):
            full_path = f"transformer.{layer_name}"
        else:
            full_path = f"model.{layer_name}"

        full_hook_manager = HookManager(model)
        full_scrubber = CausalScrubber(model, full_hook_manager)

        value = full_scrubber.scrub_node(
            layer=full_path,
            clean_input=clean_input,
            strategy=args.strategy,
            metric_fn=metric,
        )
        if isinstance(value, torch.Tensor):
            value = value.item()
        results[layer_name] = value

    # Sort by impact (absolute difference from baseline)
    baseline_diff = metric(model(**inputs)).item()

    if args.output_json:
        out = {
            "baseline": baseline_diff,
            "strategy": args.strategy,
            "results": results,
        }
        print(json.dumps(out, indent=2, default=str))
        return 0

    print(f"Baseline logit diff: {baseline_diff:.4f}\n")
    print(f"{'Layer':<35} {'Ablated':>10} {'Delta':>10}")
    print("-" * 57)

    for layer, val in results.items():
        delta = val - baseline_diff
        bar = "!" * min(int(abs(delta) * 10), 20)
        print(f"  {layer:<33} {val:>10.4f} {delta:>+10.4f}  {bar}")

    return 0


def _cmd_extract(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModel, AutoTokenizer

    from glassboxllms.instrumentation import ActivationExtractor

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)

    extractor = ActivationExtractor(model)
    print(f"Extracting from {len(args.layers)} layers, {len(args.texts)} texts...")

    activations = extractor.extract(
        texts=args.texts,
        tokenizer=tokenizer,
        layers=args.layers,
        pooling=args.pooling,
        return_type="torch",
    )

    # Save
    torch.save(activations, args.output)
    print(f"\nSaved to {args.output}")
    for layer_name, tensor in activations.items():
        print(f"  {layer_name}: {tensor.shape}")

    return 0


def _cmd_experiments(args: argparse.Namespace) -> int:
    from glassboxllms.experiments import list_experiments, get_experiment

    names = list_experiments()
    print(f"\n{len(names)} registered experiments:\n")
    for name in names:
        exp = get_experiment(name)
        config = exp.default_config
        print(f"  {name}")
        print(f"    Default config: {json.dumps(config, indent=6, default=str)}")
        print()
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from glassboxllms.experiments import run_experiment

    config = {}
    if args.config:
        try:
            config = json.loads(args.config)
        except json.JSONDecodeError:
            # Try as file path
            with open(args.config) as f:
                config = json.load(f)

    print(f"Running experiment '{args.experiment}'...")
    result = run_experiment(args.experiment, config)
    print(result.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
