"""
Pipeline glue layer for end-to-end interpretability workflows.

Provides high-level functions that chain model → activations → SAE/probes →
features → visualization, handling type conversions and boilerplate automatically.

Users can go from model → results in a few lines:

    >>> from glassboxllms.pipeline import train_sae_on_model, train_probe_on_model
    >>> sae, features = train_sae_on_model("gpt2", texts, layer=5)
    >>> probe, direction = train_probe_on_model("gpt2", pos_texts, neg_texts, layer=5)
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .features import SparseAutoencoder, SAETrainer, FeatureSet
from .primitives.probes.linear import LinearProbe
from .analysis.circuits.graph import CircuitGraph
from .analysis.circuits.node import NodeType, EdgeType


def _to_torch(x: Union[np.ndarray, torch.Tensor], dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert numpy array to torch tensor if needed."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(dtype)
    return x.to(dtype)


def _to_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert torch tensor to numpy array if needed."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _load_model(model_name: str, model_class: str = "auto") -> Any:
    """Load a HuggingFace model wrapped in ModelWrapper.

    Args:
        model_name: HuggingFace model identifier.
        model_class: ``"auto"`` or ``"causal_lm"`` — passed to
            :class:`TransformersModelWrapper`.
    """
    from .models.huggingface import TransformersModelWrapper

    return TransformersModelWrapper(model_name, model_class=model_class)


def extract_activations(
    model_name: str,
    texts: List[str],
    layers: List[str],
    return_type: str = "torch",
    model: Optional[Any] = None,
) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
    """
    Extract activations from a model at specified layers.

    Wraps ModelWrapper.get_activations() with automatic type handling.

    Args:
        model_name: HuggingFace model name (e.g., "gpt2").
        texts: Input texts to extract activations for.
        layers: Layer identifiers to extract from.
        return_type: "torch" or "numpy" — output format.
        model: Optional pre-loaded ModelWrapper (avoids reloading).

    Returns:
        Dict mapping layer names to activation tensors/arrays.
        Shape: (batch_size, seq_len, hidden_dim) per layer.
    """
    if model is None:
        model = _load_model(model_name)

    # Always extract as torch for internal consistency
    raw = model.get_activations(texts, layers, return_type="torch")

    result = {}
    for layer_name, acts in raw.items():
        if return_type == "numpy":
            result[layer_name] = _to_numpy(acts)
        else:
            result[layer_name] = _to_torch(acts)

    return result


def train_sae_on_model(
    model_name: str,
    texts: List[str],
    layer: str,
    feature_dim: int = 2048,
    k: int = 64,
    n_epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = "cpu",
    model: Optional[Any] = None,
) -> Tuple[SparseAutoencoder, FeatureSet]:
    """
    Train a Sparse Autoencoder on model activations end-to-end.

    Extracts activations → creates DataLoader → trains SAE → returns FeatureSet.

    Args:
        model_name: HuggingFace model name (e.g., "gpt2").
        texts: Training texts.
        layer: Layer to extract activations from.
        feature_dim: Number of SAE features.
        k: Top-k sparsity parameter.
        n_epochs: Training epochs.
        batch_size: Training batch size.
        lr: Learning rate.
        device: Device for training.
        model: Optional pre-loaded ModelWrapper.

    Returns:
        Tuple of (trained SparseAutoencoder, FeatureSet).
    """
    # Extract activations
    acts_dict = extract_activations(
        model_name, texts, [layer], return_type="torch", model=model
    )
    activations = acts_dict[layer]  # (batch, seq_len, hidden_dim)

    # Flatten to 2D: (batch * seq_len, hidden_dim)
    if activations.ndim == 3:
        activations = activations.reshape(-1, activations.shape[-1])
    activations = activations.to(torch.float32)

    input_dim = activations.shape[-1]

    # Create DataLoader
    dataset = TensorDataset(activations)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create and train SAE
    sae = SparseAutoencoder(
        input_dim=input_dim,
        feature_dim=feature_dim,
        k=k,
        device=device,
    )
    sae.initialize_geometric_median(activations[:min(len(activations), 10000)])

    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    trainer = SAETrainer(sae, dataloader, optimizer=optimizer, device=device)
    stats = trainer.train(n_epochs=n_epochs)

    # Convert to FeatureSet
    feature_set = trainer.get_feature_set(model_name=model_name, layer=layer)

    return sae, feature_set


def train_probe_on_model(
    model_name: str,
    positive_texts: List[str],
    negative_texts: List[str],
    layer: str,
    probe_type: str = "logistic",
    model: Optional[Any] = None,
) -> Tuple[LinearProbe, np.ndarray]:
    """
    Train a linear probe on model activations end-to-end.

    Extracts activations for positive/negative examples → trains probe →
    returns probe and learned direction vector.

    Args:
        model_name: HuggingFace model name (e.g., "gpt2").
        positive_texts: Texts representing the positive concept.
        negative_texts: Texts representing the negative concept.
        layer: Layer to extract activations from.
        probe_type: Type of probe ("logistic", "cav", etc.).
        model: Optional pre-loaded ModelWrapper.

    Returns:
        Tuple of (trained LinearProbe, direction vector as numpy array).
    """
    if model is None:
        model = _load_model(model_name)

    # Extract activations for both sets
    all_texts = positive_texts + negative_texts
    acts_dict = extract_activations(
        model_name, all_texts, [layer], return_type="numpy", model=model
    )
    activations = acts_dict[layer]  # (batch, seq_len, hidden_dim)

    # Average over sequence length if 3D
    if activations.ndim == 3:
        activations = activations.mean(axis=1)

    # Create labels: 1 for positive, 0 for negative
    labels = np.array(
        [1] * len(positive_texts) + [0] * len(negative_texts),
        dtype=np.int64,
    )

    # Train probe
    probe = LinearProbe(
        layer=layer,
        direction=f"{probe_type}_probe",
        model_type=probe_type,
    )
    probe.fit(activations, labels)

    direction = probe.get_direction()

    return probe, direction


def _extract_metric(output, metric_fn: Callable) -> float:
    """Apply *metric_fn* to a model output, handling common output wrappers."""
    if hasattr(output, "logits"):
        raw = metric_fn(output.logits)
    elif hasattr(output, "last_hidden_state"):
        raw = metric_fn(output.last_hidden_state)
    else:
        raw = metric_fn(output)
    return float(raw) if isinstance(raw, torch.Tensor) else float(raw)


def discover_circuit(
    model_name: str,
    text: str,
    metric_fn: Callable,
    layers: Optional[List[str]] = None,
    corrupted_text: Optional[str] = None,
    strategy: str = "zero",
    model: Optional[Any] = None,
) -> CircuitGraph:
    """
    Discover a circuit by scanning layers with ablation.

    Runs ablation at each layer, measures metric impact, and populates
    a CircuitGraph with the results.  In addition to input→layer and
    layer→output edges, heuristic inter-layer edges are added between
    consecutive layers weighted by ``min(impact_i, impact_j)``.

    Args:
        model_name: HuggingFace model name (e.g., "gpt2").
        text: Clean input text.
        metric_fn: Function(model_output) -> scalar measuring behavior.
        layers: Layers to scan (defaults to all model layers).
        corrupted_text: Corrupted input **required** for ``"patch"`` strategy.
        strategy: Ablation strategy — ``"zero"``, ``"mean"``, ``"random"``,
            or ``"patch"``.
        model: Optional pre-loaded ModelWrapper.

    Returns:
        CircuitGraph populated with nodes for each scanned layer,
        input→layer, layer→output, and inter-layer edges.
    """
    if strategy == "patch" and corrupted_text is None:
        raise ValueError("strategy='patch' requires corrupted_text")

    if model is None:
        model = _load_model(model_name)

    wrapper_model = model.model
    device = model.device
    if layers is None:
        layers = model.layer_names

    # Tokenize inputs — move to correct device
    tokens = model.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    clean_input = {k: v.to(device) for k, v in tokens.items()}

    # Pre-compute corrupted activations for "patch" strategy
    corrupted_acts: Dict[str, torch.Tensor] = {}
    if corrupted_text is not None:
        corrupted_tokens = model.tokenizer(
            corrupted_text, return_tensors="pt", padding=True, truncation=True
        )
        corrupted_input = {k: v.to(device) for k, v in corrupted_tokens.items()}

        # Collect all corrupted layer outputs in one forward pass
        _hooks = []
        for layer_name in layers:
            ln = layer_name[0] if isinstance(layer_name, tuple) else layer_name
            try:
                mod = model.get_layer_module(ln)

                def _capture(name):
                    def _hook(module, inp, out):
                        corrupted_acts[name] = (
                            out[0].detach() if isinstance(out, tuple) else out.detach()
                        )
                    return _hook

                _hooks.append(mod.register_forward_hook(_capture(ln)))
            except (ValueError, AttributeError):
                pass
        with torch.no_grad():
            wrapper_model(**corrupted_input)
        for h in _hooks:
            h.remove()

    # Pre-compute running mean for "mean" strategy (one clean forward pass)
    clean_acts: Dict[str, torch.Tensor] = {}
    if strategy == "mean":
        _hooks = []
        for layer_name in layers:
            ln = layer_name[0] if isinstance(layer_name, tuple) else layer_name
            try:
                mod = model.get_layer_module(ln)

                def _capture_clean(name):
                    def _hook(module, inp, out):
                        act = out[0].detach() if isinstance(out, tuple) else out.detach()
                        clean_acts[name] = act.mean(dim=1, keepdim=True).expand_as(act)
                    return _hook

                _hooks.append(mod.register_forward_hook(_capture_clean(ln)))
            except (ValueError, AttributeError):
                pass
        with torch.no_grad():
            wrapper_model(**clean_input)
        for h in _hooks:
            h.remove()

    # Get baseline metric
    with torch.no_grad():
        baseline_output = wrapper_model(**clean_input)
    baseline_val = _extract_metric(baseline_output, metric_fn)

    # Build circuit graph
    graph = CircuitGraph(
        model=model_name,
        name=f"circuit_scan_{model_name}",
        metadata={
            "strategy": strategy,
            "baseline_metric": baseline_val,
            "text": text,
        },
    )

    # Add input node
    graph.add_node("input", node_type=NodeType.EMBEDDING, layer=0)
    # Add output node
    graph.add_node("output", node_type=NodeType.UNEMBEDDING)

    # Scan each layer
    layer_impacts: Dict[str, float] = {}
    ordered_node_ids: List[str] = []
    for i, layer_name in enumerate(layers):
        if isinstance(layer_name, tuple):
            layer_name = layer_name[0]

        node_id = f"layer.{i}.{layer_name}"
        ordered_node_ids.append(node_id)

        # Determine node type
        lower_name = str(layer_name).lower()
        if "attn" in lower_name or "attention" in lower_name:
            node_type = NodeType.ATTENTION_HEAD
        elif "mlp" in lower_name or "ff" in lower_name:
            node_type = NodeType.MLP_LAYER
        else:
            node_type = NodeType.RESIDUAL_STREAM

        graph.add_node(node_id, node_type=node_type, layer=i)

        # Try to measure impact of ablating this layer
        try:
            module = model.get_layer_module(layer_name)

            def make_ablation_hook(strat, ln):
                def hook(module, input, output):
                    def _replace(tensor):
                        if not isinstance(tensor, torch.Tensor):
                            return tensor
                        if strat == "zero":
                            return torch.zeros_like(tensor)
                        elif strat == "random":
                            return torch.randn_like(tensor)
                        elif strat == "mean":
                            mean_act = clean_acts.get(ln)
                            if mean_act is not None:
                                return mean_act
                            return tensor.mean(dim=1, keepdim=True).expand_as(tensor)
                        elif strat == "patch":
                            patched = corrupted_acts.get(ln)
                            if patched is not None:
                                # Truncate/pad to match shape
                                min_seq = min(patched.shape[1], tensor.shape[1])
                                result = tensor.clone()
                                result[:, :min_seq] = patched[:, :min_seq]
                                return result
                            return tensor
                        return tensor

                    if isinstance(output, tuple):
                        return tuple(_replace(o) for o in output)
                    return _replace(output)
                return hook

            handle = module.register_forward_hook(make_ablation_hook(strategy, layer_name))
            try:
                with torch.no_grad():
                    ablated_output = wrapper_model(**clean_input)
            finally:
                handle.remove()

            ablated_val = _extract_metric(ablated_output, metric_fn)
            impact = abs(baseline_val - ablated_val)
            layer_impacts[node_id] = impact

            # Add edges with impact as weight
            graph.add_edge("input", node_id, weight=impact, edge_type=EdgeType.INFERRED)
            graph.add_edge(node_id, "output", weight=impact, edge_type=EdgeType.INFERRED)

        except (ValueError, AttributeError, RuntimeError):
            layer_impacts[node_id] = 0.0
            graph.add_edge("input", node_id, weight=0.0, edge_type=EdgeType.INFERRED)
            graph.add_edge(node_id, "output", weight=0.0, edge_type=EdgeType.INFERRED)

    # Add heuristic inter-layer edges between consecutive layers
    for j in range(len(ordered_node_ids) - 1):
        src = ordered_node_ids[j]
        tgt = ordered_node_ids[j + 1]
        src_impact = layer_impacts.get(src, 0.0)
        tgt_impact = layer_impacts.get(tgt, 0.0)
        if src_impact > 0 and tgt_impact > 0:
            weight = min(src_impact, tgt_impact)
            graph.add_edge(src, tgt, weight=weight, edge_type=EdgeType.INFERRED)

    return graph


def run_logit_lens(
    model_name: str,
    text: str,
    top_k: int = 5,
    model: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run logit lens analysis on a model.

    Projects each layer's hidden states through the unembedding matrix to see
    what tokens the model is "thinking about" at each layer.

    Requires a CausalLM model (with ``lm_head``).  Pass a pre-loaded wrapper
    created with ``model_class="causal_lm"`` to avoid duplicate model loads.

    Args:
        model_name: HuggingFace model name (e.g., "gpt2").
        text: Input text to analyze.
        top_k: Number of top predicted tokens to return per position.
        model: Optional pre-loaded ``TransformersModelWrapper``
            (must have been loaded with ``model_class="causal_lm"``).

    Returns:
        Dict with keys:
            - "tokens": List of input token strings
            - "logit_lens_data": np.ndarray of shape (n_layers, seq_len)
                with probability of correct next token at each layer
            - "top_k_tokens": List[List[List[str]]] — top-k token predictions
                per layer per position
            - "probabilities": np.ndarray of shape (n_layers, seq_len, vocab_size)
    """
    if model is None:
        model = _load_model(model_name, model_class="causal_lm")

    hf_model = model.model
    tokenizer = model.tokenizer
    device = model.device

    # Ensure the model has an lm_head
    lm_head = model.lm_head  # raises AttributeError if not CausalLM

    # Detect final layer norm (e.g., GPT-2 has transformer.ln_f)
    final_ln = None
    for name in ("transformer.ln_f", "model.norm", "model.layer_norm"):
        ln = dict(hf_model.named_modules()).get(name)
        if ln is not None:
            final_ln = ln
            break

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    tokens = [tokenizer.decode(t) for t in input_ids[0]]

    # Forward pass with hidden states
    with torch.no_grad():
        outputs = hf_model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states  # tuple of (batch, seq_len, hidden)

    n_layers = len(hidden_states) - 1  # exclude embedding layer
    seq_len = input_ids.shape[1]

    logit_lens_data = np.zeros((n_layers, seq_len))
    top_k_tokens_all = []
    all_probs = []

    for layer_idx in range(1, len(hidden_states)):  # skip embedding layer
        hidden = hidden_states[layer_idx]

        with torch.no_grad():
            # Apply final layer norm before projecting through lm_head
            # (critical for correct logit lens — raw hidden states are not
            # in the right scale/space for the unembedding matrix)
            normed = final_ln(hidden) if final_ln is not None else hidden
            logits = lm_head(normed)  # (batch, seq_len, vocab_size)
            probs = torch.softmax(logits[0], dim=-1)  # (seq_len, vocab_size)

        all_probs.append(_to_numpy(probs))

        # For each position, get probability of the "correct" next token
        layer_top_k = []
        for pos in range(seq_len):
            # Top-k tokens at this position
            top_vals, top_ids = torch.topk(probs[pos], top_k)
            top_tokens = [tokenizer.decode(tid.item()) for tid in top_ids]
            layer_top_k.append(top_tokens)

            # Correct next token probability
            if pos < seq_len - 1:
                next_token_id = input_ids[0, pos + 1].item()
                logit_lens_data[layer_idx - 1, pos] = float(probs[pos, next_token_id])
            else:
                logit_lens_data[layer_idx - 1, pos] = 0.0

        top_k_tokens_all.append(layer_top_k)

    return {
        "tokens": tokens,
        "logit_lens_data": logit_lens_data,
        "top_k_tokens": top_k_tokens_all,
        "probabilities": np.stack(all_probs, axis=0),
    }


def steer_on_model(
    model_name: str,
    texts: List[str],
    layer: str,
    direction: Union[np.ndarray, torch.Tensor],
    strength: float = 3.0,
    model: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Apply directional steering and capture before/after activations.

    Extracts activations at *layer* before steering, then re-runs the
    forward pass with a :class:`DirectionalSteering` hook applied and
    captures the steered activations.  Both activation sets are returned
    so they can be fed to the steering visualization adapter.

    Args:
        model_name: HuggingFace model name (e.g., "gpt2").
        texts: Input texts to steer.
        layer: Layer to apply steering at (e.g., ``"transformer.h.5"``).
        direction: Steering direction vector — typically from
            ``probe.get_direction()``.  Shape must be ``(hidden_dim,)``.
        strength: Scalar multiplier for the steering magnitude.
        model: Optional pre-loaded ModelWrapper.

    Returns:
        Dict with keys:
            - ``"activations_before"``: np.ndarray (n_samples, hidden_dim)
            - ``"activations_after"``: np.ndarray (n_samples, hidden_dim)
            - ``"direction"``: np.ndarray (hidden_dim,)
            - ``"strength"``: float
            - ``"layer"``: str
    """
    from .interventions.steering import DirectionalSteering

    if model is None:
        model = _load_model(model_name)

    # --- Before steering: extract baseline activations ---
    acts_before = extract_activations(
        model_name, texts, [layer], return_type="numpy", model=model,
    )[layer]  # (batch, seq_len, hidden_dim)
    # Mean-pool over sequence
    if acts_before.ndim == 3:
        acts_before = acts_before.mean(axis=1)

    # --- After steering: apply hook and extract activations ---
    direction_tensor = _to_torch(direction)
    if direction_tensor.ndim != 1:
        raise ValueError(
            f"direction must be 1-D (hidden_dim,), got shape {direction_tensor.shape}"
        )

    steering = DirectionalSteering(
        layer=layer, direction=direction_tensor, strength=strength,
    )

    # We need to capture the *steered* activations.  Register both the
    # steering hook and a capture hook on the same layer.
    target_module = model.get_layer_module(layer)
    steered_acts: List[torch.Tensor] = []

    def capture_hook(module, input, output):
        act = output[0].detach() if isinstance(output, tuple) else output.detach()
        steered_acts.append(act.cpu())

    # Steering hook fires first (modifies output), capture fires second
    steering_handle = target_module.register_forward_hook(steering.hook_fn)
    capture_handle = target_module.register_forward_hook(capture_hook)

    try:
        model.forward(texts)
    finally:
        steering_handle.remove()
        capture_handle.remove()

    # Combine captured activations
    acts_after_t = torch.cat(steered_acts, dim=0)  # (batch, seq_len, hidden)
    acts_after = _to_numpy(acts_after_t)
    if acts_after.ndim == 3:
        acts_after = acts_after.mean(axis=1)

    return {
        "activations_before": acts_before,
        "activations_after": acts_after,
        "direction": _to_numpy(direction_tensor),
        "strength": strength,
        "layer": layer,
    }
