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


def _load_model(model_name: str) -> Any:
    """Load a HuggingFace model wrapped in ModelWrapper."""
    from .models.huggingface import TransformersModelWrapper

    class _ConcreteWrapper(TransformersModelWrapper):
        """Concrete wrapper since TransformersModelWrapper is abstract."""
        pass

    return _ConcreteWrapper(model_name)


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
    Discover a circuit by scanning layers with CausalScrubber.

    Runs ablation at each layer, measures metric impact, and populates
    a CircuitGraph with the results.

    Args:
        model_name: HuggingFace model name (e.g., "gpt2").
        text: Clean input text.
        metric_fn: Function(model_output) -> scalar measuring behavior.
        layers: Layers to scan (defaults to all model layers).
        corrupted_text: Optional corrupted input for patch strategy.
        strategy: Ablation strategy ("zero", "mean", "random", "patch").
        model: Optional pre-loaded ModelWrapper.

    Returns:
        CircuitGraph populated with nodes for each scanned layer and
        edges weighted by metric impact.
    """
    from .analysis.circuits.causal_scrubbing import CausalScrubber
    from .primitives.probes.activation_store import ActivationStore

    if model is None:
        model = _load_model(model_name)

    wrapper_model = model.model
    if layers is None:
        layers = model.layer_names

    # Tokenize inputs
    tokens = model.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    clean_input = tokens["input_ids"]

    corrupted_input = None
    if corrupted_text is not None:
        corrupted_tokens = model.tokenizer(
            corrupted_text, return_tensors="pt", padding=True, truncation=True
        )
        corrupted_input = corrupted_tokens["input_ids"]

    # Get baseline metric
    with torch.no_grad():
        baseline_output = wrapper_model(clean_input)
        if hasattr(baseline_output, "logits"):
            baseline_metric = metric_fn(baseline_output.logits)
        elif hasattr(baseline_output, "last_hidden_state"):
            baseline_metric = metric_fn(baseline_output.last_hidden_state)
        else:
            baseline_metric = metric_fn(baseline_output)
    baseline_val = float(baseline_metric) if isinstance(baseline_metric, torch.Tensor) else float(baseline_metric)

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
    layer_impacts = {}
    for i, layer_name in enumerate(layers):
        if isinstance(layer_name, tuple):
            layer_name = layer_name[0]

        node_id = f"layer.{i}.{layer_name}"

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

            # Create a zeroing/ablation hook
            original_output = [None]

            def make_ablation_hook(strat):
                def hook(module, input, output):
                    if strat == "zero":
                        if isinstance(output, tuple):
                            return tuple(torch.zeros_like(o) if isinstance(o, torch.Tensor) else o for o in output)
                        return torch.zeros_like(output)
                    elif strat == "random":
                        if isinstance(output, tuple):
                            return tuple(torch.randn_like(o) if isinstance(o, torch.Tensor) else o for o in output)
                        return torch.randn_like(output)
                    return output
                return hook

            handle = module.register_forward_hook(make_ablation_hook(strategy))
            with torch.no_grad():
                ablated_output = wrapper_model(clean_input)
                if hasattr(ablated_output, "logits"):
                    ablated_metric = metric_fn(ablated_output.logits)
                elif hasattr(ablated_output, "last_hidden_state"):
                    ablated_metric = metric_fn(ablated_output.last_hidden_state)
                else:
                    ablated_metric = metric_fn(ablated_output)
            handle.remove()

            ablated_val = float(ablated_metric) if isinstance(ablated_metric, torch.Tensor) else float(ablated_metric)
            impact = abs(baseline_val - ablated_val)
            layer_impacts[node_id] = impact

            # Add edges with impact as weight
            graph.add_edge("input", node_id, weight=impact, edge_type=EdgeType.INFERRED)
            graph.add_edge(node_id, "output", weight=impact, edge_type=EdgeType.INFERRED)

        except (ValueError, AttributeError, RuntimeError):
            # Layer couldn't be hooked — still add edges with zero weight
            graph.add_edge("input", node_id, weight=0.0, edge_type=EdgeType.INFERRED)
            graph.add_edge(node_id, "output", weight=0.0, edge_type=EdgeType.INFERRED)

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

    Args:
        model_name: HuggingFace model name (e.g., "gpt2").
        text: Input text to analyze.
        top_k: Number of top predicted tokens to return per position.
        model: Optional pre-loaded ModelWrapper.

    Returns:
        Dict with keys:
            - "tokens": List of input token strings
            - "logit_lens_data": np.ndarray of shape (n_layers, seq_len)
                with probability of correct next token at each layer
            - "top_k_tokens": List[List[List[str]]] — top-k token predictions
                per layer per position
            - "probabilities": np.ndarray of shape (n_layers, seq_len, vocab_size)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
    hf_model.eval()

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    tokens = [tokenizer.decode(t) for t in input_ids[0]]

    # Forward pass with hidden states
    with torch.no_grad():
        outputs = hf_model(**inputs)

    hidden_states = outputs.hidden_states  # tuple of (batch, seq_len, hidden)
    lm_head = hf_model.lm_head if hasattr(hf_model, "lm_head") else None

    if lm_head is None:
        raise ValueError(f"Model {model_name} does not have an lm_head for logit lens")

    n_layers = len(hidden_states) - 1  # exclude embedding layer
    seq_len = input_ids.shape[1]

    logit_lens_data = np.zeros((n_layers, seq_len))
    top_k_tokens_all = []
    all_probs = []

    for layer_idx in range(1, len(hidden_states)):  # skip embedding layer
        hidden = hidden_states[layer_idx]

        with torch.no_grad():
            logits = lm_head(hidden)  # (batch, seq_len, vocab_size)
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
