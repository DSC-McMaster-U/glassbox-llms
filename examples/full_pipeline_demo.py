"""
End-to-end interpretability pipeline demo using GPT-2.

Demonstrates the full flow: model → logit lens → activations → probe →
SAE → circuit discovery → visualization-ready outputs.

Usage:
    python examples/full_pipeline_demo.py

Requires: torch, transformers, scikit-learn
"""

from glassboxllms.pipeline import (
    extract_activations,
    train_sae_on_model,
    train_probe_on_model,
    discover_circuit,
    run_logit_lens,
)

MODEL_NAME = "gpt2"
TARGET_LAYER = "transformer.h.5"


def main():
    # ----------------------------------------------------------------
    # 1. Logit Lens — see what the model predicts at each layer
    # ----------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Logit Lens Analysis")
    print("=" * 60)

    logit_results = run_logit_lens(MODEL_NAME, "The capital of France is", top_k=5)

    print(f"Tokens: {logit_results['tokens']}")
    print(f"Logit lens data shape: {logit_results['logit_lens_data'].shape}")
    print(f"Top predictions at final layer, last token: {logit_results['top_k_tokens'][-1][-1]}")

    # The logit_lens_data can be fed directly to plot_logit_lens:
    #   from glassboxllms.visualization import plot_logit_lens
    #   fig = plot_logit_lens(logit_results["logit_lens_data"], logit_results["tokens"],
    #                         top_k_tokens=logit_results["top_k_tokens"])
    #   fig.savefig("logit_lens.png")

    # ----------------------------------------------------------------
    # 2. Extract Activations — raw hidden states from a layer
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2: Extract Activations")
    print("=" * 60)

    texts = [
        "The cat sat on the mat.",
        "Machine learning is transforming science.",
        "The president gave a speech yesterday.",
    ]

    acts = extract_activations(MODEL_NAME, texts, [TARGET_LAYER], return_type="torch")
    print(f"Activation shape for {TARGET_LAYER}: {acts[TARGET_LAYER].shape}")

    # Also works with numpy output
    acts_np = extract_activations(MODEL_NAME, texts, [TARGET_LAYER], return_type="numpy")
    print(f"Numpy activation shape: {acts_np[TARGET_LAYER].shape}")

    # ----------------------------------------------------------------
    # 3. Train Linear Probe — test if sentiment is encoded
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3: Train Linear Probe")
    print("=" * 60)

    positive_texts = [
        "I love this movie, it was fantastic!",
        "This is wonderful and amazing.",
        "Great job, I'm so happy with the results.",
        "The food was delicious and the service excellent.",
        "What a beautiful day, everything is perfect!",
    ]

    negative_texts = [
        "I hate this, it was terrible.",
        "This is awful and disappointing.",
        "Terrible work, I'm very upset.",
        "The food was disgusting and the service horrible.",
        "What an ugly mess, nothing works right.",
    ]

    probe, direction = train_probe_on_model(
        MODEL_NAME,
        positive_texts,
        negative_texts,
        layer=TARGET_LAYER,
    )
    print(f"Probe fitted: {probe.is_fitted}")
    print(f"Direction vector shape: {direction.shape}")
    print(f"Direction vector norm: {float(sum(direction**2)**0.5):.4f}")

    # ----------------------------------------------------------------
    # 4. Train SAE — extract sparse features from activations
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 4: Train Sparse Autoencoder")
    print("=" * 60)

    training_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Scientists discovered a new species of deep-sea fish.",
        "The stock market reached an all-time high today.",
        "She played a beautiful melody on the piano.",
        "The algorithm processes data in logarithmic time.",
        "Rain is expected throughout the weekend.",
        "The team won the championship in overtime.",
        "Quantum computers can solve certain problems faster.",
    ]

    sae, feature_set = train_sae_on_model(
        MODEL_NAME,
        training_texts,
        layer=TARGET_LAYER,
        feature_dim=512,
        k=32,
        n_epochs=3,
        batch_size=16,
    )
    print(f"\nSAE: {sae}")
    print(f"FeatureSet: {feature_set}")
    print(f"Number of features: {len(feature_set)}")

    # Access individual features
    feature_0 = feature_set[0]
    print(f"Feature 0 decoder vector shape: {feature_0.decoder_vector.shape}")

    # ----------------------------------------------------------------
    # 5. Discover Circuit — find important components
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 5: Circuit Discovery")
    print("=" * 60)

    def simple_metric(output):
        """Measure the norm of the output as a simple metric."""
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state[0, -1].norm()
        return output[0, -1].norm()

    # Use a subset of layers for demo speed
    demo_layers = [f"transformer.h.{i}" for i in range(6)]

    graph = discover_circuit(
        MODEL_NAME,
        text="The capital of France is",
        metric_fn=simple_metric,
        layers=demo_layers,
        strategy="zero",
    )
    print(f"Circuit graph: {graph}")
    print(f"Graph summary: {graph.summary()}")

    # The graph can be fed to visualization:
    #   from glassboxllms.visualization import plot_circuit_graph
    #   fig = plot_circuit_graph(graph)
    #   fig.savefig("circuit.png")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print("All pipeline functions work end-to-end.")
    print("Results can be passed directly to visualization functions:")
    print("  - plot_logit_lens(logit_results['logit_lens_data'], ...)")
    print("  - plot_circuit_graph(graph)")
    print("  - plot_sae_training_curves(stats)")
    print("  - plot_probe_accuracy(probe_result)")


if __name__ == "__main__":
    main()
