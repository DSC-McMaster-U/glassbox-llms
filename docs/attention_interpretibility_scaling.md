# How Attention Scales in Long-Document Inputs

This document expands on the ```notebooks/attention_visualization.ipynb``` .

## Background: Self-Attention in Transformers

Transformers process input using self-attention, where each token attends to every other token in the sequence.

Multi-head attention allows different attention heads to focus on different patterns — some on syntax (e.g. verb–object), others on broader semantic or positional information.

This design enables **long-range dependencies**: even if two tokens are far apart in the input, the model can link them directly.

## How Attention Propagates Across Layers

At each layer, a token's representation is updated by aggregating information from the tokens it attends to.

If a token (e.g., a pronoun like "it") attends to a distant token (e.g., "the animal"), that connection is carried forward and refined across deeper layers.

Over many layers, the model can merge and propagate information from across the entire text — enabling it to form a contextualized understanding of references, themes, and arguments.

Attention thus acts as the mechanism that binds together semantically related ideas, even if they appear in different paragraphs.

## Tying Together Distant Concepts

In long texts (paragraphs, articles), attention heads help track:

- **Coreference** (e.g., linking "she" back to "Dr. Smith")
- **Thematic continuation** (e.g., tracing arguments or character development)
- **Causal connections** (e.g., understanding why something happened based on earlier context)

Certain heads become specialized in tracking global information across long distances, while others focus locally.

Some heads can even perform tasks like coreference resolution implicitly, without being explicitly trained to do so.

## How Attention Mechanisms Adapt as Input Length Increases

As language models scale to longer inputs, several adaptations help maintain meaningful attention and context retention:

### 1. Layer-Wise Expansion of Contextual Scope

- In early layers, attention heads often focus on local structure — attending mostly to neighboring tokens to capture grammar or phrase structure
- As we move to deeper layers, attention heads begin to cover broader spans — linking distant tokens across sentences or paragraphs
- This hierarchical processing allows models to build from local syntax up to global document-level coherence

### 2. Specialization of Attention Heads

Research shows that some heads specialize in long-distance tasks:

- **Coreference heads** consistently link pronouns to antecedents far earlier in the text
- **Thematic heads** connect related concepts that reappear across paragraphs

Others act as local pattern recognizers, allowing the model to offload global linking to a smaller subset of heads.

### 3. Implicit Chunking and Bridging

- Even in dense attention models (e.g. BERT, GPT-2), models often implicitly chunk documents into smaller regions of focus
- Specialized heads act as bridges between chunks, linking e.g., an introduction to a conclusion or a definition to its later usage
- This dynamic allows models to reconstruct global understanding from overlapping local windows

### 4. Use of Positional Encoding to Track Distance

- Absolute or relative positional encodings help the model track how far a token is from another
- Even if the content is unrelated, the model learns patterns like: "tokens 500 positions apart are less likely to be related than those 5 apart — unless..."
- Some heads learn to override positional decay and attend far back when certain linguistic cues are present (e.g., named entities, citations)

### 5. Attention Pooling via Summary Tokens

- Some models assign special roles to tokens like `[CLS]` or summary prompts
- Many heads collapse attention into these tokens over time, effectively turning them into information sinks that summarize global context
- This allows later layers to attend to a single representation of earlier context rather than re-scanning the entire sequence

### 6. Causal Mechanisms in Decoder-Only Models

- In decoder-only models (e.g. GPT-2, GPT-4), causal masking restricts attention to prior tokens
- Still, models learn long-term dependencies by encoding prior content into intermediate representations that propagate forward
- They preserve relevance across hundreds or thousands of tokens via repeated internal references and distributed memory

## Challenges of Scaling to Long Inputs

### 1. Computational Cost

Self-attention scales with **O(n²)** in both memory and compute, where n is the number of tokens.

Long documents (hundreds or thousands of tokens) result in enormous attention matrices — quickly becoming computationally expensive to process.

### 2. Memory and Speed Constraints

For each attention layer, storing an n×n matrix for each head becomes infeasible as sequence length increases.

This is why models like GPT-3 and GPT-4 set maximum context lengths (e.g. 2,048–32,000 tokens), and why new architectures (Longformer, BigBird, etc.) use sparse or local attention patterns to handle longer sequences efficiently.

### 3. Visualization Difficulty

Traditional attention heatmaps become unreadable for long texts.

With hundreds of tokens and multiple layers/heads, attention matrices are too large to interpret directly.

Making sense of which parts of a document a model is "focusing on" requires smarter visualization and aggregation methods.

### 4. Model Limitations

Even with high context limits, models cannot inherently reason across multiple documents or ultra-long content without retrieval mechanisms.

Techniques like chunking, retrieval-augmented generation (RAG), or memory tokens are often used to bypass attention window limits — but change how attention operates internally.

## Visualizing Attention at Document Scale

Researchers have developed several techniques to handle larger inputs:

### 1. Interactive Tools (e.g. BertViz, LIT)

These allow users to inspect attention per head, per layer, and select a focus token in long text to trace what it attends to.

Makes it practical to explore attention behavior one token at a time, even on longer paragraphs or pages of input.

### 2. Sentence-Level Aggregation (e.g. SAVIS)

Instead of token-to-token matrices, tools like SAVIS compute inter-sentence attention matrices.

This summarizes attention between sentences: e.g., Sentence 6 attends heavily to Sentence 2.

Greatly reduces dimensionality: for 1,000 tokens, you may only have 40–50 sentences, making the matrix far more interpretable.

### 3. Top-K Attention Filtering

Rather than showing all attention weights, only the top few strongest connections are visualized.

This highlights the most meaningful long-range connections — e.g., a conclusion paragraph strongly linking back to an earlier hypothesis.

### 4. Attention Flow and Rollout

Attention rollout aggregates attention across layers to measure cumulative influence between tokens.

Useful for tracing how information from one part of the input affects another — e.g., how an early sentence contributes to the model's final output.

## Quantifying Attention Behavior on Long Inputs

Beyond visuals, these metrics help analyze how LLMs focus over longer contexts:

### 1. Attention Distance

Measures how far apart tokens are when one attends to the other.

Long-range heads will have higher average distances, signaling they carry global information.

### 2. Entropy (Attention Concentration)

- **Low entropy**: attention is focused on a few tokens
- **High entropy**: attention is spread broadly, possibly gathering overall context

Helps classify whether a head is local vs. global.

### 3. Inter-Sentence Attention Scores

Aggregates attention between sentences.

Identifies key sentence linkages, useful for story structure or argumentative flow.

### 4. Token Influence

Measures how much total attention a specific token (e.g. [CLS], "Einstein") receives from the rest of the sequence.

Helps detect whether a model is summarizing into a token or highlighting an important entity.

### 5. Head Function Metrics

Tracks if a head exhibits certain patterns:

- **Induction heads** copy or extend previous patterns
- **Positional bias heads** always attend to specific positions (e.g. beginning of document)

These can be measured and categorized to understand functional roles.

### 6. Output Impact

Measures how much certain inputs influence the model's final decision/output, based on attention weight flow.

Helpful for explainability in question answering or classification.

## Key Takeaways

- As input length grows, attention mechanisms adapt hierarchically — early layers are local, deeper layers are global
- Despite quadratic cost, LLMs achieve long-range reasoning by head specialization, positional bias, and attention pooling
- Visualization tools like BertViz, SAVIS, and top-k attention filtering allow interpretable exploration of these mechanisms
- Quantitative metrics (distance, entropy, sentence attention) reveal how LLMs distribute focus and preserve coherence
- Understanding attention scaling is crucial for interpreting LLM behavior on real-world inputs like articles, reports, or books

## Sources

- Clark et al. (2019) – What Does BERT Look At? Coreference and syntactic head analysis
- Vig (2019), Tenney et al. (2020) – BertViz and LIT: interactive attention visualization
- Seo et al. (2025) – SAVIS: Sentence-level visualization for long documents
- Tang et al. (2024) – Ltri-LLM: Empirical study of local chunking in long inputs
- Morgan (2023, Comet ML) – Attention linking examples and explainability caveats
- Gary Fan (2023) – Hands-on tutorial using BertViz on long inputs