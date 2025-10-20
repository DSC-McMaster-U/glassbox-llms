## How does an LLM work?

An input string needs to be converted into a numerical representation. To do this, we **tokenize** the string using a **tokenizer** into a set of numbers representing each token. Each token has a numerical representation called an **embedding**, a high-dimensional vector.

Semantic similarity of words can be quantified by vectorial closeness of embedding. The model's **self-attention mechanism** performs an operation to weigh the relative importance of tokens in a sequence. **Positional embeddings** must be used, since unlike RNNs or CNNs, Transformers models do not have inherent 'knowledge' of a token's position: these are added to the embeddings. **Attention scores** are computed using $(Q,K,V)$ values, measures of how much each token affects others, and fed through **feed-forward networks**, neural networks that process the attention scores. Specifically, FFNs take the scores, do a linear transform based on the model's weights and biases, some sort of activation function (which clamps values to a specific range/form, while introducing nonlinearity), and linear transforms it again.

This process is repeated $n$ times, where $n$ is the number of attention/FFN layers. Layering helps provide better depth in abstraction, as well as better approximation to the complexities of writing (which might not always be linear).

Attention is done in parallel using a **multi-head attention** scheme, which is why Transformers models are so fast and scalable compared to previous architectures such as RNNs. In particular, the **self-attention** means all positions in a sequence can be connected in a constant number of operations, where recurrent/convolutional networks scale proportionally with sequence length, and may even require layering to cover a sequence fully.

Once it passes through all the stacks of MHA and FFNs, last processing includes a linearization and **softmax** to make sure the probabilities sum to 1, resulting in a set of output logits.
## The Blackbox

Since embeddings are designed to be a 'computer-translation' of human text, they difficult for  humans to decipher. Not only are dimensions unlabeled (and even if they were, the labels probably would not clear), the connections between meaning and embedding are nonlinear, and innumerable dimensions may have effect on any given characteristic. This problem is compounded the more layers there are, and after $n\approx>10$ layers, it can get practically intractable.

Additionally, all the nonlinearity added by activation functions can cause outputs that have no or very complex connection to its inputs, even if the activation function itself is mathematically defined via a formula or algorithm.

Finally, the model learns patterns from huge amounts of data. This leads to emergent behavior and connections that are not apparent to humans or visible by direct observation.