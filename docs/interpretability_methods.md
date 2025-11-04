# Interpretability Methods — Quick Reference

## Overview
“Interpretability methods” are techniques used to understand and explain how ML models — especially deep neural networks — arrive at their predictions.

---

## 1. Feature Attribution Methods
**Goal:** Identify which input features most influence the model’s output.

**Examples:**
- **Saliency maps / Gradients** — show sensitivity of the output to each input feature.  
- **Integrated Gradients** — accumulate gradients from a baseline input to the actual input.  
- **SHAP / LIME** — approximate the contribution of features by perturbing/removing them.

**Use:** Explains what parts of the input drive decisions (e.g., which words made a sentiment model predict “positive”).

---

## 2. Probing
**Goal:** Test what information is represented inside hidden layers/neurons.

**How it works:** Train a simple probe on model activations to predict a known label (e.g., POS tags). Strong probe performance suggests that information is encoded in that layer.

**Use:** Identifies where in the model specific linguistic or factual knowledge is stored.

---

## 3. Activation Patching (Circuit Analysis)
**Goal:** Understand causal relationships between internal components (neurons, attention heads, layers).

**How it works:** Run the model twice—once normally and once with some internal activations replaced (“patched”) with activations from another input—and observe output changes.

**Use:** Reveals which parts of the network cause specific behaviors (e.g., an attention head tracking indirect objects).

---

## 4. Toy Models & Mechanistic Interpretability
**Goal:** Build smaller, simplified models or focus on tiny subsystems that can be reverse-engineered.

**Examples:** Reverse-engineering small transformers (e.g., induction heads, IOI tasks), analyzing weights/attention patterns to find circuits.

**Use:** Provides mathematical insight into how computations emerge from parameters.

---

## 5. Concept Activation & Representation Methods
**Goal:** Connect high-level human concepts to internal representations.

**Examples:**
- **Network Dissection** — align neurons/units with interpretable concepts.  
- **CAVs (Concept Activation Vectors)** — measure alignment of internal representations with a user-defined concept (e.g., “striped,” “democracy”).

**Use:** Bridges human-understandable ideas and abstract neural features.

---

# Open Challenges in Interpretability

## White box interpretation - Decomposability
With a traditional black box model, we can only analyze inputs and outputs, leading to being unable to detect potentially harmful cognition. This harmful cognition includes but is not limited to; intentionally underperforming on evaluations, "sandbagging”, as well as giving dishonest or misleading responses to conform to a user’s biases or beliefs. However “White-box evaluation methods could serve as tools to detect potential biases that arise when models learn to use spurious correlations” (Sharkey et al.). Performing these evaluations however are not feasible today, at the least not on the proper scale. This is because we lack one key component, that being decomposability. We need to be able to see and manage what these models are doing, individual nodes, node groups, etc. This is an open problem because, while white-box models do exist, they are decomposable due to their simplicity, whereas black box models are efficient and accurate due to their complexity. Think of trying to map the brain of a fruit fly vs the brain of a human, there is a significant jump in complexity of understanding how their neurons connect, which is directly correlated to the ability to solve complex problems. One of the biggest problems of the white box approach is explainability vs efficiency.


# Faithfulness vs Plausability
An AI model can produce a series of steps to reproduce an outcome. For an example of what this means, if you ask an AI, “What is 3 + 5” it will say 8. If you ask how it got that answer, it will break down the problem into steps. This chain of thought, or CoT is what is called a plausible explanation. It makes sense to us and seems to indicate this is how the AI is functioning when given this problem. “It’s important to note that, while modern LLMs can generate plausible reasonings that are convincing to humans, they do not inherently understand truth or factual accuracy.” (Agarwal et al.) The problem of faithfulness then is, is the plausible explanation an accurate portrayal of the underlying processes of these models? Evaluating faithfulness is no easy task as there are no ground truth explanations (correct and verifiable data) for self explanatory AI answers. To alleviate this problem, studies have used input-output to estimate faithfulness, which, due to the black-box and proprietary characteristics of most modern AI models, is unreliable. “Overall, self-explanations lack faithfulness guarantees and currently, there are no universally agreed-upon metrics to quantify the faithfulness of self-explanations, and consensus on the notion of faithfulness remains elusive. Furthermore, the community must pursue avenues to enhance the faithfulness of self-explanations before their widespread adoption as ‘plausible yet unfaithful explanations foster a misplaced sense of trustworthiness in LLMs’.” (Agarwal et al.)


# References

Olah, C., et al. (2022). Mechanistic Interpretability, Variables, and the Importance of Interpretable Bases. Transformer Circuits.
https://www.transformer-circuits.pub/2022/mech-interp-essay

Anthropic Research Team (2021-2023). Transformer Circuits Series. Anthropic Interpretability Research.
https://transformer-circuits.pub

OpenAI (2023). Interpretability Overview. OpenAI Research Blog.
https://openai.com/research

Garbin, C. (2022). Machine Learning Interpretability with Feature Attribution. CGarbin Blog.
https://cgarbin.github.io/machine-learning-interpretability-feature-attribution

Nanda, N., et al. (2025). Open Problems in Mechanistic Interpretability. arXiv preprint arXiv:2501.16496.
https://arxiv.org/abs/2501.16496

Two Sigma Data Science Team (2021). Interpretability Methods in Machine Learning: A Brief Survey.
https://www.twosigma.com/articles/interpretability-methods-in-machine-learning-a-brief-survey

Sharkey, Lee, et al. "Open Problems in Mechanistic Interpretability." arXiv, 2025, arXiv:2501.16496. https://arxiv.org/html/2501.16496v1 
Agarwal, Chirag, Sree Harsha Tanneru, and Himabindu Lakkaraju. "Faithfulness vs. Plausibility: On the 

(Un)Reliability of Explanations from Large Language Models." arXiv, 2024, arXiv:2402.04614. https://arxiv.org/html/2402.04614v1 
# Interpretability Methods — Quick Reference

# Overview
“Interpretability methods” are techniques used to understand and explain how ML models — especially deep neural networks — arrive at their predictions.

---

# 1. Feature Attribution Methods
**Goal:** Identify which input features most influence the model’s output.

**Examples:**
- **Saliency maps / Gradients** — show sensitivity of the output to each input feature.  
- **Integrated Gradients** — accumulate gradients from a baseline input to the actual input.  
- **SHAP / LIME** — approximate the contribution of features by perturbing/removing them.

**Use:** Explains what parts of the input drive decisions (e.g., which words made a sentiment model predict “positive”).

---

# 2. Probing
**Goal:** Test what information is represented inside hidden layers/neurons.

**How it works:** Train a simple probe on model activations to predict a known label (e.g., POS tags). Strong probe performance suggests that information is encoded in that layer.

**Use:** Identifies where in the model specific linguistic or factual knowledge is stored.

---

# 3. Activation Patching (Circuit Analysis)
**Goal:** Understand causal relationships between internal components (neurons, attention heads, layers).

**How it works:** Run the model twice—once normally and once with some internal activations replaced (“patched”) with activations from another input—and observe output changes.

**Use:** Reveals which parts of the network cause specific behaviors (e.g., an attention head tracking indirect objects).

---

# 4. Toy Models & Mechanistic Interpretability
**Goal:** Build smaller, simplified models or focus on tiny subsystems that can be reverse-engineered.

**Examples:** Reverse-engineering small transformers (e.g., induction heads, IOI tasks), analyzing weights/attention patterns to find circuits.

**Use:** Provides mathematical insight into how computations emerge from parameters.

---

# 5. Concept Activation & Representation Methods
**Goal:** Connect high-level human concepts to internal representations.

**Examples:**
- **Network Dissection** — align neurons/units with interpretable concepts.  
- **CAVs (Concept Activation Vectors)** — measure alignment of internal representations with a user-defined concept (e.g., “striped,” “democracy”).

**Use:** Bridges human-understandable ideas and abstract neural features.

---

# Open Challenges in Interpretability

# White box interpretation - Decomposability
With a traditional black box model, we can only analyze inputs and outputs, leading to being unable to detect potentially harmful cognition. This harmful cognition includes but is not limited to; intentionally underperforming on evaluations, "sandbagging”, as well as giving dishonest or misleading responses to conform to a user’s biases or beliefs. However “White-box evaluation methods could serve as tools to detect potential biases that arise when models learn to use spurious correlations” (Sharkey et al.). Performing these evaluations however are not feasible today, at the least not on the proper scale. This is because we lack one key component, that being decomposability. We need to be able to see and manage what these models are doing, individual nodes, node groups, etc. This is an open problem because, while white-box models do exist, they are decomposable due to their simplicity, whereas black box models are efficient and accurate due to their complexity. Think of trying to map the brain of a fruit fly vs the brain of a human, there is a significant jump in complexity of understanding how their neurons connect, which is directly correlated to the ability to solve complex problems. One of the biggest problems of the white box approach is explainability vs efficiency.


# Faithfulness vs Plausability
An AI model can produce a series of steps to reproduce an outcome. For an example of what this means, if you ask an AI, “What is 3 + 5” it will say 8. If you ask how it got that answer, it will break down the problem into steps. This chain of thought, or CoT is what is called a plausible explanation. It makes sense to us and seems to indicate this is how the AI is functioning when given this problem. “It’s important to note that, while modern LLMs can generate plausible reasonings that are convincing to humans, they do not inherently understand truth or factual accuracy.” (Agarwal et al.) The problem of faithfulness then is, is the plausible explanation an accurate portrayal of the underlying processes of these models? Evaluating faithfulness is no easy task as there are no ground truth explanations (correct and verifiable data) for self explanatory AI answers. To alleviate this problem, studies have used input-output to estimate faithfulness, which, due to the black-box and proprietary characteristics of most modern AI models, is unreliable. “Overall, self-explanations lack faithfulness guarantees and currently, there are no universally agreed-upon metrics to quantify the faithfulness of self-explanations, and consensus on the notion of faithfulness remains elusive. Furthermore, the community must pursue avenues to enhance the faithfulness of self-explanations before their widespread adoption as ‘plausible yet unfaithful explanations foster a misplaced sense of trustworthiness in LLMs’.” (Agarwal et al.)


# References

Olah, C., et al. (2022). Mechanistic Interpretability, Variables, and the Importance of Interpretable Bases. Transformer Circuits.
https://www.transformer-circuits.pub/2022/mech-interp-essay

Anthropic Research Team (2021-2023). Transformer Circuits Series. Anthropic Interpretability Research.
https://transformer-circuits.pub

OpenAI (2023). Interpretability Overview. OpenAI Research Blog.
https://openai.com/research

Garbin, C. (2022). Machine Learning Interpretability with Feature Attribution. CGarbin Blog.
https://cgarbin.github.io/machine-learning-interpretability-feature-attribution

Nanda, N., et al. (2025). Open Problems in Mechanistic Interpretability. arXiv preprint arXiv:2501.16496.
https://arxiv.org/abs/2501.16496

Two Sigma Data Science Team (2021). Interpretability Methods in Machine Learning: A Brief Survey.
https://www.twosigma.com/articles/interpretability-methods-in-machine-learning-a-brief-survey

Sharkey, Lee, et al. "Open Problems in Mechanistic Interpretability." arXiv, 2025, arXiv:2501.16496. https://arxiv.org/html/2501.16496v1 
Agarwal, Chirag, Sree Harsha Tanneru, and Himabindu Lakkaraju. "Faithfulness vs. Plausibility: On the 

(Un)Reliability of Explanations from Large Language Models." arXiv, 2024, arXiv:2402.04614. https://arxiv.org/html/2402.04614v1 