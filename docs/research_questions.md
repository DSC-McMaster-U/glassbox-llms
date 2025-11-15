### On the Biology of a Large Language Model (Anthropic)

Anthropic's Transformer Circuits publication discusses a multitude of things that can be found when probing an LLM's thought process. They accomplish this through attribution graphs, which in turn are created using a replacement model, a model with more interpretable components than that of a regular LLM.

Their research investigates many subjects such as jailbreaks, refusals, hidden thinking and processing, and multilingual recognition. However, their paper is focused on academic applications, and while they note that their work may be applied for production purposes, it is made clear that the work completed is merely theoretical and luck-based (§ 15.2).

Therefore, looking at this specific paper, we can recognize that while their novel replacement model-attribution graph method has found success, the method can be inaccurate and thus is not suitable for real-world usecases.

This paper also outlines its limitations: one of the most interesting ones is that the lack of activity was not considered (§ 14.1), such as a model not performing an expected logical connection. This usecase is important for production, such as failing to cohere to RAG-provided data.

Another interesting limitation is that the method does not exactly capture the attention patterns generated, which is a problem, as it prevents study of the complexities of how the attention layers take in previous context. This is also important. For example, it is not useful to know that a model answered a question because it was simply the most popular choice chosen by an attribution feature for the concept of "answer a question on this topic". Anthropic suggests probing the query-key pairs, which are weighted, suggesting that it is possible to probe activations based on this. However, doing this naively has quadratic complexity, and may not tell the whole story, as multiple attention heads activate at once. This is important since without an understanding of how attention affects everything, we can only understand a small portion of the story, affecting every possible usecase this project could have in the real world.

Additionally, they have [released](https://www.anthropic.com/research/open-source-circuit-tracing) the library used to perform this research, which may be of interest.

### Other Works

The paper [Mechanistic Interpretability for AI Safety](https://alphaxiv.org/pdf/2404.14082v2) outlines other ways interpretability has been conducted, though they are a bit older.
- Diagnostic Classifiers
  - The term 'probing' refers to testing for specific activations, to try and detect properties and concepts when they appear. This helps find where concepts are stored and when they are activated. This method fails when it comes to detecting usage, and by how much: some concepts may be activated often but not used, or used in some sort of superimposed fashion.
- Circuit Analysis
  - This method involves breaking down a larger model into only the neurons and weights that implement a certain defined case, then studying how those neurons activate and interact. This is one of the more promising methods available in mechanical interpretability since it works pretty well.
- Sparse Autoencoders
  - This is a method that seems to automate the breaking down of a model into its learned features, whereas most other methods required a lot of manual work and trial and error.

Additionally, it should be mentioned how interpretability and its current limitations affect real-world usecases.

- Biases
  - If we can investigate how exactly an LLM processes tokens and how attention can positively or negatively weight certain features, we can get a clearer understanding of biases present in LLMs. This could be experimentally tested by varying subjective bias in datasets.
- Precise Outputs
  - Some workloads, like medical usecases, are mission-critical: error cannot be permitted. These workloads may also not have feasible ways to verify or confirm LLM output, meaning being able to probe exactly how an LLM reached a certain conclusion is valuable. Current research lacks the precision or generalizability to track 'thought' processes that closely.
- Failure Analysis
  - Models can fail for infinitely many reasons, from training to inference. If we can manage to find the specific neurons or layers that cause malaligned associations, we can more efficiently train and fix these errors. While current research has found some ways to ['edit' out](https://www.alphaxiv.org/abs/2412.02104v1) failed weights, a more reliable and consistent method is required.

## Questions

**To what extent can lack of activations be studied, and can they be connected to interpretable output?**

**How can we probe attention's effects?**

**What are feasible ways to probe a production-suitable LLM with minimal error?**
