# TODO: Officially implement other tokenization functions

from typing import Any, Dict, List, Optional, Union

from transformers import PreTrainedTokenizer


def tokenize_dataset(
    dataset,
    tokenizer: PreTrainedTokenizer,
    text_column: str = "text",
    max_length: int = 512,
    truncation: bool = True,
    padding: Union[bool, str] = "max_length",
    add_special_tokens: bool = True,
    return_tensors: str = "pt",
    return_attention_mask: bool = True,
    return_token_type_ids: bool = True,
) -> Any:
    """
    Tokenizes a HuggingFace dataset with a specific tokenizer.

    Args:
        dataset: The HuggingFace dataset to tokenize
        tokenizer: The tokenizer to use for tokenization
        text_column: Name of the column containing text to tokenize
        max_length: Maximum length of the tokenized sequence
        truncation: Whether to truncate sequences longer than max_length
        padding: Padding strategy ('longest', 'max_length', False)
        add_special_tokens: Whether to add special tokens (CLS, SEP, etc.)
        return_tensors: Type of tensors to return ('pt', 'tf', 'np', 'jax', None)
        return_attention_mask: Whether to include attention mask
        return_token_type_ids: Whether to include token type IDs (for models like BERT)

    Returns:
        The tokenized dataset with added columns for token_ids, attention_mask, etc.
    """

    def tokenize_function(examples):
        # works for both string and list of strings
        texts = examples[text_column]
        if isinstance(texts, str):
            texts = [texts]

        # Filter out None and empty strings to prevent tokenizer errors
        texts = [str(t) if t is not None else "" for t in texts]
        texts = [t if t.strip() else " " for t in texts]  # Replace empty/whitespace-only with space

        tokenized = tokenizer(
            texts,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            add_special_tokens=add_special_tokens,
            return_tensors=None,  # map() will handle the batching
        )

        result = {"input_ids": tokenized["input_ids"]}

        if return_attention_mask:
            result["attention_mask"] = tokenized.get("attention_mask", [1] * max_length)

        if return_token_type_ids and "token_type_ids" in tokenized:
            result["token_type_ids"] = tokenized["token_type_ids"]

        return result

    return dataset.map(tokenize_function, batched=True)
