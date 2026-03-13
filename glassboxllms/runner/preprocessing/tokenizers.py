"""
This module provides functions for tokenizing text data using various tokenizers
and applying different tokenization strategies. These are designed to be callable
from an experiment's code as needed.
"""

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
        if isinstance(examples[text_column], str):
            examples[text_column] = [examples[text_column]]

        tokenized = tokenizer(
            examples[text_column],
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


def tokenize_with_truncation(
    dataset,
    tokenizer: PreTrainedTokenizer,
    text_column: str = "text",
    max_length: int = 512,
    stride: int = 0,
    truncation_strategy: str = "longest_first",
) -> Any:
    """
    Tokenizes dataset with specific truncation settings.

    Args:
        dataset: The HuggingFace dataset to tokenize
        tokenizer: The tokenizer to use
        text_column: Name of the text column
        max_length: Maximum sequence length
        stride: Number of overlapping tokens between truncated sequences
        truncation_strategy: Strategy for truncation ('longest_first', 'only_first', 'only_second')

    Returns:
        Tokenized dataset with truncation applied
    """

    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            max_length=max_length,
            stride=stride,
            truncation=truncation_strategy,
            return_overflowing_tokens=True,
        )

    return dataset.map(tokenize_function, batched=True)


def tokenize_with_special_tokens(
    dataset,
    tokenizer: PreTrainedTokenizer,
    text_column: str = "text",
    special_tokens: Optional[Dict[str, Any]] = None,
    max_length: int = 512,
) -> Any:
    """
    Tokenizes dataset with custom special tokens.

    Args:
        dataset: The HuggingFace dataset to tokenize
        tokenizer: The tokenizer to use
        text_column: Name of the text column
        special_tokens: Dictionary of special tokens to add (e.g., {"bos_token": "<bos>"})
        max_length: Maximum sequence length

    Returns:
        Tokenized dataset with special tokens applied
    """
    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)

    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

    return dataset.map(tokenize_function, batched=True)


def tokenize_multimodal(
    dataset,
    tokenizer: PreTrainedTokenizer,
    text_columns: List[str],
    max_length: int = 512,
    separator: str = " ",
) -> Any:
    """
    Tokenizes dataset with multiple text columns (e.g., question and answer).

    Args:
        dataset: The HuggingFace dataset to tokenize
        tokenizer: The tokenizer to use
        text_columns: List of column names to tokenize
        max_length: Maximum sequence length
        separator: Separator to use when combining multiple text columns

    Returns:
        Tokenized dataset with combined text from multiple columns
    """

    def tokenize_function(examples):
        # Combine multiple text columns
        combined_text = []
        for i in range(len(examples[text_columns[0]])):
            texts = [examples[col][i] for col in text_columns if examples[col][i]]
            combined_text.append(separator.join(texts))

        return tokenizer(
            combined_text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

    return dataset.map(tokenize_function, batched=True)


def tokenize_with_encoding(
    dataset,
    tokenizer: PreTrainedTokenizer,
    text_column: str = "text",
    encoding: str = "utf-8",
    max_length: int = 512,
) -> Any:
    """
    Tokenizes dataset with specific encoding handling.

    Args:
        dataset: The HuggingFace dataset to tokenize
        tokenizer: The tokenizer to use
        text_column: Name of the text column
        encoding: Text encoding to use
        max_length: Maximum sequence length

    Returns:
        Tokenized dataset with encoding applied
    """

    def tokenize_function(examples):
        # Ensure text is properly encoded
        encoded_texts = []
        for text in examples[text_column]:
            try:
                # Try to decode if it's bytes
                if isinstance(text, bytes):
                    text = text.decode(encoding)
                encoded_texts.append(text)
            except (UnicodeDecodeError, AttributeError):
                encoded_texts.append(str(text))

        return tokenizer(
            encoded_texts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

    return dataset.map(tokenize_function, batched=True)


def batch_tokenize(
    dataset,
    tokenizer: PreTrainedTokenizer,
    text_column: str = "text",
    batch_size: int = 32,
    max_length: int = 512,
    num_proc: int = 4,
) -> Any:
    """
    Tokenizes dataset with multiprocessing for faster processing.

    Args:
        dataset: The HuggingFace dataset to tokenize
        tokenizer: The tokenizer to use
        text_column: Name of the text column
        batch_size: Batch size for tokenization
        max_length: Maximum sequence length
        num_proc: Number of processes to use

    Returns:
        Tokenized dataset with multiprocessing applied
    """

    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

    return dataset.map(
        tokenize_function,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
    )


def tokenize_and_align_labels(
    dataset,
    tokenizer: PreTrainedTokenizer,
    text_column: str = "text",
    labels_column: str = "labels",
    max_length: int = 512,
    padding: str = "max_length",
) -> Any:
    """
    Tokenizes dataset and aligns labels with tokenized sequences.

    Useful for token classification tasks where labels need to be
    aligned with tokens after tokenization.

    Args:
        dataset: The HuggingFace dataset to tokenize
        tokenizer: The tokenizer to use
        text_column: Name of the text column
        labels_column: Name of the labels column
        max_length: Maximum sequence length
        padding: Padding strategy

    Returns:
        Tokenized dataset with aligned labels
    """

    def tokenize_and_align_labels_function(examples) -> Any:
        tokenized: Any = tokenizer(
            examples[text_column],
            max_length=max_length,
            truncation=True,
            padding=padding,
        )

        # Align labels with tokens (assuming 1:1 mapping or special handling)
        # This is a simplified version - actual alignment may vary by task
        labels = []
        input_ids = tokenized["input_ids"]
        for i, label in enumerate(examples[labels_column]):
            # Simple case: one label per token
            if isinstance(label, list):
                aligned_label = label[: len(input_ids[i])]
                # Pad labels if needed
                aligned_label = aligned_label + [-100] * (
                    max_length - len(aligned_label)
                )
            else:
                # Single label for entire sequence
                aligned_label = [label] + [-100] * (max_length - 1)

            labels.append(aligned_label)

        tokenized["labels"] = labels
        return tokenized

    return dataset.map(tokenize_and_align_labels_function, batched=True)
