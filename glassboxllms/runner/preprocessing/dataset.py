"""
This module provides modular preprocessing functions that transform datasets according to their specific requirements.
The actual processing happens in core.py
"""

import re
from typing import Any, Callable, Dict, List, Optional, Union

from .custom import *


def select_columns(
    dataset,
    columns: List[str],
) -> Any:
    """
    Selects specific columns from a dataset.

    Args:
        dataset: The HuggingFace dataset to select columns from
        columns: List of column names to keep

    Returns:
        Dataset with only the selected columns
    """
    return dataset.select_columns(columns)


def rename_columns(
    dataset,
    column_mapping: Dict[str, str],
) -> Any:
    """
    Renames columns in a dataset.

    Args:
        dataset: The HuggingFace dataset to rename columns in
        column_mapping: Dictionary mapping old column names to new column names

    Returns:
        Dataset with renamed columns
    """

    def rename_function(examples):
        for old_name, new_name in column_mapping.items():
            if old_name in examples:
                examples[new_name] = examples.pop(old_name)
        return examples

    return dataset.map(rename_function)


def sample_dataset(
    dataset,
    num_samples: int,
    seed: Optional[int] = None,
    shuffle: bool = True,
) -> Any:
    """
    Samples a subset of the dataset.

    Args:
        dataset: The HuggingFace dataset to sample from
        num_samples: Number of samples to take
        seed: Random seed for reproducibility
        shuffle: Whether to shuffle before sampling

    Returns:
        Sampled dataset
    """
    if shuffle:
        if seed is not None:
            dataset = dataset.shuffle(seed=seed)
        else:
            dataset = dataset.shuffle()

    return dataset.select(range(min(num_samples, len(dataset))))


def clean_text(
    dataset,
    text_column: str = "text",
    lowercase: bool = True,
    remove_special_chars: bool = True,
    normalize_whitespace: bool = True,
) -> Any:
    """
    Cleans text in a dataset column.

    Args:
        dataset: The HuggingFace dataset to clean
        text_column: Name of the column containing text to clean
        lowercase: Whether to convert text to lowercase
        remove_special_chars: Whether to remove special characters
        normalize_whitespace: Whether to normalize whitespace

    Returns:
        Dataset with cleaned text
    """

    def clean_function(examples):
        cleaned_texts = []
        for text in examples[text_column]:
            if not isinstance(text, str):
                text = str(text)

            if lowercase:
                text = text.lower()

            if remove_special_chars:
                text = remove_special_characters(text)

            if normalize_whitespace:
                text = re.sub(r"\s+", " ", text).strip()

            cleaned_texts.append(text)

        examples[text_column] = cleaned_texts
        return examples

    return dataset.map(clean_function)


def lowercase_text(
    dataset,
    text_column: str = "text",
) -> Any:
    """
    Converts text to lowercase.

    Args:
        dataset: The HuggingFace dataset to process
        text_column: Name of the column containing text to convert

    Returns:
        Dataset with lowercase text
    """

    def lowercase_function(examples):
        examples[text_column] = [
            str(text).lower() if text is not None else None
            for text in examples[text_column]
        ]
        return examples

    return dataset.map(lowercase_function)


def remove_special_characters(
    text: str,
    keep_punctuation: bool = False,
    keep_numbers: bool = True,
) -> str:
    """
    Removes special characters from text.

    Args:
        text: The text to clean
        keep_punctuation: Whether to keep punctuation marks
        keep_numbers: Whether to keep numeric characters

    Returns:
        Cleaned text with special characters removed
    """
    if not isinstance(text, str):
        return ""

    if keep_punctuation:
        # Keep letters, numbers, spaces, and common punctuation
        pattern = r"[^\w\s.,!?;:'\"()-]"
    else:
        # Keep only letters and spaces
        pattern = r"[^\w\s]" if keep_numbers else r"[^\s]"

    cleaned = re.sub(pattern, "", text)

    if not keep_numbers:
        cleaned = re.sub(r"\d+", "", cleaned)

    return cleaned.strip()


def max_length(
    dataset,
    text_column: str = "text",
    max_length: int = 512,
    truncation_column: Optional[str] = None,
) -> Any:
    """
    Truncates text to maximum length.

    Args:
        dataset: The HuggingFace dataset to process
        text_column: Name of the column containing text to truncate
        max_length: Maximum length of text
        truncation_column: Optional column name to store original length before truncation

    Returns:
        Dataset with truncated text
    """

    def truncate_function(examples):
        truncated_texts = []
        lengths = []

        for text in examples[text_column]:
            if not isinstance(text, str):
                text = str(text) if text is not None else ""

            original_length = len(text)
            lengths.append(original_length)

            if len(text) > max_length:
                truncated_texts.append(text[:max_length])
            else:
                truncated_texts.append(text)

        examples[text_column] = truncated_texts

        if truncation_column:
            examples[truncation_column] = lengths

        return examples

    return dataset.map(truncate_function)


def min_length(
    dataset,
    text_column: str = "text",
    min_length: int = 10,
    drop_short: bool = True,
) -> Any:
    """
    Filters or pads text to minimum length.

    Args:
        dataset: The HuggingFace dataset to process
        text_column: Name of the column containing text to process
        min_length: Minimum length of text
        drop_short: Whether to drop samples shorter than min_length

    Returns:
        Dataset with filtered/padded text
    """

    def min_length_function(examples):
        valid_indices = []

        for i, text in enumerate(examples[text_column]):
            if not isinstance(text, str):
                text = str(text) if text is not None else ""

            if len(text) >= min_length:
                valid_indices.append(i)

        if drop_short:
            if len(valid_indices) < len(examples[text_column]):
                return dataset.select(valid_indices)
            return dataset
        else:
            # Pad short texts
            padded_texts = []
            for text in examples[text_column]:
                if not isinstance(text, str):
                    text = str(text) if text is not None else ""

                if len(text) < min_length:
                    padded_texts.append(text + " " * (min_length - len(text)))
                else:
                    padded_texts.append(text)

            examples[text_column] = padded_texts
            return examples

    return min_length_function(dataset)


from typing import Any, Callable, Iterable, List


def apply_transforms(
    dataset,
    transform_fns: Iterable[Callable[[str], str]],
    text_column: str = "text",
) -> Any:
    """
    Applies a sequence of transformation functions to text in daisy-chain order.

    Args:
        dataset: The HuggingFace dataset to transform
        transform_fns: A list/iterable of functions to apply in order
        text_column: Name of the column containing text to transform

    Returns:
        Dataset with transformed text
    """
    # Convert to list to facilitate iteration
    fns = list(transform_fns)

    def transform_function(examples):
        transformed_texts = []

        for text in examples[text_column]:
            if text is None:
                transformed_texts.append(None)
                continue

            # apply functions in sequence
            val = str(text)
            for fn in fns:
                val = fn(val)
            transformed_texts.append(val)

        examples[text_column] = transformed_texts
        return examples

    return dataset.map(transform_function)


def normalize_text(
    dataset,
    text_column: str = "text",
    strip_whitespace: bool = True,
    normalize_spaces: bool = True,
    remove_accents: bool = True,
) -> Any:
    """
    Normalizes text in a dataset column.

    Args:
        dataset: The HuggingFace dataset to normalize
        text_column: Name of the column containing text to normalize
        strip_whitespace: Whether to strip leading/trailing whitespace
        normalize_spaces: Whether to normalize multiple spaces to single space
        remove_accents: Whether to remove accent marks from characters

    Returns:
        Dataset with normalized text
    """

    def normalize_function(examples):
        normalized_texts = []

        for text in examples[text_column]:
            if not isinstance(text, str):
                text = str(text) if text is not None else ""

            if strip_whitespace:
                text = text.strip()

            if normalize_spaces:
                text = re.sub(r"\s+", " ", text)

            if remove_accents:
                # Remove accents using Unicode normalization
                text = re.sub(r"[^\u0000-\u007F]", "", text)

            normalized_texts.append(text)

        examples[text_column] = normalized_texts
        return examples

    return dataset.map(normalize_function)


def encode_labels(
    dataset,
    label_column: str = "label",
    label_mapping: Optional[Dict[Union[str, int], int]] = None,
) -> Any:
    """
    Encodes label columns to numeric values.

    Args:
        dataset: The HuggingFace dataset to encode
        label_column: Name of the column containing labels
        label_mapping: Optional dictionary mapping labels to numeric values.
                      If not provided, labels will be automatically mapped.

    Returns:
        Dataset with encoded labels
    """
