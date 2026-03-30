import logging
from typing import Any

from .dataset import (
    apply_transforms,
    clean_text,
    max_length,
    min_length,
    normalize_text,
    rename_columns,
    sample_dataset,
    select_columns,
)
from .tokenizers import tokenize_dataset


def start_preprocess(dataset, cfg) -> Any:
    """Apply preprocessing transforms based on config conditions."""
    preprocess_config = cfg.dataset.preprocess
    logging.info(f"Applying preprocessing: {list(preprocess_config.keys())}")

    if "rename" in preprocess_config:
        # this one is high priority
        namedict = preprocess_config.get("rename", None)
        dataset = rename_columns(dataset, column_mapping=namedict)
        logging.info(f"Renamed columns with mapping: {namedict}")

    if "apply_transform" in preprocess_config:
        transform = preprocess_config.get("apply_transform")
        transform_columns = transform.get("text_column")
        transform_funcs = transform.get("transform_functions")
        dataset = apply_transforms(dataset, transform_funcs, transform_columns)
        logging.info(f"Ran custom transform(s): {transform_funcs}")

    # Apply sample_dataset if num_samples is specified
    if "num_samples" in preprocess_config:
        num_samples_cfg = preprocess_config.get("num_samples")
        num_samples = num_samples_cfg.get("count", 1000)
        seed = num_samples_cfg.get("seed", None)
        shuffle = num_samples_cfg.get("shuffle", True)
        dataset = sample_dataset(
            dataset, num_samples=num_samples, seed=seed, shuffle=shuffle
        )
        logging.info(f"Sampled {num_samples} samples from dataset")

    # apply select_columns if columns are specified
    if "columns" in preprocess_config:
        columns = preprocess_config.get("columns")
        dataset = select_columns(dataset, columns=columns)
        logging.info(f"Selected columns: {columns}")

    # apply all text cleaning/normalization to raw text strings
    if "clean_text" in preprocess_config:
        clean_config = preprocess_config.get("clean_text")
        text_column = clean_config.get("text_column", "text")
        lowercase = clean_config.get("lowercase", True)
        remove_special_chars = clean_config.get("remove_special_chars", True)
        normalize_whitespace = clean_config.get("normalize_whitespace", True)
        dataset = clean_text(
            dataset,
            text_column=text_column,
            lowercase=lowercase,
            remove_special_chars=remove_special_chars,
            normalize_whitespace=normalize_whitespace,
        )
        logging.info(f"Applied text cleaning to column '{text_column}'")

    if "normalize_text" in preprocess_config:
        norm_config = preprocess_config.get("normalize_text")
        text_column = norm_config.get("text_column", "text")
        strip_whitespace = norm_config.get("strip_whitespace", True)
        normalize_spaces = norm_config.get("normalize_spaces", True)
        remove_accents = norm_config.get("remove_accents", True)
        dataset = normalize_text(
            dataset,
            text_column=text_column,
            strip_whitespace=strip_whitespace,
            normalize_spaces=normalize_spaces,
            remove_accents=remove_accents,
        )
        logging.info(f"Applied text normalization to column '{text_column}'")

    # MIN/MAX_TOKENS WILL TAKE PRIORITY OVER MIN/MAX_LENGTH

    # First handle character-based length limits (raw string lengths)
    if "max_length" in preprocess_config and "max_tokens" not in preprocess_config:
        max_len_group = preprocess_config.get("max_length")
        max_len_value = max_len_group.get("max_len")
        max_len: int = int(max_len_value)
        text_column = max_len_group.get("text_column", "text")
        truncation_column = max_len_group.get("truncation_column", None)
        dataset = max_length(
            dataset,
            text_column=text_column,
            max_length=max_len,
            truncation_column=truncation_column,
        )
        logging.info(
            f"Applied character max_length={max_len} to column '{text_column}'"
        )

    if "min_length" in preprocess_config and "min_tokens" not in preprocess_config:
        min_len_group = preprocess_config.get("min_length")
        min_len_value = min_len_group.get("min_len")
        min_len: int = int(min_len_value)
        text_column = min_len_group.get("text_column", "text")
        drop_short = min_len_group.get("drop_short", True)
        dataset = min_length(
            dataset,
            text_column=text_column,
            min_length=min_len,
            drop_short=drop_short,
        )
        logging.info(
            f"Applied character min_length={min_len} to column '{text_column}'"
        )

    # if we need token-based limits, first tokenize, then apply token length filters
    # MAX_TOKENS TAKES PRIORITY!

    if "max_tokens" in preprocess_config or "min_tokens" in preprocess_config:
        # load tokenizer from model config
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.model.checkpoint)

        # Process max_tokens group if specified
        if "max_tokens" in preprocess_config:
            max_tokens_group = preprocess_config["max_tokens"]
            max_tok = int(max_tokens_group["max_tok"])
            text_column = max_tokens_group.get("text_column", "text")
            # Tokenize the specified column with max length truncation
            dataset = tokenize_dataset(
                dataset,
                tokenizer=tokenizer,
                text_column=text_column,
                max_length=max_tok,
                truncation=True,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            # Enforce max token limit by filtering any remaining long sequences
            dataset = dataset.filter(lambda x: len(x["input_ids"]) <= max_tok)
            logging.info(f"Applied max token limit {max_tok} to column '{text_column}'")

        # Process min_tokens group if specified (supports separate text column)

        if "min_tokens" in preprocess_config:
            min_tokens_group = preprocess_config["min_tokens"]
            min_tok = int(min_tokens_group["min_tok"])
            text_column = min_tokens_group.get("text_column", "text")
            # Tokenize the column if it hasn't been tokenized yet
            if "input_ids" not in dataset.column_names:
                dataset = tokenize_dataset(
                    dataset,
                    tokenizer=tokenizer,
                    text_column=text_column,
                    max_length=10**6,  # Use large value to avoid chopping anything
                    truncation=True,
                    padding=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )
            # Filter out sequences that don't meet the minimum token requirement
            dataset = dataset.filter(lambda x: len(x["input_ids"]) >= min_tok)
            logging.info(f"Applied min token limit {min_tok} to column '{text_column}'")

    return dataset
