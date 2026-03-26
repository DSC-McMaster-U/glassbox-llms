# Preprocessing Configuration Guide

This document describes how to use the transforms available in the `preprocess` configuration section of your `config.json` file.

## Overview

The preprocessing system applies transforms in a specific order based on the configuration options you provide. Each transform is triggered by the presence of specific keys in the `preprocess` section.

### Transform Order

Here are the available transforms, ordered by order of operation (lower number == earlier):

1. **rename_columns** - if "rename" is in preprocess_config
2. **apply_transforms** - if "apply_transform" is in preprocess_config
3. **sample_dataset** - if "num_samples" is in preprocess_config
4. **select_columns** - if "columns" is in preprocess_config
5. **clean_text** - if "clean_text" is in preprocess_config
6. **normalize_text** - if "normalize_text" is in preprocess_config
7. **max_length** - if "max_length" is in preprocess_config AND "max_tokens" is NOT
8. **min_length** - if "min_length" is in preprocess_config AND "min_tokens" is NOT
9. **max_tokens** - if "min_tokens" is in preprocess_config
10. **min_tokens** - if "min_tokens" is in preprocess_config

## Configuration Structure

All preprocessing options go under the `preprocess` key in your dataset configuration:

```json
{
  "dataset": {
    "path": "data/my_dataset",
    "split": "train",
    "preprocess": {
      // preprocessing options go here
    }
  }
}
```

---

## Transform Details

### 1. rename_columns

Renames columns in your dataset using a mapping dictionary.

**Trigger:** Presence of `"rename"` key in preprocess_config

**Configuration:**
```json
{
  "preprocess": {
    "rename": {
      "old_column_name": "new_column_name"
      "old_column_name_1": "new_column_name_1"
      ...
    }
  }
}
```

**Example:**
```json
{
  "preprocess": {
    "rename": {
      "text_content": "text",
      "original_source": "oc"
    }
  }
}
```

**What it does:**
- Maps old column names to new column names
- Only renames columns that exist in the dataset
- Useful for standardizing column names across different datasets

---

### 2. apply_transforms

Applies a sequence of custom transformation functions to text in a column.

**Trigger:** Presence of `"apply_transform"` key in preprocess_config

**Configuration:**
```json
{
  "preprocess": {
    "apply_transform": {
      "text_column": "text",
      "transform_functions": ["func1", "func2", "func3"]
    }
  }
}
```

**Example:**
```json
{
  "preprocess": {
    "apply_transform": {
      "text_column": "text",
      "transform_functions": [
        "remove_urls",
        "remove_mentions",
        "expand_contractions"
      ]
    }
  }
}
```

**What it does:**
- Applies transformation functions in sequence (daisy-chain order)
- Functions are applied to the specified text column
- Custom transform functions must be defined elsewhere in your codebase and accessible 

---

### 3. sample_dataset

Samples a subset of the dataset for faster experimentation.

**Trigger:** Presence of `"num_samples"` key in preprocess_config

**Configuration:**
```json
{
  "preprocess": {
    "num_samples": {
      "count": 1000,
      "seed": 42,
      "shuffle": true
    }
  }
}
```

**Example:**
```json
{
  "preprocess": {
    "num_samples": {
      "count": 500,
      "seed": 123,
      "shuffle": true
    }
  }
}
```

**Parameters:**
- `count` (required): Number of samples to take
- `seed` (optional): Random seed for reproducibility
- `shuffle` (optional, default: true): Whether to shuffle before sampling

**What it does:**
- Takes a random subset of the dataset
- Shuffles the dataset before sampling (if shuffle=true)
- Useful for quick testing without processing the entire dataset

---

### 4. select_columns

Selects specific columns from the dataset, dropping all others.

**Trigger:** Presence of `"columns"` key in preprocess_config

**Configuration:**
```json
{
  "preprocess": {
    "columns": ["column1", "column2", "column3"]
  }
}
```

**Example:**
```json
{
  "preprocess": {
    "columns": ["text", "label", "metadata"]
  }
}
```

**What it does:**
- Keeps only the specified columns
- Drops all other columns from the dataset
- Useful for reducing memory usage and focusing on relevant data

---

### 5. clean_text

Cleans text in a dataset column with various text cleaning options.

**Trigger:** Presence of `"clean_text"` key in preprocess_config

**Configuration:**
```json
{
  "preprocess": {
    "clean_text": {
      "text_column": "text",
      "lowercase": true,
      "remove_special_chars": true,
      "normalize_whitespace": true
    }
  }
}
```

**Example:**
```json
{
  "preprocess": {
    "clean_text": {
      "text_column": "content",
      "lowercase": true,
      "remove_special_chars": true,
      "normalize_whitespace": true
    }
  }
}
```

**Parameters:**
- `text_column` (optional, default: "text"): Name of the column containing text to clean
- `lowercase` (optional, default: true): Whether to convert text to lowercase
- `remove_special_chars` (optional, default: true): Whether to remove special characters
- `normalize_whitespace` (optional, default: true): Whether to normalize whitespace (multiple spaces to single)

**What it does:**
- Converts text to lowercase (if enabled)
- Removes special characters (if enabled)
- Normalizes whitespace (if enabled)
- Handles None values gracefully

---

### 6. normalize_text

Normalizes text with more advanced normalization options including accent removal.

**Trigger:** Presence of `"normalize_text"` key in preprocess_config

**Configuration:**
```json
{
  "preprocess": {
    "normalize_text": {
      "text_column": "text",
      "strip_whitespace": true,
      "normalize_spaces": true,
      "remove_accents": true
    }
  }
}
```

**Example:**
```json
{
  "preprocess": {
    "normalize_text": {
      "text_column": "text",
      "strip_whitespace": true,
      "normalize_spaces": true,
      "remove_accents": true
    }
  }
}
```

**Parameters:**
- `text_column` (optional, default: "text"): Name of the column containing text to normalize
- `strip_whitespace` (optional, default: true): Whether to strip leading/trailing whitespace
- `normalize_spaces` (optional, default: true): Whether to normalize multiple spaces to single space
- `remove_accents` (optional, default: true): Whether to remove accent marks from characters

**What it does:**
- Strips leading/trailing whitespace (if enabled)
- Normalizes multiple spaces to single space (if enabled)
- Removes accent marks from characters using Unicode normalization (if enabled)
- Useful for text normalization in NLP tasks

---

### 7. max_length

Truncates text to a maximum character length.

**Trigger:** Presence of `"max_length"` key in preprocess_config AND absence of `"max_tokens"`

**Configuration:**
```json
{
  "preprocess": {
    "max_length": {
      "text_column": "text",
      "max_len": 512,
      "truncation_column": null
    }
  }
}
```

**Example:**
```json
{
  "preprocess": {
    "max_length": {
      "text_column": "content",
      "max_len": 1024,
      "truncation_column": "original_length"
    }
  }
}
```

**Parameters:**
- `text_column` (optional, default: "text"): Name of the column containing text to truncate
- `max_len` (required): Maximum length of text in characters
- `truncation_column` (optional, default: null): Optional column name to store original length before truncation

**What it does:**
- Truncates text to the specified maximum length
- Stores original length in a separate column if `truncation_column` is specified
- Works with character-based length limits (not token-based)

**Note:** This is skipped if `max_tokens` is specified, as token-based limits take priority.

---

### 8. min_length

Filters or pads text to minimum length.

**Trigger:** Presence of `"min_length"` key in preprocess_config AND absence of `"min_tokens"`

**Configuration:**
```json
{
  "preprocess": {
    "min_length": {
      "text_column": "text",
      "min_len": 10,
      "drop_short": true
    }
  }
}
```

**Example:**
```json
{
  "preprocess": {
    "min_length": {
      "text_column": "content",
      "min_len": 50,
      "drop_short": true
    }
  }
}
```

**Parameters:**
- `text_column` (optional, default: "text"): Name of the column containing text to process
- `min_len` (required): Minimum length of text in characters
- `drop_short` (optional, default: true): Whether to drop samples shorter than min_length

**What it does:**
- If `drop_short=true`: Removes samples shorter than min_length from the dataset
- If `drop_short=false`: Pads short texts with spaces to reach min_length
- Works with character-based length limits (not token-based)

**Note:** This is skipped if `min_tokens` is specified, as token-based limits take priority.

---

### 9. tokenize_dataset

Tokenizes text using a model's tokenizer and adds token-based columns.

**Trigger:** Presence of `"max_tokens"` OR `"min_tokens"` key in preprocess_config

**Configuration:**
```json
{
  "preprocess": {
    "max_tokens": {
      "text_column": "text",
      "max_tok": 512
    }
  }
}
```

**Example with max_tokens:**
```json
{
  "preprocess": {
    "max_tokens": {
      "text_column": "content",
      "max_tok": 1024
    }
  }
}
```

**Example with min_tokens:**
```json
{
  "preprocess": {
    "min_tokens": {
      "text_column": "content",
      "min_tok": 10
    }
  }
}
```

**Example with both:**
```json
{
  "preprocess": {
    "max_tokens": {
      "text_column": "content",
      "max_tok": 1024
    },
    "min_tokens": {
      "text_column": "content",
      "min_tok": 50
    }
  }
}
```

**Parameters:**
- `text_column` (optional, default: "text"): Name of the column containing text to tokenize
- `max_tok` (required for max_tokens): Maximum number of tokens
- `min_tok` (required for min_tokens): Minimum number of tokens

**What it does:**
- Uses the tokenizer from your model checkpoint (specified in `model.checkpoint`)
- Tokenizes the specified text column
- Adds columns: `input_ids`, `attention_mask`, and optionally `token_type_ids`
- For `max_tokens`: Truncates sequences and filters any remaining long sequences
- For `min_tokens`: Filters out sequences that don't meet the minimum token requirement
- Token-based limits take priority over character-based limits

**Important Notes:**
- When both `max_tokens` and `min_tokens` are specified, both filters are applied
- The tokenizer is automatically loaded from your model checkpoint
- Tokenization happens after all character-based preprocessing

---

## Complete Example

Here's a complete example showing multiple transforms in action:

```json
{
  "model": {
    "name": "llama-3.3-8b",
    "checkpoint": "meta-llama/Meta-Llama-3.3-8B",
    "device": "cuda",
    "dtype": "float16",
    "wrapper_type": "transformers"
  },
  "dataset": {
    "path": "data/my_dataset",
    "split": "train",
    "preprocess": {
      "rename": {
        "text_content": "text",
        "label_value": "label"
      },
      "columns": ["text", "label"],
      "clean_text": {
        "text_column": "text, label",
        "lowercase": true,
        "remove_special_chars": true,
        "normalize_whitespace": true
      },
      "normalize_text": {
        "text_column": "text",
        "strip_whitespace": true,
        "normalize_spaces": true,
        "remove_accents": true
      },
      "max_length": {
        "text_column": "text",
        "max_len": 512
      },
      "num_samples": {
        "count": 1000,
        "seed": 67,
        "shuffle": true
      }
    }
  },
  "experiment": {
    "type": "dummy",
    "parameters": {
      "lr": 2e-5,
      "bsz": 8,
      "time": 3
    },
    "seed": 0
  },
  "tracking": {
    "enabled": true,
    "type": "wandb",
    "project": "example-project",
    "entity": "awesome-team",
    "tags": ["llama", "baseline"],
    "config": {
      "optimizer": "adamw-8bit"
    }
  },
  "output": {
    "base_dir": "runs/",
    "name": "run-001"
  }
}
```

## Transform Execution Order

When multiple transforms are configured, they are executed in this order:

1. **rename_columns** - First, rename any columns you need
2. **apply_transforms** - Then apply any custom transform functions
3. **sample_dataset** - Sample the dataset if needed
4. **select_columns** - Select only the columns you need
5. **clean_text** - Clean the text (lowercase, remove special chars, etc.)
6. **normalize_text** - Further normalize the text (strip whitespace, remove accents, etc.)
7. **max_length** - Truncate text to max character length (if no token limits specified)
8. **min_length** - Filter/pad text to min character length (if no token limits specified)
9. **tokenize_dataset** - Tokenize the text (if token limits are specified)
10. **filter** - Apply token-based filtering

## Tips

- **Character vs Token Limits**: Use `max_length`/`min_length` for character-based limits or `max_tokens`/`min_tokens` for token-based limits. **Token-based limits take priority**.
- **Order Matters**: The order of transforms is fixed. Plan your preprocessing pipeline accordingly.
- **Sampling**: Use `num_samples` for quick testing, but remember it samples before other transforms!.
- **Column Selection**: Use `columns` to reduce memory usage by keeping only necessary columns.
- **Text Cleaning**: `clean_text` and `normalize_text` are designed to be used together.
