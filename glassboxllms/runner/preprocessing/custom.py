# Place your custom functions here (to be used with apply_transforms)

from typing import Any

def dummy(text: str) -> str:
    # For testing.
    return text

def example_transform(text: str, prefix: str = "", suffix: str = "") -> str:
    # An example of a custom transformation function that accepts additional parameters.
    return f"{prefix}{text}{suffix}"
