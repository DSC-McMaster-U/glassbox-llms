# Place your custom functions here (to be used with apply_transforms)

from typing import Any, List


def example(
    dataset,
    columns: List[str],
) -> Any:
    return dataset.select_columns(columns[0])

def dummy() -> Any:
    return None
