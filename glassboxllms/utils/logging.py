import logging
from typing import Optional

_LOGGER_NAME = "glassboxllms"

def get_logger(
    name: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Get a GlassboxLLMs logger.

    Parameters
    ----------
    name : str, optional
        Sub-logger name (e.g. "models", "instrumentation.hooks").
        If None, returns the root glassboxllms logger.
    level : int
        Logging level (e.g. logging.INFO, logging.DEBUG).

    Returns
    -------
    logging.Logger
    """
    logger_name = _LOGGER_NAME if name is None else f"{_LOGGER_NAME}.{name}"
    logger = logging.getLogger(logger_name)

    # Avoid duplicate handlers
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level)
    logger.propagate = False

    return logger
"""
EXAMPLE USAGE

from glassboxllms.utils.logging import get_logger

logger = get_logger("instrumentation.hooks")

class HookManager:
    def add_hook(self, module):
        logger.debug(f"Adding hook to {module}")
        
"""