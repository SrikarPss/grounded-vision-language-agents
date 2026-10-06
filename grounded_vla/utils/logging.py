"""Thin wrapper so every module uses the same `rich`-flavored logger."""
from __future__ import annotations

import logging
import os

try:
    from rich.logging import RichHandler

    _HANDLER: logging.Handler = RichHandler(rich_tracebacks=True, markup=True, show_path=False)
except ImportError:  # rich is a core dep but stay defensive
    _HANDLER = logging.StreamHandler()
    _HANDLER.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )

_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = os.environ.get("GVLA_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level, handlers=[_HANDLER], force=True)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    _configure_root()
    return logging.getLogger(name)
