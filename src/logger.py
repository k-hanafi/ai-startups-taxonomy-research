"""Structured logging: rich console handler + rotating file handler.

Every module in the pipeline imports the stdlib logging module and calls
logger = logging.getLogger(__name__).  This module configures the root
logger once so all output is routed through two handlers:

  - Console: rich markup, colored by level, human-readable.
  - File:    outputs/run.log, machine-readable, survives terminal close.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler

_LOG_DIR = Path(__file__).resolve().parents[1] / "outputs"
_LOG_FILE = _LOG_DIR / "run.log"

_CONFIGURED = False


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with console and file handlers.

    Safe to call multiple times. Only the first invocation attaches
    handlers, so imported modules can call it defensively.
    """
    global _CONFIGURED  # noqa: PLW0603
    if _CONFIGURED:
        return

    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    console = RichHandler(
        rich_tracebacks=True,
        show_path=False,
        markup=True,
    )
    console.setLevel(level)

    file_handler = RotatingFileHandler(
        _LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10 MB per file
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s  %(name)-20s  %(levelname)-8s  %(message)s")
    )

    root.addHandler(console)
    root.addHandler(file_handler)

    _CONFIGURED = True
