"""Run the March-2023 V1 classifier in an isolated output namespace."""

from __future__ import annotations

import os
import sys
from pathlib import Path

NAMESPACE = "wayback_2023"
os.environ["CLASSIFY_NS"] = NAMESPACE

from single_pass_classifier.cli import main as classifier_main  # noqa: E402

from .paths import CLASSIFIER_INPUT_2023_CSV  # noqa: E402

DEFAULT_HISTORICAL_INPUT = Path(CLASSIFIER_INPUT_2023_CSV)
_DATA_COMMANDS = frozenset({"prepare", "submit", "retry", "test", "run"})


def main() -> None:
    """Delegate to V1 after binding safe historical paths."""
    subcommand = sys.argv[1] if len(sys.argv) > 1 else None
    if subcommand in _DATA_COMMANDS and "--data" not in sys.argv:
        sys.argv[2:2] = ["--data", str(DEFAULT_HISTORICAL_INPUT)]
    classifier_main()


if __name__ == "__main__":
    main()
