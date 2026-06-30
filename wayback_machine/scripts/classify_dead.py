#!/usr/bin/env python3
"""Stage E CLI (survivorship): run the UNCHANGED classifier in an isolated workspace.

Sets ``CLASSIFY_NS=wayback_dead`` BEFORE importing the classifier, which reroutes
all batch state + the output CSV under ``outputs/wayback_dead/`` (see
``src/paths.py``). The finished modern run under ``outputs/batch_data`` and
``outputs/production_csvs`` is therefore physically untouchable from here.

Every ``classify.py`` subcommand works through this wrapper — same model, prompt,
and schema as the live cohort. Only the input evidence differs:

    python wayback_machine/scripts/classify_dead.py run \\
        --data wayback_machine/outputs/processed/classifier_input_dead.csv
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Bind the namespace before any src import so src/paths picks it up at import.
os.environ.setdefault("CLASSIFY_NS", "wayback_dead")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DEAD_INPUT = (
    PROJECT_ROOT / "wayback_machine" / "outputs" / "processed" / "classifier_input_dead.csv"
)

from classify import main  # noqa: E402

if __name__ == "__main__":
    if "--data" not in sys.argv:
        insert_at = 2 if len(sys.argv) > 1 else len(sys.argv)
        sys.argv[insert_at:insert_at] = ["--data", str(DEFAULT_DEAD_INPUT)]
    main()
