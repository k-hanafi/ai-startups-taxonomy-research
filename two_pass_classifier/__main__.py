"""Run the production classifier CLI with ``python -m two_pass_classifier``."""

from __future__ import annotations

import sys

from .cli import main


if __name__ == "__main__":
    sys.exit(main())
