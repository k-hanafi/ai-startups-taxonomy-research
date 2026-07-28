"""Filesystem paths for live website crawl inputs and artifacts."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

TAVILY_DIR = OUTPUTS_DIR / "tavilycrawl"
TAVILY_RAW_DIR = TAVILY_DIR / "raw"
TAVILY_PROCESSED_DIR = TAVILY_DIR / "processed"
LOGS_DIR = OUTPUTS_DIR / "logs"
