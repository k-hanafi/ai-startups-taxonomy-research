from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_historical_wrapper_binds_isolated_v1_artifacts() -> None:
    project_root = Path(__file__).resolve().parents[2]
    script = """
import json
import os
import wayback_machine.classify_2023 as wrapper
from single_pass_classifier import paths

print(json.dumps({
    "namespace": os.environ["CLASSIFY_NS"],
    "batch_state": str(paths.BATCH_STATE_FILE),
    "output_csv": str(paths.DEFAULT_CLASSIFICATION_OUTPUT_CSV),
    "default_input": str(wrapper.DEFAULT_HISTORICAL_INPUT),
    "live_batch_state": str(paths.OUTPUTS_DIR / "batch_data" / "state.json"),
    "live_output_csv": str(
        paths.OUTPUTS_DIR
        / "production_csvs"
        / "production_classifications.csv"
    ),
}))
"""
    env = dict(os.environ)
    env["OPENAI_API_KEY"] = "placeholder"
    env["CLASSIFY_NS"] = ""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["namespace"] == "wayback_2023"
    assert payload["batch_state"].endswith(
        "outputs/wayback_2023/batch_data/state.json"
    )
    assert payload["output_csv"].endswith(
        "outputs/wayback_2023/wayback_2023_classifications.csv"
    )
    assert payload["default_input"].endswith(
        "wayback_machine/outputs/processed/classifier_input_2023.csv"
    )
    assert payload["batch_state"] != payload["live_batch_state"]
    assert payload["output_csv"] != payload["live_output_csv"]
