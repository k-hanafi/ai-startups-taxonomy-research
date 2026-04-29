#!/usr/bin/env python3
"""Migrate a v2 classified_startups CSV to the v2.1 taxonomy.

The v2.1 taxonomy collapses the 11-class system to 10 classes:
- 0C-THIN, 0C-THICK promote to AI-native and renumber to 1C, 1D
- 0D merges into 0B (the AI-Augmented / AI-Adjacent bucket)
- The AI-native side reorders: foundation -> tooling -> wrappers -> applied -> autonomous -> generative

RAD becomes a perfect function of ai_native:
- ai_native = 1 (1A-1G): always RAD-H/M/L
- ai_native = 0 (0A, 0B, 0E): always RAD-NA, conf_rad = null

The migration normalizes any v2 row that violates the new RAD invariant:
- ai_native=0 rows with non-NA RAD are forced to RAD-NA with conf_rad=null
  (this happens in the v2 CSV because the LLM occasionally assigned RAD-H/M
  to old 0B companies)
- ai_native=1 rows with RAD-NA cannot be auto-fixed and are reported but
  left untouched. These rows need to be re-classified by re-running the
  pipeline on those CompanyIDs.

Usage:
    python scripts/migrate_taxonomy_v21.py
    python scripts/migrate_taxonomy_v21.py --input outputs/classified_startups_v2.csv \
                                           --output outputs/classified_startups_v21_migrated.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = _PROJECT_ROOT / "outputs" / "classified_startups_v2.csv"
DEFAULT_OUTPUT = _PROJECT_ROOT / "outputs" / "classified_startups_v21_migrated.csv"

SUBCLASS_REMAP: dict[str, str] = {
    "1A": "1A",
    "1B": "1E",
    "1C": "1B",
    "1D": "1F",
    "1E": "1G",
    "0A": "0A",
    "0B": "0B",
    "0C-THIN": "1C",
    "0C-THICK": "1D",
    "0D": "0B",
    "0E": "0E",
}

PROMOTED_TO_NATIVE: set[str] = {"0C-THIN", "0C-THICK"}


def migrate_row(row: dict, stats: Counter) -> dict:
    """Apply the v2 -> v2.1 transformation to a single row.

    Mutates `stats` to count normalization events and anomalies.
    """
    old_subclass = row.get("subclass", "")
    new_subclass = SUBCLASS_REMAP.get(old_subclass, old_subclass)

    row["subclass"] = new_subclass

    if old_subclass in PROMOTED_TO_NATIVE:
        row["ai_native"] = "1"

    ai_native = row.get("ai_native", "")
    rad = row.get("rad_score", "")

    if ai_native == "0" and rad != "RAD-NA":
        row["rad_score"] = "RAD-NA"
        row["conf_rad"] = ""
        stats["normalized_ai0_to_NA"] += 1
    elif ai_native == "1" and rad == "RAD-NA":
        stats["anomaly_ai1_with_NA"] += 1

    return row


def migrate(input_path: Path, output_path: Path) -> tuple[int, Counter, Counter, Counter]:
    """Stream the CSV row-by-row and write the migrated output.

    Returns (rows_processed, old_counter, new_counter, normalization_stats).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    old_counts: Counter = Counter()
    new_counts: Counter = Counter()
    stats: Counter = Counter()
    n = 0

    with open(input_path, newline="", encoding="utf-8") as in_f, \
         open(output_path, "w", newline="", encoding="utf-8") as out_f:
        reader = csv.DictReader(in_f)
        if reader.fieldnames is None:
            raise ValueError(f"No header row found in {input_path}")

        unknown_fields = set(reader.fieldnames) - {
            "CompanyID", "CompanyName", "ai_native", "subclass", "rad_score",
            "cohort", "conf_classification", "conf_rad", "reasons_3_points",
            "sources_used", "verification_critique",
        }
        if unknown_fields:
            print(f"Note: input has unexpected fields (will be passed through): {unknown_fields}")

        writer = csv.DictWriter(out_f, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            old_counts[row.get("subclass", "")] += 1
            migrated = migrate_row(row, stats)
            new_counts[migrated.get("subclass", "")] += 1
            writer.writerow(migrated)
            n += 1

    return n, old_counts, new_counts, stats


def _print_summary(old_counts: Counter, new_counts: Counter) -> None:
    """Show the before/after subclass distribution."""
    all_codes = sorted(set(old_counts) | set(new_counts))
    width = max((len(c) for c in all_codes), default=8)
    print()
    print(f"{'subclass'.ljust(width)}    old        new       delta")
    print(f"{'-' * width}    -------    -------   -------")
    for code in all_codes:
        old = old_counts.get(code, 0)
        new = new_counts.get(code, 0)
        delta = new - old
        sign = "+" if delta > 0 else " "
        print(f"{code.ljust(width)}    {old:>7,}    {new:>7,}   {sign}{delta:>6,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help=f"Source v2 CSV (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Destination v2.1 CSV (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input CSV not found: {args.input}")

    print(f"Reading  {args.input}")
    print(f"Writing  {args.output}")

    n, old_counts, new_counts, stats = migrate(args.input, args.output)

    print(f"\nMigrated {n:,} rows.")
    _print_summary(old_counts, new_counts)

    print()
    normalized = stats.get("normalized_ai0_to_NA", 0)
    if normalized:
        print(f"Normalized {normalized:,} ai_native=0 rows from RAD-H/M/L to RAD-NA "
              "(conf_rad cleared).")
    anomalies = stats.get("anomaly_ai1_with_NA", 0)
    if anomalies:
        print(f"WARNING: {anomalies:,} ai_native=1 rows have RAD-NA. These violate the "
              "new invariant and require pipeline re-classification to repair.")


if __name__ == "__main__":
    main()
