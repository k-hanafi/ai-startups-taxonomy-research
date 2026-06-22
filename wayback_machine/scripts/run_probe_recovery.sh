#!/usr/bin/env bash
# Recovery re-probe: retries companies that failed CDX (skips status=ok).
# Run from project root OR via: bash wayback_machine/scripts/run_probe_recovery.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="$ROOT/wayback_machine/logs"
mkdir -p "$LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/probe_recovery_${STAMP}.log"
LATEST="$LOG_DIR/LATEST_RUN.txt"

cd "$ROOT"

{
  echo "=== Wayback probe recovery ==="
  echo "Started: $(date)"
  echo "Log: $LOG"
  echo "Results CSV: wayback_machine/data/death_coverage.csv"
  echo "To watch live: tail -f $LOG"
  echo ""
} | tee "$LOG"

{
  echo "log=$LOG"
  echo "pid=$$"
  echo "started=$(date -Iseconds)"
} > "$LATEST"

# IA CDX limit: 60 req/min hard cap; we target 48/min (80% headroom). concurrency=1
# keeps one global rate limiter honest. Skips resolved org_uuids; retries error rows.
caffeinate -ims python3 wayback_machine/scripts/probe_death_coverage.py \
  --sample-size 0 \
  --concurrency 1 \
  --rpm 48 \
  --retries 8 \
  --timeout 60 \
  --output wayback_machine/data/death_coverage.csv \
  2>&1 | tee -a "$LOG"

echo "Finished: $(date)" | tee -a "$LOG"
