#!/usr/bin/env bash
# Run the DGX Spark local scale matrix and generate summary.csv.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/dgx_spark_run_matrix.sh [options]

Default matrix:
  - trainside standard
  - sglang scale4
  - sglang scale10
  - vllmomni scale4
  - vllmomni scale10

Options:
  --watch / --no-watch     Enable/disable watcher (default: --watch)
  --interval SECONDS       Watcher sample interval (default: 5)
  --log-dir DIR            Watcher output dir (default: outputs/dgx-spark-observe)
  --dry-run                Print commands without executing
  --continue-on-error      Continue matrix after a failed cell
  -h, --help               Show help
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

WATCH=1
INTERVAL=5
LOG_DIR=outputs/dgx-spark-observe
DRY_RUN=0
CONTINUE_ON_ERROR=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --watch) WATCH=1; shift ;;
    --no-watch) WATCH=0; shift ;;
    --interval) INTERVAL="${2:?missing --interval value}"; shift 2 ;;
    --log-dir) LOG_DIR="${2:?missing --log-dir value}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --continue-on-error) CONTINUE_ON_ERROR=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

MATRIX=(
  "trainside standard"
  "sglang scale4"
  "sglang scale10"
  "vllmomni scale4"
  "vllmomni scale10"
)

status=0
for cell in "${MATRIX[@]}"; do
  read -r engine profile <<<"$cell"
  echo
  echo "## matrix cell: engine=$engine profile=$profile"
  cmd=(scripts/dgx_spark_run_smoke.sh --engine "$engine" --profile "$profile" --interval "$INTERVAL" --log-dir "$LOG_DIR")
  [[ "$WATCH" == 1 ]] && cmd+=(--watch)
  [[ "$DRY_RUN" == 1 ]] && cmd+=(--dry-run)
  if ! "${cmd[@]}"; then
    status=$?
    echo "matrix cell failed: engine=$engine profile=$profile status=$status" >&2
    if [[ "$CONTINUE_ON_ERROR" != 1 ]]; then
      exit "$status"
    fi
  fi
done

if [[ "$DRY_RUN" != 1 ]]; then
  mkdir -p "$LOG_DIR"
  python3 scripts/dgx_spark_summarize_runs.py --log-dir "$LOG_DIR" --limit 100 --format csv > "$LOG_DIR/summary.csv"
  python3 scripts/dgx_spark_summarize_runs.py --log-dir "$LOG_DIR" --limit 20
  echo
  echo "summary_csv=$LOG_DIR/summary.csv"
fi

exit "$status"
