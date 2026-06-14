#!/usr/bin/env bash
# Local validation gate for the DGX Spark fork.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/dgx_spark_validate.sh [options]

Options:
  --static-only      Run non-GPU checks only (default)
  --gpu-smoke        Also run a tiny vllmomni LoRA command smoke
  -h, --help         Show this help
EOF
}

STATIC_ONLY=1
GPU_SMOKE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --static-only) STATIC_ONLY=1; shift ;;
    --gpu-smoke) GPU_SMOKE=1; STATIC_ONLY=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -x .venv-spark/bin/python ]]; then
  PY=.venv-spark/bin/python
else
  PY=python3
fi

step() { printf '\n== %s ==\n' "$*"; }

step "checkpoint tests"
"$PY" -m pytest tests/test_checkpoint_mode.py tests/test_lora_state_merge.py tests/test_lora_manifest.py -q

step "shell syntax"
bash -n scripts/dgx_spark_clean.sh \
  scripts/dgx_spark_run_smoke.sh \
  scripts/dgx_spark_checkpoint_smoke.sh \
  scripts/dgx_spark_run_vllmomni_lora.sh \
  scripts/dgx_spark_checkpoint_cleanup.sh \
  scripts/dgx_spark_validate.sh

step "clean dry-run"
scripts/dgx_spark_clean.sh --dry-run --no-ray --kill-orphans --yes >/tmp/dgx_spark_validate_clean.log
sed -n '1,40p' /tmp/dgx_spark_validate_clean.log

step "report dry-run"
python3 scripts/dgx_spark_report.py --limit 5 --stdout >/tmp/dgx_spark_validate_report.md
sed -n '1,20p' /tmp/dgx_spark_validate_report.md

if [[ "$GPU_SMOKE" == 1 ]]; then
  step "gpu smoke"
  scripts/dgx_spark_run_vllmomni_lora.sh --rollouts 1 --checkpoint-interval 1 --watch --label dgx-validate-vllmomni-r1
fi

step "validation complete"
