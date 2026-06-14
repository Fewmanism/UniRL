#!/usr/bin/env bash
# Report and optionally prune DGX Spark UniRL checkpoint artifacts.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/dgx_spark_checkpoint_cleanup.sh [options]

Options:
  --root DIR       Checkpoint root (default: local_runs/checkpoints)
  --dry-run        Report only (default)
  --delete-full    Delete checkpoint.pt files but keep lora_adapter.pt
  --yes            Required with --delete-full
  -h, --help       Show this help
EOF
}

ROOT=local_runs/checkpoints
DELETE_FULL=0
YES=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="${2:?missing --root value}"; shift 2 ;;
    --dry-run) DELETE_FULL=0; shift ;;
    --delete-full) DELETE_FULL=1; shift ;;
    --yes) YES=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -d "$ROOT" ]]; then
  echo "checkpoint root not found: $ROOT" >&2
  exit 1
fi

mapfile -t full_files < <(find "$ROOT" -type f -name checkpoint.pt | sort)
mapfile -t lora_files < <(find "$ROOT" -type f -name lora_adapter.pt | sort)

lora_only=0
for lora in "${lora_files[@]}"; do
  dir="$(dirname "$lora")"
  if [[ ! -f "$dir/checkpoint.pt" ]]; then
    lora_only=$((lora_only + 1))
  fi
done

bytes_for() {
  if [[ $# -eq 0 ]]; then
    echo 0
    return
  fi
  du -cb "$@" 2>/dev/null | awk 'END {print $1+0}'
}

full_bytes=$(bytes_for "${full_files[@]}")
lora_bytes=$(bytes_for "${lora_files[@]}")

printf 'checkpoint root: %s\n' "$ROOT"
printf 'full checkpoints: %d\n' "${#full_files[@]}"
printf 'lora adapters: %d\n' "${#lora_files[@]}"
printf 'lora-only checkpoints: %d\n' "$lora_only"
printf 'full checkpoint bytes: %s\n' "$full_bytes"
printf 'lora adapter bytes: %s\n' "$lora_bytes"

printf '\ndelete candidates (full checkpoint.pt files):\n'
if [[ ${#full_files[@]} -eq 0 ]]; then
  printf 'none\n'
else
  printf '%s\n' "${full_files[@]}"
fi

if [[ "$DELETE_FULL" == 1 ]]; then
  if [[ "$YES" != 1 ]]; then
    echo "refusing to delete without --yes" >&2
    exit 2
  fi
  for f in "${full_files[@]}"; do
    rm -f -- "$f"
  done
  printf '\ndeleted full checkpoint.pt files: %d\n' "${#full_files[@]}"
fi
