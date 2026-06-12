#!/usr/bin/env bash
# Checkpoint/resume smoke for the local DGX Spark fork.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/dgx_spark_checkpoint_smoke.sh [options]

Runs:
  1. trainside quick, one rollout, save full checkpoint + LoRA adapter
  2. trainside quick, one rollout, resume from that checkpoint

Options:
  --watch             Wrap each phase with dgx_spark_watch.py
  --interval SECONDS  Watcher interval (default: 5)
  --base-dir DIR      Checkpoint base dir (default: local_runs/checkpoints)
  --dry-run           Print commands only
  -h, --help          Show help
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

WATCH=0
INTERVAL=5
BASE_DIR=local_runs/checkpoints
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --watch) WATCH=1; shift ;;
    --interval) INTERVAL="${2:?missing --interval value}"; shift 2 ;;
    --base-dir) BASE_DIR="${2:?missing --base-dir value}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

stamp="$(date +%Y%m%d-%H%M%S)"
run_dir="$BASE_DIR/$stamp-trainside-quick"
ckpt_dir="$run_dir/save"
resume_from="$ckpt_dir/rollout-000001"

common=(
  --engine trainside
  --profile quick
  --interval "$INTERVAL"
)
[[ "$WATCH" == 1 ]] && common+=(--watch)
[[ "$DRY_RUN" == 1 ]] && common+=(--dry-run)

save_extra=(
  +checkpoint_dir="$ckpt_dir"
  +checkpoint_interval=1
  +save_lora_checkpoint=true
)
resume_extra=(
  +resume_checkpoint_dir="$resume_from"
  +checkpoint_dir="$run_dir/resume"
  +checkpoint_interval=1
  +save_lora_checkpoint=true
)

run_phase() {
  local label="$1"; shift
  local -a extra=("$@")
  echo
  echo "== checkpoint phase: $label =="
  cmd=(scripts/dgx_spark_run_smoke.sh "${common[@]}" --label "dgx-checkpoint-$label-$stamp")
  cmd+=(--)
}

# dgx_spark_run_smoke.sh intentionally accepts only its own options, so append
# checkpoint overrides by directly reconstructing the validated trainside command.
make_train_cmd() {
  local -n out=$1
  shift
  local -a overrides=("$@")
  out=(
    /home/spark01/.local/bin/npx --yes @dotenvx/dotenvx run -f .env --
    bash -lc "source '.venv-spark/bin/activate'; export REPORT_TO_WANDB=false HF_HUB_DISABLE_TELEMETRY=1 HYDRA_FULL_ERROR=1 PRETRAINED_MODEL=stabilityai/stable-diffusion-3.5-medium; exec \"\$@\"" _
    python -m unirl.train_diffusion --config-name=diffusion/sd3_trainside
    num_devices=1 +devices_per_node=1 batch_size=2 +num_rollouts=1
    data_source.args.algorithm.prompts_per_rollout=2
    sampling.samples_per_prompt=2 sampling.num_inference_steps=2
    sampling.scheduler.num_timesteps=2 sampling.scheduler.num_sde_steps=1
    'sampling.scheduler.timestep_fraction=[0,1]'
    rollout.forward_batch_size=1 reward.backend.config.batch_size=1
    stack.micro_batch_size=1 stack.num_updates_per_batch=1
    "${overrides[@]}"
  )
}

execute_phase() {
  local label="$1"; shift
  local -a cmd
  make_train_cmd cmd "$@"
  printf 'cmd:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  if [[ "$DRY_RUN" == 1 ]]; then
    return 0
  fi
  ray stop --force >/tmp/unirl-dgx-spark-checkpoint-ray-stop.log 2>&1 || true
  if [[ "$WATCH" == 1 ]]; then
    python3 scripts/dgx_spark_watch.py --label "dgx-checkpoint-$label-$stamp" --interval "$INTERVAL" -- "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
}

execute_phase save "${save_extra[@]}"

if [[ "$DRY_RUN" != 1 ]]; then
  test -f "$resume_from/checkpoint.pt"
  test -f "$resume_from/lora_adapter.pt"
  du -sh "$resume_from"
fi

execute_phase resume "${resume_extra[@]}"

if [[ "$DRY_RUN" != 1 ]]; then
  test -f "$run_dir/resume/rollout-000001/checkpoint.pt"
  test -f "$run_dir/resume/rollout-000001/lora_adapter.pt"
  echo "checkpoint_smoke_dir=$run_dir"
fi
