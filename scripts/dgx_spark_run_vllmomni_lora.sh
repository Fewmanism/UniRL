#!/usr/bin/env bash
# Reproducible VLLM-Omni LoRA-only runner for DGX Spark / GB10.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/dgx_spark_run_vllmomni_lora.sh [options]

Options:
  --rollouts N              Number of rollouts (default: 20)
  --scale 10|20             Denoising profile (default: 10)
  --checkpoint-interval N   Save LoRA adapter every N rollouts (default: 20)
  --resume-lora DIR         Resume from DIR/lora_adapter.pt using resume_checkpoint_mode=lora
  --batch-size N            Batch size (default: 2)
  --prompts-per-rollout N   Prompts per rollout (default: 2)
  --samples-per-prompt N    Samples per prompt (default: 2)
  --watch                   Wrap with scripts/dgx_spark_watch.py
  --label LABEL             Watcher label (default: exp-vllmomni-scaleS-rR-lora-TIMESTAMP)
  --interval SECONDS        Watcher interval (default: 10)
  --log-dir DIR             Watcher output dir (default: outputs/dgx-spark-observe)
  --no-ray-stop             Do not stop Ray before running
  --print-command           Print the rendered command and exit
  --dry-run                 Alias for --print-command
  -h, --help                Show this help

Examples:
  scripts/dgx_spark_run_vllmomni_lora.sh --rollouts 100 --checkpoint-interval 20 --watch
  scripts/dgx_spark_run_vllmomni_lora.sh --scale 20 --rollouts 10 --checkpoint-interval 5 --watch
  scripts/dgx_spark_run_vllmomni_lora.sh --rollouts 10 --resume-lora local_runs/checkpoints/.../rollout-000100 --watch
EOF
}

ROLL_OUTS=20
SCALE=10
CHECKPOINT_INTERVAL=20
RESUME_LORA=""
BATCH_SIZE=2
PROMPTS_PER_ROLLOUT=2
SAMPLES_PER_PROMPT=2
WATCH=0
LABEL=""
INTERVAL=10
LOG_DIR=outputs/dgx-spark-observe
RAY_STOP=1
PRINT_COMMAND=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rollouts) ROLL_OUTS="${2:?missing --rollouts value}"; shift 2 ;;
    --scale) SCALE="${2:?missing --scale value}"; shift 2 ;;
    --checkpoint-interval) CHECKPOINT_INTERVAL="${2:?missing --checkpoint-interval value}"; shift 2 ;;
    --resume-lora) RESUME_LORA="${2:?missing --resume-lora value}"; shift 2 ;;
    --batch-size) BATCH_SIZE="${2:?missing --batch-size value}"; shift 2 ;;
    --prompts-per-rollout) PROMPTS_PER_ROLLOUT="${2:?missing --prompts-per-rollout value}"; shift 2 ;;
    --samples-per-prompt) SAMPLES_PER_PROMPT="${2:?missing --samples-per-prompt value}"; shift 2 ;;
    --watch) WATCH=1; shift ;;
    --label) LABEL="${2:?missing --label value}"; shift 2 ;;
    --interval) INTERVAL="${2:?missing --interval value}"; shift 2 ;;
    --log-dir) LOG_DIR="${2:?missing --log-dir value}"; shift 2 ;;
    --no-ray-stop) RAY_STOP=0; shift ;;
    --print-command|--dry-run) PRINT_COMMAND=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$SCALE" in
  10) STEPS=10; SDE_STEPS=3 ;;
  20) STEPS=20; SDE_STEPS=5 ;;
  *) echo "Invalid --scale: $SCALE (expected 10 or 20)" >&2; exit 2 ;;
esac

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STAMP="${DGX_SPARK_STAMP:-$(date +%Y%m%d-%H%M%S)}"
if [[ -z "$LABEL" ]]; then
  LABEL="exp-vllmomni-scale${SCALE}-r${ROLL_OUTS}-lora-${STAMP}"
fi
CKPT_DIR="local_runs/checkpoints/${STAMP}-vllmomni-scale${SCALE}-r${ROLL_OUTS}-lora"

BASE_CMD=(
  /home/spark01/.local/bin/npx --yes @dotenvx/dotenvx run -f .env --
  bash -lc 'source .venv-vllm/bin/activate; export REPORT_TO_WANDB=false HF_HUB_DISABLE_TELEMETRY=1 HYDRA_FULL_ERROR=1 PRETRAINED_MODEL=stabilityai/stable-diffusion-3.5-medium; exec "$@"' _
  python -m unirl.train_diffusion --config-name=diffusion/sd3_vllmomni
  num_devices=1 +devices_per_node=1 "batch_size=${BATCH_SIZE}" "+num_rollouts=${ROLL_OUTS}"
  "data_source.args.algorithm.prompts_per_rollout=${PROMPTS_PER_ROLLOUT}"
  "sampling.samples_per_prompt=${SAMPLES_PER_PROMPT}" "sampling.num_inference_steps=${STEPS}"
  "sampling.scheduler.num_timesteps=${STEPS}" "sampling.scheduler.num_sde_steps=${SDE_STEPS}"
  'sampling.scheduler.timestep_fraction=[0,0.5]'
  "rollout.config.default_num_inference_steps=${STEPS}"
  reward.backend.config.batch_size=1 stack.micro_batch_size=1 stack.num_updates_per_batch=1
  "+checkpoint_dir=${CKPT_DIR}" "+checkpoint_interval=${CHECKPOINT_INTERVAL}" +checkpoint_mode=lora
)

if [[ -n "$RESUME_LORA" ]]; then
  BASE_CMD+=("+resume_checkpoint_dir=${RESUME_LORA}" +resume_checkpoint_mode=lora)
fi

if [[ "$WATCH" == 1 ]]; then
  RUN_CMD=(python3 scripts/dgx_spark_watch.py --label "$LABEL" --log-dir "$LOG_DIR" --interval "$INTERVAL" -- "${BASE_CMD[@]}")
else
  RUN_CMD=("${BASE_CMD[@]}")
fi

print_cmd() {
  printf '%q ' "${RUN_CMD[@]}"
  printf '\n'
}

if [[ "$PRINT_COMMAND" == 1 ]]; then
  print_cmd
  exit 0
fi

if [[ "$RAY_STOP" == 1 && -x .venv-vllm/bin/ray ]]; then
  .venv-vllm/bin/ray stop --force || true
fi

"${RUN_CMD[@]}"
python3 scripts/dgx_spark_report.py --limit 100
