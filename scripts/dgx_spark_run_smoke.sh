#!/usr/bin/env bash
# DGX Spark local smoke runner for this fork.
#
# Runs known-good single-GPU UniRL smoke profiles on linux-aarch64 / GB10 / CUDA 13.
# Secrets are loaded through dotenvx when .env exists, and command output can be
# wrapped by scripts/dgx_spark_watch.py for local observability.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/dgx_spark_run_smoke.sh [options]

Options:
  --engine ENGINE     trainside | sglang | vllmomni | all (default: trainside)
  --profile PROFILE   quick | scale4 | scale10 | standard (default: quick)
  --watch             Wrap the run with scripts/dgx_spark_watch.py
  --label LABEL       Label used by watcher (default: dgx-ENGINE-PROFILE)
  --log-dir DIR       Watcher output dir (default: outputs/dgx-spark-observe)
  --interval SECONDS  Watcher sample interval (default: 5)
  --no-ray-stop       Do not run 'ray stop --force' before each engine
  --dry-run           Print commands but do not execute
  -h, --help          Show this help

Profiles:
  quick:
    batch_size=2, prompts_per_rollout=2, samples_per_prompt=2,
    num_inference_steps=2, num_sde_steps=1, timestep_fraction=[0,1]

  scale4:
    batch_size=2, prompts_per_rollout=2, samples_per_prompt=2,
    num_inference_steps=4, num_sde_steps=1, timestep_fraction=[0,1]

  scale10:
    batch_size=2, prompts_per_rollout=2, samples_per_prompt=2,
    num_inference_steps=10, num_sde_steps=3, timestep_fraction=[0,0.5]

  standard:
    trainside: batch_size=16, prompts_per_rollout=16,
    samples_per_prompt=2, num_inference_steps=10, num_sde_steps=3,
    timestep_fraction=[0,0.5]
    sglang/vllmomni currently map to scale10.

Examples:
  scripts/dgx_spark_run_smoke.sh --engine trainside --watch
  scripts/dgx_spark_run_smoke.sh --engine sglang --profile scale4 --watch
  scripts/dgx_spark_run_smoke.sh --engine vllmomni --profile scale10 --watch
  scripts/dgx_spark_run_smoke.sh --engine all --profile quick --watch
EOF
}

ENGINE=trainside
PROFILE=quick
WATCH=0
LABEL=""
LOG_DIR=outputs/dgx-spark-observe
INTERVAL=5
RAY_STOP=1
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --engine) ENGINE="${2:?missing --engine value}"; shift 2 ;;
    --profile) PROFILE="${2:?missing --profile value}"; shift 2 ;;
    --watch) WATCH=1; shift ;;
    --label) LABEL="${2:?missing --label value}"; shift 2 ;;
    --log-dir) LOG_DIR="${2:?missing --log-dir value}"; shift 2 ;;
    --interval) INTERVAL="${2:?missing --interval value}"; shift 2 ;;
    --no-ray-stop) RAY_STOP=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$ENGINE" in trainside|sglang|vllmomni|all) ;; *) echo "Invalid --engine: $ENGINE" >&2; exit 2 ;; esac
case "$PROFILE" in quick|scale4|scale10|standard) ;; *) echo "Invalid --profile: $PROFILE" >&2; exit 2 ;; esac

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DOTENVX="${DOTENVX:-}"
if [[ -z "$DOTENVX" ]]; then
  if command -v dotenvx >/dev/null 2>&1; then
    DOTENVX=dotenvx
  elif [[ -x "$HOME/.local/bin/npx" ]]; then
    DOTENVX="$HOME/.local/bin/npx --yes @dotenvx/dotenvx"
  elif command -v npx >/dev/null 2>&1; then
    DOTENVX="npx --yes @dotenvx/dotenvx"
  fi
fi

run_cmd() {
  local label="$1"; shift
  local -a cmd=("$@")
  echo "== $label =="
  printf 'cwd: %s\n' "$ROOT_DIR"
  printf 'cmd:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  if [[ "$DRY_RUN" == 1 ]]; then
    return 0
  fi
  if [[ "$RAY_STOP" == 1 ]]; then
    for ray_bin in .venv-spark/bin/ray .venv-sglang/bin/ray .venv-vllm/bin/ray ray; do
      if command -v "$ray_bin" >/dev/null 2>&1; then
        "$ray_bin" stop --force >>/tmp/unirl-dgx-spark-ray-stop.log 2>&1 || true
      fi
    done
  fi
  if [[ "$WATCH" == 1 ]]; then
    local run_label="${LABEL:-$label}"
    python3 scripts/dgx_spark_watch.py --label "$run_label" --log-dir "$LOG_DIR" --interval "$INTERVAL" -- "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
}

with_env() {
  local venv="$1"; shift
  local -a inner=("$@")
  local -a script=(bash -lc "source '$venv/bin/activate'; export REPORT_TO_WANDB=false HF_HUB_DISABLE_TELEMETRY=1 HYDRA_FULL_ERROR=1 PRETRAINED_MODEL=stabilityai/stable-diffusion-3.5-medium; exec \"\$@\"" _ "${inner[@]}")
  if [[ -f .env && -n "$DOTENVX" ]]; then
    # shellcheck disable=SC2206
    local -a dx=( $DOTENVX )
    printf '%s\0' "${dx[@]}" run -f .env -- "${script[@]}"
  else
    printf '%s\0' "${script[@]}"
  fi
}

materialize_cmd() {
  local -n out_ref=$1
  shift
  mapfile -d '' -t out_ref < <(with_env "$@")
}

add_profile_overrides() {
  local -n arr=$1
  local engine="$2"
  local profile="$3"
  local effective="$profile"
  if [[ "$profile" == standard && "$engine" != trainside ]]; then
    effective=scale10
  fi
  case "$effective" in
    quick)
      arr+=(batch_size=2 data_source.args.algorithm.prompts_per_rollout=2 sampling.num_inference_steps=2 sampling.scheduler.num_timesteps=2 sampling.scheduler.num_sde_steps=1 'sampling.scheduler.timestep_fraction=[0,1]')
      ;;
    scale4)
      arr+=(batch_size=2 data_source.args.algorithm.prompts_per_rollout=2 sampling.num_inference_steps=4 sampling.scheduler.num_timesteps=4 sampling.scheduler.num_sde_steps=1 'sampling.scheduler.timestep_fraction=[0,1]')
      ;;
    scale10)
      arr+=(batch_size=2 data_source.args.algorithm.prompts_per_rollout=2 sampling.num_inference_steps=10 sampling.scheduler.num_timesteps=10 sampling.scheduler.num_sde_steps=3 'sampling.scheduler.timestep_fraction=[0,0.5]')
      ;;
    standard)
      arr+=(batch_size=16 data_source.args.algorithm.prompts_per_rollout=16 sampling.num_inference_steps=10 sampling.scheduler.num_timesteps=10 sampling.scheduler.num_sde_steps=3 'sampling.scheduler.timestep_fraction=[0,0.5]')
      ;;
  esac
}

trainside_cmd() {
  local -a common=(python -m unirl.train_diffusion --config-name=diffusion/sd3_trainside num_devices=1 +devices_per_node=1 sampling.samples_per_prompt=2 rollout.forward_batch_size=1 reward.backend.config.batch_size=1 stack.micro_batch_size=1 stack.num_updates_per_batch=1 +num_rollouts=1)
  add_profile_overrides common trainside "$PROFILE"
  local -a cmd
  materialize_cmd cmd .venv-spark "${common[@]}"
  run_cmd "dgx-trainside-$PROFILE" "${cmd[@]}"
}

sglang_cmd() {
  local -a common=(python -m unirl.train_diffusion --config-name=diffusion/sd3_sglang_replay_colocate num_devices=1 +devices_per_node=1 num_rollouts=1 sampling.samples_per_prompt=2 rollout.config.forward_batch_size=1 rollout.config.num_gpus=1 rollout.config.tp_size=1 reward.backend.config.batch_size=1 stack.micro_batch_size=1 stack.num_updates_per_batch=1)
  add_profile_overrides common sglang "$PROFILE"
  local -a cmd
  materialize_cmd cmd .venv-sglang "${common[@]}"
  run_cmd "dgx-sglang-$PROFILE" "${cmd[@]}"
}

vllmomni_cmd() {
  local -a common=(python -m unirl.train_diffusion --config-name=diffusion/sd3_vllmomni num_devices=1 +devices_per_node=1 +num_rollouts=1 sampling.samples_per_prompt=2 reward.backend.config.batch_size=1 stack.micro_batch_size=1 stack.num_updates_per_batch=1)
  add_profile_overrides common vllmomni "$PROFILE"
  case "$PROFILE" in
    quick) common+=(rollout.config.default_num_inference_steps=2) ;;
    scale4) common+=(rollout.config.default_num_inference_steps=4) ;;
    scale10|standard) common+=(rollout.config.default_num_inference_steps=10) ;;
  esac
  local -a cmd
  materialize_cmd cmd .venv-vllm "${common[@]}"
  run_cmd "dgx-vllmomni-$PROFILE" "${cmd[@]}"
}

case "$ENGINE" in
  trainside) trainside_cmd ;;
  sglang) sglang_cmd ;;
  vllmomni) vllmomni_cmd ;;
  all)
    trainside_cmd
    sglang_cmd
    vllmomni_cmd
    ;;
esac
