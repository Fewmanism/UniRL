#!/usr/bin/env bash
# Safe cleanup helper for local DGX Spark UniRL runs.
# Defaults are conservative: stop Ray and remove Python bytecode only.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/dgx_spark_clean.sh [options]

Default action:
  - ray stop --force
  - remove __pycache__ / *.pyc under the repo
  - print candidate old output directories, but do not delete outputs

Options:
  --dry-run               Print what would be done
  --no-ray                Do not stop Ray
  --kill-orphans          Kill leftover UniRL/SGLang/vLLM worker processes after Ray stop
  --outputs-days N        Delete outputs/dgx-spark-observe run dirs older than N days
  --hydra-days N          Delete outputs/YYYY-MM-DD Hydra run dirs older than N days
  --yes                   Required for deleting outputs or killing orphans
  -h, --help              Show this help

Notes:
  - Hugging Face cache is never deleted by this script.
  - .env and .env.keys are never touched.
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DRY_RUN=0
RAY=1
KILL_ORPHANS=0
OUTPUTS_DAYS=""
HYDRA_DAYS=""
YES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --no-ray) RAY=0; shift ;;
    --kill-orphans) KILL_ORPHANS=1; shift ;;
    --outputs-days) OUTPUTS_DAYS="${2:?missing --outputs-days value}"; shift 2 ;;
    --hydra-days) HYDRA_DAYS="${2:?missing --hydra-days value}"; shift 2 ;;
    --yes) YES=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

run() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
  if [[ "$DRY_RUN" != 1 ]]; then
    "$@"
  fi
}

require_yes() {
  local action="$1"
  if [[ "$YES" != 1 ]]; then
    echo "Refusing to $action without --yes" >&2
    exit 2
  fi
}

if [[ "$RAY" == 1 ]]; then
  stopped=0
  for ray_bin in .venv-spark/bin/ray .venv-sglang/bin/ray .venv-vllm/bin/ray ray; do
    if command -v "$ray_bin" >/dev/null 2>&1; then
      run "$ray_bin" stop --force || true
      stopped=1
    fi
  done
  if [[ "$stopped" == 0 ]]; then
    echo 'ray command not found in known venvs; skipping Ray stop'
  fi
fi

printf '\n== remove Python bytecode ==\n'
if [[ "$DRY_RUN" == 1 ]]; then
  find . \
    \( -path './.git' -o -path './.venv-*' -o -path './outputs' \) -prune -o \
    \( -type d -name __pycache__ -o -type f -name '*.pyc' \) -print | sed -n '1,200p'
else
  find . \
    \( -path './.git' -o -path './.venv-*' -o -path './outputs' \) -prune -o \
    -type d -name __pycache__ -print0 | xargs -0 -r rm -rf
  find . \
    \( -path './.git' -o -path './.venv-*' -o -path './outputs' \) -prune -o \
    -type f -name '*.pyc' -print0 | xargs -0 -r rm -f
  echo 'bytecode removed'
fi

find_orphan_pids() {
  python3 - <<'PY'
import os
import subprocess

# Avoid killing this cleanup process, its parents, and Hermes shell wrappers whose
# command line may contain words like "sglang" only because they launched us.
self_pid = os.getpid()
ancestors = {self_pid}
pid = self_pid
while True:
    try:
        with open(f"/proc/{pid}/stat") as fh:
            ppid = int(fh.read().split()[3])
    except Exception:
        break
    if ppid <= 1 or ppid in ancestors:
        break
    ancestors.add(ppid)
    pid = ppid

out = subprocess.check_output(["ps", "-eo", "pid=,comm=,args="], text=True, errors="replace")
for line in out.splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        pid_s, comm, args = line.split(None, 2)
        proc_pid = int(pid_s)
    except ValueError:
        continue
    if proc_pid in ancestors:
        continue
    if "dgx_spark_clean.sh" in args or "hermes" in args:
        continue
    kill = False
    if "unirl.train_diffusion" in args:
        kill = True
    elif "sgl_diffusion::scheduler" in args:
        kill = True
    elif "raylet" in args or "gcs_server" in args:
        kill = True
    elif "/.venv-sglang/bin/python" in args and "resource_tracker" in args:
        kill = True
    elif "/.venv-vllm/bin/python" in args and "resource_tracker" in args:
        kill = True
    if kill:
        print(proc_pid)
PY
}

if [[ "$KILL_ORPHANS" == 1 ]]; then
  require_yes 'kill orphan UniRL/SGLang/vLLM processes'
  printf '\n== kill orphan training/engine processes ==\n'
  mapfile -t pids < <(find_orphan_pids)
  if [[ ${#pids[@]} -eq 0 ]]; then
    echo 'none'
  else
    printf 'pids: %s\n' "${pids[*]}"
    if [[ "$DRY_RUN" != 1 ]]; then
      kill "${pids[@]}" || true
      sleep 3
      mapfile -t still < <(find_orphan_pids)
      [[ ${#still[@]} -eq 0 ]] || kill -9 "${still[@]}" || true
    fi
  fi
fi

if [[ -n "$OUTPUTS_DAYS" ]]; then
  require_yes "delete watcher outputs older than $OUTPUTS_DAYS days"
  printf '\n== delete old watcher outputs ==\n'
  if [[ -d outputs/dgx-spark-observe ]]; then
    if [[ "$DRY_RUN" == 1 ]]; then
      find outputs/dgx-spark-observe -mindepth 1 -maxdepth 1 -type d -mtime +"$OUTPUTS_DAYS" -print
    else
      find outputs/dgx-spark-observe -mindepth 1 -maxdepth 1 -type d -mtime +"$OUTPUTS_DAYS" -print -exec rm -rf {} +
    fi
  fi
fi

if [[ -n "$HYDRA_DAYS" ]]; then
  require_yes "delete Hydra outputs older than $HYDRA_DAYS days"
  printf '\n== delete old Hydra outputs ==\n'
  if [[ -d outputs ]]; then
    if [[ "$DRY_RUN" == 1 ]]; then
      find outputs -mindepth 2 -maxdepth 2 -type d -path 'outputs/20??-??-??/*' -mtime +"$HYDRA_DAYS" -print
    else
      find outputs -mindepth 2 -maxdepth 2 -type d -path 'outputs/20??-??-??/*' -mtime +"$HYDRA_DAYS" -print -exec rm -rf {} +
    fi
  fi
fi

printf '\n== remaining candidate processes ==\n'
pgrep -af 'raylet|gcs_server|unirl.train_diffusion|sglang|vllm' | sed -n '1,80p' || echo 'none'

printf '\n== git status ==\n'
git status --short --branch || true
