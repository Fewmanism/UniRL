#!/usr/bin/env bash
# Print a compact local health report for this DGX Spark UniRL fork.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

json_get() {
  python3 - "$1" "$2" <<'PY'
import json, sys
path, key = sys.argv[1], sys.argv[2]
try:
    data = json.load(open(path))
except Exception as e:
    print(f"<error:{e}>")
    raise SystemExit(0)
cur = data
for part in key.split('.'):
    if isinstance(cur, dict):
        cur = cur.get(part)
    else:
        cur = None
        break
print(cur if cur is not None else "")
PY
}

section() { printf '\n== %s ==\n' "$1"; }

section repo
printf 'root: %s\n' "$ROOT_DIR"
git status --short --branch || true
git log --oneline -3 || true

section host
uname -a
printf 'python3: '; python3 --version
printf 'date: '; date -Is

section gpu
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,index,driver_version,memory.used,memory.total,temperature.gpu,power.draw,utilization.gpu --format=csv,noheader,nounits || true
else
  echo 'nvidia-smi: missing'
fi

section venvs
for venv in .venv-spark .venv-sglang .venv-vllm; do
  if [[ -x "$venv/bin/python" ]]; then
    size=$(du -sh "$venv" 2>/dev/null | awk '{print $1}')
    printf '%s: present size=%s python=%s\n' "$venv" "${size:-?}" "$($venv/bin/python -V 2>&1)"
  else
    printf '%s: missing\n' "$venv"
  fi
done

section python-packages
for pair in \
  '.venv-spark torch diffusers transformers ray' \
  '.venv-sglang torch sglang sglang-kernel flash-attn-4 flashinfer-python' \
  '.venv-vllm torch vllm vllm-omni flashinfer-python'; do
  read -r venv pkgs <<<"$pair"
  [[ -x "$venv/bin/python" ]] || continue
  echo "-- $venv --"
  "$venv/bin/python" - "$pkgs" <<'PY'
import importlib.metadata as md, sys
for name in sys.argv[1].split():
    try:
        print(f"{name}=={md.version(name)}")
    except Exception as e:
        print(f"{name}: missing ({type(e).__name__})")
PY
done

section secrets
if [[ -f .env ]]; then
  echo '.env: present'
  if grep -q '^HF_TOKEN=encrypted:' .env 2>/dev/null; then
    echo 'HF_TOKEN: encrypted dotenvx entry present'
  elif grep -q '^HF_TOKEN=' .env 2>/dev/null; then
    echo 'HF_TOKEN: plaintext entry present (consider dotenvx encrypt)'
  else
    echo 'HF_TOKEN: no .env entry found'
  fi
else
  echo '.env: missing'
fi
[[ -f .env.keys ]] && echo '.env.keys: present (local secret key; do not commit)' || echo '.env.keys: missing'

section storage
for path in . outputs outputs/dgx-spark-observe "$HOME/.cache/huggingface"; do
  [[ -e "$path" ]] || continue
  du -sh "$path" 2>/dev/null || true
done
df -h .

section ray-processes
pgrep -af 'raylet|gcs_server|unirl.train_diffusion|sglang|vllm' | sed -n '1,80p' || echo 'none'

section recent-watch-runs
if [[ -d outputs/dgx-spark-observe ]]; then
  while IFS= read -r summary; do
    dir=$(dirname "$summary")
    label=$(json_get "$summary" label)
    code=$(json_get "$summary" exit_code)
    elapsed=$(json_get "$summary" elapsed_s)
    printf '%s label=%s exit=%s elapsed=%s\n' "$dir" "$label" "$code" "$elapsed"
  done < <(find outputs/dgx-spark-observe -maxdepth 2 -name summary.json -printf '%T@ %p\n' | sort -nr | head -10 | cut -d' ' -f2-)
else
  echo 'no outputs/dgx-spark-observe directory'
fi
