#!/usr/bin/env python3
"""Run a command with lightweight DGX Spark observability.

The watcher records:

- command stdout/stderr to ``command.log``
- periodic host/GPU samples to ``metrics.jsonl``
- a compact final ``summary.json``

It is intentionally dependency-free and safe for minimal virtual environments.
It does not inspect environment variable values, so secrets such as HF_TOKEN are
not written by the watcher itself.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

_TOKEN_RE = re.compile(r"hf_[A-Za-z0-9_\-]+")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def redact(text: str) -> str:
    return _TOKEN_RE.sub("<HF_TOKEN_REDACTED>", text)


def run_text(cmd: list[str], timeout: float = 5.0) -> str | None:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return out.strip()


def read_meminfo() -> dict[str, float]:
    vals: dict[str, float] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, raw = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                vals[f"{key}_mib"] = float(raw.strip().split()[0]) / 1024.0
    except OSError:
        pass
    return vals


def read_loadavg() -> dict[str, float]:
    try:
        one, five, fifteen, *_ = Path("/proc/loadavg").read_text().split()
        return {"load1": float(one), "load5": float(five), "load15": float(fifteen)}
    except OSError:
        return {}


def sample_nvidia_smi() -> list[dict[str, Any]] | None:
    query = ",".join(
        [
            "timestamp",
            "name",
            "index",
            "utilization.gpu",
            "memory.used",
            "memory.total",
            "temperature.gpu",
            "power.draw",
        ]
    )
    out = run_text(
        [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        timeout=5.0,
    )
    if not out:
        return None
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 8:
            continue
        timestamp, name, index, util, mem_used, mem_total, temp, power = parts
        def num(x: str) -> float | None:
            try:
                return float(x)
            except ValueError:
                return None
        gpus.append(
            {
                "timestamp": timestamp,
                "name": name,
                "index": int(float(index)) if num(index) is not None else index,
                "utilization_gpu_pct": num(util),
                "memory_used_mib": num(mem_used),
                "memory_total_mib": num(mem_total),
                "temperature_c": num(temp),
                "power_draw_w": num(power),
            }
        )
    return gpus


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def git_metadata() -> dict[str, Any]:
    """Return compact git state for run registry records."""
    branch = run_text(["git", "branch", "--show-current"], timeout=3.0)
    sha = run_text(["git", "rev-parse", "HEAD"], timeout=3.0)
    status = run_text(["git", "status", "--porcelain"], timeout=3.0)
    return {
        "branch": branch,
        "sha": sha,
        "dirty": bool(status),
    }


def package_versions() -> dict[str, dict[str, str]]:
    """Collect key package versions from known local venvs when present."""
    packages = {
        ".venv-spark": ["torch", "diffusers", "transformers", "ray"],
        ".venv-sglang": ["torch", "sglang", "sglang-kernel", "flash-attn-4", "flashinfer-python"],
        ".venv-vllm": ["torch", "vllm", "vllm-omni", "flashinfer-python"],
    }
    out: dict[str, dict[str, str]] = {}
    for venv, names in packages.items():
        py = Path(venv) / "bin" / "python"
        if not py.exists():
            continue
        code = (
            "import importlib.metadata as md, json, sys; "
            "d={}; "
            "\nfor n in sys.argv[1:]:\n"
            "    try: d[n]=md.version(n)\n"
            "    except Exception: d[n]=None\n"
            "print(json.dumps(d, sort_keys=True))"
        )
        text = run_text([str(py), "-c", code, *names], timeout=10.0)
        if text:
            try:
                out[venv] = json.loads(text)
            except json.JSONDecodeError:
                out[venv] = {"error": text}
    return out


def append_registry(registry_path: Path, record: dict[str, Any]) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("a", buffering=1) as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def stream_output(proc: subprocess.Popen[str], log_path: Path) -> None:
    assert proc.stdout is not None
    with log_path.open("a", buffering=1) as fh:
        for line in proc.stdout:
            text = redact(line)
            fh.write(text)
            print(text, end="")


def monitor(metrics_path: Path, interval: float, stop: threading.Event) -> None:
    with metrics_path.open("a", buffering=1) as fh:
        while not stop.is_set():
            sample: dict[str, Any] = {"ts": utc_now()}
            sample.update(read_loadavg())
            sample.update(read_meminfo())
            gpus = sample_nvidia_smi()
            if gpus is not None:
                sample["gpus"] = gpus
            fh.write(json.dumps(sample, sort_keys=True) + "\n")
            stop.wait(interval)


def summarize_metrics(metrics_path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {"samples": 0, "gpu": {}}
    if not metrics_path.exists():
        return summary
    gpu_max: dict[str, dict[str, float]] = {}
    for line in metrics_path.read_text().splitlines():
        if not line.strip():
            continue
        summary["samples"] += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        for gpu in row.get("gpus", []) or []:
            idx = str(gpu.get("index", "unknown"))
            cur = gpu_max.setdefault(idx, {})
            for src, dst in [
                ("utilization_gpu_pct", "max_utilization_gpu_pct"),
                ("memory_used_mib", "max_memory_used_mib"),
                ("temperature_c", "max_temperature_c"),
                ("power_draw_w", "max_power_draw_w"),
            ]:
                val = gpu.get(src)
                if isinstance(val, (int, float)):
                    cur[dst] = max(float(val), float(cur.get(dst, val)))
            if gpu.get("name"):
                cur["name"] = gpu["name"]
    summary["gpu"] = gpu_max
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="run", help="Human-readable run label used in the directory name")
    parser.add_argument("--log-dir", default="outputs/dgx-spark-observe", help="Base directory for logs")
    parser.add_argument("--interval", type=float, default=5.0, help="Metric sampling interval in seconds")
    parser.add_argument("--registry", default="local_runs/index.jsonl", help="Append run records here; pass '' to disable")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run after --")
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("missing command after --")

    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.label).strip("-") or "run"
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.log_dir) / f"{stamp}-{safe_label}"
    run_dir.mkdir(parents=True, exist_ok=False)

    command_log = run_dir / "command.log"
    metrics_jsonl = run_dir / "metrics.jsonl"
    summary_json = run_dir / "summary.json"
    metadata = {
        "label": args.label,
        "started_at": utc_now(),
        "cwd": os.getcwd(),
        "command": command,
        "interval_s": args.interval,
        "git": git_metadata(),
        "package_versions": package_versions(),
    }
    write_json(run_dir / "metadata.json", metadata)

    stop = threading.Event()
    mon = threading.Thread(target=monitor, args=(metrics_jsonl, args.interval, stop), daemon=True)
    mon.start()

    start = time.monotonic()
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    reader = threading.Thread(target=stream_output, args=(proc, command_log), daemon=True)
    reader.start()

    def terminate(signum: int, _frame: Any) -> None:
        try:
            os.killpg(proc.pid, signum)
        except ProcessLookupError:
            pass

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)

    rc = proc.wait()
    reader.join(timeout=10)
    stop.set()
    mon.join(timeout=max(1.0, args.interval + 1.0))
    elapsed = time.monotonic() - start

    summary = summarize_metrics(metrics_jsonl)
    summary.update(
        {
            "label": args.label,
            "started_at": metadata["started_at"],
            "finished_at": utc_now(),
            "elapsed_s": elapsed,
            "exit_code": rc,
            "run_dir": str(run_dir),
            "command_log": str(command_log),
            "metrics_jsonl": str(metrics_jsonl),
        }
    )
    write_json(summary_json, summary)
    if args.registry:
        append_registry(
            Path(args.registry),
            {
                "schema_version": 1,
                "label": args.label,
                "run_dir": str(run_dir),
                "metadata": metadata,
                "summary": summary,
            },
        )
    print(f"\n[dgx_spark_watch] summary: {summary_json}")
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
