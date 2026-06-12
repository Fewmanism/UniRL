#!/usr/bin/env python3
"""Summarize DGX Spark watcher runs.

Reads ``outputs/dgx-spark-observe/*/summary.json`` plus ``command.log`` and prints
a compact table for experiment continuity. The script is intentionally stdlib-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROLLOUT_RE = re.compile(
    r"rollout\s+(?P<rollout>\d+/\d+)\s+"
    r"reward=(?P<reward>[-+0-9.eE]+)\s+"
    r"loss=(?P<loss>[-+0-9.eE]+)\s+"
    r"grad_norm=(?P<grad_norm>[-+0-9.eE]+)\s+"
    r"lr=(?P<lr>[-+0-9.eE]+)"
)


@dataclass
class RunRow:
    path: str
    label: str
    exit_code: int | None
    elapsed_s: float | None
    samples: int | None
    gpu_name: str
    max_gpu_util_pct: float | None
    max_gpu_mem_mib: float | None
    max_temp_c: float | None
    max_power_w: float | None
    rollout: str
    reward: float | None
    loss: float | None
    grad_norm: float | None
    lr: float | None
    command: str


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def parse_rollout(command_log: Path) -> dict[str, Any]:
    if not command_log.exists():
        return {}
    last: dict[str, Any] = {}
    for line in command_log.read_text(errors="replace").splitlines():
        match = ROLLOUT_RE.search(line)
        if not match:
            continue
        last = match.groupdict()
    if not last:
        return {}
    for key in ("reward", "loss", "grad_norm", "lr"):
        try:
            last[key] = float(last[key])
        except Exception:
            last[key] = None
    return last


def row_from_dir(run_dir: Path) -> RunRow | None:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None
    summary = load_json(summary_path)
    metadata = load_json(run_dir / "metadata.json")
    rollout = parse_rollout(run_dir / "command.log")
    gpu0 = (summary.get("gpu") or {}).get("0") or {}
    command = metadata.get("command") or []
    if isinstance(command, list):
        command_str = " ".join(shlex.quote(str(part)) for part in command)
    else:
        command_str = str(command)
    return RunRow(
        path=str(run_dir),
        label=str(summary.get("label") or metadata.get("label") or run_dir.name),
        exit_code=summary.get("exit_code"),
        elapsed_s=summary.get("elapsed_s"),
        samples=summary.get("samples"),
        gpu_name=str(gpu0.get("name") or ""),
        max_gpu_util_pct=gpu0.get("max_utilization_gpu_pct"),
        max_gpu_mem_mib=gpu0.get("max_memory_used_mib"),
        max_temp_c=gpu0.get("max_temperature_c"),
        max_power_w=gpu0.get("max_power_draw_w"),
        rollout=str(rollout.get("rollout") or ""),
        reward=rollout.get("reward"),
        loss=rollout.get("loss"),
        grad_norm=rollout.get("grad_norm"),
        lr=rollout.get("lr"),
        command=command_str,
    )


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def collect_runs(log_dir: Path, limit: int = 20) -> list[dict[str, Any]]:
    rows: list[RunRow] = []
    if log_dir.exists():
        dirs = sorted(
            [p.parent for p in log_dir.glob("*/summary.json")],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for run_dir in dirs[:limit]:
            row = row_from_dir(run_dir)
            if row is not None:
                rows.append(row)
    return [r.__dict__ for r in rows]


def print_table(rows: list[RunRow]) -> None:
    headers = [
        "label",
        "exit",
        "elapsed_s",
        "reward",
        "grad_norm",
        "gpu_util%",
        "gpu_mem_mib",
        "temp_c",
        "power_w",
        "samples",
        "path",
    ]
    table = []
    for r in rows:
        table.append(
            [
                r.label,
                fmt(r.exit_code),
                fmt(r.elapsed_s, 1),
                fmt(r.reward, 4),
                fmt(r.grad_norm, 4),
                fmt(r.max_gpu_util_pct, 1),
                fmt(r.max_gpu_mem_mib, 0),
                fmt(r.max_temp_c, 0),
                fmt(r.max_power_w, 1),
                fmt(r.samples),
                r.path,
            ]
        )
    widths = [len(h) for h in headers]
    for row in table:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in table:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", default="outputs/dgx-spark-observe")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--format", choices=["table", "json", "csv"], default="table")
    parser.add_argument("--show-command", action="store_true", help="Include wrapped command in table output")
    args = parser.parse_args()

    base = Path(args.log_dir)
    rows: list[RunRow] = []
    if base.exists():
        dirs = sorted(
            [p.parent for p in base.glob("*/summary.json")],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for run_dir in dirs[: args.limit]:
            row = row_from_dir(run_dir)
            if row is not None:
                rows.append(row)

    payload = [r.__dict__ for r in rows]
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.format == "csv":
        if payload:
            writer = csv.DictWriter(__import__("sys").stdout, fieldnames=list(payload[0].keys()))
            writer.writeheader()
            writer.writerows(payload)
    else:
        print_table(rows)
        if args.show_command:
            for r in rows:
                print(f"\n== {r.label} ==\ncd {Path(r.path).parents[1]}\n{r.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
