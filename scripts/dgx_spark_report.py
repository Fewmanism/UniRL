#!/usr/bin/env python3
"""Generate a local Markdown experiment report from DGX Spark watcher runs.

The report is intentionally local-only by default. It reads watcher summaries and
`local_runs/index.jsonl`, then writes a human-readable Markdown report under
`local_runs/reports/` plus a `latest.md` copy.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

from dgx_spark_summarize_runs import collect_runs


def read_registry(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        if abs(value) >= 100:
            return f"{value:.1f}"
        return f"{value:.{digits}f}"
    return str(value)


def md_escape(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def short_sha(sha: Any) -> str:
    if not sha:
        return ""
    return str(sha)[:12]


def classify(label: str) -> str:
    label = label.lower()
    if "checkpoint" in label:
        return "checkpoint"
    if "matrix" in label:
        return "matrix"
    if "scale" in label or "standard" in label:
        return "scale"
    if "quick" in label:
        return "quick"
    if "probe" in label or "selftest" in label:
        return "probe"
    return "other"


def registry_by_run_dir(registry: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in registry:
        run_dir = row.get("run_dir") or row.get("summary", {}).get("run_dir")
        if run_dir:
            out[str(run_dir)] = row
    return out


def pick_extreme(rows: list[dict[str, Any]], key: str, *, max_value: bool = True) -> dict[str, Any] | None:
    valid = [r for r in rows if isinstance(r.get(key), (int, float))]
    if not valid:
        return None
    return max(valid, key=lambda r: float(r[key])) if max_value else min(valid, key=lambda r: float(r[key]))


def build_report(
    *,
    rows: list[dict[str, Any]],
    registry: list[dict[str, Any]],
    limit: int,
    title: str,
    log_dir: Path,
    registry_path: Path,
) -> str:
    now = dt.datetime.now().astimezone()
    rows = rows[:limit]
    reg_map = registry_by_run_dir(registry)
    success = sum(1 for r in rows if r.get("exit_code") == 0)
    failed = sum(1 for r in rows if r.get("exit_code") not in (0, None))
    best = pick_extreme([r for r in rows if r.get("exit_code") == 0], "reward")
    slowest = pick_extreme(rows, "elapsed_s")
    hottest = pick_extreme(rows, "max_temp_c")
    power = pick_extreme(rows, "max_power_w")

    latest_registry = registry[-1] if registry else {}
    latest_meta = latest_registry.get("metadata", {}) if isinstance(latest_registry, dict) else {}
    latest_git = latest_meta.get("git", {}) if isinstance(latest_meta, dict) else {}

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"Generated: {now.isoformat(timespec='seconds')}")
    lines.append(f"Log dir: `{log_dir}`")
    lines.append(f"Registry: `{registry_path}`")
    lines.append("")

    lines.append("## Executive summary")
    lines.append("")
    lines.append(f"- Runs shown: {len(rows)}")
    lines.append(f"- Success / failed: {success} / {failed}")
    if latest_git:
        lines.append(
            f"- Latest registry git: `{latest_git.get('branch', '')}` @ `{short_sha(latest_git.get('sha'))}` "
            f"dirty={latest_git.get('dirty')}"
        )
    if best:
        lines.append(f"- Best reward: {fmt(best.get('reward'))} from `{best.get('label')}`")
    if slowest:
        lines.append(f"- Slowest run: {fmt(slowest.get('elapsed_s'), 1)}s from `{slowest.get('label')}`")
    if hottest:
        lines.append(f"- Hottest run: {fmt(hottest.get('max_temp_c'), 1)}C from `{hottest.get('label')}`")
    if power:
        lines.append(f"- Highest power sample: {fmt(power.get('max_power_w'), 1)}W from `{power.get('label')}`")
    lines.append("")

    lines.append("## Run table")
    lines.append("")
    header = ["label", "class", "exit", "elapsed_s", "reward", "grad_norm", "gpu_util%", "temp_c", "power_w", "path"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        vals = [
            r.get("label"),
            classify(str(r.get("label", ""))),
            r.get("exit_code"),
            fmt(r.get("elapsed_s"), 1),
            fmt(r.get("reward")),
            fmt(r.get("grad_norm")),
            fmt(r.get("max_gpu_util_pct"), 1),
            fmt(r.get("max_temp_c"), 1),
            fmt(r.get("max_power_w"), 1),
            f"`{r.get('path')}`",
        ]
        lines.append("| " + " | ".join(md_escape(v) for v in vals) + " |")
    lines.append("")

    failures = [r for r in rows if r.get("exit_code") not in (0, None)]
    lines.append("## Failed runs")
    lines.append("")
    if failures:
        for r in failures:
            lines.append(f"- `{r.get('label')}` exit={r.get('exit_code')} path=`{r.get('path')}`")
    else:
        lines.append("- None in selected window.")
    lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Summary CSV: `{log_dir / 'summary.csv'}`")
    lines.append(f"- Registry JSONL: `{registry_path}`")
    lines.append("- Checkpoints: `local_runs/checkpoints/`")
    lines.append("- Reports: `local_runs/reports/`")
    lines.append("")

    lines.append("## Registry details")
    lines.append("")
    if not registry:
        lines.append("- No registry rows found.")
    else:
        recent = registry[-min(10, len(registry)):]
        for item in recent:
            label = item.get("label")
            meta = item.get("metadata", {})
            git = meta.get("git", {}) if isinstance(meta, dict) else {}
            summary = item.get("summary", {})
            lines.append(
                f"- `{label}` exit={summary.get('exit_code')} sha=`{short_sha(git.get('sha'))}` "
                f"dirty={git.get('dirty')} run_dir=`{item.get('run_dir')}`"
            )
    lines.append("")

    lines.append("## Notes / next actions")
    lines.append("")
    lines.append("- Extend `num_rollouts` beyond smoke values and keep checkpoint cadence enabled.")
    lines.append("- Compare scale10 SGLang/VLLM-Omni timings over repeated runs.")
    lines.append("- Add W&B online/offline reporting once local long-run stability is established.")
    lines.append("- Treat failed setup/debug runs as useful history; keep them in the report window when investigating regressions.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", default="outputs/dgx-spark-observe")
    parser.add_argument("--registry", default="local_runs/index.jsonl")
    parser.add_argument("--out-dir", default="local_runs/reports")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--title", default=None)
    parser.add_argument("--stdout", action="store_true", help="Print instead of writing files")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    registry_path = Path(args.registry)
    out_dir = Path(args.out_dir)
    rows = collect_runs(log_dir, limit=max(1, args.limit))
    registry = read_registry(registry_path)
    title = args.title or f"DGX Spark UniRL Experiment Report - {dt.date.today().isoformat()}"
    report = build_report(
        rows=rows,
        registry=registry,
        limit=max(1, args.limit),
        title=title,
        log_dir=log_dir,
        registry_path=registry_path,
    )

    if args.stdout:
        print(report)
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    date_path = out_dir / f"{dt.date.today().isoformat()}.md"
    latest_path = out_dir / "latest.md"
    date_path.write_text(report + "\n")
    shutil.copyfile(date_path, latest_path)
    print(f"report={date_path}")
    print(f"latest={latest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
