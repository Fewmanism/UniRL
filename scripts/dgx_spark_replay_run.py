#!/usr/bin/env python3
"""Print or rerun a command captured by dgx_spark_watch.py metadata.

This is a reproducibility helper, not a checkpoint/resume implementation. It lets
us take a watcher run directory and reconstruct the exact wrapped command so the
experiment can be repeated after cleanup or reboot.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any


def latest_run(log_dir: Path) -> Path:
    candidates = [p.parent for p in log_dir.glob("*/metadata.json")]
    if not candidates:
        raise SystemExit(f"No watcher runs found under {log_dir}")
    return max(candidates, key=lambda p: (p / "metadata.json").stat().st_mtime)


def load_metadata(run_dir: Path) -> dict[str, Any]:
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        raise SystemExit(f"Missing metadata.json: {meta_path}")
    return json.loads(meta_path.read_text())


def quote_cmd(command: list[Any]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", help="Watcher run directory. Defaults to latest under --log-dir.")
    parser.add_argument("--log-dir", default="outputs/dgx-spark-observe")
    parser.add_argument("--execute", action="store_true", help="Execute the captured command instead of only printing it")
    parser.add_argument("--watch", action="store_true", help="When executing, wrap the replay with dgx_spark_watch.py")
    parser.add_argument("--label", default=None, help="Label for --watch replay runs")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else latest_run(Path(args.log_dir))
    metadata = load_metadata(run_dir)
    cwd = Path(metadata.get("cwd") or ".").resolve()
    command = metadata.get("command")
    if not isinstance(command, list) or not command:
        raise SystemExit(f"metadata command is not a non-empty list in {run_dir}")

    print(f"# source_run: {run_dir}")
    print(f"cd {shlex.quote(str(cwd))}")
    if args.watch:
        label = args.label or f"replay-{metadata.get('label') or run_dir.name}"
        replay_command = [
            "python3",
            "scripts/dgx_spark_watch.py",
            "--label",
            label,
            "--log-dir",
            args.log_dir,
            "--",
            *[str(part) for part in command],
        ]
    else:
        replay_command = [str(part) for part in command]
    print(quote_cmd(replay_command))

    if not args.execute:
        return 0

    return subprocess.call(replay_command, cwd=str(cwd), env=os.environ.copy())


if __name__ == "__main__":
    raise SystemExit(main())
