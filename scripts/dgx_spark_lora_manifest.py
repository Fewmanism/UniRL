#!/usr/bin/env python3
"""Create and verify LoRA adapter manifests for DGX Spark checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unirl.train.fsdp_utils import verify_lora_manifest, write_lora_manifest  # noqa: E402


def lora_dirs(root: Path) -> list[Path]:
    return sorted({p.parent for p in root.rglob("lora_adapter.pt")})


def write_one(path: Path, *, overwrite: bool = True) -> None:
    adapter = path / "lora_adapter.pt"
    if not adapter.exists():
        raise FileNotFoundError(f"missing lora_adapter.pt: {adapter}")
    manifest = path / "lora_manifest.json"
    if manifest.exists() and not overwrite:
        print(f"SKIP {path} manifest exists")
        return
    state = torch.load(adapter, map_location="cpu")
    write_lora_manifest(path, state)
    print(f"WROTE {manifest}")


def verify_one(path: Path) -> None:
    verify_lora_manifest(path)
    print(f"OK {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path, help="write manifest for one checkpoint directory")
    parser.add_argument("--verify", type=Path, help="verify one checkpoint directory")
    parser.add_argument("--root", type=Path, default=Path("local_runs/checkpoints"), help="checkpoint root")
    parser.add_argument("--write-missing", action="store_true", help="write missing manifests under --root")
    parser.add_argument("--verify-all", action="store_true", help="verify all manifests under --root")
    parser.add_argument("--overwrite", action="store_true", help="rewrite existing manifests when used with --write-missing")
    args = parser.parse_args(argv)

    actions = [args.write is not None, args.verify is not None, args.write_missing, args.verify_all]
    if sum(actions) != 1:
        parser.error("choose exactly one of --write, --verify, --write-missing, --verify-all")

    try:
        if args.write is not None:
            write_one(args.write, overwrite=True)
            return 0
        if args.verify is not None:
            verify_one(args.verify)
            return 0
        if args.write_missing:
            count = 0
            for d in lora_dirs(args.root):
                if args.overwrite or not (d / "lora_manifest.json").exists():
                    write_one(d, overwrite=True)
                    count += 1
            print(f"written: {count}")
            return 0
        if args.verify_all:
            count = 0
            for d in lora_dirs(args.root):
                verify_one(d)
                count += 1
            print(f"verified: {count}")
            return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
