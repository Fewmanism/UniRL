import json
from pathlib import Path
import subprocess

import torch

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    return subprocess.run(args, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def write_adapter(directory: Path, value: float = 1.0):
    directory.mkdir(parents=True)
    torch.save({"layer.lora_A.default.weight": torch.tensor([value])}, directory / "lora_adapter.pt")


def test_lora_manifest_cli_write_and_verify(tmp_path: Path):
    ckpt = tmp_path / "run" / "rollout-000001"
    write_adapter(ckpt)

    write_result = run("python3", "scripts/dgx_spark_lora_manifest.py", "--write", str(ckpt))
    assert write_result.returncode == 0, write_result.stdout
    manifest = json.loads((ckpt / "lora_manifest.json").read_text())
    assert manifest["format"] == "unirl-lora-adapter-v1"
    assert manifest["adapter_file"] == "lora_adapter.pt"

    verify_result = run("python3", "scripts/dgx_spark_lora_manifest.py", "--verify", str(ckpt))
    assert verify_result.returncode == 0, verify_result.stdout
    assert "OK" in verify_result.stdout


def test_lora_manifest_cli_verify_detects_corruption(tmp_path: Path):
    ckpt = tmp_path / "run" / "rollout-000001"
    write_adapter(ckpt)
    assert run("python3", "scripts/dgx_spark_lora_manifest.py", "--write", str(ckpt)).returncode == 0
    (ckpt / "lora_adapter.pt").write_bytes(b"corrupt")

    result = run("python3", "scripts/dgx_spark_lora_manifest.py", "--verify", str(ckpt))

    assert result.returncode != 0
    assert "mismatch" in result.stdout


def test_lora_manifest_cli_write_missing_and_verify_all(tmp_path: Path):
    ckpt1 = tmp_path / "run1" / "rollout-000001"
    ckpt2 = tmp_path / "run2" / "rollout-000002"
    write_adapter(ckpt1, 1.0)
    write_adapter(ckpt2, 2.0)

    result = run("python3", "scripts/dgx_spark_lora_manifest.py", "--root", str(tmp_path), "--write-missing")
    assert result.returncode == 0, result.stdout
    assert (ckpt1 / "lora_manifest.json").exists()
    assert (ckpt2 / "lora_manifest.json").exists()

    verify = run("python3", "scripts/dgx_spark_lora_manifest.py", "--root", str(tmp_path), "--verify-all")
    assert verify.returncode == 0, verify.stdout
    assert "verified: 2" in verify.stdout
