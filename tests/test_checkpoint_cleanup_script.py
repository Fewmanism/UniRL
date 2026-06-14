from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    return subprocess.run(args, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def test_cleanup_policy_reports_full_and_lora_checkpoints(tmp_path: Path):
    full_dir = tmp_path / "full" / "rollout-000001"
    lora_dir = tmp_path / "lora" / "rollout-000001"
    full_dir.mkdir(parents=True)
    lora_dir.mkdir(parents=True)
    (full_dir / "checkpoint.pt").write_bytes(b"full")
    (full_dir / "lora_adapter.pt").write_bytes(b"lora")
    (lora_dir / "lora_adapter.pt").write_bytes(b"lora-only")

    result = run(
        "bash",
        "scripts/dgx_spark_checkpoint_cleanup.sh",
        "--root",
        str(tmp_path),
        "--dry-run",
    )

    assert result.returncode == 0, result.stdout
    assert "full checkpoints: 1" in result.stdout
    assert "lora-only checkpoints: 1" in result.stdout
    assert "delete candidates" in result.stdout
    assert "checkpoint.pt" in result.stdout


def test_cleanup_policy_deletes_full_checkpoint_only_with_yes(tmp_path: Path):
    rollout = tmp_path / "full" / "rollout-000001"
    rollout.mkdir(parents=True)
    checkpoint = rollout / "checkpoint.pt"
    adapter = rollout / "lora_adapter.pt"
    checkpoint.write_bytes(b"full")
    adapter.write_bytes(b"lora")

    result = run(
        "bash",
        "scripts/dgx_spark_checkpoint_cleanup.sh",
        "--root",
        str(tmp_path),
        "--delete-full",
        "--yes",
    )

    assert result.returncode == 0, result.stdout
    assert not checkpoint.exists()
    assert adapter.exists()
