from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    return subprocess.run(args, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def test_vllmomni_lora_script_renders_scale10_command_without_execute():
    result = run(
        "bash",
        "scripts/dgx_spark_run_vllmomni_lora.sh",
        "--rollouts",
        "7",
        "--checkpoint-interval",
        "3",
        "--print-command",
    )

    assert result.returncode == 0, result.stdout
    assert "--config-name=diffusion/sd3_vllmomni" in result.stdout
    assert "+num_rollouts=7" in result.stdout
    assert "sampling.num_inference_steps=10" in result.stdout
    assert "+checkpoint_interval=3" in result.stdout
    assert "+checkpoint_mode=lora" in result.stdout


def test_vllmomni_lora_script_renders_scale20_and_resume_lora():
    result = run(
        "bash",
        "scripts/dgx_spark_run_vllmomni_lora.sh",
        "--scale",
        "20",
        "--rollouts",
        "2",
        "--resume-lora",
        "local_runs/checkpoints/example/rollout-000100",
        "--print-command",
    )

    assert result.returncode == 0, result.stdout
    assert "sampling.num_inference_steps=20" in result.stdout
    assert "sampling.scheduler.num_sde_steps=5" in result.stdout
    assert "+resume_checkpoint_dir=local_runs/checkpoints/example/rollout-000100" in result.stdout
    assert "+resume_checkpoint_mode=lora" in result.stdout


def test_validate_script_can_run_static_checks_only():
    result = run("bash", "scripts/dgx_spark_validate.sh", "--static-only")

    assert result.returncode == 0, result.stdout
    assert "checkpoint tests" in result.stdout
    assert "clean dry-run" in result.stdout
    assert "report dry-run" in result.stdout
