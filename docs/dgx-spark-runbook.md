# DGX Spark UniRL local runbook

This runbook is for the local DGX Spark / GB10 / linux-aarch64 fork branch.
It assumes the repo is checked out at `/home/spark01/Projects/UniRL` and that
`.venv-spark` / `.venv-vllm` have already been created.

## Recommended default

Use VLLM-Omni scale10 with LoRA-only checkpoints.

```bash
scripts/dgx_spark_clean.sh --kill-orphans --yes
scripts/dgx_spark_run_vllmomni_lora.sh --rollouts 100 --checkpoint-interval 20 --watch
```

Validated boundary:

- scale10 r20/r50/r100 all completed.
- LoRA-only checkpoint and LoRA-only resume completed.
- r100 stayed below 80C on the tested host.
- LoRA adapters are about 37.7 MB each.

## Lightweight resume

Resume from a saved `rollout-*` directory containing `lora_adapter.pt`:

```bash
scripts/dgx_spark_run_vllmomni_lora.sh \
  --rollouts 20 \
  --checkpoint-interval 10 \
  --resume-lora local_runs/checkpoints/<run>/rollout-000100 \
  --watch
```

LoRA resume loads adapter weights only and starts optimizer/scheduler state fresh.
It verifies `lora_manifest.json` by default when the manifest exists. Old adapters
without a manifest log a warning and still load; use the manifest tool below to
backfill them.

## LoRA manifest audit

Write missing manifests for existing adapters:

```bash
scripts/dgx_spark_lora_manifest.py --root local_runs/checkpoints --write-missing
```

Verify every LoRA checkpoint:

```bash
scripts/dgx_spark_lora_manifest.py --root local_runs/checkpoints --verify-all
```

A valid `lora_manifest.json` records adapter file size, SHA256, tensor count, and
sorted tensor keys.

## Validation gate

Run static/local checks after code changes:

```bash
scripts/dgx_spark_validate.sh --static-only
```

This runs checkpoint/manifest tests, shell syntax checks, cleanup dry-run, and
report dry-run. Add `--gpu-smoke` only when a short GPU run is desired.

## Reports

Refresh and inspect reports:

```bash
scripts/dgx_spark_report.py --limit 100
sed -n '1,80p' local_runs/reports/latest.md
scripts/dgx_spark_summarize_runs.py --limit 20
```

## Disk policy

Inspect checkpoint usage:

```bash
scripts/dgx_spark_checkpoint_cleanup.sh --dry-run
```

Keep one known-good trainside full checkpoint for exact full-resume reference and
delete other full `checkpoint.pt` files when disk pressure grows:

```bash
scripts/dgx_spark_checkpoint_cleanup.sh \
  --delete-full \
  --keep '*trainside-standard-r20/rollout-000020/checkpoint.pt' \
  --yes
```

This keeps LoRA adapters and manifests.

## Scale20 boundary

Scale20 is useful for short high-reward probes, not default long runs.

Validated:

- scale20 r5 OK
- scale20 r10 OK
- scale20 r20 hit Ray OOM

Use only for short experiments:

```bash
scripts/dgx_spark_run_vllmomni_lora.sh --scale 20 --rollouts 10 --checkpoint-interval 5 --watch
```

## Known holds

- SGLang multi-rollout is on hold due to Ray OOM / lifecycle issues.
- W&B online smoke is blocked until `wandb login` or `WANDB_API_KEY` / project
  configuration is available.
