# DGX Spark / GB10 / linux-aarch64 bring-up

This page tracks the supported path for running UniRL on DGX Spark-class hosts:

- CPU/OS architecture: `aarch64`
- GPU: NVIDIA GB10
- CUDA user stack: 13.x
- glibc: 2.39+
- Python: 3.12 aarch64

The CUDA driver/runtime can be present while Python packages and native CUDA
extensions are still the blocker. Treat the stack as layers and validate them in
order.

## Current support boundary

The `spark` extra is the DGX Spark smoke-test stack. It deliberately does not
install rollout engines (`sglang`, `vllm`, `vllm-omni`, `flash-attn`,
`sglang-kernel`) because those packages frequently lag linux-aarch64/GB10/CUDA 13
wheel availability.

Expected first milestone:

1. PyTorch CUDA imports and sees GB10.
2. Diffusers/Transformers/PEFT/Ray import.
3. UniRL core imports.
4. Hydra configs compose.

This milestone has been verified on DGX Spark with Python 3.12.3,
glibc 2.39, CUDA 13.0 user wheels, `torch==2.12.0+cu130`, and GB10 reporting
`sm_121`. Only after this layer passes should we try the engine extras.

## Install

Use a fresh virtual environment. `uv` is preferred because this project declares
the CUDA torch indexes in `pyproject.toml`.

```bash
python3 -m venv .venv-spark
source .venv-spark/bin/activate
python -m pip install -U pip uv
uv pip install -e ".[spark,train,infer,dev]"
```

If using `pip` directly, pass the CUDA 13 PyTorch index explicitly:

```bash
python3 -m venv .venv-spark
source .venv-spark/bin/activate
python -m pip install -U pip
python -m pip install -e ".[spark,train,infer,dev]" \
  --extra-index-url https://download.pytorch.org/whl/cu130
```

## Validate layers

```bash
source .venv-spark/bin/activate
python scripts/dgx_spark_probe.py
python -m unirl.train_diffusion --config-name=diffusion/sd3_trainside --cfg job --resolve
```

A passing DGX Spark smoke test should report `torch.cuda.is_available=true`,
`torch.version.cuda=13.0`, `torch.cuda.device=NVIDIA GB10`, and a successful CUDA
tensor smoke test. The probe exits non-zero if the required layer fails. It also
prints optional failures for `sglang`, `vllm`, `flash_attn`, and
`sglang_kernel`; those are expected until their linux-aarch64 CUDA 13 wheels (or a
working local source-build recipe) are available.

## Run one tiny end-to-end diffusion rollout

The default SD3.5 checkpoint is gated on Hugging Face. For an ungated functional
smoke run, override it with a tiny random SD3 repo and shrink the batch/step
counts:

```bash
source .venv-spark/bin/activate
export PRETRAINED_MODEL=optimum-internal-testing/tiny-random-stable-diffusion-3
export REPORT_TO_WANDB=false
python -m unirl.train_diffusion --config-name=diffusion/sd3_trainside \
  num_devices=1 +devices_per_node=1 batch_size=1 \
  data_source.args.algorithm.prompts_per_rollout=1 \
  sampling.samples_per_prompt=1 sampling.num_inference_steps=1 \
  sampling.scheduler.num_timesteps=1 sampling.scheduler.num_sde_steps=0 \
  rollout.forward_batch_size=1 reward.backend.config.batch_size=1 \
  stack.num_updates_per_batch=1 +num_rollouts=1
```

This exercises Ray worker setup, dataset loading, model component loading,
trainside rollout, reward attachment, and the train loop without requiring a gated
or large production checkpoint. On DGX Spark this completed one rollout with
`unirl_tiny_sd3_status=0`; with a single sample it may log a PyTorch warning about
reward standard deviation degrees of freedom and skip an optimizer step when no
micro-batch reports backward.

## Real SD3.5 trainside smoke run

The gated SD3.5 checkpoint was validated on DGX Spark using a read-only Hugging
Face token with access to `stabilityai/stable-diffusion-3.5-medium`:

```bash
source .venv-spark/bin/activate
export PRETRAINED_MODEL=stabilityai/stable-diffusion-3.5-medium
export REPORT_TO_WANDB=false
python -m unirl.train_diffusion --config-name=diffusion/sd3_trainside \
  num_devices=1 +devices_per_node=1 batch_size=2 \
  data_source.args.algorithm.prompts_per_rollout=2 \
  sampling.samples_per_prompt=2 sampling.num_inference_steps=2 \
  sampling.scheduler.num_timesteps=2 sampling.scheduler.num_sde_steps=1 \
  sampling.scheduler.timestep_fraction=[0,1] \
  rollout.forward_batch_size=1 reward.backend.config.batch_size=1 \
  stack.micro_batch_size=1 stack.num_updates_per_batch=1 +num_rollouts=1
```

This reaches `rollout 1/1` and reports a non-zero `grad_norm`. On the tested GB10
host, the same trainside path also completed with `batch_size=16`,
`prompts_per_rollout=16`, `samples_per_prompt=2`, `num_inference_steps=10`, and
`num_sde_steps=3`.

Two scheduler details matter for reduced tests:

- `num_sde_steps=0` is not a valid FlowGRPO optimizer-step smoke test because
  FlowGRPO expects an `sde_logp` anchor.
- `num_inference_steps=1` with `num_sde_steps=1` can fail because the scheduler
  timestep pool is empty. Use at least the 2-step settings above.

## Engine triage order and validated engines

After the smoke-test stack passes, test native engines one at a time in separate
venvs. Do not mix `sglang` and `vllm` in the same environment; their CUDA/PyTorch
pins are intentionally conflicting.

### SGLang

```bash
python3 -m venv .venv-sglang
source .venv-sglang/bin/activate
python -m pip install -U pip uv
uv pip install -e ".[sglang,train,infer]" --prerelease=allow
python -m sglang.check_env
python -m unirl.train_diffusion --config-name=diffusion/sd3_sglang_replay_colocate \
  num_devices=1 +devices_per_node=1 batch_size=2 num_rollouts=1 \
  data_source.args.algorithm.prompts_per_rollout=2 \
  sampling.samples_per_prompt=2 sampling.num_inference_steps=2 \
  sampling.scheduler.num_timesteps=2 sampling.scheduler.num_sde_steps=1 \
  sampling.scheduler.timestep_fraction=[0,1] \
  rollout.config.forward_batch_size=1 rollout.config.num_gpus=1 rollout.config.tp_size=1 \
  reward.backend.config.batch_size=1 stack.micro_batch_size=1 stack.num_updates_per_batch=1
```

This reaches `rollout 1/1` through `SGLangRolloutEngine`. If `xatlas` builds from
source on a minimal Ubuntu/Python install, Python development headers are
required. Without them the first failure is `fatal error: Python.h: No such file
or directory`. The validated host used `python3.12-dev` headers; installing the
system package is preferred when available.

Some SGLang SD3.5 text-encoder loaders may report duplicate safetensors keys and
fall back to native loaders when both `model.safetensors` and
`model.fp16.safetensors` are present. This was non-fatal in the validated smoke
run.

### vLLM / vLLM-Omni

```bash
python3 -m venv .venv-vllm
source .venv-vllm/bin/activate
python -m pip install -U pip uv
uv pip install -e ".[vllm,train,infer]"
python -m unirl.train_diffusion --config-name=diffusion/sd3_vllmomni \
  num_devices=1 +devices_per_node=1 batch_size=2 +num_rollouts=1 \
  data_source.args.algorithm.prompts_per_rollout=2 \
  sampling.samples_per_prompt=2 sampling.num_inference_steps=2 \
  sampling.scheduler.num_timesteps=2 sampling.scheduler.num_sde_steps=1 \
  sampling.scheduler.timestep_fraction=[0,1] \
  rollout.config.default_num_inference_steps=2 \
  reward.backend.config.batch_size=1 stack.micro_batch_size=1 stack.num_updates_per_batch=1
```

This reaches `rollout 1/1` through `VLLMOmniRolloutEngine`. Use
`+num_rollouts=1` for this recipe because `num_rollouts` is not present in the
base config.

For every native-extension failure, record:

- package and version
- whether a linux-aarch64 wheel exists
- Python tag (`cp312` here)
- CUDA version encoded by the wheel or build
- PyTorch version and CUDA ABI
- compiler and CUDA header versions if building from source
- GB10 compute capability reported by `scripts/dgx_spark_probe.py`

## Local smoke runner and observability

Use `scripts/dgx_spark_run_smoke.sh` for this fork's DGX Spark day-to-day health
check. It chooses the right local venv per engine, loads `.env` through dotenvx
when available, disables WandB/telemetry for smoke tests, optionally stops Ray,
and can wrap every run with the watcher below.

```bash
scripts/dgx_spark_run_smoke.sh --engine all --profile quick --watch
scripts/dgx_spark_run_smoke.sh --engine trainside --profile standard --watch
scripts/dgx_spark_run_smoke.sh --engine sglang --profile quick --watch
scripts/dgx_spark_run_smoke.sh --engine vllmomni --profile quick --watch
```

Engines: `trainside`, `sglang`, `vllmomni`, or `all`. The `quick` profile uses
the 2-step optimizer-smoke settings validated above. The `standard` profile uses
the validated medium trainside settings and currently keeps SGLang/VLLM-Omni on
quick until those engines are scaled locally.

Use `scripts/dgx_spark_watch.py` directly to wrap arbitrary long smoke runs and
keep their logs and host/GPU samples together. It writes a per-run directory
containing:

- `command.log` — redacted stdout/stderr from the wrapped command
- `metrics.jsonl` — periodic load, memory, and `nvidia-smi` samples
- `metadata.json` — command and run metadata
- `summary.json` — exit code, elapsed time, and peak GPU stats

Example:

```bash
source .venv-spark/bin/activate
python scripts/dgx_spark_watch.py \
  --label sd35-trainside-b2 \
  --log-dir outputs/dgx-spark-observe \
  --interval 5 \
  -- python -m unirl.train_diffusion --config-name=diffusion/sd3_trainside \
    num_devices=1 +devices_per_node=1 batch_size=2 \
    data_source.args.algorithm.prompts_per_rollout=2 \
    sampling.samples_per_prompt=2 sampling.num_inference_steps=2 \
    sampling.scheduler.num_timesteps=2 sampling.scheduler.num_sde_steps=1 \
    sampling.scheduler.timestep_fraction=[0,1] \
    rollout.forward_batch_size=1 reward.backend.config.batch_size=1 \
    stack.micro_batch_size=1 stack.num_updates_per_batch=1 +num_rollouts=1
```

`outputs/` is ignored by git. The watcher redacts Hugging Face tokens in captured
output, but the wrapped command should still avoid printing secrets.

## Known risks

- x86_64 wheels are not usable on DGX Spark.
- Docker images for research ML projects often publish only `linux/amd64`.
- CUDA 13 support does not imply GB10/Blackwell tuning support.
- Source builds can fail independently on Python 3.12, PyTorch ABI, CUDA headers,
  compute capability flags, or Arm-specific compiler issues.
