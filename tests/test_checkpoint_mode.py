import pytest

from unirl.trainer.diffusion import _checkpoint_actions, _checkpoint_load_actions


def test_full_checkpoint_mode_saves_full_state_and_optional_lora():
    assert _checkpoint_actions("full", save_lora_checkpoint=True) == (True, True)
    assert _checkpoint_actions("full", save_lora_checkpoint=False) == (True, False)


def test_lora_checkpoint_mode_saves_only_lora_adapter():
    assert _checkpoint_actions("lora", save_lora_checkpoint=True) == (False, True)
    assert _checkpoint_actions("lora", save_lora_checkpoint=False) == (False, True)


def test_invalid_checkpoint_mode_is_rejected():
    with pytest.raises(ValueError, match="checkpoint_mode"):
        _checkpoint_actions("weights", save_lora_checkpoint=True)


def test_full_resume_loads_full_checkpoint_only():
    assert _checkpoint_load_actions("full") == (True, False)


def test_lora_resume_loads_lora_adapter_only():
    assert _checkpoint_load_actions("lora") == (False, True)


def test_lora_resume_can_disable_manifest_verification():
    from unirl.trainer.diffusion import DiffusionTrainer

    class Backend:
        def __init__(self):
            self.calls = []

        def load_lora(self, path, *, verify_manifest=True):
            self.calls.append((path, verify_manifest))

    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer.backend = Backend()
    trainer._init_wandb = lambda **_: None
    trainer._finish_wandb = lambda: None

    trainer.train(
        num_rollouts=0,
        resume_checkpoint_dir="checkpoint-dir",
        resume_checkpoint_mode="lora",
        verify_lora_manifest=False,
    )

    assert trainer.backend.calls == [("checkpoint-dir", False)]


def test_invalid_resume_checkpoint_mode_is_rejected():
    with pytest.raises(ValueError, match="resume_checkpoint_mode"):
        _checkpoint_load_actions("weights")
