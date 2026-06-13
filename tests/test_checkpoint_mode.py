import pytest

from unirl.trainer.diffusion import _checkpoint_actions


def test_full_checkpoint_mode_saves_full_state_and_optional_lora():
    assert _checkpoint_actions("full", save_lora_checkpoint=True) == (True, True)
    assert _checkpoint_actions("full", save_lora_checkpoint=False) == (True, False)


def test_lora_checkpoint_mode_saves_only_lora_adapter():
    assert _checkpoint_actions("lora", save_lora_checkpoint=True) == (False, True)
    assert _checkpoint_actions("lora", save_lora_checkpoint=False) == (False, True)


def test_invalid_checkpoint_mode_is_rejected():
    with pytest.raises(ValueError, match="checkpoint_mode"):
        _checkpoint_actions("weights", save_lora_checkpoint=True)
