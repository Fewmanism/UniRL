from unirl.train.fsdp_utils import merge_lora_state_dict


def test_merge_lora_state_dict_overrides_only_lora_keys():
    full = {
        "block.base.weight": "base",
        "block.lora_A.default.weight": "old_a",
        "block.lora_B.default.weight": "old_b",
    }
    adapter = {
        "block.lora_A.default.weight": "new_a",
        "block.lora_B.default.weight": "new_b",
    }

    assert merge_lora_state_dict(full, adapter) == {
        "block.base.weight": "base",
        "block.lora_A.default.weight": "new_a",
        "block.lora_B.default.weight": "new_b",
    }


def test_merge_lora_state_dict_rejects_unknown_adapter_keys():
    full = {"block.lora_A.default.weight": "old_a"}
    adapter = {"missing.lora_A.default.weight": "new_a"}

    try:
        merge_lora_state_dict(full, adapter)
    except KeyError as exc:
        assert "missing.lora_A.default.weight" in str(exc)
    else:
        raise AssertionError("expected KeyError")
