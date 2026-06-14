import json
from pathlib import Path

import torch

from unirl.train.fsdp_utils import write_lora_manifest


def test_write_lora_manifest_records_file_checksum_and_keys(tmp_path: Path):
    adapter_path = tmp_path / "lora_adapter.pt"
    state = {
        "layer.lora_A.default.weight": torch.ones((1, 2), dtype=torch.float32),
        "layer.lora_B.default.weight": torch.zeros((2, 1), dtype=torch.float32),
    }
    torch.save(state, adapter_path)

    manifest = write_lora_manifest(tmp_path, state)

    manifest_path = tmp_path / "lora_manifest.json"
    assert manifest_path.exists()
    loaded = json.loads(manifest_path.read_text())
    assert loaded == manifest
    assert manifest["format"] == "unirl-lora-adapter-v1"
    assert manifest["adapter_file"] == "lora_adapter.pt"
    assert manifest["adapter_size_bytes"] == adapter_path.stat().st_size
    assert len(manifest["adapter_sha256"]) == 64
    assert manifest["num_tensors"] == 2
    assert manifest["keys"] == sorted(state)
