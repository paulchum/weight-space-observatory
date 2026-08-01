from pathlib import Path

import pytest
import torch

from wso.experiments.redteam import WeightSpaceCovertChannel


def test_round_trip_in_memory() -> None:
    model = torch.nn.Linear(16, 16)
    state_dict = model.state_dict()
    channel = WeightSpaceCovertChannel()
    payload = b"inert telemetry marker"

    channel.inject(state_dict, "weight", payload)

    assert channel.extract(state_dict, "weight") == payload


def test_round_trip_after_safe_torch_load(tmp_path: Path) -> None:
    model = torch.nn.Linear(16, 16)
    state_dict = model.state_dict()
    channel = WeightSpaceCovertChannel()
    payload = b"serialized inert marker"

    channel.inject(state_dict, "weight", payload)
    model_path = tmp_path / "model.pt"
    torch.save({"model": state_dict}, model_path)

    loaded = torch.load(model_path, weights_only=True)

    assert channel.extract(loaded["model"], "weight") == payload


def test_rejects_non_bytes_payload() -> None:
    model = torch.nn.Linear(16, 16)
    channel = WeightSpaceCovertChannel()

    with pytest.raises(TypeError, match="payload must be bytes"):
        channel.inject(model.state_dict(), "weight", "not bytes")  # type: ignore[arg-type]


def test_rejects_payload_larger_than_layer_capacity() -> None:
    model = torch.nn.Linear(2, 2)
    channel = WeightSpaceCovertChannel()

    with pytest.raises(ValueError, match="Payload requires"):
        channel.inject(model.state_dict(), "weight", b"too large")


def test_extract_requires_header_at_start() -> None:
    model = torch.nn.Linear(16, 16)
    channel = WeightSpaceCovertChannel()

    with pytest.raises(ValueError, match="Magic header"):
        channel.extract(model.state_dict(), "weight")
