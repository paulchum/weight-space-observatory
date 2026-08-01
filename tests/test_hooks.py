import torch
import pytest

from wso.adapters.tflow_placeholder import (
    collect_tflow_batched_weighted_lora_telemetry,
    collect_tflow_weighted_lora_telemetry,
)
from wso.telemetry.hooks import (
    collect_batched_weighted_lora_event_telemetry,
    collect_peft_lora_telemetry,
    collect_state_dict_delta_telemetry,
    collect_weighted_lora_event_telemetry,
    materialize_weighted_lora_delta,
)


class TinyLoRALinear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(4, 2, bias=False)})
        self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 3, bias=False)})
        self.scaling = {"default": 2.0}


def test_collect_peft_lora_telemetry_from_tiny_module() -> None:
    torch.manual_seed(5)
    module = TinyLoRALinear()
    record = collect_peft_lora_telemetry(
        module,
        run_id="run",
        module_id="tiny.lora",
        adapter_id="default",
        task_family="demo",
        model_id="tiny",
    )
    assert record.collection_mode == "lora_hook"
    assert record.layer_shape == [3, 4]
    assert record.adapter_id == "default"
    assert record.frobenius_norm > 0
    assert record.provenance["raw_tensor_persisted"] is False


def test_collect_state_dict_delta_telemetry() -> None:
    before = {"weight": torch.zeros(3, 4)}
    after = {"weight": torch.ones(3, 4)}
    record = collect_state_dict_delta_telemetry(
        before,
        after,
        parameter_name="weight",
        run_id="run",
        module_id="linear.weight",
    )
    assert record.collection_mode == "state_dict_delta"
    assert record.layer_shape == [3, 4]
    assert record.frobenius_norm > 0


def test_tflow_weighted_lora_delta_preserves_sender_terms() -> None:
    a1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    b1 = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    a2 = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    b2 = torch.tensor([[1.0, 0.0], [1.0, 3.0]])
    entries = [(a1, b1, 0.5, 0.25), (a2, b2, 2.0, 0.75)]

    expected = 0.25 * 0.5 * (b1 @ a1) + 0.75 * 2.0 * (b2 @ a2)

    assert torch.allclose(materialize_weighted_lora_delta(entries), expected)


def test_collect_weighted_lora_event_telemetry_infers_fusion_metrics() -> None:
    a1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    b1 = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    a2 = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    b2 = torch.tensor([[1.0, 0.0], [1.0, 3.0]])

    records = collect_weighted_lora_event_telemetry(
        {"receiver.layers.0.mlp.down_proj.weight": [(a1, b1, 0.5, 0.25), (a2, b2, 2.0, 0.75)]},
        run_id="run",
        task_family="demo",
        model_id="event-shaped-local",
        sender_ids=["sender_a", "sender_b"],
    )

    record = records[0]
    assert record.collection_mode == "transient_lora_fusion_hook"
    assert record.module_id == "receiver.layers.0.mlp.down_proj"
    assert record.sender_count == 2
    assert record.sender_ids == ["sender_a", "sender_b"]
    assert record.rank_bound == 4
    assert record.fusion_gate_logits is not None
    assert record.fusion_entropy is not None
    assert record.fusion_gate_probs == pytest.approx([0.25, 0.75])
    assert record.provenance["collector"] == (
        "wso.telemetry.hooks.collect_weighted_lora_event_telemetry"
    )
    assert record.provenance["raw_tensor_persisted"] is False


def test_collect_batched_weighted_lora_event_telemetry_slices_samples() -> None:
    a = torch.stack(
        [
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([[2.0, 0.0], [0.0, 1.0]]),
        ]
    )
    b = torch.stack(
        [
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([[0.5, 0.0], [0.0, 1.5]]),
        ]
    )
    records = collect_batched_weighted_lora_event_telemetry(
        {"receiver.layers.0.mlp.down_proj.weight": [(a, b, 1.0, torch.tensor([0.2, 0.8]))]},
        run_ids=["run-0", "run-1"],
        task_families=["demo", "demo"],
        prompt_hashes=["hash-0", "hash-1"],
    )

    assert [record.run_id for record in records] == ["run-0", "run-1"]
    assert [record.prompt_hash for record in records] == ["hash-0", "hash-1"]
    assert all(record.layer_shape == [2, 2] for record in records)
    assert all(record.sender_count == 1 for record in records)


def test_collect_tflow_weighted_lora_telemetry_from_official_entry_shape() -> None:
    a = torch.randn(2, 4)
    b = torch.randn(3, 2)
    records = collect_tflow_weighted_lora_telemetry(
        {"receiver.layers.0.mlp.down_proj.weight": [(a, b, 1.0, 1.0)]},
        run_id="run",
        task_family="demo",
        model_id="official-tflow-shaped-local",
    )

    record = records[0]
    assert record.collection_mode == "tflow_inference_hook"
    assert record.runtime_framework == "official_tflow"
    assert record.module_id == "receiver.layers.0.mlp.down_proj"
    assert record.layer_shape == [3, 4]
    assert record.sender_count == 1
    assert record.rank_bound == 2
    assert record.provenance["collector"] == (
        "wso.telemetry.hooks.collect_weighted_lora_event_telemetry"
    )
    assert record.provenance["adapter_collector"] == (
        "wso.adapters.tflow_placeholder.collect_tflow_weighted_lora_telemetry"
    )
    assert record.provenance["raw_tensor_persisted"] is False


def test_collect_tflow_batched_weighted_lora_telemetry_delegates_to_hooks() -> None:
    a = torch.randn(2, 2, 4)
    b = torch.randn(2, 3, 2)
    records = collect_tflow_batched_weighted_lora_telemetry(
        {"receiver.layers.0.mlp.down_proj.weight": [(a, b, 1.0, torch.tensor([0.4, 0.6]))]},
        run_ids=["run-0", "run-1"],
        model_id="official-tflow-shaped-local",
    )

    assert [record.run_id for record in records] == ["run-0", "run-1"]
    assert all(record.collection_mode == "tflow_inference_hook" for record in records)
    assert all(record.runtime_framework == "official_tflow" for record in records)
    assert all(
        record.provenance["collector"]
        == "wso.telemetry.hooks.collect_weighted_lora_event_telemetry"
        for record in records
    )
