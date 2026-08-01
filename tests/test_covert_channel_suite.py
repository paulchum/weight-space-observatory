import json
from pathlib import Path

from wso.cli import main
from wso.experiments.covert_channel_suite import (
    AuditGateDetectionConfig,
    FusionGateTimingConfig,
    LoraWatermarkAuditConfig,
    TFlowObserverComparisonConfig,
    determine_dynamic_lora_rank,
    run_audit_gate_detection,
    run_fusion_gate_timing_audit,
    run_lora_watermark_audit,
    run_tflow_observer_comparison,
)

import torch


def test_dynamic_rank_is_clamped_to_configured_bounds() -> None:
    delta = torch.diag(torch.tensor([5.0, 4.0, 2.0, 0.2, 0.1, 0.05]))
    rank = determine_dynamic_lora_rank(delta, min_rank=2, max_rank=4, energy_threshold=0.95)
    assert 2 <= rank <= 4


def test_lora_watermark_audit_blind_retrieval_controls(tmp_path: Path) -> None:
    result = run_lora_watermark_audit(
        tmp_path,
        LoraWatermarkAuditConfig(trials=100, dim=32, seed=17),
    )
    assert result.metrics["blind_decoding_accuracy"] == 1.0
    assert result.metrics["max_relative_perplexity_shift"] < 0.002
    assert result.metrics["textual_monitor_accuracy"] <= 0.55
    assert Path(result.artifacts["trial_csv"]).exists()


def test_fusion_gate_timing_audit_reaches_low_ber(tmp_path: Path) -> None:
    result = run_fusion_gate_timing_audit(
        tmp_path,
        FusionGateTimingConfig(bits=64, seed=19),
    )
    assert result.metrics["best_bit_error_rate"] <= 0.005
    assert result.metrics["textual_monitor_accuracy"] <= 0.55
    assert result.metrics["max_gate_error_percent"] <= 5.0


def test_audit_gate_detector_separates_weight_space_snapshots(tmp_path: Path) -> None:
    result = run_audit_gate_detection(
        tmp_path,
        AuditGateDetectionConfig(
            legitimate_sequences=12,
            covert_sequences=12,
            snapshots_per_sequence=4,
            dim=32,
            seed=23,
        ),
    )
    assert result.metrics["weight_space_auroc"] >= 0.98
    assert result.metrics["textual_output_monitor_auroc"] <= 0.60
    assert Path(result.artifacts["snapshot_feature_csv"]).exists()


def test_tflow_observer_comparison_writes_table(tmp_path: Path) -> None:
    result = run_tflow_observer_comparison(
        tmp_path,
        TFlowObserverComparisonConfig(
            task_families=("math_reasoning", "code_generation"),
            module_ids=(
                "receiver.layers.3.self_attn.q_proj",
                "receiver.layers.7.mlp.down_proj",
            ),
            ranks=(2, 4),
            sender_counts=(1, 3),
            fusion_noise_sigmas=(0.0, 0.16),
            baseline_samples_per_group=5,
            heldout_samples_per_group=3,
            stress_samples_per_condition=1,
            dim=20,
            seed=29,
        ),
    )
    table = Path(result.artifacts["observer_comparison_md"])
    assert table.exists()
    text = table.read_text(encoding="utf-8")
    assert "text_only_observer" in text
    assert "telemetry_observer" in text
    assert "tensor_observer" in text
    assert result.metrics["heldout_benign_records"] > 0
    assert result.metrics["telemetry_observer_auroc"] > result.metrics["text_observer_auroc"]


def test_cli_suite_quick_writes_summary(tmp_path: Path) -> None:
    out_dir = tmp_path / "suite"
    assert main(["suite", "--out-dir", str(out_dir), "--quick", "--seed", "5"]) == 0
    summary = json.loads((out_dir / "suite_summary.json").read_text(encoding="utf-8"))
    assert "A_lora_watermark_blind_retrieval_audit" in summary["experiments"]
    assert "D_tflow_observer_detectability_comparison" in summary["experiments"]
    assert "scalar_jsonl" in summary["dashboard"]
