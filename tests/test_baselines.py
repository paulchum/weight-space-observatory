import json
from pathlib import Path

from wso.baselines.detectors import aggregate_risk, analyze_records, load_findings_json, write_findings_json
from wso.baselines.task_conditioned import (
    build_baseline_profiles,
    load_baseline_profiles,
    write_baseline_profiles,
)
from wso.telemetry.schemas import TransientLoRATelemetry


def _record(run_id: str, drift: float, norm: float = 1.0) -> TransientLoRATelemetry:
    return TransientLoRATelemetry(
        run_id=run_id,
        module_id="layers.0",
        task_family="code",
        collection_mode="synthetic",
        sender_count=1,
        target_rank=2,
        frobenius_norm=norm,
        singular_values=[1.0, 0.5],
        effective_rank=1.5,
        spectral_gap=0.25,
        grassmann_drift=drift,
    )


def test_build_and_load_baseline_profiles(tmp_path: Path) -> None:
    profiles = build_baseline_profiles([_record(f"run-{idx}", 0.1 + idx * 0.01) for idx in range(8)])
    assert len(profiles) == 1
    assert "grassmann_drift" in profiles[0].metric_bands

    out = tmp_path / "baseline.json"
    write_baseline_profiles(profiles, out)
    loaded = load_baseline_profiles(out)
    assert loaded[0].baseline_id == profiles[0].baseline_id


def test_analyze_records_flags_out_of_band_metrics() -> None:
    baseline_records = [_record(f"base-{idx}", 0.1 + idx * 0.01) for idx in range(12)]
    profiles = build_baseline_profiles(baseline_records)
    findings = analyze_records([_record("live", 1.5, norm=9.0)], profiles)
    assert findings[0].risk_level in {"MODERATE", "HIGH"}
    assert findings[0].out_of_band_metrics
    assert aggregate_risk(findings) == findings[0].risk_level


def test_legacy_list_json_loaders(tmp_path: Path) -> None:
    profiles = build_baseline_profiles([_record(f"run-{idx}", 0.1 + idx * 0.01) for idx in range(8)])
    baseline_path = tmp_path / "legacy-baselines.json"
    baseline_path.write_text(json.dumps([profile.model_dump() for profile in profiles]), encoding="utf-8")
    assert load_baseline_profiles(baseline_path)[0].baseline_id == profiles[0].baseline_id

    findings = analyze_records([_record("live", 1.5)], profiles)
    findings_path = tmp_path / "findings.json"
    write_findings_json(findings, findings_path)
    assert load_findings_json(findings_path)[0].run_id == "live"
