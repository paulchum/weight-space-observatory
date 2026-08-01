import pytest

from wso.telemetry.schemas import (
    AnomalyFinding,
    AuditReport,
    BaselineProfile,
    MetricBand,
    TransientLoRATelemetry,
)


def test_telemetry_schema_serializes() -> None:
    record = TransientLoRATelemetry(
        run_id="run",
        trace_id="0" * 32,
        span_id="1" * 16,
        deployment_id="demo-deployment",
        module_id="layers.0.mlp.down_proj",
        sender_count=2,
        rank_bound=8,
        runtime_framework="torch_peft_compatible",
        target_rank=4,
        observer_level="derived_weight_space_telemetry",
        otel_resource={"service.namespace": "agent-security"},
        frobenius_norm=1.0,
        singular_values=[1.0, 0.5],
    )
    dumped = record.model_dump()
    assert dumped["run_id"] == "run"
    assert dumped["rank_bound"] == 8
    assert dumped["schema_version"] == "1.0"
    assert dumped["collection_mode"] == "synthetic"
    assert dumped["deployment_id"] == "demo-deployment"
    assert dumped["otel_resource"]["service.namespace"] == "agent-security"


def test_audit_report_schema() -> None:
    report = AuditReport(
        deployment_name="demo",
        total_inferences_audited=1,
        mean_fusion_entropy=0.5,
        mean_spectral_gap=0.2,
        max_grassmann_drift=0.1,
        risk_level="LOW",
    )
    assert report.risk_level == "LOW"


def test_telemetry_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError):
        TransientLoRATelemetry(
            run_id="run",
            module_id="layers.0",
            sender_count=1,
            target_rank=1,
            frobenius_norm=float("nan"),
            singular_values=[1.0],
        )


def test_baseline_and_finding_schemas() -> None:
    band = MetricBand(metric="frobenius_norm", lower=0.0, upper=2.0, sample_count=5)
    profile = BaselineProfile(
        baseline_id="baseline",
        task_family="code",
        module_id="layers.0",
        collection_mode="synthetic",
        group_key="code|layers.0|synthetic|any-model",
        record_count=5,
        metric_bands={"frobenius_norm": band},
    )
    finding = AnomalyFinding(
        run_id="run",
        baseline_id=profile.baseline_id,
        module_id="layers.0",
        collection_mode="synthetic",
        risk_level="LOW",
    )
    assert profile.metric_bands["frobenius_norm"].contains(1.0)
    assert finding.risk_level == "LOW"
