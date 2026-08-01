from pathlib import Path

from wso.reports.report_generator import (
    generate_audit_report,
    load_telemetry_jsonl,
    render_investor_markdown_report,
    render_markdown_report,
    write_report,
)
from wso.telemetry.schemas import TransientLoRATelemetry


def _record() -> TransientLoRATelemetry:
    return TransientLoRATelemetry(
        run_id="run-123456",
        module_id="layers.29.mlp.down_proj",
        task_family="code_generation",
        prompt_hash="abcd",
        sender_count=3,
        rank_bound=12,
        target_rank=4,
        fusion_entropy=0.5,
        frobenius_norm=1.0,
        singular_values=[3.0, 2.0, 1.0, 0.5, 0.25],
        effective_rank=3.2,
        spectral_gap=0.25,
        grassmann_drift=0.4,
    )


def test_generate_audit_report() -> None:
    report = generate_audit_report("demo", [_record()])
    assert report.deployment_name == "demo"
    assert report.total_inferences_audited == 1
    assert report.risk_level == "LOW"


def test_render_markdown_report_contains_tables() -> None:
    record = _record()
    report = generate_audit_report("demo", [record])
    rendered = render_markdown_report(report, [record])
    assert "Passing vs. Failing Audit Signals" in rendered
    assert "layers.29.mlp.down_proj" in rendered


def test_render_investor_report_contains_commercial_sections() -> None:
    record = _record()
    report = generate_audit_report("demo", [record])
    rendered = render_investor_markdown_report(report, [record])
    assert "Hidden Agent Communication Surface Audit" in rendered
    assert "Observability Gap" in rendered
    assert "Remediation Path" in rendered


def test_write_report_round_trip(tmp_path: Path) -> None:
    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(_record().model_dump_json() + "\n", encoding="utf-8")
    out = tmp_path / "report.md"
    report = write_report(telemetry, out, "demo")
    loaded = load_telemetry_jsonl(telemetry)
    assert len(loaded) == 1
    assert report.total_inferences_audited == 1
    assert out.exists()
    assert "Weight-Space Observability Audit" in out.read_text(encoding="utf-8")


def test_write_investor_report(tmp_path: Path) -> None:
    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(_record().model_dump_json() + "\n", encoding="utf-8")
    out = tmp_path / "investor.html"
    report = write_report(telemetry, out, "demo", "html", template="investor")
    assert report.total_inferences_audited == 1
    assert "Hidden Agent Communication Surface Audit" in out.read_text(encoding="utf-8")
