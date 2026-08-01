from pathlib import Path

from wso.cli import main
from wso.reports.report_generator import load_telemetry_jsonl


def test_cli_pipeline_toy_baseline_analyze_report(tmp_path: Path) -> None:
    telemetry = tmp_path / "toy.jsonl"
    baseline = tmp_path / "baseline.json"
    findings = tmp_path / "findings.json"
    report = tmp_path / "report.html"

    assert main(["toy-demo", "--output", str(telemetry)]) == 0
    assert main(["validate", str(telemetry)]) == 0
    assert main(["baseline", "build", str(telemetry), "--out", str(baseline)]) == 0
    assert main(["analyze", str(telemetry), "--baseline", str(baseline), "--out", str(findings)]) == 0
    assert (
        main(
            [
                "report",
                str(telemetry),
                "--out",
                str(report),
                "--format",
                "html",
                "--baseline",
                str(baseline),
                "--findings",
                str(findings),
            ]
        )
        == 0
    )
    assert report.exists()
    assert "Baseline Coverage" in report.read_text(encoding="utf-8")


def test_cli_hook_demo_writes_valid_telemetry(tmp_path: Path) -> None:
    telemetry = tmp_path / "hooked.jsonl"
    assert main(["hook-demo", "--output", str(telemetry)]) == 0
    records = load_telemetry_jsonl(telemetry)
    assert records[0].collection_mode == "lora_hook"


def test_cli_export_otel_writes_json(tmp_path: Path) -> None:
    telemetry = tmp_path / "hooked.jsonl"
    otel = tmp_path / "otel.json"
    assert main(["hook-demo", "--output", str(telemetry)]) == 0
    assert main(["export", "otel", str(telemetry), "--out", str(otel), "--deployment", "Demo"]) == 0
    assert "resourceSpans" in otel.read_text(encoding="utf-8")


def test_cli_pilot_writes_audit_package(tmp_path: Path) -> None:
    out_dir = tmp_path / "pilot"
    assert (
        main(
            [
                "pilot",
                "--deployment",
                "Demo Agent Platform",
                "--out-dir",
                str(out_dir),
                "--format",
                "html",
                "--baseline-records",
                "6",
            ]
        )
        == 0
    )
    assert (out_dir / "baseline_telemetry.jsonl").exists()
    assert (out_dir / "audit_telemetry.jsonl").exists()
    assert (out_dir / "telemetry_otel.json").exists()
    report = out_dir / "hidden_agent_communication_surface_audit.html"
    assert "Hidden Agent Communication Surface Audit" in report.read_text(encoding="utf-8")


def test_cli_design_partner_writes_evidence_pack(tmp_path: Path) -> None:
    config = tmp_path / "wso.yaml"
    out_dir = tmp_path / "design_partner"
    config.write_text(
        """
schema_version: "1.0"
mode: demo
deployment:
  name: "Design Partner Agent Platform"
runtime:
  framework: "torch_peft_compatible"
  model_id: "tiny-local-peft-pilot"
observability:
  service_name: "weight-space-observatory"
  service_namespace: "agent-security"
  destination_mode: file
modules:
  - module_id: "pilot.receiver.layers.0.lora"
    adapter_id: "default"
    task_family: "agent_tool_use"
baseline_records: 6
report:
  format: html
""",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "design-partner",
                "--config",
                str(config),
                "--out-dir",
                str(out_dir),
                "--format",
                "html",
            ]
        )
        == 0
    )
    assert (out_dir / "baseline_telemetry.jsonl").exists()
    assert (out_dir / "audit_telemetry.jsonl").exists()
    assert (out_dir / "telemetry_otel.json").exists()
    assert (out_dir / "design_partner_pilot.md").exists()
    assert (out_dir / "security_buyer_checklist.md").exists()
    assert (out_dir / "safety_boundary.md").exists()
    assert "Silicon Valley CISO Investments" in (out_dir / "manifest.json").read_text(
        encoding="utf-8"
    )
    report = out_dir / "hidden_agent_communication_surface_audit.html"
    assert "Hidden Agent Communication Surface Audit" in report.read_text(encoding="utf-8")


def test_cli_capacity_benchmark_writes_artifacts(tmp_path: Path) -> None:
    csv_path = tmp_path / "frontier.csv"
    json_path = tmp_path / "frontier.json"
    plot_path = tmp_path / "frontier.png"
    assert (
        main(
            [
                "benchmark",
                "capacity",
                "--out",
                str(csv_path),
                "--json-out",
                str(json_path),
                "--plot",
                str(plot_path),
                "--steps",
                "4",
                "--samples-per-strength",
                "40",
                "--benign-samples",
                "80",
            ]
        )
        == 0
    )
    assert "grassmann_drift_auc" in csv_path.read_text(encoding="utf-8")
    assert json_path.exists()
    assert plot_path.read_bytes().startswith(b"\x89PNG")
