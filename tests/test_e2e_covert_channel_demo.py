from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

from wso.reports.report_generator import load_telemetry_jsonl


def _load_demo_module() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "examples" / "e2e_covert_channel_demo.py"
    spec = importlib.util.spec_from_file_location("e2e_covert_channel_demo", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e2e_demo_flags_lsb_injection(tmp_path: Path) -> None:
    module = _load_demo_module()
    payload_path = tmp_path / "payload.bin"
    payload_path.write_bytes(
        b"inert payload file for local red-team detection validation " * 4
    )

    summary = module.run_demo(
        out_dir=tmp_path / "demo",
        payload_file=payload_path,
        seed=13,
        dim=256,
        rank=4,
    )

    assert summary["aggregate_risk"] == "HIGH"
    assert summary["clean_control"]["risk_level"] == "LOW"
    assert summary["infected_lsb_event"]["risk_level"] == "HIGH"
    assert {
        "effective_rank",
        "spectral_gap",
        "grassmann_drift",
    }.issubset(set(summary["infected_lsb_event"]["out_of_band_metrics"]))

    artifact_paths = summary["artifact_paths"]
    telemetry_records = load_telemetry_jsonl(artifact_paths["telemetry_jsonl"])
    assert len(telemetry_records) == 26
    assert telemetry_records[-1].run_id == "infected-lsb-write-event"

    detection_rows = [
        json.loads(line)
        for line in Path(artifact_paths["detection_jsonl"]).read_text(encoding="utf-8").splitlines()
    ]
    assert [row["risk_level"] for row in detection_rows] == ["LOW", "HIGH"]
    assert Path(artifact_paths["baseline_json"]).exists()
    assert Path(artifact_paths["findings_json"]).exists()
    assert Path(artifact_paths["summary_json"]).exists()
