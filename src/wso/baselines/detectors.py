"""Task-conditioned anomaly analysis for defensive telemetry baselines."""

from __future__ import annotations

import json
from pathlib import Path

from wso.telemetry.schemas import AnomalyFinding, BaselineProfile, MetricBand, TransientLoRATelemetry


RISK_ORDER = {"LOW": 0, "MODERATE": 1, "HIGH": 2}


class QuantileBand(MetricBand):
    """Backward-compatible alias for the metric-band schema."""


def analyze_records(
    records: list[TransientLoRATelemetry],
    profiles: list[BaselineProfile],
) -> list[AnomalyFinding]:
    """Compare telemetry records with matching task-conditioned baselines."""

    profile_index = {_profile_key(profile): profile for profile in profiles}
    findings: list[AnomalyFinding] = []
    for record in records:
        profile = profile_index.get(_record_key(record))
        if profile is None:
            findings.append(_missing_baseline_finding(record))
            continue
        findings.append(_analyze_record(record, profile))
    return findings


def aggregate_risk(findings: list[AnomalyFinding]) -> str:
    """Return the highest risk level across a list of findings."""

    risk = "LOW"
    for finding in findings:
        risk = risk if RISK_ORDER[risk] >= RISK_ORDER[finding.risk_level] else finding.risk_level
    return risk


def write_findings_json(findings: list[AnomalyFinding], output: str | Path) -> None:
    payload = {
        "schema_version": "1.0",
        "risk_level": aggregate_risk(findings),
        "finding_count": len(findings),
        "findings": [finding.model_dump() for finding in findings],
    }
    Path(output).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_findings_json(path: str | Path) -> list[AnomalyFinding]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_findings = payload if isinstance(payload, list) else payload.get("findings", [])
    return [AnomalyFinding.model_validate(finding) for finding in raw_findings]


def _analyze_record(record: TransientLoRATelemetry, profile: BaselineProfile) -> AnomalyFinding:
    values = _metric_values(record)
    flags: list[str] = []
    messages: list[str] = []

    for metric, band in profile.metric_bands.items():
        value = values.get(metric)
        if value is None:
            continue
        if not band.contains(value):
            flags.append(metric)
            direction = "below" if value < band.lower else "above"
            messages.append(
                f"{metric}={value:.4f} is {direction} baseline band "
                f"[{band.lower:.4f}, {band.upper:.4f}]"
            )

    risk_level = "LOW"
    if flags:
        risk_level = "MODERATE"
    if len(flags) >= 3 or "grassmann_drift" in flags and len(flags) >= 2:
        risk_level = "HIGH"

    if not messages:
        messages.append("All available metrics fall inside the matched baseline bands.")

    return AnomalyFinding(
        run_id=record.run_id,
        baseline_id=profile.baseline_id,
        module_id=record.module_id,
        task_family=record.task_family,
        collection_mode=record.collection_mode,
        model_id=record.model_id,
        risk_level=risk_level,
        metric_values=values,
        out_of_band_metrics=flags,
        messages=messages,
    )


def _missing_baseline_finding(record: TransientLoRATelemetry) -> AnomalyFinding:
    return AnomalyFinding(
        run_id=record.run_id,
        module_id=record.module_id,
        task_family=record.task_family,
        collection_mode=record.collection_mode,
        model_id=record.model_id,
        risk_level="MODERATE",
        metric_values=_metric_values(record),
        out_of_band_metrics=[],
        messages=[
            "No matching task-conditioned baseline profile was available for this record."
        ],
    )


def _metric_values(record: TransientLoRATelemetry) -> dict[str, float]:
    pairs = {
        "fusion_entropy": record.fusion_entropy,
        "frobenius_norm": record.frobenius_norm,
        "effective_rank": record.effective_rank,
        "spectral_gap": record.spectral_gap,
        "grassmann_drift": record.grassmann_drift,
    }
    return {metric: value for metric, value in pairs.items() if value is not None}


def _profile_key(profile: BaselineProfile) -> tuple[str, str, str, str | None]:
    return (
        profile.task_family,
        profile.module_id,
        profile.collection_mode,
        profile.model_id,
    )


def _record_key(record: TransientLoRATelemetry) -> tuple[str, str, str, str | None]:
    return (
        record.task_family or "unknown",
        record.module_id,
        record.collection_mode,
        record.model_id,
    )
