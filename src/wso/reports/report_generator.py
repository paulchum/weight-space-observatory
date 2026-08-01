from __future__ import annotations

import html
import json
from pathlib import Path
from statistics import mean
from typing import Sequence

from wso.baselines.detectors import aggregate_risk, load_findings_json
from wso.baselines.task_conditioned import load_baseline_profiles
from wso.telemetry.schemas import (
    AnomalyFinding,
    AuditReport,
    BaselineProfile,
    CapacityFrontierReport,
    TransientLoRATelemetry,
)


RISK_ORDER = {"LOW": 0, "MODERATE": 1, "HIGH": 2}


def load_telemetry_jsonl(path: str | Path) -> list[TransientLoRATelemetry]:
    """Load newline-delimited telemetry records into validated Pydantic models."""
    path = Path(path)
    records: list[TransientLoRATelemetry] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(TransientLoRATelemetry.model_validate_json(stripped))
            except Exception as exc:  # pragma: no cover - pydantic formats vary by version
                raise ValueError(f"Invalid telemetry JSONL at line {line_no}: {exc}") from exc
    return records


def generate_audit_report(
    deployment_name: str,
    records: list[TransientLoRATelemetry],
    risk_level: str | None = None,
    notes: list[str] | None = None,
    findings: list[AnomalyFinding] | None = None,
) -> AuditReport:
    """Generate a diagnostic audit report summary from telemetry records."""
    if not records:
        return AuditReport(
            deployment_name=deployment_name,
            total_inferences_audited=0,
            mean_fusion_entropy=0.0,
            mean_spectral_gap=0.0,
            max_grassmann_drift=0.0,
            risk_level=risk_level or aggregate_risk(findings or []),
            notes=notes or [],
        )

    entropy_vals = [r.fusion_entropy for r in records if r.fusion_entropy is not None]
    gap_vals = [r.spectral_gap for r in records if r.spectral_gap is not None]
    drift_vals = [r.grassmann_drift for r in records if r.grassmann_drift is not None]

    inferred_risk = risk_level or (aggregate_risk(findings) if findings else infer_risk_level(records))

    return AuditReport(
        deployment_name=deployment_name,
        total_inferences_audited=len(records),
        mean_fusion_entropy=float(mean(entropy_vals)) if entropy_vals else 0.0,
        mean_spectral_gap=float(mean(gap_vals)) if gap_vals else 0.0,
        max_grassmann_drift=float(max(drift_vals)) if drift_vals else 0.0,
        risk_level=inferred_risk,
        notes=notes or recommended_controls(records, inferred_risk),
    )


def infer_risk_level(records: list[TransientLoRATelemetry]) -> str:
    """Simple conservative demo risk policy for transient adapter telemetry.

    This is not a universal security policy. Real deployments should set thresholds from
    task-conditioned baselines and customer-specific operating ranges.
    """
    risk = "LOW"
    for record in records:
        if record.spectral_gap is not None and record.spectral_gap < 0.05:
            risk = _max_risk(risk, "MODERATE")
        if record.grassmann_drift is not None and record.grassmann_drift > 1.25:
            risk = _max_risk(risk, "MODERATE")
        if record.rank_bound is not None and record.effective_rank is not None:
            if record.effective_rank > max(record.target_rank + 1.5, 0.75 * record.rank_bound):
                risk = _max_risk(risk, "MODERATE")
        if record.spectral_gap is not None and record.spectral_gap < 0.01:
            risk = _max_risk(risk, "HIGH")
        if record.grassmann_drift is not None and record.grassmann_drift > 1.75:
            risk = _max_risk(risk, "HIGH")
    return risk


def recommended_controls(records: list[TransientLoRATelemetry], risk_level: str) -> list[str]:
    controls = [
        "Persist perturbation provenance summaries for each instrumented inference.",
        "Compare fusion-gate entropy and singular spectra against task-conditioned baselines.",
        "Track Grassmannian subspace drift for transient LoRA updates by module and task family.",
    ]
    if risk_level in {"MODERATE", "HIGH"}:
        controls.append(
            "Tighten alert thresholds and expand baseline collection before deploying adapter routing at scale."
        )
    if any(r.spectral_gap is not None and r.spectral_gap < 0.05 for r in records):
        controls.append(
            "Review low spectral-gap events; unstable subspaces reduce confidence in telemetry reproducibility."
        )
    return controls


def render_markdown_report(
    report: AuditReport,
    records: list[TransientLoRATelemetry],
    *,
    baseline_profiles: list[BaselineProfile] | None = None,
    findings: list[AnomalyFinding] | None = None,
    frontier: CapacityFrontierReport | None = None,
) -> str:
    """Render a boardroom-readable Markdown audit report."""
    module_count = len({r.module_id for r in records})
    task_families = sorted({r.task_family or "unknown" for r in records})
    collection_modes = sorted({r.collection_mode for r in records})
    lines = [
        f"# Weight-Space Observability Audit: {report.deployment_name}",
        "",
        "## Executive Summary",
        "",
        f"**Risk level:** {report.risk_level}",
        "",
        (
            f"Audited {report.total_inferences_audited} transient adapter telemetry "
            f"record(s) across {module_count} module(s)."
        ),
        "",
        "This report summarizes non-text communication-surface telemetry for agent or "
        "adapter-routed systems. It stores derived spectral and routing signatures rather "
        "than raw model tensors.",
        "",
        "## Key Metrics",
        "",
        "| Metric | Value | Interpretation |",
        "|---|---:|---|",
        f"| Mean fusion entropy | {report.mean_fusion_entropy:.4f} | Sender contribution concentration/distribution |",
        f"| Mean spectral gap | {report.mean_spectral_gap:.4f} | Principal subspace stability signal |",
        f"| Max Grassmannian drift | {report.max_grassmann_drift:.4f} | Largest observed subspace movement |",
        f"| Task families | {', '.join(task_families) if task_families else 'n/a'} | Baseline conditioning label(s) |",
        f"| Collection modes | {', '.join(collection_modes) if collection_modes else 'n/a'} | Telemetry source path(s) |",
        "",
        "## Passing vs. Failing Audit Signals",
        "",
        "| Signal | Passing Example | Failing Example | Why It Matters |",
        "|---|---:|---:|---|",
        "| Fusion-gate entropy | Inside task baseline | Outside task baseline | Shows whether sender routing changed unexpectedly |",
        "| Spectral gap | Stable gap above policy floor | Near-degenerate gap | Low gap makes subspace telemetry unstable |",
        "| Grassmannian drift | Within empirical quantile | Outside empirical quantile | Detects movement outside normal task-conditioned subspaces |",
        "| Perturbation provenance | Model/generator/run hashes present | Missing provenance | Enables replay, audit, and root-cause analysis |",
        "| Effective rank | Within rank-bound expectation | Rank inflation or collapse | Flags unusual adapter update geometry |",
        "",
        "## Telemetry Records",
        "",
        "| Run ID | Module | Task | Fusion Entropy | Spectral Gap | Grassmann Drift | Effective Rank |",
        "|---|---|---|---:|---:|---:|---:|",
    ]

    for record in records:
        lines.append(
            "| "
            f"{record.run_id[:8]} | "
            f"{record.module_id} | "
            f"{record.task_family or 'unknown'} | "
            f"{_fmt(record.fusion_entropy)} | "
            f"{_fmt(record.spectral_gap)} | "
            f"{_fmt(record.grassmann_drift)} | "
            f"{_fmt(record.effective_rank)} |"
        )

    if baseline_profiles:
        lines.extend(
            [
                "",
                "## Baseline Coverage",
                "",
                "| Baseline ID | Task | Module | Mode | Model | Records | Metrics |",
                "|---|---|---|---|---|---:|---|",
            ]
        )
        for profile in baseline_profiles:
            lines.append(
                "| "
                f"{profile.baseline_id} | "
                f"{profile.task_family} | "
                f"{profile.module_id} | "
                f"{profile.collection_mode} | "
                f"{profile.model_id or 'any'} | "
                f"{profile.record_count} | "
                f"{', '.join(sorted(profile.metric_bands))} |"
            )

    if findings:
        lines.extend(
            [
                "",
                "## Anomaly Findings",
                "",
                "| Run ID | Risk | Baseline | Flags | Summary |",
                "|---|---|---|---|---|",
            ]
        )
        for finding in findings:
            lines.append(
                "| "
                f"{finding.run_id[:8]} | "
                f"{finding.risk_level} | "
                f"{finding.baseline_id or 'missing'} | "
                f"{', '.join(finding.out_of_band_metrics) or 'none'} | "
                f"{'; '.join(finding.messages)} |"
            )

    if frontier:
        lines.extend(
            [
                "",
                "## Capacity-Detectability Frontier",
                "",
                "| Signal | Recoverability AUC | Anomaly Rate |",
                "|---:|---:|---:|",
            ]
        )
        for point in frontier.points:
            lines.append(
                f"| {point.signal_strength:.4f} | "
                f"{point.recoverability_auc:.4f} | {point.anomaly_rate:.4f} |"
            )

    lines.extend(
        [
            "",
            "## Recommended Controls",
            "",
        ]
    )
    for note in report.notes:
        lines.append(f"- {note}")

    lines.extend(
        [
            "",
            "## Responsible Research Boundary",
            "",
            "This report is for defensive observability and audit planning. It does not "
            "include payload-hiding code, keyed extraction logic, or prompt-search steganography.",
            "",
        ]
    )
    return "\n".join(lines)


def render_html_report(
    report: AuditReport,
    records: list[TransientLoRATelemetry],
    *,
    baseline_profiles: list[BaselineProfile] | None = None,
    findings: list[AnomalyFinding] | None = None,
    frontier: CapacityFrontierReport | None = None,
) -> str:
    """Render a lightweight standalone HTML audit report."""
    task_families = sorted({r.task_family or "unknown" for r in records})
    collection_modes = sorted({r.collection_mode for r in records})
    metric_rows = [
        ("Mean fusion entropy", f"{report.mean_fusion_entropy:.4f}"),
        ("Mean spectral gap", f"{report.mean_spectral_gap:.4f}"),
        ("Max Grassmannian drift", f"{report.max_grassmann_drift:.4f}"),
        ("Task families", ", ".join(task_families) or "n/a"),
        ("Collection modes", ", ".join(collection_modes) or "n/a"),
    ]
    telemetry_rows = [
        [
            record.run_id[:8],
            record.module_id,
            record.task_family or "unknown",
            record.collection_mode,
            _fmt(record.fusion_entropy),
            _fmt(record.spectral_gap),
            _fmt(record.grassmann_drift),
            _fmt(record.effective_rank),
        ]
        for record in records
    ]
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Weight-Space Observability Audit</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 1080px; margin: 40px auto; line-height: 1.55; color: #111827; }}
    h1, h2 {{ line-height: 1.2; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f4f6; }}
    .risk {{ display: inline-block; padding: 4px 10px; border-radius: 999px; font-weight: 700; background: {_risk_color(report.risk_level)}; color: white; }}
    .note {{ background: #f8fafc; border-left: 4px solid #64748b; padding: 12px 16px; }}
  </style>
</head>
<body>
  <h1>Weight-Space Observability Audit: {_esc(report.deployment_name)}</h1>
  <p><span class=\"risk\">{_esc(report.risk_level)}</span></p>
  <p>Audited {_esc(str(report.total_inferences_audited))} transient adapter telemetry record(s). The report stores derived spectral and routing signatures rather than raw model tensors.</p>
  <h2>Key Metrics</h2>
  {_html_table(["Metric", "Value"], metric_rows)}
  <h2>Telemetry Records</h2>
  {_html_table(["Run ID", "Module", "Task", "Mode", "Fusion Entropy", "Spectral Gap", "Grassmann Drift", "Effective Rank"], telemetry_rows)}
  {_html_baselines(baseline_profiles or [])}
  {_html_findings(findings or [])}
  {_html_frontier(frontier)}
  <h2>Recommended Controls</h2>
  <ul>{''.join(f'<li>{_esc(note)}</li>' for note in report.notes)}</ul>
  <h2>Responsible Research Boundary</h2>
  <p class=\"note\">This report is for defensive observability and audit planning. It does not include payload-hiding code, keyed extraction logic, or prompt-search steganography.</p>
</body>
</html>
"""


def render_investor_markdown_report(
    report: AuditReport,
    records: list[TransientLoRATelemetry],
    *,
    baseline_profiles: list[BaselineProfile] | None = None,
    findings: list[AnomalyFinding] | None = None,
    frontier: CapacityFrontierReport | None = None,
) -> str:
    """Render the commercial diagnostic report template."""

    task_families = sorted({r.task_family or "unknown" for r in records})
    collection_modes = sorted({r.collection_mode for r in records})
    observer_levels = sorted({r.observer_level for r in records})
    finding_count = len(findings or [])
    flagged_count = len([finding for finding in findings or [] if finding.out_of_band_metrics])
    lines = [
        f"# Hidden Agent Communication Surface Audit: {report.deployment_name}",
        "",
        "## Executive Verdict",
        "",
        f"**Risk level:** {report.risk_level}",
        "",
        (
            f"WSO audited {report.total_inferences_audited} adapter/weight-space telemetry "
            f"record(s) and found {flagged_count} out-of-band event(s) across "
            f"{finding_count} analyzed finding(s)."
        ),
        "",
        "The diagnostic is designed for agent deployments where prompts, tool calls, "
        "and outputs are visible but transient adapter routing or LoRA perturbations "
        "may carry operational signal outside normal text logs.",
        "",
        "## Observability Gap",
        "",
        "| Surface | Current audit signal | Buyer relevance |",
        "|---|---|---|",
        (
            f"| Adapter / LoRA perturbations | {', '.join(collection_modes) or 'n/a'} | "
            "Shows whether runtime model updates are inspectable |"
        ),
        (
            f"| Task-conditioned baseline | {len(baseline_profiles or [])} profile(s) | "
            "Separates normal task geometry from suspicious drift |"
        ),
        (
            f"| Observer level | {', '.join(observer_levels) or 'n/a'} | "
            "Makes the monitoring blind spot explicit |"
        ),
        (
            f"| Task families | {', '.join(task_families) or 'n/a'} | "
            "Maps risk to concrete deployment workflows |"
        ),
        "",
        "## Commercial Readout",
        "",
        "| Metric | Value | Why a buyer cares |",
        "|---|---:|---|",
        (
            f"| Mean fusion entropy | {report.mean_fusion_entropy:.4f} | "
            "Flags unexpected sender/router concentration |"
        ),
        (
            f"| Mean spectral gap | {report.mean_spectral_gap:.4f} | "
            "Measures stability of the inspected perturbation subspace |"
        ),
        (
            f"| Max Grassmannian drift | {report.max_grassmann_drift:.4f} | "
            "Captures movement outside expected task geometry |"
        ),
        (
            f"| Analyzed findings | {finding_count} | "
            "Turns raw telemetry into an audit queue |"
        ),
    ]

    if baseline_profiles:
        lines.extend(
            [
                "",
                "## Baseline Coverage",
                "",
                "| Baseline ID | Task | Module | Mode | Model | Records | Metrics |",
                "|---|---|---|---|---|---:|---|",
            ]
        )
        for profile in baseline_profiles:
            lines.append(
                "| "
                f"{profile.baseline_id} | "
                f"{profile.task_family} | "
                f"{profile.module_id} | "
                f"{profile.collection_mode} | "
                f"{profile.model_id or 'any'} | "
                f"{profile.record_count} | "
                f"{', '.join(sorted(profile.metric_bands))} |"
            )

    if findings:
        lines.extend(
            [
                "",
                "## Anomaly Findings",
                "",
                "| Run ID | Risk | Baseline | Flags | Summary |",
                "|---|---|---|---|---|",
            ]
        )
        for finding in findings:
            lines.append(
                "| "
                f"{finding.run_id[:8]} | "
                f"{finding.risk_level} | "
                f"{finding.baseline_id or 'missing'} | "
                f"{', '.join(finding.out_of_band_metrics) or 'none'} | "
                f"{'; '.join(finding.messages)} |"
            )

    if frontier:
        lines.extend(
            [
                "",
                "## Capacity-Detectability Frontier",
                "",
                "| Signal | Recoverability AUC | Anomaly Rate |",
                "|---:|---:|---:|",
            ]
        )
        for point in frontier.points:
            lines.append(
                f"| {point.signal_strength:.4f} | "
                f"{point.recoverability_auc:.4f} | {point.anomaly_rate:.4f} |"
            )

    lines.extend(
        [
            "",
            "## Remediation Path",
            "",
            "1. Instrument adapter and transient LoRA surfaces in the agent runtime.",
            "2. Collect task-conditioned benign baselines before broad deployment.",
            "3. Export derived telemetry to the existing observability or SIEM pipeline.",
            "4. Alert on out-of-band drift, rank inflation, low spectral gaps, and missing provenance.",
            "",
            "## Recommended Controls",
            "",
        ]
    )
    for note in report.notes:
        lines.append(f"- {note}")

    lines.extend(
        [
            "",
            "## Responsible Research Boundary",
            "",
            "This diagnostic stores derived telemetry only. It does not include payload-hiding "
            "code, keyed extraction, monitor bypass logic, or operational covert-channel tooling.",
            "",
        ]
    )
    return "\n".join(lines)


def render_investor_html_report(
    report: AuditReport,
    records: list[TransientLoRATelemetry],
    *,
    baseline_profiles: list[BaselineProfile] | None = None,
    findings: list[AnomalyFinding] | None = None,
    frontier: CapacityFrontierReport | None = None,
) -> str:
    """Render a standalone HTML version of the commercial diagnostic."""

    task_families = sorted({r.task_family or "unknown" for r in records})
    collection_modes = sorted({r.collection_mode for r in records})
    observer_levels = sorted({r.observer_level for r in records})
    flagged_count = len([finding for finding in findings or [] if finding.out_of_band_metrics])
    metric_rows = [
        ["Risk level", report.risk_level],
        ["Telemetry records", str(report.total_inferences_audited)],
        ["Out-of-band events", str(flagged_count)],
        ["Mean fusion entropy", f"{report.mean_fusion_entropy:.4f}"],
        ["Mean spectral gap", f"{report.mean_spectral_gap:.4f}"],
        ["Max Grassmannian drift", f"{report.max_grassmann_drift:.4f}"],
    ]
    gap_rows = [
        ["Adapter / LoRA perturbations", ", ".join(collection_modes) or "n/a"],
        ["Task-conditioned baselines", str(len(baseline_profiles or []))],
        ["Observer level", ", ".join(observer_levels) or "n/a"],
        ["Task families", ", ".join(task_families) or "n/a"],
    ]
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Hidden Agent Communication Surface Audit</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 1120px; margin: 40px auto; line-height: 1.55; color: #111827; }}
    h1, h2 {{ line-height: 1.2; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f4f6; }}
    .risk {{ display: inline-block; padding: 4px 10px; border-radius: 999px; font-weight: 700; background: {_risk_color(report.risk_level)}; color: white; }}
    .note {{ background: #f8fafc; border-left: 4px solid #2563eb; padding: 12px 16px; }}
  </style>
</head>
<body>
  <h1>Hidden Agent Communication Surface Audit: {_esc(report.deployment_name)}</h1>
  <p><span class=\"risk\">{_esc(report.risk_level)}</span></p>
  <p class=\"note\">WSO audits agent deployments where ordinary prompt, tool, and output logs may miss transient adapter routing or LoRA perturbation behavior.</p>
  <h2>Executive Verdict</h2>
  {_html_table(["Signal", "Value"], metric_rows)}
  <h2>Observability Gap</h2>
  {_html_table(["Surface", "Current audit signal"], gap_rows)}
  {_html_baselines(baseline_profiles or [])}
  {_html_findings(findings or [])}
  {_html_frontier(frontier)}
  <h2>Remediation Path</h2>
  <ol>
    <li>Instrument adapter and transient LoRA surfaces in the agent runtime.</li>
    <li>Collect task-conditioned benign baselines before broad deployment.</li>
    <li>Export derived telemetry to the existing observability or SIEM pipeline.</li>
    <li>Alert on out-of-band drift, rank inflation, low spectral gaps, and missing provenance.</li>
  </ol>
  <h2>Recommended Controls</h2>
  <ul>{''.join(f'<li>{_esc(note)}</li>' for note in report.notes)}</ul>
  <h2>Responsible Research Boundary</h2>
  <p class=\"note\">This diagnostic stores derived telemetry only. It does not include payload-hiding code, keyed extraction, monitor bypass logic, or operational covert-channel tooling.</p>
</body>
</html>
"""


def write_report(
    telemetry_path: str | Path,
    out_path: str | Path,
    deployment_name: str = "Demo Deployment",
    fmt: str | None = None,
    baseline_path: str | Path | None = None,
    findings_path: str | Path | None = None,
    frontier_path: str | Path | None = None,
    template: str = "standard",
) -> AuditReport:
    """Load telemetry JSONL, render an audit report, and write Markdown or HTML."""
    records = load_telemetry_jsonl(telemetry_path)
    baseline_profiles = load_baseline_profiles(baseline_path) if baseline_path else []
    findings = load_findings_json(findings_path) if findings_path else []
    frontier = load_frontier_report(frontier_path) if frontier_path else None
    report = generate_audit_report(deployment_name, records, findings=findings)
    out_path = Path(out_path)
    fmt = (fmt or out_path.suffix.lstrip(".") or "md").lower()

    if fmt in {"md", "markdown"}:
        renderer = (
            render_investor_markdown_report
            if template == "investor"
            else render_markdown_report
        )
        content = renderer(
            report,
            records,
            baseline_profiles=baseline_profiles,
            findings=findings,
            frontier=frontier,
        )
    elif fmt == "html":
        html_renderer = render_investor_html_report if template == "investor" else render_html_report
        content = html_renderer(
            report,
            records,
            baseline_profiles=baseline_profiles,
            findings=findings,
            frontier=frontier,
        )
    elif fmt == "json":
        content = json.dumps(report.model_dump(), indent=2)
    else:
        raise ValueError(f"Unsupported report format: {fmt}")

    out_path.write_text(content, encoding="utf-8")
    return report


def load_frontier_report(path: str | Path) -> CapacityFrontierReport:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return CapacityFrontierReport.model_validate(payload)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _max_risk(left: str, right: str) -> str:
    return left if RISK_ORDER[left] >= RISK_ORDER[right] else right


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _risk_color(risk_level: str) -> str:
    return {"LOW": "#15803d", "MODERATE": "#b45309", "HIGH": "#b91c1c"}.get(
        risk_level, "#374151"
    )


def _html_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    head = "".join(f"<th>{_esc(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_esc(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _html_baselines(profiles: list[BaselineProfile]) -> str:
    if not profiles:
        return ""
    rows = [
        [
            profile.baseline_id,
            profile.task_family,
            profile.module_id,
            profile.collection_mode,
            profile.model_id or "any",
            str(profile.record_count),
            ", ".join(sorted(profile.metric_bands)),
        ]
        for profile in profiles
    ]
    return "<h2>Baseline Coverage</h2>" + _html_table(
        ["Baseline ID", "Task", "Module", "Mode", "Model", "Records", "Metrics"], rows
    )


def _html_findings(findings: list[AnomalyFinding]) -> str:
    if not findings:
        return ""
    rows = [
        [
            finding.run_id[:8],
            finding.risk_level,
            finding.baseline_id or "missing",
            ", ".join(finding.out_of_band_metrics) or "none",
            "; ".join(finding.messages),
        ]
        for finding in findings
    ]
    return "<h2>Anomaly Findings</h2>" + _html_table(
        ["Run ID", "Risk", "Baseline", "Flags", "Summary"], rows
    )


def _html_frontier(frontier: CapacityFrontierReport | None) -> str:
    if frontier is None:
        return ""
    rows = [
        [
            f"{point.signal_strength:.4f}",
            f"{point.recoverability_auc:.4f}",
            f"{point.anomaly_rate:.4f}",
        ]
        for point in frontier.points
    ]
    return "<h2>Capacity-Detectability Frontier</h2>" + _html_table(
        ["Signal", "Recoverability AUC", "Anomaly Rate"], rows
    )
