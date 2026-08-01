"""Local commercial pilot workflow for Weight-Space Observatory."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import yaml
from pydantic import BaseModel, Field

from wso.baselines.detectors import analyze_records, aggregate_risk, write_findings_json
from wso.baselines.task_conditioned import build_baseline_profiles, write_baseline_profiles
from wso.reports.report_generator import load_telemetry_jsonl, write_report
from wso.telemetry.hooks import collect_peft_lora_telemetry
from wso.telemetry.otel import write_otel_json
from wso.telemetry.provenance import hash_text
from wso.telemetry.schemas import TransientLoRATelemetry


@dataclass(frozen=True)
class PilotArtifacts:
    """Paths and headline result for a generated audit pilot package."""

    out_dir: Path
    baseline_telemetry: Path
    audit_telemetry: Path
    baseline: Path
    findings: Path
    otel: Path
    report: Path
    manifest: Path
    risk_level: str


class DeploymentConfig(BaseModel):
    name: str
    id: str | None = None


class RuntimeConfig(BaseModel):
    framework: str = "torch_peft_compatible"
    model_id: str | None = "customer-agent-runtime"
    observer_level: str = "derived_weight_space_telemetry"


class ObservabilityConfig(BaseModel):
    service_name: str = "weight-space-observatory"
    service_namespace: str = "agent-security"
    destination_mode: Literal["file"] = "file"


class ModuleSelectorConfig(BaseModel):
    module_id: str
    adapter_id: str | None = "default"
    task_family: str = "agent_tool_use"


class TelemetryInputConfig(BaseModel):
    baseline_telemetry: Path | None = None
    audit_telemetry: Path | None = None


class ReportConfig(BaseModel):
    format: Literal["md", "html", "json"] = "html"


class DesignPartnerConfig(BaseModel):
    """Config for a buyer-grade design-partner audit evidence pack."""

    schema_version: str = "1.0"
    mode: Literal["demo", "customer"] = "demo"
    deployment: DeploymentConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    modules: list[ModuleSelectorConfig] = Field(default_factory=list)
    inputs: TelemetryInputConfig = Field(default_factory=TelemetryInputConfig)
    baseline_records: int = 12
    report: ReportConfig = Field(default_factory=ReportConfig)


@dataclass(frozen=True)
class DesignPartnerArtifacts:
    """Paths and headline result for a design-partner evidence package."""

    out_dir: Path
    baseline_telemetry: Path
    audit_telemetry: Path
    baseline: Path
    findings: Path
    otel: Path
    report: Path
    manifest: Path
    pilot_proposal: Path
    buyer_checklist: Path
    safety_boundary: Path
    risk_level: str


def run_local_audit_pilot(
    *,
    deployment_name: str,
    out_dir: str | Path,
    report_format: str = "html",
    baseline_records: int = 12,
) -> PilotArtifacts:
    """Generate a self-contained real-stack audit pilot package.

    The pilot uses a tiny PEFT-compatible LoRA module so it can run offline while
    exercising the same collector path a customer integration would use.
    """

    if baseline_records < 6:
        raise ValueError("pilot baseline requires at least 6 records")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    deployment_id = _slug_id(deployment_name)
    baseline_path = out_dir / "baseline_telemetry.jsonl"
    audit_path = out_dir / "audit_telemetry.jsonl"
    baseline_json = out_dir / "baseline.json"
    findings_json = out_dir / "findings.json"
    otel_json = out_dir / "telemetry_otel.json"
    report_path = out_dir / f"hidden_agent_communication_surface_audit.{_report_suffix(report_format)}"
    manifest_path = out_dir / "manifest.json"

    baseline = [
        _collect_pilot_record(
            seed=200 + idx,
            scale=1.0 + (idx % 3) * 0.04,
            deployment_id=deployment_id,
            prompt=f"baseline enterprise agent task {idx}",
            scenario="baseline",
        )
        for idx in range(baseline_records)
    ]
    audit = [
        _collect_pilot_record(
            seed=700,
            scale=1.02,
            deployment_id=deployment_id,
            prompt="summarize an internal policy document",
            scenario="audit_normal_policy_summary",
        ),
        _collect_pilot_record(
            seed=701,
            scale=1.04,
            deployment_id=deployment_id,
            prompt="route a customer support ticket to the right tool",
            scenario="audit_normal_tool_routing",
        ),
        _collect_pilot_record(
            seed=702,
            scale=4.0,
            deployment_id=deployment_id,
            prompt="coordinate a multi-agent analysis with unusually strong adapter routing",
            scenario="audit_shifted_adapter_surface",
        ),
    ]

    _write_jsonl(baseline, baseline_path)
    _write_jsonl(audit, audit_path)
    profiles = build_baseline_profiles(baseline)
    write_baseline_profiles(profiles, baseline_json)
    findings = analyze_records(audit, profiles)
    write_findings_json(findings, findings_json)
    write_otel_json(
        audit,
        otel_json,
        deployment_name=deployment_name,
    )
    report = write_report(
        audit_path,
        report_path,
        deployment_name,
        report_format,
        baseline_path=baseline_json,
        findings_path=findings_json,
        template="investor",
    )
    risk_level = aggregate_risk(findings) if findings else report.risk_level

    manifest = {
        "schema_version": "1.0",
        "deployment_name": deployment_name,
        "deployment_id": deployment_id,
        "risk_level": risk_level,
        "artifacts": {
            "baseline_telemetry": str(baseline_path),
            "audit_telemetry": str(audit_path),
            "baseline": str(baseline_json),
            "findings": str(findings_json),
            "otel": str(otel_json),
            "report": str(report_path),
        },
        "commercial_wedge": (
            "Runtime observability for adapter-mediated and weight-space agent communication."
        ),
        "safety_boundary": (
            "Derived telemetry only; no payload-hiding code, keyed extraction, or evasion tooling."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return PilotArtifacts(
        out_dir=out_dir,
        baseline_telemetry=baseline_path,
        audit_telemetry=audit_path,
        baseline=baseline_json,
        findings=findings_json,
        otel=otel_json,
        report=report_path,
        manifest=manifest_path,
        risk_level=risk_level,
    )


def load_design_partner_config(config_path: str | Path) -> DesignPartnerConfig:
    """Load and validate a design-partner YAML or JSON config."""

    path = Path(config_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        raise ValueError(f"empty design-partner config: {path}")
    if not isinstance(payload, dict):
        raise ValueError("design-partner config must be a mapping")

    config = DesignPartnerConfig.model_validate(payload)
    if config.baseline_records < 6:
        raise ValueError("design-partner baseline requires at least 6 records")
    if not config.modules:
        raise ValueError("design-partner config requires at least one module selector")
    if config.mode == "customer":
        if config.inputs.baseline_telemetry is None or config.inputs.audit_telemetry is None:
            raise ValueError("customer mode requires baseline_telemetry and audit_telemetry inputs")
        config = _resolve_config_input_paths(config, path.parent)
    return config


def run_design_partner_pilot(
    *,
    config_path: str | Path,
    out_dir: str | Path | None = None,
    report_format: str | None = None,
) -> DesignPartnerArtifacts:
    """Generate a self-hosted design-partner audit evidence package.

    Demo mode creates a deterministic offline PEFT-compatible audit. Customer mode
    packages real telemetry JSONL produced by the WSO hooks or an equivalent
    integration in the buyer environment.
    """

    config = load_design_partner_config(config_path)
    resolved_out_dir = Path(out_dir) if out_dir is not None else Path("wso_design_partner")
    resolved_out_dir.mkdir(parents=True, exist_ok=True)

    deployment_name = config.deployment.name
    deployment_id = config.deployment.id or _slug_id(deployment_name)
    baseline_path = resolved_out_dir / "baseline_telemetry.jsonl"
    audit_path = resolved_out_dir / "audit_telemetry.jsonl"
    baseline_json = resolved_out_dir / "baseline.json"
    findings_json = resolved_out_dir / "findings.json"
    otel_json = resolved_out_dir / "telemetry_otel.json"
    fmt = report_format or config.report.format
    report_path = (
        resolved_out_dir / f"hidden_agent_communication_surface_audit.{_report_suffix(fmt)}"
    )
    manifest_path = resolved_out_dir / "manifest.json"
    pilot_proposal_path = resolved_out_dir / "design_partner_pilot.md"
    buyer_checklist_path = resolved_out_dir / "security_buyer_checklist.md"
    safety_boundary_path = resolved_out_dir / "safety_boundary.md"

    if config.mode == "demo":
        baseline, audit = _build_design_partner_demo_records(config, deployment_id)
    else:
        baseline, audit = _load_customer_records(config)

    if not baseline:
        raise ValueError("design-partner baseline telemetry is empty")
    if not audit:
        raise ValueError("design-partner audit telemetry is empty")

    _write_jsonl(baseline, baseline_path)
    _write_jsonl(audit, audit_path)
    profiles = build_baseline_profiles(baseline)
    write_baseline_profiles(profiles, baseline_json)
    findings = analyze_records(audit, profiles)
    write_findings_json(findings, findings_json)
    write_otel_json(
        audit,
        otel_json,
        service_name=config.observability.service_name,
        deployment_name=deployment_name,
    )
    report = write_report(
        audit_path,
        report_path,
        deployment_name,
        fmt,
        baseline_path=baseline_json,
        findings_path=findings_json,
        template="investor",
    )
    risk_level = aggregate_risk(findings) if findings else report.risk_level

    manifest = _design_partner_manifest(
        config=config,
        deployment_id=deployment_id,
        out_dir=resolved_out_dir,
        baseline_path=baseline_path,
        audit_path=audit_path,
        baseline_json=baseline_json,
        findings_json=findings_json,
        otel_json=otel_json,
        report_path=report_path,
        pilot_proposal_path=pilot_proposal_path,
        buyer_checklist_path=buyer_checklist_path,
        safety_boundary_path=safety_boundary_path,
        risk_level=risk_level,
        baseline_count=len(baseline),
        audit_count=len(audit),
        profile_count=len(profiles),
        finding_count=len(findings),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pilot_proposal_path.write_text(
        _render_design_partner_pilot_proposal(
            config=config,
            deployment_id=deployment_id,
            risk_level=risk_level,
            baseline_count=len(baseline),
            audit_count=len(audit),
            profile_count=len(profiles),
            finding_count=len(findings),
        ),
        encoding="utf-8",
    )
    buyer_checklist_path.write_text(
        _render_buyer_checklist(config, risk_level, len(profiles), len(findings)),
        encoding="utf-8",
    )
    safety_boundary_path.write_text(_render_safety_boundary(), encoding="utf-8")

    return DesignPartnerArtifacts(
        out_dir=resolved_out_dir,
        baseline_telemetry=baseline_path,
        audit_telemetry=audit_path,
        baseline=baseline_json,
        findings=findings_json,
        otel=otel_json,
        report=report_path,
        manifest=manifest_path,
        pilot_proposal=pilot_proposal_path,
        buyer_checklist=buyer_checklist_path,
        safety_boundary=safety_boundary_path,
        risk_level=risk_level,
    )


class _TinyPEFTLoRALinear(torch.nn.Module):
    def __init__(self, *, scale: float) -> None:
        super().__init__()
        self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(8, 2, bias=False)})
        self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 6, bias=False)})
        self.scaling = {"default": scale}


def _collect_pilot_record(
    *,
    seed: int,
    scale: float,
    deployment_id: str,
    prompt: str,
    scenario: str,
    module_id: str = "pilot.receiver.layers.0.lora",
    adapter_id: str | None = "default",
    task_family: str = "agent_tool_use",
    runtime_framework: str = "torch_peft_compatible",
    model_id: str = "tiny-local-peft-pilot",
    observer_level: str = "derived_weight_space_telemetry",
    service_namespace: str = "agent-security",
    integration: str = "local_peft_compatible_pilot",
) -> TransientLoRATelemetry:
    torch.manual_seed(seed)
    module = _TinyPEFTLoRALinear(scale=scale)
    return collect_peft_lora_telemetry(
        module,
        run_id=str(uuid.uuid4()),
        module_id=module_id,
        adapter_id=adapter_id,
        task_family=task_family,
        prompt_hash=hash_text(prompt),
        deployment_id=deployment_id,
        runtime_framework=runtime_framework,
        model_id=model_id,
        observer_level=observer_level,
        otel_resource={
            "service.namespace": service_namespace,
            "wso.integration": integration,
        },
        provenance={
            "pilot_scenario": scenario,
            "seed": seed,
            "scale": scale,
            "external_tflow_required": False,
            "tflow_compatible_surface": True,
        },
    )


def _build_design_partner_demo_records(
    config: DesignPartnerConfig,
    deployment_id: str,
) -> tuple[list[TransientLoRATelemetry], list[TransientLoRATelemetry]]:
    baseline: list[TransientLoRATelemetry] = []
    audit: list[TransientLoRATelemetry] = []
    for module_idx, module in enumerate(config.modules):
        seed_offset = module_idx * 1_000
        baseline.extend(
            _collect_pilot_record(
                seed=200 + seed_offset + idx,
                scale=1.0 + (idx % 3) * 0.04,
                deployment_id=deployment_id,
                prompt=f"baseline design-partner agent task {idx}",
                scenario="design_partner_baseline",
                module_id=module.module_id,
                adapter_id=module.adapter_id,
                task_family=module.task_family,
                runtime_framework=config.runtime.framework,
                model_id=config.runtime.model_id or "tiny-local-peft-pilot",
                observer_level=config.runtime.observer_level,
                service_namespace=config.observability.service_namespace,
                integration="design_partner_demo",
            )
            for idx in range(config.baseline_records)
        )
        audit.extend(
            [
                _collect_pilot_record(
                    seed=700 + seed_offset,
                    scale=1.02,
                    deployment_id=deployment_id,
                    prompt="summarize an internal policy document",
                    scenario="design_partner_normal_policy_summary",
                    module_id=module.module_id,
                    adapter_id=module.adapter_id,
                    task_family=module.task_family,
                    runtime_framework=config.runtime.framework,
                    model_id=config.runtime.model_id or "tiny-local-peft-pilot",
                    observer_level=config.runtime.observer_level,
                    service_namespace=config.observability.service_namespace,
                    integration="design_partner_demo",
                ),
                _collect_pilot_record(
                    seed=701 + seed_offset,
                    scale=1.04,
                    deployment_id=deployment_id,
                    prompt="route a customer support ticket to the right tool",
                    scenario="design_partner_normal_tool_routing",
                    module_id=module.module_id,
                    adapter_id=module.adapter_id,
                    task_family=module.task_family,
                    runtime_framework=config.runtime.framework,
                    model_id=config.runtime.model_id or "tiny-local-peft-pilot",
                    observer_level=config.runtime.observer_level,
                    service_namespace=config.observability.service_namespace,
                    integration="design_partner_demo",
                ),
                _collect_pilot_record(
                    seed=702 + seed_offset,
                    scale=4.0,
                    deployment_id=deployment_id,
                    prompt="coordinate a multi-agent analysis with unusually strong adapter routing",
                    scenario="design_partner_shifted_adapter_surface",
                    module_id=module.module_id,
                    adapter_id=module.adapter_id,
                    task_family=module.task_family,
                    runtime_framework=config.runtime.framework,
                    model_id=config.runtime.model_id or "tiny-local-peft-pilot",
                    observer_level=config.runtime.observer_level,
                    service_namespace=config.observability.service_namespace,
                    integration="design_partner_demo",
                ),
            ]
        )
    return baseline, audit


def _load_customer_records(
    config: DesignPartnerConfig,
) -> tuple[list[TransientLoRATelemetry], list[TransientLoRATelemetry]]:
    baseline_path = config.inputs.baseline_telemetry
    audit_path = config.inputs.audit_telemetry
    if baseline_path is None or audit_path is None:
        raise ValueError("customer mode requires telemetry input paths")
    return load_telemetry_jsonl(baseline_path), load_telemetry_jsonl(audit_path)


def _resolve_config_input_paths(
    config: DesignPartnerConfig,
    base_dir: Path,
) -> DesignPartnerConfig:
    inputs = config.inputs.model_copy(
        update={
            "baseline_telemetry": _resolve_relative_path(
                config.inputs.baseline_telemetry,
                base_dir,
            ),
            "audit_telemetry": _resolve_relative_path(
                config.inputs.audit_telemetry,
                base_dir,
            ),
        }
    )
    return config.model_copy(update={"inputs": inputs})


def _resolve_relative_path(path: Path | None, base_dir: Path) -> Path | None:
    if path is None or path.is_absolute():
        return path
    return base_dir / path


def _write_jsonl(records: list[TransientLoRATelemetry], output: Path) -> None:
    output.write_text(
        "".join(record.model_dump_json() + "\n" for record in records),
        encoding="utf-8",
    )


def _report_suffix(report_format: str) -> str:
    fmt = report_format.lower()
    if fmt in {"md", "markdown"}:
        return "md"
    if fmt in {"html", "json"}:
        return fmt
    raise ValueError(f"Unsupported report format: {report_format}")


def _slug_id(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-")
    slug = "-".join(part for part in slug.split("-") if part)
    return slug or "demo-deployment"


def _design_partner_manifest(
    *,
    config: DesignPartnerConfig,
    deployment_id: str,
    out_dir: Path,
    baseline_path: Path,
    audit_path: Path,
    baseline_json: Path,
    findings_json: Path,
    otel_json: Path,
    report_path: Path,
    pilot_proposal_path: Path,
    buyer_checklist_path: Path,
    safety_boundary_path: Path,
    risk_level: str,
    baseline_count: int,
    audit_count: int,
    profile_count: int,
    finding_count: int,
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "package_type": "design_partner_evidence_pack",
        "mode": config.mode,
        "deployment_name": config.deployment.name,
        "deployment_id": deployment_id,
        "risk_level": risk_level,
        "counts": {
            "baseline_records": baseline_count,
            "audit_records": audit_count,
            "baseline_profiles": profile_count,
            "findings": finding_count,
        },
        "artifacts": {
            "baseline_telemetry": str(baseline_path),
            "audit_telemetry": str(audit_path),
            "baseline": str(baseline_json),
            "findings": str(findings_json),
            "otel": str(otel_json),
            "report": str(report_path),
            "pilot_proposal": str(pilot_proposal_path),
            "security_buyer_checklist": str(buyer_checklist_path),
            "safety_boundary": str(safety_boundary_path),
        },
        "output_directory": str(out_dir),
        "runtime": {
            "framework": config.runtime.framework,
            "model_id": config.runtime.model_id,
            "observer_level": config.runtime.observer_level,
        },
        "modules": [module.model_dump() for module in config.modules],
        "observability": config.observability.model_dump(),
        "commercial_wedge": (
            "Runtime observability for adapter-mediated and weight-space agent communication."
        ),
        "investor_target": {
            "name": "Silicon Valley CISO Investments (SVCI)",
            "why": (
                "CISO-led cybersecurity angels with early-stage buyer access and AI security "
                "portfolio adjacency."
            ),
            "url": "https://www.svci.io/",
        },
        "pilot_acceptance_criteria": [
            "Adapter or transient LoRA telemetry is collected for selected modules.",
            "Task-conditioned baselines are generated from benign records.",
            "Audit records are analyzed into a findings queue.",
            "Derived telemetry is exportable through OTLP-shaped JSON.",
            "The executive report states risk, coverage, controls, and safety boundary.",
        ],
        "safety_boundary": (
            "Derived telemetry only; no payload-hiding code, keyed extraction, evasion logic, "
            "or operational covert-channel tooling."
        ),
    }


def _render_design_partner_pilot_proposal(
    *,
    config: DesignPartnerConfig,
    deployment_id: str,
    risk_level: str,
    baseline_count: int,
    audit_count: int,
    profile_count: int,
    finding_count: int,
) -> str:
    module_rows = "\n".join(
        f"| {module.module_id} | {module.adapter_id or 'n/a'} | {module.task_family} |"
        for module in config.modules
    )
    return f"""# Design-Partner Pilot: {config.deployment.name}

## Objective

Run a one-week, self-hosted audit of non-text agent communication surfaces and prove
whether adapter, LoRA, or weight-space runtime behavior is observable enough for
enterprise security review.

## Deployment

| Field | Value |
|---|---|
| Deployment ID | {deployment_id} |
| Mode | {config.mode} |
| Runtime framework | {config.runtime.framework} |
| Model ID | {config.runtime.model_id or 'n/a'} |
| Observer level | {config.runtime.observer_level} |
| OTEL destination | {config.observability.destination_mode} |

## Instrumented Modules

| Module | Adapter | Task family |
|---|---|---|
{module_rows}

## Evidence Pack Summary

| Evidence | Count / Result |
|---|---:|
| Baseline telemetry records | {baseline_count} |
| Audit telemetry records | {audit_count} |
| Baseline profiles | {profile_count} |
| Analyzed findings | {finding_count} |
| Aggregate risk | {risk_level} |

## Pilot Success Criteria

1. The buyer can run the workflow locally without sharing raw tensors or prompts.
2. WSO produces task-conditioned baseline profiles for selected modules.
3. Audit records become a concrete findings queue for the security/platform team.
4. OTLP-shaped JSON can be routed into the buyer's observability or SIEM workflow.
5. The executive report gives the CISO a buy/no-buy risk readout.

## Investor Outreach Target

Primary target: Silicon Valley CISO Investments (SVCI). The outreach angle is a
CISO-led design-partner round for a runtime observability layer that closes the blind
spot below prompt, tool, and output logs in agent deployments.
"""


def _render_buyer_checklist(
    config: DesignPartnerConfig,
    risk_level: str,
    profile_count: int,
    finding_count: int,
) -> str:
    return f"""# Security Buyer Checklist

## Buyer Fit

- Owns or evaluates agent deployments that use adapters, LoRA, fusion gates, or transient model updates.
- Needs evidence beyond prompt, tool-call, and output logs.
- Can provide benign task families for baseline calibration.
- Wants local-first telemetry that avoids raw tensor and prompt retention.

## Technical Acceptance

- [ ] Selected modules match the deployed adapter or LoRA surfaces.
- [ ] Baseline coverage exists for each high-risk task family.
- [ ] Findings are triaged by a security or platform owner.
- [ ] OTLP-shaped JSON can be ingested by the existing observability stack.
- [ ] The responsible-research boundary is acceptable to legal and security review.

## Current Evidence

| Signal | Value |
|---|---:|
| Deployment | {config.deployment.name} |
| Aggregate risk | {risk_level} |
| Baseline profiles | {profile_count} |
| Findings | {finding_count} |
"""


def _render_safety_boundary() -> str:
    return """# Safety Boundary

Weight-Space Observatory is defensive telemetry for adapter-mediated and weight-space
agent communication surfaces.

The design-partner workflow stores derived summaries such as singular spectra,
effective rank, Frobenius norm, spectral gap, fusion entropy, provenance, and
task-conditioned anomaly findings. It does not persist raw model tensors by default.

This package does not include payload-hiding code, keyed extraction, prompt-search
steganography, monitor bypass logic, or operational covert-channel tooling.
"""
