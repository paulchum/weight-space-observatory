from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

from rich.console import Console

from wso.baselines.detectors import analyze_records, aggregate_risk, write_findings_json
from wso.baselines.task_conditioned import (
    build_baseline_profiles,
    load_baseline_profiles,
    write_baseline_profiles,
)
from wso.experiments.capacity_curve import (
    build_frontier_report,
    default_signal_strengths,
    simulate_frontier,
    summarize_frontier,
    write_frontier_csv,
    write_frontier_json,
    write_frontier_plot,
)
from wso.experiments.covert_channel_suite import (
    AuditGateDetectionConfig,
    FusionGateTimingConfig,
    LoraWatermarkAuditConfig,
    TFlowObserverComparisonConfig,
    run_weight_space_detection_suite,
)
from wso.pilot import run_design_partner_pilot, run_local_audit_pilot
from wso.reports.report_generator import load_telemetry_jsonl, write_report
from wso.telemetry.otel import write_otel_json

console = Console()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="wso-audit",
        description="Weight-Space Observatory defensive telemetry CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    toy = sub.add_parser("toy-demo", help="Run the toy transient LoRA telemetry demo")
    toy.add_argument("--output", default="toy_telemetry.jsonl", help="Telemetry JSONL output path")

    hook_demo = sub.add_parser("hook-demo", help="Run a no-network LoRA hook telemetry demo")
    hook_demo.add_argument(
        "--output", default="hooked_telemetry.jsonl", help="Telemetry JSONL output path"
    )

    pilot = sub.add_parser("pilot", help="Generate a local enterprise audit pilot package")
    pilot.add_argument("--deployment", default="Demo Deployment", help="Deployment name")
    pilot.add_argument("--out-dir", type=Path, default=Path("wso_pilot"), help="Artifact directory")
    pilot.add_argument("--format", choices=["md", "html", "json"], default="html")
    pilot.add_argument("--baseline-records", type=int, default=12)

    design_partner = sub.add_parser(
        "design-partner",
        help="Generate a buyer-grade design-partner evidence package",
    )
    design_partner.add_argument("--config", type=Path, required=True, help="wso YAML config path")
    design_partner.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Artifact directory; defaults to wso_design_partner",
    )
    design_partner.add_argument(
        "--format",
        choices=["md", "html", "json"],
        default=None,
        help="Override report format from the config",
    )

    validate = sub.add_parser("validate", help="Validate telemetry JSONL records")
    validate.add_argument("telemetry", help="Path to telemetry JSONL")

    report = sub.add_parser("report", help="Generate a Markdown/HTML/JSON audit report")
    report.add_argument("telemetry", help="Path to telemetry JSONL")
    report.add_argument("--out", default="weight_space_audit_report.md", help="Report output path")
    report.add_argument("--deployment", default="Demo Deployment", help="Deployment name")
    report.add_argument("--format", choices=["md", "html", "json"], default=None)
    report.add_argument("--baseline", default=None, help="Optional baseline JSON path")
    report.add_argument("--findings", default=None, help="Optional anomaly findings JSON path")
    report.add_argument("--frontier", default=None, help="Optional capacity frontier JSON path")
    report.add_argument("--template", choices=["standard", "investor"], default="standard")

    export = sub.add_parser("export", help="Export telemetry into external observability formats")
    export_sub = export.add_subparsers(dest="export_command", required=True)
    export_otel = export_sub.add_parser("otel", help="Export telemetry as OTLP-friendly JSON")
    export_otel.add_argument("telemetry", help="Path to telemetry JSONL")
    export_otel.add_argument("--out", type=Path, default=Path("telemetry_otel.json"))
    export_otel.add_argument("--service-name", default="weight-space-observatory")
    export_otel.add_argument("--deployment", default=None, help="Optional deployment name")

    benchmark = sub.add_parser("benchmark", help="Run defensive benchmark workflows")
    benchmark_sub = benchmark.add_subparsers(dest="benchmark_command", required=True)
    capacity = benchmark_sub.add_parser(
        "capacity", help="Run the synthetic capacity-detectability frontier"
    )
    capacity.add_argument("--out", type=Path, default=Path("frontier.csv"), help="CSV output path")
    capacity.add_argument("--json-out", type=Path, default=None, help="JSON summary output path")
    capacity.add_argument("--plot", type=Path, default=None, help="Optional PNG plot output path")
    capacity.add_argument("--seed", type=int, default=7, help="Random seed")
    capacity.add_argument("--max-strength", type=float, default=3.0, help="Maximum signal strength")
    capacity.add_argument("--steps", type=int, default=13, help="Number of frontier points")
    capacity.add_argument("--samples-per-strength", type=int, default=400)
    capacity.add_argument("--benign-samples", type=int, default=1000)
    capacity.add_argument("--noise-std", type=float, default=1.0)

    baseline = sub.add_parser("baseline", help="Build and inspect baseline profiles")
    baseline_sub = baseline.add_subparsers(dest="baseline_command", required=True)
    baseline_build = baseline_sub.add_parser("build", help="Build task-conditioned baselines")
    baseline_build.add_argument("telemetry", help="Path to telemetry JSONL")
    baseline_build.add_argument("--out", type=Path, default=Path("baseline.json"))
    baseline_build.add_argument("--quantile-low", type=float, default=0.05)
    baseline_build.add_argument("--quantile-high", type=float, default=0.95)

    analyze = sub.add_parser("analyze", help="Analyze telemetry against baseline profiles")
    analyze.add_argument("telemetry", help="Path to telemetry JSONL")
    analyze.add_argument("--baseline", required=True, help="Baseline JSON path")
    analyze.add_argument("--out", type=Path, default=Path("findings.json"))

    suite = sub.add_parser(
        "suite",
        help="Run the defensive weight-space covert-channel detection suite",
    )
    suite.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/weight_space_detection_suite"),
        help="Artifact directory",
    )
    suite.add_argument("--seed", type=int, default=7, help="Root random seed")
    suite.add_argument(
        "--quick",
        action="store_true",
        help="Use a smaller smoke-test profile for local CI",
    )

    args = parser.parse_args(argv)

    if args.command == "toy-demo":
        return _run_toy_demo(Path(args.output))
    if args.command == "hook-demo":
        return _run_hook_demo(Path(args.output))
    if args.command == "pilot":
        return _run_pilot(args)
    if args.command == "design-partner":
        return _run_design_partner(args)
    if args.command == "validate":
        records = load_telemetry_jsonl(args.telemetry)
        console.print(f"[bold green]Validated {len(records)} telemetry record(s).[/bold green]")
        return 0
    if args.command == "report":
        audit_report = write_report(
            args.telemetry,
            args.out,
            args.deployment,
            args.format,
            baseline_path=args.baseline,
            findings_path=args.findings,
            frontier_path=args.frontier,
            template=args.template,
        )
        console.print(
            f"[bold green]Wrote report:[/bold green] {args.out} "
            f"([bold]risk={audit_report.risk_level}[/bold])"
        )
        return 0
    if args.command == "export" and args.export_command == "otel":
        return _run_export_otel(args)
    if args.command == "benchmark" and args.benchmark_command == "capacity":
        return _run_capacity_benchmark(args)
    if args.command == "baseline" and args.baseline_command == "build":
        return _run_baseline_build(args)
    if args.command == "analyze":
        return _run_analyze(args)
    if args.command == "suite":
        return _run_suite(args)

    parser.print_help()
    return 1


def _run_toy_demo(output_path: Path) -> int:
    """Run a packaged toy transient LoRA telemetry demo."""
    import hashlib
    import uuid

    from rich.table import Table

    from wso.adapters.toy_lora import simulate_transient_lora_update
    from wso.metrics.fusion import calculate_fusion_entropy
    from wso.metrics.grassmann import get_principal_subspace, grassmann_geodesic_distance
    from wso.metrics.spectra import (
        calculate_spectral_gap,
        compute_full_spectrum,
        compute_spectral_signature,
        effective_rank_from_singular_values,
        frobenius_norm,
    )
    from wso.telemetry.schemas import TransientLoRATelemetry

    def hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    console.print("[bold green]Initializing Weight-Space Security Observatory...[/bold green]")
    console.print("Generating task-conditioned baseline manifold anchor...")
    baseline_delta, _, _ = simulate_transient_lora_update(seed=7)
    baseline_subspace = get_principal_subspace(baseline_delta, rank=4)

    console.print("Simulating active inference telemetry (Module: layers.29.mlp.down_proj)...")
    live_delta, live_logits, live_weights = simulate_transient_lora_update(seed=13)

    full_spectrum = compute_full_spectrum(live_delta)
    singular_vals = compute_spectral_signature(live_delta, top_k=8)
    gap = calculate_spectral_gap(full_spectrum, k=4)
    entropy = calculate_fusion_entropy(live_logits)
    f_norm = frobenius_norm(live_delta)
    eff_rank = effective_rank_from_singular_values(full_spectrum)
    live_subspace = get_principal_subspace(live_delta, rank=4)
    drift = grassmann_geodesic_distance(baseline_subspace, live_subspace)

    prompt = "Write a Python module that parses CSV rows."
    telemetry_record = TransientLoRATelemetry(
        run_id=str(uuid.uuid4()),
        module_id="layers.29.mlp.down_proj",
        task_family="code_generation",
        prompt_hash=hash_text(prompt),
        collection_mode="synthetic",
        model_id="toy-lora-router",
        adapter_id="simulated-transient",
        layer_shape=list(live_delta.shape),
        sender_count=3,
        rank_bound=12,
        sender_ids=["deepseek-coder", "qwen2.5-math", "llama3-8b"],
        target_rank=4,
        fusion_gate_logits=live_logits.tolist(),
        fusion_gate_probs=live_weights.tolist(),
        fusion_entropy=entropy,
        frobenius_norm=f_norm,
        singular_values=singular_vals.tolist(),
        effective_rank=eff_rank,
        spectral_gap=gap,
        grassmann_drift=drift,
        provenance={
            "hook_version": "0.1.0",
            "demo": "toy_lora_router",
            "source": "wso-audit",
            "raw_tensor_persisted": False,
            "baseline_seed": 7,
            "live_seed": 13,
        },
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(telemetry_record.model_dump_json() + "\n", encoding="utf-8")

    table = Table(title="Transient Adapter Telemetry (Diagnostic Audit)")
    table.add_column("Metric", justify="right", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    table.add_column("Example Policy", justify="right", style="green")
    table.add_row("Fusion Entropy", f"{entropy:.4f}", "< 1.05")
    table.add_row("Spectral Gap (σ4-σ5)", f"{gap:.4f}", "> 0.10")
    table.add_row("Grassmannian Drift", f"{drift:.4f}", "< 0.85")
    table.add_row("Effective Rank", f"{eff_rank:.4f}", "< 12.50")
    table.add_row("Frobenius Norm", f"{f_norm:.4f}", "task baseline")
    console.print(table)
    console.print(f"[bold green]Telemetry artifact:[/bold green] {output_path}")
    return 0


def _run_hook_demo(output_path: Path) -> int:
    """Run a no-network LoRA telemetry hook demo."""

    import torch

    from wso.telemetry.hooks import collect_peft_lora_telemetry

    class TinyLoRALinear(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(8, 2, bias=False)})
            self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 6, bias=False)})
            self.scaling = {"default": 2.0}

    torch.manual_seed(23)
    module = TinyLoRALinear()
    record = collect_peft_lora_telemetry(
        module,
        run_id=str(uuid.uuid4()),
        module_id="tiny.layers.0.lora",
        adapter_id="default",
        task_family="hook_demo",
        model_id="tiny-local-lora",
        provenance={"demo": "hook-demo", "network_access_required": False},
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(record.model_dump_json() + "\n", encoding="utf-8")
    console.print("[bold green]Collected LoRA hook telemetry without network access.[/bold green]")
    console.print(f"[bold green]Telemetry artifact:[/bold green] {output_path}")
    return 0


def _run_pilot(args: argparse.Namespace) -> int:
    artifacts = run_local_audit_pilot(
        deployment_name=args.deployment,
        out_dir=args.out_dir,
        report_format=args.format,
        baseline_records=args.baseline_records,
    )
    console.print(
        f"[bold green]Wrote pilot package:[/bold green] {artifacts.out_dir} "
        f"([bold]risk={artifacts.risk_level}[/bold])"
    )
    console.print(f"[bold green]Report:[/bold green] {artifacts.report}")
    console.print(f"[bold green]OTEL export:[/bold green] {artifacts.otel}")
    return 0


def _run_design_partner(args: argparse.Namespace) -> int:
    artifacts = run_design_partner_pilot(
        config_path=args.config,
        out_dir=args.out_dir,
        report_format=args.format,
    )
    console.print(
        f"[bold green]Wrote design-partner evidence pack:[/bold green] {artifacts.out_dir} "
        f"([bold]risk={artifacts.risk_level}[/bold])"
    )
    console.print(f"[bold green]Report:[/bold green] {artifacts.report}")
    console.print(f"[bold green]Pilot proposal:[/bold green] {artifacts.pilot_proposal}")
    console.print(f"[bold green]OTEL export:[/bold green] {artifacts.otel}")
    return 0


def _run_export_otel(args: argparse.Namespace) -> int:
    records = load_telemetry_jsonl(args.telemetry)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_otel_json(
        records,
        args.out,
        service_name=args.service_name,
        deployment_name=args.deployment,
    )
    console.print(
        f"[bold green]Wrote OTEL JSON:[/bold green] {args.out} "
        f"([bold]{len(records)} span(s)[/bold])"
    )
    return 0


def _run_capacity_benchmark(args: argparse.Namespace) -> int:
    strengths = default_signal_strengths(args.max_strength, args.steps)
    points = simulate_frontier(
        strengths,
        samples_per_strength=args.samples_per_strength,
        benign_samples=args.benign_samples,
        noise_std=args.noise_std,
        seed=args.seed,
    )
    report = build_frontier_report(
        points,
        samples_per_strength=args.samples_per_strength,
        benign_samples=args.benign_samples,
        noise_std=args.noise_std,
        seed=args.seed,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_frontier_csv(points, args.out)
    json_out = args.json_out or args.out.with_suffix(".json")
    json_out.parent.mkdir(parents=True, exist_ok=True)
    write_frontier_json(report, json_out)
    if args.plot is not None:
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        write_frontier_plot(points, args.plot)
        console.print(f"[bold green]Wrote frontier plot:[/bold green] {args.plot}")

    console.print(summarize_frontier(points))
    console.print(f"[bold green]Wrote frontier CSV:[/bold green] {args.out}")
    console.print(f"[bold green]Wrote frontier JSON:[/bold green] {json_out}")
    return 0


def _run_baseline_build(args: argparse.Namespace) -> int:
    records = load_telemetry_jsonl(args.telemetry)
    profiles = build_baseline_profiles(
        records,
        quantile_low=args.quantile_low,
        quantile_high=args.quantile_high,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_baseline_profiles(profiles, args.out)
    console.print(
        f"[bold green]Wrote {len(profiles)} baseline profile(s):[/bold green] {args.out}"
    )
    return 0


def _run_analyze(args: argparse.Namespace) -> int:
    records = load_telemetry_jsonl(args.telemetry)
    profiles = load_baseline_profiles(args.baseline)
    findings = analyze_records(records, profiles)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_findings_json(findings, args.out)
    console.print(
        f"[bold green]Wrote {len(findings)} finding(s):[/bold green] {args.out} "
        f"([bold]risk={aggregate_risk(findings)}[/bold])"
    )
    return 0


def _run_suite(args: argparse.Namespace) -> int:
    if args.quick:
        summary = run_weight_space_detection_suite(
            args.out_dir,
            seed=args.seed,
            experiment_a=LoraWatermarkAuditConfig(trials=20, dim=32, seed=args.seed + 101),
            experiment_b=FusionGateTimingConfig(bits=16, seed=args.seed + 202),
            experiment_c=AuditGateDetectionConfig(
                legitimate_sequences=10,
                covert_sequences=10,
                snapshots_per_sequence=4,
                dim=32,
                seed=args.seed + 303,
            ),
            experiment_d=TFlowObserverComparisonConfig(
                task_families=("math_reasoning", "code_generation"),
                module_ids=(
                    "receiver.layers.3.self_attn.q_proj",
                    "receiver.layers.7.mlp.down_proj",
                ),
                ranks=(2, 4),
                sender_counts=(1, 3),
                fusion_noise_sigmas=(0.0, 0.16),
                baseline_samples_per_group=6,
                heldout_samples_per_group=4,
                stress_samples_per_condition=1,
                dim=24,
                seed=args.seed + 404,
            ),
        )
    else:
        summary = run_weight_space_detection_suite(args.out_dir, seed=args.seed)
    experiments = summary["experiments"]
    console.print("[bold green]Wrote detection suite artifacts:[/bold green] " f"{args.out_dir}")
    for name, result in experiments.items():
        passed = all(result["criteria"].values())
        status = "PASS" if passed else "REVIEW"
        console.print(f"[bold]{status}[/bold] {name}")
    console.print(f"[bold green]Summary:[/bold green] {args.out_dir / 'suite_summary.json'}")
    if "tensorboard_logdir" in summary.get("dashboard", {}):
        console.print(
            "[bold green]TensorBoard logdir:[/bold green] "
            f"{summary['dashboard']['tensorboard_logdir']}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
