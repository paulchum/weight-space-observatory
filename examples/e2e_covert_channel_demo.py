from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.table import Table

from wso.baselines.detectors import aggregate_risk, analyze_records, write_findings_json
from wso.baselines.task_conditioned import build_baseline_profiles, write_baseline_profiles
from wso.experiments.redteam import WeightSpaceCovertChannel
from wso.metrics.grassmann import get_principal_subspace, grassmann_geodesic_distance
from wso.metrics.spectra import (
    calculate_spectral_gap,
    compute_full_spectrum,
    compute_spectral_signature,
    effective_rank_from_singular_values,
    frobenius_norm,
)
from wso.telemetry.schemas import AnomalyFinding, TransientLoRATelemetry


DEFAULT_PAYLOAD = (
    b"SIMULATED_RED_TEAM_PAYLOAD:"
    b"inert-local-marker-for-weight-space-observatory-detection-demo;"
    b"no-network-no-execution-no-secrets;"
    b"funding-proof-of-value"
)
MODULE_ID = "tiny_linear.weight"
TASK_FAMILY = "redteam_lsb_demo"
COLLECTION_MODE = "redteam_e2e"
MODEL_ID = "tiny-linear-fp32"
BASELINE_RECORDS = 24
TOP_K = 8
console = Console()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_demo(
        out_dir=args.out_dir,
        payload_file=args.payload_file,
        seed=args.seed,
        dim=args.dim,
        rank=args.rank,
    )
    _print_summary(result)
    return 0 if result["aggregate_risk"] == "HIGH" else 1


def run_demo(
    *,
    out_dir: Path,
    payload_file: Path | None = None,
    seed: int = 7,
    dim: int = 256,
    rank: int = 4,
) -> dict[str, Any]:
    """Run the bounded red-team/blue-team LSB telemetry demo."""
    if dim <= 0:
        raise ValueError("dim must be positive")
    if rank <= 0 or rank > dim:
        raise ValueError("rank must satisfy 0 < rank <= dim")

    payload = _load_payload(payload_file)
    out_dir.mkdir(parents=True, exist_ok=True)

    anchor_delta = _make_low_rank_anchor(dim=dim, rank=rank, seed=seed + 100)
    anchor_subspace = get_principal_subspace(anchor_delta, rank=rank)
    baseline_records = [
        _telemetry_record(
            _benign_low_rank_delta(anchor_delta, index=idx, rank=rank, seed=seed + 200),
            anchor_subspace=anchor_subspace,
            run_id=f"baseline-{idx:02d}",
            role="blue_team_baseline",
            payload_metadata=None,
            rank=rank,
        )
        for idx in range(BASELINE_RECORDS)
    ]

    profiles = build_baseline_profiles(baseline_records)
    clean_control = _telemetry_record(
        _benign_low_rank_delta(anchor_delta, index=BASELINE_RECORDS + 1, rank=rank, seed=seed + 200),
        anchor_subspace=anchor_subspace,
        run_id="clean-control",
        role="blue_team_clean_control",
        payload_metadata=None,
        rank=rank,
    )

    torch.manual_seed(seed)
    model = torch.nn.Linear(dim, dim, bias=False)
    state_dict = model.state_dict()
    clean_weight = state_dict["weight"].detach().clone()
    capacity_bytes = max(0, clean_weight.numel() // 8 - WeightSpaceCovertChannel.HEADER_SIZE)
    if len(payload) > capacity_bytes:
        raise ValueError(
            f"Payload is {len(payload)} bytes but layer capacity is {capacity_bytes} bytes"
        )

    channel = WeightSpaceCovertChannel()
    channel.inject(state_dict, "weight", payload)
    extracted = channel.extract(state_dict, "weight")
    if extracted != payload:
        raise RuntimeError("Injected payload failed the defensive fixture roundtrip")

    infected_weight = state_dict["weight"].detach().clone()
    injection_delta = infected_weight - clean_weight
    payload_metadata = {
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "payload_bytes": len(payload),
        "layer_capacity_bytes": capacity_bytes,
        "payload_source": str(payload_file) if payload_file is not None else "default_inert_fixture",
        "changed_weight_elements": int((injection_delta != 0).sum().item()),
        "max_abs_weight_delta": float(injection_delta.abs().max().item()),
    }
    infected_record = _telemetry_record(
        injection_delta,
        anchor_subspace=anchor_subspace,
        run_id="infected-lsb-write-event",
        role="red_team_lsb_injection_delta",
        payload_metadata=payload_metadata,
        rank=rank,
    )

    findings = analyze_records([clean_control, infected_record], profiles)
    aggregate = aggregate_risk(findings)
    telemetry_records = [*baseline_records, clean_control, infected_record]

    telemetry_path = out_dir / "telemetry.jsonl"
    detection_path = out_dir / "detection.jsonl"
    baseline_path = out_dir / "baseline.json"
    findings_path = out_dir / "findings.json"
    summary_path = out_dir / "summary.json"

    _write_telemetry_jsonl(telemetry_path, telemetry_records)
    _write_detection_jsonl(detection_path, findings)
    write_baseline_profiles(profiles, baseline_path)
    write_findings_json(findings, findings_path)

    summary = {
        "schema_version": "1.0",
        "demo": "e2e_covert_channel_demo",
        "aggregate_risk": aggregate,
        "artifact_paths": {
            "telemetry_jsonl": str(telemetry_path),
            "detection_jsonl": str(detection_path),
            "baseline_json": str(baseline_path),
            "findings_json": str(findings_path),
            "summary_json": str(summary_path),
        },
        "payload": payload_metadata,
        "clean_control": _finding_summary(findings[0]),
        "infected_lsb_event": _finding_summary(findings[1]),
        "detection_claim": (
            "WSO telemetry flagged the LSB write-event perturbation as anomalous "
            "against the clean low-rank baseline."
        ),
        "safety_boundary": (
            "The demo stores only derived telemetry and payload metadata. It does "
            "not persist raw payload bytes, model checkpoints, or raw tensors."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the bounded red-team/blue-team FP32 LSB detection demo."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/e2e_covert_channel_demo"),
        help="Artifact directory",
    )
    parser.add_argument(
        "--payload-file",
        type=Path,
        default=None,
        help="Optional local payload file. Only SHA-256 and length are persisted.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--dim", type=int, default=256, help="Linear layer width")
    parser.add_argument("--rank", type=int, default=4, help="Clean baseline rank")
    return parser.parse_args(argv)


def _load_payload(payload_file: Path | None) -> bytes:
    if payload_file is None:
        return DEFAULT_PAYLOAD
    return payload_file.read_bytes()


def _make_low_rank_anchor(
    *,
    dim: int,
    rank: int,
    seed: int,
    target_norm: float = 1e-6,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    left = torch.linalg.qr(torch.randn(dim, rank, generator=generator)).Q[:, :rank]
    right = torch.randn(rank, dim, generator=generator)
    delta = left @ right
    return delta * (target_norm / torch.linalg.norm(delta, ord="fro"))


def _benign_low_rank_delta(
    anchor_delta: torch.Tensor,
    *,
    index: int,
    rank: int,
    seed: int,
    jitter_scale: float = 0.03,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed + index)
    anchor_subspace = get_principal_subspace(anchor_delta, rank=rank)
    jitter = anchor_subspace @ torch.randn(rank, anchor_delta.shape[1], generator=generator)
    anchor_norm = torch.linalg.norm(anchor_delta, ord="fro")
    jitter = jitter * (anchor_norm / torch.linalg.norm(jitter, ord="fro"))
    delta = anchor_delta + jitter_scale * jitter
    return delta * (anchor_norm / torch.linalg.norm(delta, ord="fro"))


def _telemetry_record(
    delta_w: torch.Tensor,
    *,
    anchor_subspace: torch.Tensor,
    run_id: str,
    role: str,
    payload_metadata: dict[str, Any] | None,
    rank: int,
) -> TransientLoRATelemetry:
    singular_values = compute_full_spectrum(delta_w)
    current_subspace = get_principal_subspace(delta_w, rank=rank)
    provenance: dict[str, Any] = {
        "demo": "e2e_covert_channel_demo",
        "role": role,
        "raw_tensor_persisted": False,
        "raw_payload_persisted": False,
    }
    if payload_metadata is not None:
        provenance.update(payload_metadata)

    return TransientLoRATelemetry(
        run_id=run_id,
        module_id=MODULE_ID,
        task_family=TASK_FAMILY,
        collection_mode=COLLECTION_MODE,
        model_id=MODEL_ID,
        adapter_id="none",
        observer_level="derived_weight_space_telemetry",
        layer_shape=list(delta_w.shape),
        sender_count=1,
        rank_bound=rank,
        sender_ids=["local-red-team-fixture"],
        target_rank=rank,
        fusion_gate_probs=[1.0],
        fusion_entropy=0.0,
        frobenius_norm=frobenius_norm(delta_w),
        singular_values=compute_spectral_signature(delta_w, top_k=TOP_K).tolist(),
        effective_rank=effective_rank_from_singular_values(singular_values),
        spectral_gap=calculate_spectral_gap(singular_values, k=rank),
        grassmann_drift=grassmann_geodesic_distance(anchor_subspace, current_subspace),
        provenance=provenance,
    )


def _write_telemetry_jsonl(path: Path, records: list[TransientLoRATelemetry]) -> None:
    path.write_text(
        "".join(record.model_dump_json() + "\n" for record in records),
        encoding="utf-8",
    )


def _write_detection_jsonl(path: Path, findings: list[AnomalyFinding]) -> None:
    path.write_text(
        "".join(finding.model_dump_json() + "\n" for finding in findings),
        encoding="utf-8",
    )


def _finding_summary(finding: AnomalyFinding) -> dict[str, Any]:
    return {
        "run_id": finding.run_id,
        "risk_level": finding.risk_level,
        "out_of_band_metrics": finding.out_of_band_metrics,
        "metric_values": finding.metric_values,
        "messages": finding.messages,
    }


def _print_summary(summary: dict[str, Any]) -> None:
    table = Table(title="Red-Team vs. Blue-Team LSB Detection Demo")
    table.add_column("Record", style="cyan", no_wrap=True)
    table.add_column("Risk", style="magenta", no_wrap=True)
    table.add_column("Out-of-band metrics", style="green")

    clean = summary["clean_control"]
    infected = summary["infected_lsb_event"]
    table.add_row(clean["run_id"], clean["risk_level"], ", ".join(clean["out_of_band_metrics"]) or "none")
    table.add_row(
        infected["run_id"],
        infected["risk_level"],
        ", ".join(infected["out_of_band_metrics"]) or "none",
    )
    console.print(table)
    console.print(f"[bold]Aggregate risk:[/bold] {summary['aggregate_risk']}")
    console.print(f"[bold]Payload SHA-256:[/bold] {summary['payload']['payload_sha256']}")
    console.print(f"[bold]Artifacts:[/bold] {summary['artifact_paths']['summary_json']}")


if __name__ == "__main__":
    raise SystemExit(main())
