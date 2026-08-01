import uuid

from rich.console import Console
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
from wso.telemetry.provenance import hash_text
from wso.telemetry.schemas import TransientLoRATelemetry

console = Console()


def main() -> None:
    console.print("[bold green]Initializing Weight-Space Observatory...[/bold green]")

    console.print("Generating task-conditioned baseline manifold anchor...")
    baseline_delta, _, _ = simulate_transient_lora_update(seed=7)
    baseline_subspace = get_principal_subspace(baseline_delta, rank=4)

    console.print("Simulating active inference telemetry (Module: layers.29.mlp.down_proj)...")
    live_delta, live_logits, live_weights = simulate_transient_lora_update(seed=11)

    full_spectrum = compute_full_spectrum(live_delta)
    singular_vals = compute_spectral_signature(live_delta, top_k=8)
    gap = calculate_spectral_gap(full_spectrum, k=4)
    entropy = calculate_fusion_entropy(live_logits)
    f_norm = frobenius_norm(live_delta)
    eff_rank = effective_rank_from_singular_values(full_spectrum)

    live_subspace = get_principal_subspace(live_delta, rank=4)
    drift = grassmann_geodesic_distance(baseline_subspace, live_subspace)

    prompt = "Write a Python module that parses CSV rows."
    sender_count = 3
    target_rank = 4

    telemetry_record = TransientLoRATelemetry(
        run_id=str(uuid.uuid4()),
        module_id="layers.29.mlp.down_proj",
        task_family="code_generation",
        prompt_hash=hash_text(prompt),
        collection_mode="synthetic",
        model_id="toy-lora-router",
        adapter_id="simulated-transient",
        layer_shape=list(live_delta.shape),
        sender_count=sender_count,
        rank_bound=sender_count * target_rank,
        sender_ids=["deepseek-coder", "qwen2.5-math", "llama3-8b"],
        target_rank=target_rank,
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
            "raw_tensor_persisted": False,
        },
    )

    with open("toy_telemetry.jsonl", "w", encoding="utf-8") as f:
        f.write(telemetry_record.model_dump_json() + "\n")

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
    console.print("\n[bold yellow]Artifact Written:[/bold yellow] toy_telemetry.jsonl")


if __name__ == "__main__":
    main()
