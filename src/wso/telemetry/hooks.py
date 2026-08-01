"""Runtime hook helpers for defensive LoRA / adapter telemetry.

These helpers compute metrics from transient deltas in memory and return
``TransientLoRATelemetry`` records. Callers should persist the returned summaries,
not raw tensors.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any

import torch

from wso.metrics.fusion import calculate_fusion_entropy, fusion_probabilities
from wso.metrics.grassmann import get_principal_subspace, grassmann_geodesic_distance
from wso.metrics.spectra import (
    calculate_spectral_gap,
    compute_full_spectrum,
    compute_spectral_signature,
    effective_rank_from_singular_values,
    frobenius_norm,
)
from wso.telemetry.schemas import TransientLoRATelemetry
from wso.telemetry.schemas import EvidenceMode, TelemetrySplit


WeightedLoRAEntry = tuple[torch.Tensor, torch.Tensor, float, float | torch.Tensor]
WeightedLoRAEntries = Mapping[str, Sequence[WeightedLoRAEntry]]


class TelemetryHookRegistry:
    """Minimal registry for future adapter / LoRA telemetry hooks."""

    def __init__(self) -> None:
        self._hooks: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, hook: Callable[..., Any]) -> None:
        if not name:
            raise ValueError("hook name must be non-empty")
        self._hooks[name] = hook

    def get(self, name: str) -> Callable[..., Any] | None:
        return self._hooks.get(name)

    def names(self) -> list[str]:
        return sorted(self._hooks)


def collect_delta_telemetry(
    delta_w: torch.Tensor,
    *,
    run_id: str,
    module_id: str,
    task_family: str | None = None,
    prompt_hash: str | None = None,
    evidence_mode: EvidenceMode = "synthetic",
    sender_count: int = 1,
    target_rank: int | None = None,
    rank_bound: int | None = None,
    sender_ids: list[str] | None = None,
    fusion_gate_logits: torch.Tensor | None = None,
    collection_mode: str = "hook",
    source_kind: str | None = None,
    source_path_hashes: dict[str, str] | None = None,
    model_path: str | None = None,
    adapter_path: str | None = None,
    checkpoint_sha256: str | None = None,
    dataset_id: str | None = None,
    dataset_sha256: str | None = None,
    split: TelemetrySplit | None = None,
    collector_version: str | None = None,
    git_commit: str | None = None,
    python_version: str | None = None,
    torch_version: str | None = None,
    device: str | None = None,
    collection_timestamp: float | None = None,
    deployment_id: str | None = None,
    runtime_framework: str | None = None,
    model_id: str | None = None,
    adapter_id: str | None = None,
    observer_level: str = "derived_telemetry",
    otel_resource: dict[str, Any] | None = None,
    baseline_delta: torch.Tensor | None = None,
    provenance: dict[str, Any] | None = None,
) -> TransientLoRATelemetry:
    """Build telemetry from a 2D delta tensor without persisting the tensor."""

    if delta_w.ndim != 2:
        raise ValueError(f"delta_w must be 2D, got shape {tuple(delta_w.shape)}")
    resolved_rank = target_rank or min(4, min(delta_w.shape))
    if resolved_rank <= 0:
        raise ValueError("target rank must be positive")

    full_spectrum = compute_full_spectrum(delta_w)
    singular_values = compute_spectral_signature(delta_w, top_k=8)
    spectral_gap = calculate_spectral_gap(full_spectrum, k=min(resolved_rank, full_spectrum.numel()))
    grassmann_drift = None
    if baseline_delta is not None:
        rank = min(resolved_rank, min(delta_w.shape), min(baseline_delta.shape))
        baseline_subspace = get_principal_subspace(baseline_delta, rank=rank)
        live_subspace = get_principal_subspace(delta_w, rank=rank)
        grassmann_drift = grassmann_geodesic_distance(baseline_subspace, live_subspace)

    fusion_probs: list[float] = []
    fusion_entropy = None
    logits_list: list[float] | None = None
    if fusion_gate_logits is not None:
        logits_list = fusion_gate_logits.detach().to(dtype=torch.float32).tolist()
        fusion_probs = fusion_probabilities(fusion_gate_logits).tolist()
        fusion_entropy = calculate_fusion_entropy(fusion_gate_logits)

    return TransientLoRATelemetry(
        run_id=run_id,
        deployment_id=deployment_id,
        module_id=module_id,
        task_family=task_family,
        prompt_hash=prompt_hash,
        evidence_mode=evidence_mode,
        collection_mode=collection_mode,
        source_kind=source_kind,
        source_path_hashes=source_path_hashes or {},
        model_path=model_path,
        adapter_path=adapter_path,
        checkpoint_sha256=checkpoint_sha256,
        dataset_id=dataset_id,
        dataset_sha256=dataset_sha256,
        split=split,
        collector_version=collector_version,
        git_commit=git_commit,
        python_version=python_version,
        torch_version=torch_version,
        device=device,
        collection_timestamp=collection_timestamp,
        runtime_framework=runtime_framework,
        model_id=model_id,
        adapter_id=adapter_id,
        observer_level=observer_level,
        otel_resource=otel_resource or {},
        layer_shape=list(delta_w.shape),
        sender_count=sender_count,
        rank_bound=rank_bound,
        sender_ids=sender_ids or [],
        target_rank=resolved_rank,
        fusion_gate_logits=logits_list,
        fusion_gate_probs=fusion_probs,
        fusion_entropy=fusion_entropy,
        frobenius_norm=frobenius_norm(delta_w),
        singular_values=singular_values.tolist(),
        effective_rank=effective_rank_from_singular_values(full_spectrum),
        spectral_gap=spectral_gap,
        grassmann_drift=grassmann_drift,
        provenance={
            "collector": "wso.telemetry.hooks.collect_delta_telemetry",
            "raw_tensor_persisted": False,
            **(provenance or {}),
        },
    )


def collect_peft_lora_telemetry(
    module: Any,
    *,
    run_id: str,
    module_id: str,
    adapter_id: str | None = None,
    task_family: str | None = None,
    prompt_hash: str | None = None,
    evidence_mode: EvidenceMode = "synthetic",
    source_kind: str | None = None,
    source_path_hashes: dict[str, str] | None = None,
    model_path: str | None = None,
    adapter_path: str | None = None,
    checkpoint_sha256: str | None = None,
    dataset_id: str | None = None,
    dataset_sha256: str | None = None,
    split: TelemetrySplit | None = None,
    collector_version: str | None = None,
    git_commit: str | None = None,
    python_version: str | None = None,
    torch_version: str | None = None,
    device: str | None = None,
    collection_timestamp: float | None = None,
    deployment_id: str | None = None,
    runtime_framework: str | None = None,
    model_id: str | None = None,
    observer_level: str = "derived_telemetry",
    otel_resource: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> TransientLoRATelemetry:
    """Collect telemetry from a PEFT-style LoRA module.

    The collector supports the common PEFT layout where ``lora_A`` and ``lora_B``
    are adapter-indexed module dictionaries. It also supports tiny local modules
    exposing direct ``lora_A`` and ``lora_B`` linear layers.
    """

    lora_a, resolved_adapter = _resolve_lora_component(module, "lora_A", adapter_id)
    lora_b, _ = _resolve_lora_component(module, "lora_B", resolved_adapter)
    a_weight = _weight_tensor(lora_a)
    b_weight = _weight_tensor(lora_b)
    if a_weight.ndim != 2 or b_weight.ndim != 2:
        raise ValueError("LoRA A and B weights must be 2D")
    if b_weight.shape[1] != a_weight.shape[0]:
        raise ValueError(
            f"LoRA shapes are incompatible: B={tuple(b_weight.shape)}, A={tuple(a_weight.shape)}"
        )

    scale = _resolve_lora_scale(module, resolved_adapter, rank=a_weight.shape[0])
    delta_w = (b_weight.detach().to(dtype=torch.float32) @ a_weight.detach().to(dtype=torch.float32)) * scale
    return collect_delta_telemetry(
        delta_w,
        run_id=run_id,
        module_id=module_id,
        task_family=task_family,
        prompt_hash=prompt_hash,
        evidence_mode=evidence_mode,
        sender_count=1,
        target_rank=a_weight.shape[0],
        rank_bound=a_weight.shape[0],
        collection_mode="lora_hook",
        source_kind=source_kind,
        source_path_hashes=source_path_hashes,
        model_path=model_path,
        adapter_path=adapter_path,
        checkpoint_sha256=checkpoint_sha256,
        dataset_id=dataset_id,
        dataset_sha256=dataset_sha256,
        split=split,
        collector_version=collector_version,
        git_commit=git_commit,
        python_version=python_version,
        torch_version=torch_version,
        device=device,
        collection_timestamp=collection_timestamp,
        deployment_id=deployment_id,
        runtime_framework=runtime_framework,
        model_id=model_id,
        adapter_id=resolved_adapter,
        observer_level=observer_level,
        otel_resource=otel_resource,
        provenance={
            "collector": "wso.telemetry.hooks.collect_peft_lora_telemetry",
            "scale": scale,
            **(provenance or {}),
        },
    )


def collect_state_dict_delta_telemetry(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
    *,
    parameter_name: str,
    run_id: str,
    module_id: str,
    task_family: str | None = None,
    evidence_mode: EvidenceMode = "synthetic",
    source_kind: str | None = None,
    source_path_hashes: dict[str, str] | None = None,
    model_path: str | None = None,
    adapter_path: str | None = None,
    checkpoint_sha256: str | None = None,
    dataset_id: str | None = None,
    dataset_sha256: str | None = None,
    split: TelemetrySplit | None = None,
    collector_version: str | None = None,
    git_commit: str | None = None,
    python_version: str | None = None,
    torch_version: str | None = None,
    device: str | None = None,
    collection_timestamp: float | None = None,
    deployment_id: str | None = None,
    runtime_framework: str | None = None,
    model_id: str | None = None,
    observer_level: str = "derived_telemetry",
    otel_resource: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> TransientLoRATelemetry:
    """Collect telemetry from one 2D parameter delta between two state dicts."""

    if parameter_name not in before:
        raise KeyError(f"missing parameter in before state dict: {parameter_name}")
    if parameter_name not in after:
        raise KeyError(f"missing parameter in after state dict: {parameter_name}")
    delta_w = after[parameter_name].detach() - before[parameter_name].detach()
    return collect_delta_telemetry(
        delta_w,
        run_id=run_id,
        module_id=module_id,
        task_family=task_family,
        evidence_mode=evidence_mode,
        sender_count=1,
        target_rank=min(4, min(delta_w.shape)),
        collection_mode="state_dict_delta",
        source_kind=source_kind,
        source_path_hashes=source_path_hashes,
        model_path=model_path,
        adapter_path=adapter_path,
        checkpoint_sha256=checkpoint_sha256,
        dataset_id=dataset_id,
        dataset_sha256=dataset_sha256,
        split=split,
        collector_version=collector_version,
        git_commit=git_commit,
        python_version=python_version,
        torch_version=torch_version,
        device=device,
        collection_timestamp=collection_timestamp,
        deployment_id=deployment_id,
        runtime_framework=runtime_framework,
        model_id=model_id,
        observer_level=observer_level,
        otel_resource=otel_resource,
        provenance={
            "collector": "wso.telemetry.hooks.collect_state_dict_delta_telemetry",
            "parameter_name": parameter_name,
            **(provenance or {}),
        },
    )


def collect_weighted_lora_event_telemetry(
    weighted_entries: WeightedLoRAEntries,
    *,
    run_id: str,
    task_family: str | None = None,
    prompt_hash: str | None = None,
    evidence_mode: EvidenceMode = "synthetic",
    module_ids: set[str] | None = None,
    sender_ids: Sequence[str] | None = None,
    fusion_gate_logits: torch.Tensor | None = None,
    baseline_deltas: Mapping[str, torch.Tensor] | None = None,
    deployment_id: str | None = None,
    runtime_framework: str | None = None,
    model_id: str | None = None,
    adapter_id: str | None = None,
    collection_mode: str = "transient_lora_fusion_hook",
    source_kind: str | None = None,
    source_path_hashes: dict[str, str] | None = None,
    model_path: str | None = None,
    adapter_path: str | None = None,
    checkpoint_sha256: str | None = None,
    dataset_id: str | None = None,
    dataset_sha256: str | None = None,
    split: TelemetrySplit | None = None,
    collector_version: str | None = None,
    git_commit: str | None = None,
    python_version: str | None = None,
    torch_version: str | None = None,
    device: str | None = None,
    collection_timestamp: float | None = None,
    observer_level: str = "derived_telemetry",
    otel_resource: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> list[TransientLoRATelemetry]:
    """Collect derived telemetry from transient weighted LoRA/fusion events.

    ``weighted_entries`` maps parameter names to sender LoRA factors shaped as
    ``(A_i, B_i, scale_i, gate_weight_i)``. The fused delta is materialized only
    in memory long enough to derive telemetry summaries.
    """

    records: list[TransientLoRATelemetry] = []
    for parameter_name, entries in sorted(weighted_entries.items()):
        if not entries:
            continue
        module_id = _module_id_from_parameter_name(parameter_name)
        if module_ids is not None and module_id not in module_ids:
            continue

        delta_w = materialize_weighted_lora_delta(entries)
        ranks = [_entry_rank(entry) for entry in entries]
        logits = fusion_gate_logits
        if logits is None:
            logits = _fusion_logits_from_entries(entries)
        record = collect_delta_telemetry(
            delta_w,
            run_id=run_id,
            module_id=module_id,
            task_family=task_family,
            prompt_hash=prompt_hash,
            evidence_mode=evidence_mode,
            sender_count=len(entries),
            target_rank=max(ranks),
            rank_bound=sum(ranks),
            sender_ids=list(sender_ids or _default_sender_ids(len(entries))),
            fusion_gate_logits=logits,
            collection_mode=collection_mode,
            source_kind=source_kind,
            source_path_hashes=source_path_hashes,
            model_path=model_path,
            adapter_path=adapter_path,
            checkpoint_sha256=checkpoint_sha256,
            dataset_id=dataset_id,
            dataset_sha256=dataset_sha256,
            split=split,
            collector_version=collector_version,
            git_commit=git_commit,
            python_version=python_version,
            torch_version=torch_version,
            device=device,
            collection_timestamp=collection_timestamp,
            deployment_id=deployment_id,
            runtime_framework=runtime_framework,
            model_id=model_id,
            adapter_id=adapter_id,
            observer_level=observer_level,
            otel_resource=otel_resource,
            baseline_delta=(baseline_deltas or {}).get(module_id),
            provenance={
                "collector": "wso.telemetry.hooks.collect_weighted_lora_event_telemetry",
                "parameter_name": parameter_name,
                "delta_materialized_in_memory": True,
                "raw_tensor_persisted": False,
                **(provenance or {}),
            },
        )
        records.append(record)
    return records


def materialize_weighted_lora_delta(entries: Sequence[WeightedLoRAEntry]) -> torch.Tensor:
    """Return ``sum_i gate_i * scale_i * (B_i @ A_i)`` for single-sample entries."""

    if not entries:
        raise ValueError("at least one weighted LoRA entry is required")
    delta: torch.Tensor | None = None
    for a_factor, b_factor, scale, gate_weight in entries:
        if a_factor.ndim != 2 or b_factor.ndim != 2:
            raise ValueError(
                "materialize_weighted_lora_delta expects 2D single-sample factors; "
                "use per-sample slicing before collecting batched telemetry"
            )
        if b_factor.shape[1] != a_factor.shape[0]:
            raise ValueError(
                f"incompatible LoRA factors: A={tuple(a_factor.shape)} B={tuple(b_factor.shape)}"
            )
        contribution = b_factor.detach().to(dtype=torch.float32) @ a_factor.detach().to(
            dtype=torch.float32
        )
        contribution = contribution * float(scale) * _weight_to_float(gate_weight)
        delta = contribution if delta is None else delta + contribution
    assert delta is not None
    return delta


def collect_batched_weighted_lora_event_telemetry(
    weighted_entries: WeightedLoRAEntries,
    *,
    run_ids: Sequence[str],
    task_families: Sequence[str | None] | None = None,
    prompt_hashes: Sequence[str | None] | None = None,
    fusion_gate_logits: torch.Tensor | None = None,
    **kwargs: Any,
) -> list[TransientLoRATelemetry]:
    """Collect per-sample telemetry from batched transient LoRA/fusion events."""

    batch_size = _infer_batch_size(weighted_entries)
    if len(run_ids) != batch_size:
        raise ValueError(f"expected {batch_size} run_ids, got {len(run_ids)}")
    records: list[TransientLoRATelemetry] = []
    for sample_idx, run_id in enumerate(run_ids):
        sample_entries: dict[str, list[WeightedLoRAEntry]] = {}
        for parameter_name, entries in weighted_entries.items():
            sliced: list[WeightedLoRAEntry] = []
            for a_factor, b_factor, scale, gate_weight in entries:
                a_sample = a_factor[sample_idx] if a_factor.ndim == 3 else a_factor
                b_sample = b_factor[sample_idx] if b_factor.ndim == 3 else b_factor
                weight_sample = (
                    gate_weight[sample_idx]
                    if isinstance(gate_weight, torch.Tensor) and gate_weight.ndim > 0
                    else gate_weight
                )
                sliced.append((a_sample, b_sample, scale, weight_sample))
            sample_entries[parameter_name] = sliced

        records.extend(
            collect_weighted_lora_event_telemetry(
                sample_entries,
                run_id=run_id,
                task_family=(
                    task_families[sample_idx] if task_families is not None else None
                ),
                prompt_hash=(
                    prompt_hashes[sample_idx] if prompt_hashes is not None else None
                ),
                fusion_gate_logits=_slice_batched_fusion_gate_logits(
                    fusion_gate_logits,
                    sample_idx,
                    batch_size,
                ),
                **kwargs,
            )
        )
    return records


def _resolve_lora_component(
    module: Any,
    attribute: str,
    adapter_id: str | None,
) -> tuple[Any, str]:
    if not hasattr(module, attribute):
        raise AttributeError(f"module does not expose {attribute}")
    container = getattr(module, attribute)
    if hasattr(container, "keys") and hasattr(container, "__getitem__"):
        keys = list(container.keys())
        if not keys:
            raise ValueError(f"{attribute} does not contain any adapters")
        resolved_adapter = adapter_id or str(keys[0])
        if resolved_adapter not in container:
            raise KeyError(f"adapter {resolved_adapter!r} not found in {attribute}")
        return container[resolved_adapter], resolved_adapter
    return container, adapter_id or "default"


def _weight_tensor(component: Any) -> torch.Tensor:
    if isinstance(component, torch.Tensor):
        return component
    if hasattr(component, "weight"):
        weight = getattr(component, "weight")
        if isinstance(weight, torch.Tensor):
            return weight
    raise TypeError("LoRA component must be a tensor or expose a tensor .weight")


def _resolve_lora_scale(module: Any, adapter_id: str, *, rank: int) -> float:
    scaling = getattr(module, "scaling", None)
    if isinstance(scaling, Mapping):
        value = scaling.get(adapter_id)
        if value is not None:
            return float(value)
    if isinstance(scaling, (float, int)):
        return float(scaling)

    alpha = getattr(module, "lora_alpha", None)
    if isinstance(alpha, Mapping):
        value = alpha.get(adapter_id)
        if value is not None:
            return float(value) / float(rank)
    if isinstance(alpha, (float, int)):
        return float(alpha) / float(rank)
    return 1.0


def _module_id_from_parameter_name(parameter_name: str) -> str:
    return parameter_name[: -len(".weight")] if parameter_name.endswith(".weight") else parameter_name


def _entry_rank(entry: WeightedLoRAEntry) -> int:
    a_factor = entry[0]
    return int(a_factor.shape[-2])


def _default_sender_ids(sender_count: int) -> list[str]:
    return [f"sender_{idx}" for idx in range(sender_count)]


def _weight_to_float(weight: float | torch.Tensor) -> float:
    if isinstance(weight, torch.Tensor):
        if weight.numel() != 1:
            raise ValueError("expected scalar gate weight for single-sample telemetry")
        return float(weight.detach().to(dtype=torch.float32).item())
    return float(weight)


def _fusion_logits_from_entries(entries: Sequence[WeightedLoRAEntry]) -> torch.Tensor:
    weights = torch.tensor(
        [max(_weight_to_float(entry[3]), 1e-12) for entry in entries],
        dtype=torch.float32,
    )
    probs = weights / weights.sum().clamp_min(1e-12)
    return torch.log(probs)


def _infer_batch_size(weighted_entries: WeightedLoRAEntries) -> int:
    for entries in weighted_entries.values():
        for a_factor, b_factor, _scale, gate_weight in entries:
            if a_factor.ndim == 3:
                return int(a_factor.shape[0])
            if b_factor.ndim == 3:
                return int(b_factor.shape[0])
            if isinstance(gate_weight, torch.Tensor) and gate_weight.ndim > 0:
                return int(gate_weight.shape[0])
    return 1


def _slice_batched_fusion_gate_logits(
    fusion_gate_logits: torch.Tensor | None,
    sample_idx: int,
    batch_size: int,
) -> torch.Tensor | None:
    if fusion_gate_logits is None:
        return None
    if fusion_gate_logits.ndim > 1 and fusion_gate_logits.shape[0] == batch_size:
        return fusion_gate_logits[sample_idx]
    return fusion_gate_logits
