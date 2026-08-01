from __future__ import annotations

import math
import time
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


EvidenceMode = Literal["synthetic", "demo", "fixture", "customer", "real"]
TelemetrySplit = Literal["baseline", "heldout", "audit"]


class TransientLoRATelemetry(BaseModel):
    """Defensive telemetry summary for a transient LoRA / adapter perturbation.

    This schema intentionally stores derived signatures rather than raw tensors.
    """

    schema_version: str = Field(default="1.0", description="Telemetry schema version")
    run_id: str = Field(..., description="Audit run identifier")
    trace_id: str | None = Field(default=None, description="Distributed trace identifier")
    span_id: str | None = Field(default=None, description="Distributed span identifier")
    deployment_id: str | None = Field(default=None, description="Audited deployment identifier")
    module_id: str = Field(..., description="Target receiver layer/module")
    timestamp: float = Field(default_factory=time.time)
    task_family: str | None = Field(default=None, description="Task-conditioned baseline label")
    prompt_hash: str | None = Field(default=None, description="Hash of prompt/input, not raw text")
    evidence_mode: EvidenceMode = Field(
        default="synthetic",
        description="Evidence status: synthetic, demo, fixture, customer, or real",
    )
    collection_mode: str = Field(
        default="synthetic", description="Collection path, for example synthetic, hook, or lora"
    )
    source_kind: str | None = Field(
        default=None,
        description="External source family for customer/real evidence",
    )
    source_path_hashes: dict[str, str] = Field(
        default_factory=dict,
        description="SHA-256 hashes of external artifact paths; raw artifacts are not persisted",
    )
    model_path: str | None = Field(default=None, description="Local model path reference")
    adapter_path: str | None = Field(default=None, description="Local adapter path reference")
    checkpoint_sha256: str | None = Field(default=None, description="Checkpoint SHA-256")
    dataset_id: str | None = Field(default=None, description="Dataset or prompt-set identifier")
    dataset_sha256: str | None = Field(default=None, description="Dataset or prompt-file SHA-256")
    split: TelemetrySplit | None = Field(default=None, description="Evaluation split")
    collector_version: str | None = Field(default=None, description="Collector package/version")
    git_commit: str | None = Field(default=None, description="Collector git commit")
    python_version: str | None = Field(default=None, description="Python runtime version")
    torch_version: str | None = Field(default=None, description="Torch version")
    device: str | None = Field(default=None, description="Collection device label")
    collection_timestamp: float | None = Field(
        default=None,
        description="Collection-event timestamp when supplied",
    )
    runtime_framework: str | None = Field(
        default=None, description="Runtime or integration family that produced the telemetry"
    )
    model_id: str | None = Field(default=None, description="Model family or deployment identifier")
    adapter_id: str | None = Field(default=None, description="Adapter identifier when applicable")
    observer_level: str = Field(
        default="derived_telemetry",
        description="Observer access level represented by this record",
    )
    otel_resource: dict[str, Any] = Field(
        default_factory=dict,
        description="OpenTelemetry resource attributes to carry into exports",
    )
    baseline_id: str | None = Field(default=None, description="Baseline profile used for analysis")
    layer_shape: list[int] = Field(default_factory=list, description="Shape of the observed delta")
    metric_version: str = Field(default="1.0", description="Metric implementation version")
    sender_count: int = Field(..., description="Number of sender updates fused")
    rank_bound: int | None = Field(default=None, description="Upper bound on fused perturbation rank")
    sender_ids: list[str] = Field(default_factory=list)
    target_rank: int
    fusion_gate_logits: list[float] | None = None
    fusion_gate_probs: list[float] = Field(default_factory=list)
    fusion_entropy: float | None = None
    frobenius_norm: float
    singular_values: list[float]
    effective_rank: float | None = None
    spectral_gap: float | None = None
    grassmann_drift: float | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "timestamp",
        "fusion_entropy",
        "frobenius_norm",
        "effective_rank",
        "spectral_gap",
        "grassmann_drift",
        "collection_timestamp",
    )
    @classmethod
    def finite_optional_float(cls, value: float | None) -> float | None:
        if value is None:
            return value
        if not math.isfinite(value):
            raise ValueError("telemetry numeric fields must be finite")
        return value

    @field_validator("fusion_gate_logits", "fusion_gate_probs", "singular_values")
    @classmethod
    def finite_float_list(cls, values: list[float] | None) -> list[float] | None:
        if values is None:
            return values
        if any(not math.isfinite(value) for value in values):
            raise ValueError("telemetry list fields must contain only finite values")
        return values

    @field_validator("sender_count", "target_rank")
    @classmethod
    def positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("count and rank fields must be positive")
        return value


class AuditReport(BaseModel):
    """Final diagnostic report summarizing communication-surface risk."""

    deployment_name: str
    total_inferences_audited: int
    mean_fusion_entropy: float
    mean_spectral_gap: float
    max_grassmann_drift: float
    risk_level: str = Field(..., description="LOW, MODERATE, HIGH")
    notes: list[str] = Field(default_factory=list)


class MetricBand(BaseModel):
    """Task-conditioned acceptable range for one telemetry metric."""

    metric: str
    lower: float
    upper: float
    quantile_low: float = 0.05
    quantile_high: float = 0.95
    sample_count: int

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper


class BaselineProfile(BaseModel):
    """Calibrated metric bands for one task/module/model telemetry group."""

    schema_version: str = "1.0"
    baseline_id: str
    created_at: float = Field(default_factory=time.time)
    task_family: str
    module_id: str
    collection_mode: str
    model_id: str | None = None
    group_key: str
    record_count: int
    metric_bands: dict[str, MetricBand] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)


class AnomalyFinding(BaseModel):
    """Analysis result for one telemetry record against one baseline profile."""

    schema_version: str = "1.0"
    run_id: str
    baseline_id: str | None = None
    module_id: str
    task_family: str | None = None
    collection_mode: str
    model_id: str | None = None
    risk_level: str
    metric_values: dict[str, float] = Field(default_factory=dict)
    out_of_band_metrics: list[str] = Field(default_factory=list)
    messages: list[str] = Field(default_factory=list)


class CapacityFrontierPoint(BaseModel):
    """One point on a defensive capacity-detectability frontier."""

    signal_strength: float
    recoverability_auc: float
    anomaly_rate: float
    metric_aucs: dict[str, float] = Field(default_factory=dict)
    metric_anomaly_rates: dict[str, float] = Field(default_factory=dict)
    benign_bands: dict[str, MetricBand] = Field(default_factory=dict)


class CapacityFrontierReport(BaseModel):
    """Report-ready summary of a synthetic capacity-detectability sweep."""

    schema_version: str = "1.0"
    benchmark_id: str
    created_at: float = Field(default_factory=time.time)
    samples_per_strength: int
    benign_samples: int
    noise_std: float
    seed: int
    points: list[CapacityFrontierPoint]
    safety_boundary: str = (
        "Synthetic audit labels are offline annotations only; this benchmark does not "
        "implement payload encoders, keyed extraction, or monitor-bypass tooling."
    )
