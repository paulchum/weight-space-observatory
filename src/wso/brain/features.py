"""Stable, privacy-preserving features for weight-space decisions."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from wso.brain.types import BrainConfig
from wso.telemetry.schemas import TransientLoRATelemetry

FEATURE_SCHEMA = "weight-space-v2"
MAX_SINGULAR_VALUES = 4096
MAX_GATE_VALUES = 1024
MAX_LAYER_DIMENSIONS = 8
MAX_SENDER_IDENTIFIERS = 1024
MAX_RANK_LIKE_VALUE = 1_000_000
MAX_CONTEXT_BYTES = 4096
MAX_METADATA_NODES = 2048
MAX_METADATA_BYTES = 65_536

NUMERIC_FEATURE_NAMES = (
    "fusion_entropy",
    "log_frobenius_norm",
    "log_effective_rank",
    "log_spectral_gap",
    "log_grassmann_drift",
    "log_sender_count",
    "log_target_rank",
    "log_rank_bound",
    "sender_rank_ratio",
    "log_singular_value_count",
    "log_leading_singular_value",
    "log_singular_value_mass",
    "leading_energy_concentration",
    "spectral_entropy",
    "spectral_concentration",
    "spectral_kurtosis",
    "stable_rank_ratio",
    "second_to_first_ratio",
    "tail_to_first_ratio",
    "fusion_gate_max",
    "fusion_gate_min",
    "fusion_gate_l2",
    "fusion_gate_entropy",
    "log_layer_rows",
    "log_layer_cols",
    "missing_fusion_entropy",
    "missing_effective_rank",
    "missing_spectral_gap",
    "missing_grassmann_drift",
    "has_runtime_provenance",
)


@dataclass(frozen=True)
class FeatureBundle:
    vector: np.ndarray
    numeric: np.ndarray
    raw_numeric: dict[str, float]


class WeightSpaceFeatureEncoder:
    """Encode derived telemetry without retaining prompt or tensor payloads."""

    def __init__(self, config: BrainConfig) -> None:
        if config.numeric_feature_count != len(NUMERIC_FEATURE_NAMES):
            raise ValueError(
                f"numeric_feature_count must be {len(NUMERIC_FEATURE_NAMES)} for {FEATURE_SCHEMA}"
            )
        self.config = config

    def encode(self, record: TransientLoRATelemetry) -> FeatureBundle:
        validate_telemetry_budget(record)
        singular = np.asarray(record.singular_values, dtype=np.float64)
        singular = np.abs(singular[np.isfinite(singular)])
        singular = np.sort(singular)[::-1]
        probabilities = _energy_probabilities(singular)

        top = float(singular[0]) if singular.size else 0.0
        second = float(singular[1]) if singular.size > 1 else 0.0
        tail = float(singular[-1]) if singular.size else 0.0
        scaled_singular = singular / top if top > 0 else np.zeros_like(singular)
        scaled_squared_mass = float(np.square(scaled_singular).sum())
        leading_energy = 1.0 / scaled_squared_mass if scaled_squared_mass > 0 else 0.0
        spectral_entropy = _normalized_entropy(probabilities)
        stable_rank = scaled_squared_mass

        gate = np.asarray(record.fusion_gate_probs, dtype=np.float64)
        gate = np.clip(gate[np.isfinite(gate)], 0.0, None)
        gate_maximum = float(gate.max()) if gate.size else 0.0
        if gate_maximum > 0:
            gate = gate / gate_maximum
            gate = gate / float(gate.sum())
        gate_entropy = _normalized_entropy(gate)

        rows = record.layer_shape[0] if record.layer_shape else 0
        cols = record.layer_shape[1] if len(record.layer_shape) > 1 else 0
        rank_bound = record.rank_bound or 0
        runtime_provenance = bool(
            record.collection_timestamp
            and record.runtime_framework
            and (record.git_commit or record.collector_version)
        )

        raw_values = [
            _optional(record.fusion_entropy),
            _signed_log(record.frobenius_norm, 5.0),
            _signed_log(record.effective_rank, 4.0),
            _signed_log(record.spectral_gap, 4.0),
            _signed_log(record.grassmann_drift, 2.0),
            math.log1p(record.sender_count) / 3.0,
            math.log1p(record.target_rank) / 4.0,
            math.log1p(rank_bound) / 4.0,
            math.tanh(record.sender_count / max(1.0, float(record.target_rank))),
            math.log1p(singular.size) / 4.0,
            _signed_log(top, 5.0),
            _log_positive_mass(singular, 6.0),
            leading_energy,
            spectral_entropy,
            1.0 - spectral_entropy,
            math.tanh(_kurtosis(singular) / 8.0),
            math.tanh(stable_rank / max(1.0, float(record.target_rank))),
            second / top if top > 0 else 0.0,
            tail / top if top > 0 else 0.0,
            float(gate.max()) if gate.size else 0.0,
            float(gate.min()) if gate.size else 0.0,
            float(np.linalg.norm(gate)) if gate.size else 0.0,
            gate_entropy,
            math.log1p(max(0, rows)) / 12.0,
            math.log1p(max(0, cols)) / 12.0,
            float(record.fusion_entropy is None),
            float(record.effective_rank is None),
            float(record.spectral_gap is None),
            float(record.grassmann_drift is None),
            float(runtime_provenance),
        ]
        numeric = np.asarray(raw_values, dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise ValueError("telemetry produced non-finite weight-space features")
        numeric = np.clip(numeric, -6.0, 6.0)
        vector = np.zeros(self.config.feature_dimension, dtype=np.float64)
        vector[: numeric.size] = numeric
        self._hash_context(record, vector)
        if not np.isfinite(vector).all():
            raise ValueError("telemetry produced a non-finite feature vector")

        return FeatureBundle(
            vector=vector,
            numeric=numeric,
            raw_numeric=dict(zip(NUMERIC_FEATURE_NAMES, numeric.tolist(), strict=True)),
        )

    def _hash_context(self, record: TransientLoRATelemetry, vector: np.ndarray) -> None:
        offset = self.config.numeric_feature_count
        buckets = self.config.feature_dimension - offset
        context = {
            "task": record.task_family or "unknown",
            "module": record.module_id,
            "collection": record.collection_mode,
            "model": record.model_id or "unknown",
            "adapter": record.adapter_id or "unknown",
            "source": record.source_kind or "unknown",
            "framework": record.runtime_framework or "unknown",
            "observer": record.observer_level,
        }
        for key, value in context.items():
            token = f"{key}={value}".encode("utf-8", errors="replace")
            digest = hashlib.sha256(token).digest()
            index = int.from_bytes(digest[:8], "big") % buckets
            sign = 1.0 if digest[8] & 1 else -1.0
            vector[offset + index] += sign / math.sqrt(len(context))
        np.clip(vector, -3.0, 3.0, out=vector)


def validate_telemetry_budget(record: TransientLoRATelemetry) -> None:
    """Reject structurally valid telemetry that exceeds the brain's work budget."""

    if len(record.singular_values) > MAX_SINGULAR_VALUES:
        raise ValueError(f"telemetry singular_values exceeds {MAX_SINGULAR_VALUES} entries")
    for name, values in (
        ("fusion_gate_logits", record.fusion_gate_logits or []),
        ("fusion_gate_probs", record.fusion_gate_probs),
    ):
        if len(values) > MAX_GATE_VALUES:
            raise ValueError(f"telemetry {name} exceeds {MAX_GATE_VALUES} entries")
    if len(record.layer_shape) > MAX_LAYER_DIMENSIONS:
        raise ValueError("telemetry layer_shape has too many dimensions")
    if len(record.sender_ids) > MAX_SENDER_IDENTIFIERS:
        raise ValueError("telemetry sender_ids exceeds the brain work budget")
    for name, value in (
        ("sender_count", record.sender_count),
        ("target_rank", record.target_rank),
        ("rank_bound", record.rank_bound if record.rank_bound is not None else 0),
    ):
        if value < 0 or value > MAX_RANK_LIKE_VALUE:
            raise ValueError(f"telemetry {name} is outside the brain work budget")
    if any(value < 0 or value > 2**31 - 1 for value in record.layer_shape):
        raise ValueError("telemetry layer dimensions are outside the brain work budget")

    string_fields = (
        record.run_id,
        record.trace_id,
        record.span_id,
        record.deployment_id,
        record.module_id,
        record.task_family,
        record.prompt_hash,
        record.collection_mode,
        record.source_kind,
        record.model_path,
        record.adapter_path,
        record.checkpoint_sha256,
        record.dataset_id,
        record.dataset_sha256,
        record.collector_version,
        record.git_commit,
        record.python_version,
        record.torch_version,
        record.device,
        record.runtime_framework,
        record.model_id,
        record.adapter_id,
        record.observer_level,
        record.baseline_id,
        record.metric_version,
    )
    if any(
        value is not None and len(value.encode("utf-8", errors="replace")) > MAX_CONTEXT_BYTES
        for value in string_fields
    ):
        raise ValueError("telemetry identifier exceeds the brain work budget")
    if any(
        len(value.encode("utf-8", errors="replace")) > MAX_CONTEXT_BYTES
        for value in record.sender_ids
    ):
        raise ValueError("telemetry sender identifier exceeds the brain work budget")
    for name, metadata in (
        ("source_path_hashes", record.source_path_hashes),
        ("otel_resource", record.otel_resource),
        ("provenance", record.provenance),
    ):
        if not _metadata_within_budget(metadata):
            raise ValueError(f"telemetry {name} exceeds the brain metadata budget")


def _metadata_within_budget(value: object) -> bool:
    stack: list[tuple[object, int]] = [(value, 0)]
    nodes = 0
    byte_count = 0
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > MAX_METADATA_NODES or depth > 8:
            return False
        current_type = type(current)
        if isinstance(current, dict):
            if current_type is not dict:
                return False
            child_count = 2 * len(current)
            if nodes + len(stack) + child_count > MAX_METADATA_NODES:
                return False
            if any(type(key) is not str for key in current):
                return False
            byte_count += 2 + max(0, len(current) - 1) + len(current)
            stack.extend((item, depth + 1) for pair in current.items() for item in pair)
        elif isinstance(current, list):
            if current_type is not list:
                return False
            if nodes + len(stack) + len(current) > MAX_METADATA_NODES:
                return False
            byte_count += 2 + max(0, len(current) - 1)
            stack.extend((item, depth + 1) for item in current)
        elif isinstance(current, str):
            if current_type is not str:
                return False
            byte_count += _json_string_budget_bytes(
                current,
                remaining=MAX_METADATA_BYTES - byte_count,
            )
        elif current is None:
            byte_count += 4
        elif isinstance(current, bool):
            if current_type is not bool:
                return False
            byte_count += 4 if current else 5
        elif isinstance(current, int):
            if current_type is not int:
                return False
            byte_count += max(1, current.bit_length()) + int(current < 0)
        elif isinstance(current, float):
            if current_type is not float:
                return False
            if not math.isfinite(current):
                return False
            byte_count += len(repr(current))
        else:
            return False
        if byte_count > MAX_METADATA_BYTES:
            return False
    return True


def _json_string_budget_bytes(value: str, *, remaining: int) -> int:
    """Upper-bound canonical ensure-ASCII JSON bytes without allocating the encoding."""

    size = 2  # Quotes.
    for character in value:
        codepoint = ord(character)
        if character in {'"', "\\"} or codepoint < 0x20:
            size += 6
        elif codepoint <= 0x7F:
            size += 1
        elif codepoint <= 0xFFFF:
            size += 6
        else:
            size += 12
        if size > remaining:
            return remaining + 1
    return size


def _optional(value: float | None) -> float:
    return float(value) if value is not None else 0.0


def _signed_log(value: float | None, scale: float) -> float:
    if value is None:
        return 0.0
    return math.copysign(math.log1p(abs(float(value))) / scale, float(value))


def _energy_probabilities(values: np.ndarray) -> np.ndarray:
    if not values.size:
        return np.asarray([], dtype=np.float64)
    maximum = float(values.max())
    if maximum <= 0:
        return np.zeros_like(values)
    energy = np.square(values / maximum)
    total = float(energy.sum())
    return energy / total if total > 0 else np.zeros_like(energy)


def _log_positive_mass(values: np.ndarray, scale: float) -> float:
    """Return log1p(sum(values)) without overflowing a finite spectrum."""

    if not values.size:
        return 0.0
    maximum = float(values.max())
    if maximum <= 0:
        return 0.0
    log_total = math.log(maximum) + math.log(float(np.sum(values / maximum)))
    return float(np.logaddexp(0.0, log_total)) / scale


def _normalized_entropy(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0]
    if positive.size <= 1:
        return 0.0
    entropy = -float(np.sum(positive * np.log(positive)))
    return entropy / math.log(float(positive.size))


def _kurtosis(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    maximum = float(np.max(np.abs(values)))
    if maximum <= 0:
        return 0.0
    scaled = values / maximum
    centered = scaled - float(scaled.mean())
    variance = float(np.mean(np.square(centered)))
    if variance <= 1e-12:
        return 0.0
    return float(np.mean(np.power(centered, 4)) / (variance * variance) - 3.0)
