"""Task-conditioned baseline utilities."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from wso.telemetry.schemas import BaselineProfile, MetricBand, TransientLoRATelemetry


BASELINE_METRICS = (
    "fusion_entropy",
    "frobenius_norm",
    "effective_rank",
    "spectral_gap",
    "grassmann_drift",
)


def mean_spectral_gap(records: list[TransientLoRATelemetry]) -> float:
    vals = [r.spectral_gap for r in records if r.spectral_gap is not None]
    return float(mean(vals)) if vals else 0.0


def build_baseline_profiles(
    records: list[TransientLoRATelemetry],
    *,
    quantile_low: float = 0.05,
    quantile_high: float = 0.95,
) -> list[BaselineProfile]:
    """Build baseline profiles grouped by task, module, mode, and model."""

    if not 0.0 <= quantile_low < quantile_high <= 1.0:
        raise ValueError("baseline quantiles must satisfy 0 <= low < high <= 1")

    grouped: dict[tuple[str, str, str, str | None], list[TransientLoRATelemetry]] = defaultdict(list)
    for record in records:
        grouped[
            (
                record.task_family or "unknown",
                record.module_id,
                record.collection_mode,
                record.model_id,
            )
        ].append(record)

    profiles: list[BaselineProfile] = []
    for (task_family, module_id, collection_mode, model_id), group_records in sorted(grouped.items()):
        metric_bands = _metric_bands(group_records, quantile_low, quantile_high)
        group_key = _group_key(task_family, module_id, collection_mode, model_id)
        profiles.append(
            BaselineProfile(
                baseline_id=_baseline_id(group_key, metric_bands),
                task_family=task_family,
                module_id=module_id,
                collection_mode=collection_mode,
                model_id=model_id,
                group_key=group_key,
                record_count=len(group_records),
                metric_bands=metric_bands,
                provenance={
                    "generated_by": "wso.baselines.task_conditioned",
                    "quantile_low": quantile_low,
                    "quantile_high": quantile_high,
                },
            )
        )
    return profiles


def write_baseline_profiles(profiles: list[BaselineProfile], output: str | Path) -> None:
    """Write baseline profiles as a versioned JSON document."""

    payload = {
        "schema_version": "1.0",
        "profile_count": len(profiles),
        "profiles": [profile.model_dump() for profile in profiles],
    }
    Path(output).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_baseline_profiles(path: str | Path) -> list[BaselineProfile]:
    """Load baseline profiles from a versioned JSON document or legacy list."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_profiles = payload if isinstance(payload, list) else payload.get("profiles", [])
    return [BaselineProfile.model_validate(profile) for profile in raw_profiles]


def _metric_bands(
    records: list[TransientLoRATelemetry],
    quantile_low: float,
    quantile_high: float,
) -> dict[str, MetricBand]:
    bands: dict[str, MetricBand] = {}
    for metric in BASELINE_METRICS:
        raw_values = [_metric_value(record, metric) for record in records]
        values: list[float] = sorted(value for value in raw_values if value is not None)
        if not values:
            continue
        bands[metric] = MetricBand(
            metric=metric,
            lower=_quantile(values, quantile_low),
            upper=_quantile(values, quantile_high),
            quantile_low=quantile_low,
            quantile_high=quantile_high,
            sample_count=len(values),
        )
    return bands


def _metric_value(record: TransientLoRATelemetry, metric: str) -> float | None:
    value = getattr(record, metric)
    return float(value) if value is not None else None


def _quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("cannot compute quantile for empty values")
    idx = min(len(values) - 1, max(0, round(q * (len(values) - 1))))
    return float(values[idx])


def _group_key(
    task_family: str,
    module_id: str,
    collection_mode: str,
    model_id: str | None,
) -> str:
    model_part = model_id or "any-model"
    return f"{task_family}|{module_id}|{collection_mode}|{model_part}"


def _baseline_id(group_key: str, metric_bands: dict[str, MetricBand]) -> str:
    payload = {
        "group_key": group_key,
        "bands": {
            metric: {
                "lower": round(band.lower, 8),
                "upper": round(band.upper, 8),
                "sample_count": band.sample_count,
            }
            for metric, band in sorted(metric_bands.items())
        },
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"baseline-{digest[:16]}"
