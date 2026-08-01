"""OpenTelemetry-shaped export helpers for WSO telemetry records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from wso.telemetry.schemas import TransientLoRATelemetry


def records_to_otel_json(
    records: list[TransientLoRATelemetry],
    *,
    service_name: str = "weight-space-observatory",
    deployment_name: str | None = None,
) -> dict[str, Any]:
    """Convert WSO telemetry records to an OTLP-friendly JSON document.

    The result mirrors the OTLP trace JSON shape while staying dependency-free.
    It is intended for collectors, SIEM pipelines, and demos that can ingest JSON
    before a native OTLP exporter is warranted.
    """

    resource_attributes: dict[str, Any] = {
        "service.name": service_name,
        "telemetry.sdk.name": "weight-space-observatory",
        "wso.export.format": "otlp_json",
    }
    if deployment_name:
        resource_attributes["deployment.name"] = deployment_name
    deployment_id = _first_present(record.deployment_id for record in records)
    if deployment_id:
        resource_attributes["deployment.id"] = deployment_id

    for record in records:
        for key, value in record.otel_resource.items():
            resource_attributes.setdefault(key, value)

    spans = [_record_to_span(record) for record in records]
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": _otel_attributes(resource_attributes)},
                "scopeSpans": [
                    {
                        "scope": {
                            "name": "wso.telemetry",
                            "version": _first_present(record.schema_version for record in records)
                            or "1.0",
                        },
                        "spans": spans,
                    }
                ],
            }
        ]
    }


def write_otel_json(
    records: list[TransientLoRATelemetry],
    output: str | Path,
    *,
    service_name: str = "weight-space-observatory",
    deployment_name: str | None = None,
) -> dict[str, Any]:
    """Write OTLP-friendly JSON and return the exported payload."""

    payload = records_to_otel_json(
        records,
        service_name=service_name,
        deployment_name=deployment_name,
    )
    Path(output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _record_to_span(record: TransientLoRATelemetry) -> dict[str, Any]:
    timestamp_ns = int(record.timestamp * 1_000_000_000)
    attributes = {
        "wso.run_id": record.run_id,
        "wso.deployment_id": record.deployment_id,
        "wso.module_id": record.module_id,
        "wso.task_family": record.task_family,
        "wso.prompt_hash": record.prompt_hash,
        "wso.collection_mode": record.collection_mode,
        "wso.runtime_framework": record.runtime_framework,
        "wso.model_id": record.model_id,
        "wso.adapter_id": record.adapter_id,
        "wso.observer_level": record.observer_level,
        "wso.sender_count": record.sender_count,
        "wso.rank_bound": record.rank_bound,
        "wso.target_rank": record.target_rank,
        "wso.fusion_entropy": record.fusion_entropy,
        "wso.frobenius_norm": record.frobenius_norm,
        "wso.effective_rank": record.effective_rank,
        "wso.spectral_gap": record.spectral_gap,
        "wso.grassmann_drift": record.grassmann_drift,
        "wso.singular_values.top": record.singular_values,
    }
    return {
        "traceId": _trace_id(record),
        "spanId": _span_id(record),
        "name": "wso.adapter_telemetry",
        "kind": "SPAN_KIND_INTERNAL",
        "startTimeUnixNano": str(timestamp_ns),
        "endTimeUnixNano": str(timestamp_ns),
        "attributes": _otel_attributes(attributes),
    }


def _otel_attributes(values: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"key": key, "value": _otel_value(value)}
        for key, value in sorted(values.items())
        if value is not None
    ]


def _otel_value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int) and not isinstance(value, bool):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, (list, dict)):
        return {"stringValue": json.dumps(value, sort_keys=True)}
    return {"stringValue": str(value)}


def _trace_id(record: TransientLoRATelemetry) -> str:
    return _valid_hex_id(record.trace_id, 32) or _hex_id(record.run_id, 32)


def _span_id(record: TransientLoRATelemetry) -> str:
    return _valid_hex_id(record.span_id, 16) or _hex_id(f"{record.run_id}:{record.module_id}", 16)


def _hex_id(seed: str, length: int) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:length]


def _valid_hex_id(value: str | None, length: int) -> str | None:
    if value is None:
        return None
    normalized = value.lower()
    if len(normalized) == length and all(char in "0123456789abcdef" for char in normalized):
        return normalized
    return None


def _first_present(values: Any) -> Any:
    for value in values:
        if value:
            return value
    return None
