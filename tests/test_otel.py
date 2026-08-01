from pathlib import Path

from wso.telemetry.otel import records_to_otel_json, write_otel_json
from wso.telemetry.schemas import TransientLoRATelemetry


def _record() -> TransientLoRATelemetry:
    return TransientLoRATelemetry(
        run_id="run-otel",
        trace_id="a" * 32,
        span_id="b" * 16,
        deployment_id="demo-deployment",
        module_id="layers.0.lora",
        task_family="agent_tool_use",
        collection_mode="lora_hook",
        runtime_framework="torch_peft_compatible",
        model_id="tiny-local",
        adapter_id="default",
        sender_count=1,
        rank_bound=2,
        target_rank=2,
        frobenius_norm=1.0,
        singular_values=[1.0, 0.5],
        effective_rank=1.5,
        spectral_gap=0.5,
        otel_resource={"service.namespace": "agent-security"},
    )


def test_records_to_otel_json_shape() -> None:
    payload = records_to_otel_json([_record()], deployment_name="Demo")
    resource_span = payload["resourceSpans"][0]
    span = resource_span["scopeSpans"][0]["spans"][0]
    assert span["traceId"] == "a" * 32
    assert span["spanId"] == "b" * 16
    assert span["name"] == "wso.adapter_telemetry"
    keys = {attribute["key"] for attribute in span["attributes"]}
    assert "wso.module_id" in keys
    resource_keys = {attribute["key"] for attribute in resource_span["resource"]["attributes"]}
    assert "service.namespace" in resource_keys


def test_write_otel_json(tmp_path: Path) -> None:
    output = tmp_path / "otel.json"
    payload = write_otel_json([_record()], output)
    assert output.exists()
    assert payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["traceId"] == "a" * 32
