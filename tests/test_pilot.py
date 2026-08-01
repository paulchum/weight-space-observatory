import pytest

from wso.pilot import load_design_partner_config, run_design_partner_pilot


def test_design_partner_customer_mode_packages_existing_telemetry(tmp_path):
    demo_config = tmp_path / "demo.yaml"
    demo_out = tmp_path / "demo"
    demo_config.write_text(
        """
mode: demo
deployment:
  name: "Source Pilot"
modules:
  - module_id: "pilot.receiver.layers.0.lora"
    task_family: "agent_tool_use"
baseline_records: 6
""",
        encoding="utf-8",
    )
    source = run_design_partner_pilot(config_path=demo_config, out_dir=demo_out)

    customer_config = tmp_path / "customer.yaml"
    customer_out = tmp_path / "customer"
    customer_config.write_text(
        f"""
mode: customer
deployment:
  name: "Customer Agent Runtime"
  id: "customer-agent-runtime"
runtime:
  framework: "torch_peft_compatible"
  model_id: "customer-model"
modules:
  - module_id: "pilot.receiver.layers.0.lora"
    adapter_id: "default"
    task_family: "agent_tool_use"
inputs:
  baseline_telemetry: {source.baseline_telemetry.name}
  audit_telemetry: {source.audit_telemetry.name}
baseline_records: 6
""",
        encoding="utf-8",
    )
    source.baseline_telemetry.replace(tmp_path / source.baseline_telemetry.name)
    source.audit_telemetry.replace(tmp_path / source.audit_telemetry.name)

    packaged = run_design_partner_pilot(config_path=customer_config, out_dir=customer_out)

    assert packaged.manifest.exists()
    assert packaged.buyer_checklist.exists()
    assert "customer" in packaged.manifest.read_text(encoding="utf-8")
    assert "Customer Agent Runtime" in packaged.pilot_proposal.read_text(encoding="utf-8")


def test_design_partner_customer_mode_requires_inputs(tmp_path):
    config = tmp_path / "missing-inputs.yaml"
    config.write_text(
        """
mode: customer
deployment:
  name: "Customer Agent Runtime"
modules:
  - module_id: "pilot.receiver.layers.0.lora"
baseline_records: 6
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="customer mode requires"):
        load_design_partner_config(config)
