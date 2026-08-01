# Design-Partner Config

The `wso-audit design-partner` command turns WSO telemetry into a buyer-grade
evidence pack. It supports two modes:

- `demo`: generates a deterministic offline PEFT-compatible package for investor
  demos and internal review.
- `customer`: packages real telemetry JSONL collected by WSO hooks or an equivalent
  customer integration.

Run the demo config:

```bash
wso-audit design-partner --config docs/methodology/wso_design_partner.yaml --out-dir wso_design_partner --format html
```

For customer mode, set:

```yaml
mode: customer
inputs:
  baseline_telemetry: ./baseline_telemetry.jsonl
  audit_telemetry: ./audit_telemetry.jsonl
```

Relative telemetry input paths resolve from the config file directory. The command
copies validated JSONL into the evidence pack, builds task-conditioned baselines,
analyzes audit records, writes OTLP-shaped JSON, and generates:

- `hidden_agent_communication_surface_audit.*`
- `manifest.json`
- `design_partner_pilot.md`
- `security_buyer_checklist.md`
- `safety_boundary.md`

The workflow stores derived telemetry only and does not include payload-hiding code,
keyed extraction, evasion logic, or operational covert-channel tooling.
