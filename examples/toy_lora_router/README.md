# Toy LoRA Router Demo

This demo simulates a transient LoRA / adapter-routing event and emits a validated telemetry record.

Run:

```bash
python3 examples/toy_lora_router/run_demo.py
```

Output:

- terminal dashboard;
- `toy_telemetry.jsonl`.

The main v1 CLI path is:

```bash
wso-audit toy-demo --output toy_telemetry.jsonl
wso-audit baseline build toy_telemetry.jsonl --out baseline.json
wso-audit analyze toy_telemetry.jsonl --baseline baseline.json --out findings.json
```
