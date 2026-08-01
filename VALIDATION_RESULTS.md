# Validation Results

Public-release hardening performed:

- Added GitHub Actions CI for `pytest` and `ruff`.
- Added CLI entry point: `wso-audit`.
- Added `hook-demo`, `benchmark capacity`, `baseline build`, and `analyze` commands.
- Added task-conditioned baselines and anomaly findings.
- Added multi-metric capacity-detectability frontier CSV/JSON/PNG artifacts.
- Added Markdown / HTML / JSON audit report generation.
- Expanded sample weight-space audit report with passing/failing signals.
- Added report-generator, baseline, hook, CLI, and benchmark tests.

Local validation:

```text
pytest -q
27 passed

ruff check .
All checks passed

mypy src/wso
Success: no issues found in 22 source files

cargo check --manifest-path src/daemon/Cargo.toml
Finished dev profile

wso-audit toy-demo --output /tmp/wso/toy_telemetry.jsonl
wso-audit hook-demo --output /tmp/wso/hooked_telemetry.jsonl
wso-audit benchmark capacity --out /tmp/wso/frontier.csv --plot /tmp/wso/frontier.png
wso-audit baseline build /tmp/wso/toy_telemetry.jsonl --out /tmp/wso/baseline.json
wso-audit analyze /tmp/wso/toy_telemetry.jsonl --baseline /tmp/wso/baseline.json --out /tmp/wso/findings.json
wso-audit report /tmp/wso/toy_telemetry.jsonl --baseline /tmp/wso/baseline.json --findings /tmp/wso/findings.json --frontier /tmp/wso/frontier.json --out /tmp/wso/report.html --format html --deployment 'Toy LoRA Router'
CLI smoke test passed
```

Pilot upgrade validation:

```text
uv run pytest -q
33 passed

uv run ruff check .
All checks passed!

uv run mypy src/wso
Success: no issues found in 24 source files

cargo check --manifest-path src/daemon/Cargo.toml
Finished dev profile

uv run wso-audit pilot --deployment 'Demo Agent Platform' --out-dir /tmp/wso_pilot_demo --format html --baseline-records 6
Wrote pilot package: /tmp/wso_pilot_demo (risk=MODERATE)
CLI pilot smoke test passed
```

Red-team / blue-team LSB detection demo validation:

```text
uv run ruff check .
All checks passed!

uv run pytest -q
47 passed

uv run python examples/e2e_covert_channel_demo.py --out-dir /tmp/wso_e2e_demo
clean-control: LOW
infected-lsb-write-event: HIGH
out-of-band metrics: frobenius_norm, effective_rank, spectral_gap, grassmann_drift
Artifacts: /tmp/wso_e2e_demo/summary.json

./run_all.sh
All checks passed!
47 passed
E2E demo aggregate risk: HIGH
PASS A_lora_watermark_blind_retrieval_audit
PASS B_fusion_gate_latency_channel_audit
PASS C_audit_gate_weight_space_transfer_detection
E2E covert-channel demo artifacts written to artifacts/e2e_covert_channel_demo
Detection suite artifacts written to artifacts/weight_space_detection_suite
```
