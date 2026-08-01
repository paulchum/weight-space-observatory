# Security and Responsible Research

Weight-Space Observatory is a defensive telemetry and audit project. The repo
contains one bounded local red-team fixture for validating whether defensive
telemetry detects FP32 LSB weight injection. That fixture is for controlled
measurement, not deployment or evasion.

Please do not submit or request:

- deployment covert-channel protocols;
- monitor-bypass or evasion implementations;
- prompt-search steganography systems;
- keyed extraction protocols;
- code intended to bypass monitoring or exfiltrate data.

Issues and contributions should focus on defensive telemetry, anomaly detection, provenance, audit methodology, and reproducible measurement.

## Reporting a vulnerability

Please do not open a public issue for a vulnerability that could enable unauthorized
model actions, monitoring bypass, or sensitive-data exposure. Use the repository's
**Security → Report a vulnerability** flow so the issue can be reviewed privately.

Include the affected version, trust boundary, minimal defensive reproduction, expected
behavior, and actual behavior. Do not include live credentials, private telemetry, raw
customer prompts, model weights, or operational evasion payloads.
