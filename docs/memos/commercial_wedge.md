# Commercial Wedge: Runtime Weight-Space Observability

## Positioning

Weight-Space Observatory is runtime observability for adapter-mediated and weight-space
agent communication. The product wedge is a diagnostic audit for teams deploying agent
systems where prompts, tool calls, and output logs are visible, but adapter routing,
transient LoRA updates, or fused receiver perturbations are not.

## Buyer Problem

Enterprise agent monitoring usually assumes that important communication appears in text
logs, tool traces, or outputs. Adapter-routed and TFLOW-style systems weaken that
assumption because operational signal can move through transient model-update surfaces.
Security and platform teams need a way to prove that those surfaces are instrumented,
baselined, and exportable into their existing observability stack.

## Fundable Upgrade

The highest-leverage artifact is the local audit pilot:

```bash
wso-audit pilot --deployment "Acme Agent Platform" --out-dir wso_pilot --format html
```

The command produces:

- baseline and audit telemetry JSONL;
- task-conditioned baseline profiles;
- anomaly findings;
- OpenTelemetry-shaped JSON export;
- an investor/customer-facing HTML report;
- a manifest that lists the generated artifacts and safety boundary.

## Commercial Proof Point

The pilot turns WSO from a research toolkit into a repeatable enterprise diagnostic:

1. Instrument a real PEFT-compatible adapter surface.
2. Collect derived telemetry without persisting raw model tensors.
3. Calibrate normal task-conditioned adapter geometry.
4. Flag out-of-band runtime events.
5. Export to observability/SIEM workflows.
6. Deliver an executive report with remediation steps.

## Safety Boundary

WSO remains defensive. The commercial workflow does not include payload-hiding code,
keyed extraction, monitor bypass logic, prompt-search steganography, or operational
covert-channel tooling.
