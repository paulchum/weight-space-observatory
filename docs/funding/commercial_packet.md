# Commercial Funding Packet: Runtime Weight-Space Observability

## Product Positioning

Weight-Space Observatory is runtime observability for adapter-mediated and
weight-space agent communication. It helps security and platform teams prove that
adapter routing, transient LoRA updates, fusion gates, and receiver-side
perturbations are instrumented, baselined, and exportable into existing
observability workflows.

The commercial wedge is not another prompt firewall. WSO is the audit layer for
model-state and adapter-state behavior that sits below prompt, output, and tool
logs. The first buyer promise is simple: before an enterprise scales agentic
systems, it can run a local evidence pack showing which non-text communication
surfaces are visible, which are out of baseline, and which controls need to exist
before production rollout.

## Buyer Problem

Enterprise agent monitoring is built around prompts, outputs, tool calls, and
ordinary application logs. That misses operational signal when communication moves
through adapters, memory layers, fusion gates, or transient model-update surfaces.
Security teams need an audit layer that can measure these surfaces without
persisting raw model tensors or deploying offensive tooling in production.

This is a CISO problem because the accountable question is not "did the agent
answer safely in one demo?" It is "can we prove the runtime communication substrate
is observable, baselined, and exportable to our security operations workflow?"

## Current Product Evidence

The repo already supports:

- telemetry schemas for transient LoRA and adapter events;
- singular-spectrum, effective-rank, spectral-gap, fusion-gate, and Grassmannian
  drift metrics;
- a toy LoRA router demo and no-network LoRA hook demo;
- task-conditioned baselines and anomaly findings;
- OTLP-shaped JSON export;
- Markdown, HTML, and JSON audit reports;
- a config-driven design-partner evidence package;
- a buyer walkthrough, SVCI outreach email, and target-contact brief;
- a bounded red-team / blue-team LSB injection demo that proves the telemetry
  stack flags a live weight-space write event;
- responsible-research documentation.

Primary demo command:

```bash
wso-audit design-partner --config docs/methodology/wso_design_partner.yaml --out-dir wso_design_partner --format html
```

## Empirical Detection Demo

The repository now includes a live, runnable "Sword + Shield" proof for buyers:
a bounded red-team fixture injects an inert payload into FP32 mantissa LSBs, and
WSO's blue-team telemetry flags the resulting write-event perturbation against a
clean low-rank baseline.

Run it locally:

```bash
python examples/e2e_covert_channel_demo.py --out-dir artifacts/e2e_covert_channel_demo
```

The demo writes `telemetry.jsonl`, `detection.jsonl`, `baseline.json`,
`findings.json`, and `summary.json`. It persists only derived telemetry and
payload metadata such as SHA-256, byte length, capacity, changed-element count,
and maximum absolute weight delta. The expected buyer-facing result is simple:
the clean control remains `LOW` risk, while the infected LSB write event becomes
`HIGH` risk with out-of-band effective rank, spectral gap, and Grassmannian
drift.

For SVCI CISOs, this is the proof-of-value wedge: WSO is not merely describing a
theoretical hidden communication surface. It can show a local covert
weight-space injection and produce auditable evidence that the telemetry layer
caught it without storing raw model tensors or payload bytes.

## Pilot Offer

One-week self-hosted hidden-communication-surface audit:

1. Instrument a PEFT-compatible or adapter-like surface.
2. Collect derived telemetry summaries without persisting raw model tensors.
3. Calibrate normal task-conditioned adapter geometry.
4. Flag out-of-band runtime events.
5. Export OTEL-shaped artifacts for observability/SIEM review.
6. Deliver an executive report with risk, coverage, controls, and next steps.

Demo artifact order for a buyer walkthrough:

1. `hidden_agent_communication_surface_audit.html`
2. `design_partner_pilot.md`
3. `security_buyer_checklist.md`
4. `telemetry_otel.json`
5. `safety_boundary.md`
6. `buyer_walkthrough.md`
7. `svci_outreach_email.md`
8. `target_contact_brief.md`

The design-partner command generates all of these files into one output directory.

## Best-Match Funding Routes

| Route | Why It Fits | Eligibility Work |
| --- | --- | --- |
| NSF SBIR/STTR | Non-dilutive U.S. R&D for deep-tech commercialization and AI/cybersecurity tooling. | Requires U.S. small business and company-employed PI. |
| DARPA SBIR/STTR / AFWERX | Dual-use AI assurance, agent security, and telemetry for defense-relevant deployments. | Requires U.S. small business; AFWERX needs defense end-user pull. |
| NRC IRAP / DI Assist | Canadian SME support for dual-use and defence industrial-base innovation. | Requires Canadian incorporated SME and IRAP advisor engagement. |
| Innovative Solutions Canada | Challenge/prototype funding when trust, verification, autonomy, or defence challenge matches. | Requires Canadian incorporated SME and open challenge fit. |
| Scale AI Canada | Consortium funding for applied AI adoption and commercialization. | Requires adopter/partner coalition and value-chain productivity framing. |
| YC / SVCI | Equity route for security-market validation and CISO design partners. | Requires startup narrative, customer discovery, and willingness to take equity capital. |

## NSF SBIR Project Pitch Skeleton

**Technical innovation:** WSO adds derived weight-space telemetry for agent systems
that cannot be audited through prompt/output logs alone.

**Customer problem:** AI platform and security teams lack observability for
adapter-mediated communication surfaces and cannot prove coverage to enterprise
risk owners.

**Commercial opportunity:** AI agent deployments are moving into enterprise
operations, regulated workflows, and security-sensitive environments. Monitoring
and observability vendors need new signal layers for non-text model state.

**Phase I objective:** Validate technical feasibility against PEFT-compatible
adapter surfaces and produce a repeatable audit package with calibrated baselines,
anomaly findings, and OTEL export.

**Phase II objective:** Harden integrations, support production observability
pipelines, and run paid pilots with design partners.

## Defense/IRAP/ISC Framing

WSO supports sovereign AI assurance by measuring hidden communication surfaces in
agent systems before they are deployed in defence, critical infrastructure, or
high-trust government workflows. The product is defensive monitoring and audit
infrastructure, not a capability for hiding messages or bypassing monitors.

## Design-Partner Ask

We are seeking security and platform teams deploying agent systems with adapters,
LoRA routing, memory layers, or transient model-update mechanisms. The first pilot
goal is to produce a signed readout showing runtime telemetry coverage, risk
signals, recommended controls, and observability export compatibility.

For SVCI, the ask is narrower: review the generated evidence pack, pressure-test the
security-buyer narrative, and introduce one or two CISO design partners who own AI
platform security or runtime observability for agent deployments.
