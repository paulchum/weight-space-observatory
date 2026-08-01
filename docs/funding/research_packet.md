# Research Funding Packet: The Uninspectable Message

## Project Title

The Uninspectable Message: Auditing Hidden Communication in Weight-Space
Multi-Agent Systems

## One-Sentence Pitch

Weight-Space Observatory measures whether agent communication can move below
prompt and output logs into transient LoRA/adapter state, and identifies the
telemetry required to detect or bound that risk.

## Short Abstract

Modern agent monitoring assumes that relevant inter-agent communication appears in
prompts, tool calls, outputs, or logs. Adapter-routed and TFLOW-style systems
challenge that assumption because sender hidden states can induce transient
receiver-side LoRA perturbations, leaving no natural-language message to inspect.

Weight-Space Observatory is a defensive local audit toolkit that already emits
telemetry JSONL, validates records, collects local LoRA-hook metrics, builds
task-conditioned baselines, analyzes anomaly findings, exports OTEL-shaped data,
runs a synthetic capacity-detectability benchmark, executes a bounded red-team /
blue-team LSB injection demo, and generates Markdown/HTML reports. This project
turns that working substrate into an empirical AI safety study: which telemetry
summaries are stable, when task-independent structure becomes detectable, and
what controls should be required before weight-space communication is deployed in
safety-sensitive agent systems.

## Research Question

Can TFLOW-compatible multi-agent systems carry task-independent signals through
transient LoRA perturbations while remaining invisible to ordinary text-based
monitoring, and what telemetry is needed to detect or bound that risk?

## Method

1. Reproduce or approximate a TFLOW-style transient LoRA routing surface with
   frozen senders, generated low-rank receiver perturbations, and transient
   receiver injection.
2. Instrument only defensive summaries: singular values, effective rank,
   Frobenius norm, spectral gap, fusion entropy, Grassmannian drift, task family,
   module ID, and provenance metadata.
3. Measure a capacity-detectability frontier using safe synthetic proxy labels as
   offline audit annotations.
4. Compare observer levels: prompt/output logs, derived telemetry summaries, and
   direct perturbation access in controlled experiments.
5. Publish a paper-style report, reproducible benchmark, and deployment controls
   for teams evaluating adapter-mediated agent systems.

## Empirical Detection Demo

The repository now includes a runnable controlled exploit-and-detect experiment:
`examples/e2e_covert_channel_demo.py` initializes a local FP32 PyTorch layer,
uses a bounded `WeightSpaceCovertChannel` fixture to inject an inert payload into
mantissa LSBs, and then runs WSO telemetry on clean and infected perturbations.

The blue-team side measures the write-event delta (`infected_weight -
clean_weight`) rather than overstating whole-weight drift. It exports
`telemetry.jsonl`, `detection.jsonl`, `baseline.json`, `findings.json`, and
`summary.json`, storing only derived metrics and payload metadata. The expected
result is a clean `LOW` control and a `HIGH` infected event flagged by effective
rank, spectral gap, and Grassmannian drift.

For DARPA, NSF, and Anthropic-style reviewers, this upgrades the proposal from a
theoretical observability argument to an empirical red-team / blue-team
demonstration: a concrete zero-day-style weight-space steganography fixture,
paired with telemetry that detects the anomaly under a reproducible local
baseline.

## Workplan

| Period | Milestone | Output |
| --- | --- | --- |
| Month 1 | Reproduce or approximate transient LoRA routing; finalize observer model and telemetry schema. | Baseline task-conditioned telemetry corpus. |
| Month 2 | Implement capacity-detectability experiments with safe synthetic proxy labels. | Initial frontier curves and metric ablations. |
| Month 3 | Stress-test under benign distribution shift, implementation noise, sender-count changes, and fusion variation. | Robustness analysis and failure modes. |
| Month 4 | Package public artifact and write results. | Paper/report, benchmark, CLI/notebook, deployment guidance. |

## Expected Outputs

- Workshop-style paper or technical report.
- Open benchmark for transient adapter telemetry and observability.
- Reference implementation for spectrum, fusion-gate, provenance, and
  Grassmannian drift auditing.
- Practical monitoring recommendations for TFLOW-style and adapter-mediated
  multi-agent deployments.

## Best-Match Funders

- The Alignment Project: primary large non-dilutive research target.
- Anthropic Fellows: primary individual/fellowship target.
- Foresight AI for Science and Safety Nodes: fast individual/team grant target.
- Manifund, EA Long-Term Future Fund, and Coefficient/Open Phil: flexible
  6-12 month research sprint support.
- CAISI/CIFAR, NSERC, Mitacs, and Schmidt Sciences: partner-dependent targets if
  a university or AI institute sponsor is added.

## Budget Shapes

### Four-Month Fellowship Budget

- Founder/researcher stipend for full-time execution.
- Compute/API budget for small-model and open-model experiments.
- Cloud storage and experiment tracking.
- Paper/report preparation and reproducibility packaging.

### Six- to Twelve-Month Independent Grant Budget

- Founder/researcher stipend.
- Part-time research engineering support for benchmark hardening.
- Compute/API budget.
- Security review and responsible-disclosure review.
- Publication, workshop, and community dissemination costs.

## Safety Boundary

The project is defensive. It includes one bounded local red-team fixture for
measuring whether telemetry detects FP32 LSB injection, but it does not include a
deployment covert-channel protocol, keyed payload extraction, message codebooks,
prompt-search steganography, monitor bypass logic, or evasion tuning against
specific monitoring systems. Default demos use inert payloads and persist only
derived telemetry plus payload metadata. Synthetic labels remain audit
annotations only.
