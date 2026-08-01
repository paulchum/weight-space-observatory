# Anthropic Fellows Application: The Uninspectable Message

## Project Title

The Uninspectable Message: Auditing Hidden Communication in Weight-Space Multi-Agent Systems

## Short Pitch

Modern multi-agent LLM monitoring assumes that important inter-agent communication appears in prompts, tool calls, outputs, or logs. TFLOW-style systems challenge that assumption: sender agents can communicate by inducing transient LoRA perturbations in a receiver model, leaving no natural-language message to inspect. I have built Weight-Space Observatory v1, a defensive local audit toolkit that collects derived LoRA/adapter telemetry, calibrates task-conditioned baselines, analyzes anomaly findings, and runs synthetic capacity-detectability benchmarks. I propose a four-month empirical AI safety project that uses this foundation to measure the observability gap and determine which monitoring signals are sufficient to detect or bound task-independent hidden communication risk.

The project will not implement operational steganography, keyed extraction, or monitor bypass tooling. It will use safe synthetic proxy labels, open-source toy/small-model systems, and derived telemetry summaries to produce a public research artifact and audit toolkit.

## Anthropic Fellows Fit

The Anthropic Fellows program emphasizes empirical AI safety work, external infrastructure, mentorship, and public research outputs over a roughly four-month period. This project fits the AI Safety Fellows and AI Security Fellows tracks, with overlap in model internals and mechanistic interpretability.

The core alignment is direct:

- **AI security:** studies an emerging non-text communication surface that can evade prompt/output-only monitoring.
- **Adversarial robustness and control:** asks what control measures remain valid when agent communication moves into transient parameter space.
- **Model internals:** treats LoRA perturbation spectra, subspaces, and fusion gates as inspectable internal traces.
- **Empirical safety:** starts from runnable telemetry, baseline, analysis, reporting, and benchmark code rather than a purely conceptual argument.

## Research Question

Can TFLOW-compatible multi-agent systems carry task-independent signals through transient LoRA perturbations while remaining invisible to ordinary text-based monitoring, and what telemetry is needed to detect or bound that risk?

Subquestions:

- When sender hidden states are translated into receiver-side LoRA perturbations, which derived signatures remain stable enough for monitoring?
- How much synthetic task-independent signal can be statistically recovered from telemetry summaries before task-conditioned anomaly detectors flag the perturbation family?
- Which observer level is sufficient: prompt/output logs, derived telemetry summaries, or direct perturbation access?
- What deployment controls should be required before using TFLOW-like weight-space communication in safety-sensitive agent systems?

## Method

1. **Reproduce the communication surface.** Extend the existing local LoRA hook and toy transient-router demos into a TFLOW-compatible small-model pipeline with frozen sender agents, a parameter generator, fused low-rank receiver perturbations, and transient receiver injection.
2. **Instrument defensive telemetry.** Record only summaries such as singular values, effective rank, Frobenius norm, spectral gap, fusion entropy, Grassmannian drift, task family, module ID, and provenance metadata.
3. **Measure the capacity-detectability frontier.** Use synthetic non-task labels as offline audit annotations, then measure label separability from telemetry summaries versus anomaly score under task-conditioned benign baselines.
4. **Compare observer models.** Evaluate prompt/output-only monitoring against summary telemetry and direct perturbation access in controlled experiments.
5. **Publish controls.** Produce a paper-style report and open-source tooling that explain what telemetry must be collected for TFLOW-style MAS deployments.

## Four-Month Plan

- **Month 1:** Reproduce or approximate TFLOW-style transient LoRA routing; finalize observer model and telemetry schema; collect benign task-conditioned baselines.
- **Month 2:** Implement capacity-detectability experiments with safe synthetic proxy labels; evaluate summary metrics across task families and receiver modules.
- **Month 3:** Stress-test detectors under benign distribution shift, implementation noise, sender-count changes, and fusion-gate variation.
- **Month 4:** Write the public research report, package the audit CLI/notebook, and produce deployment guidance for monitoring transient weight-space communication.

## Expected Outputs

- Short research paper or workshop-style report.
- Open-source telemetry and audit toolkit in Weight-Space Observatory, building on the existing v1 CLI.
- Reproducible toy benchmark for transient LoRA communication observability.
- Practical monitoring recommendations for teams considering TFLOW-style multi-agent systems.

## Safety Boundary

This project is defensive. It will not include:

- operational covert-channel encoders;
- keyed payload extraction;
- codebooks for hiding messages in perturbations;
- prompt-search steganography;
- evasion tuning against specific monitoring systems.

Synthetic labels are used only as audit annotations to measure whether task-independent structure can be detected or bounded.

## Application Answer Draft

I want to study a monitoring failure mode created by weight-space multi-agent communication. Recent TFLOW-style systems allow sender agents to pass information to a receiver not as text, but as transient LoRA perturbations generated from hidden states. This is efficient, but it breaks the assumption behind most agent monitoring: that communication appears in prompts, tool calls, outputs, or logs. I have built a v1 defensive audit toolkit that already emits telemetry JSONL, validates records, collects local LoRA-hook metrics without network access, builds task-conditioned baselines, analyzes anomalies, runs a synthetic capacity-detectability frontier, and generates Markdown/HTML/JSON reports. The fellowship project would turn that working spine into a disciplined empirical study: which telemetry summaries are stable, when synthetic task-independent structure becomes recoverable, and what controls are necessary before weight-space communication is deployed in safety-sensitive agent systems.

Anthropic is the right environment because the question sits at the boundary of AI security, control, and model internals. The project is practical enough for a four-month fellowship because the engineering substrate already exists: telemetry schemas, LoRA metric code, toy routing, local hook extraction, baseline calibration, anomaly reporting, benchmark outputs, tests, and responsible-research documentation. The next step is the research work Anthropic is especially well placed to support: stress-testing the observer model, mapping the capacity-detectability frontier under realistic small-model conditions, and turning the result into deployment guidance.
