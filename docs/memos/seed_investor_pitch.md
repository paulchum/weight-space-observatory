# MEMORANDUM: The Uninspectable Message

**To:** Seed Investors / AI Security Funders  
**From:** Paul C. / Weight-Space Security Observatory  
**Subject:** Measuring Hidden Communication Capacity in Weight-Space Multi-Agent Communication

## Executive Summary

- Agent monitoring is built for prompts, outputs, and text logs. But new multi-agent systems are beginning to communicate through hidden adapter states, memory layers, and transient weight updates.
- That creates an observability gap: safety teams may see the prompt and output while missing the actual inter-agent communication substrate.
- Weight-Space Security Observatory restores visibility with a runtime telemetry and audit layer for adapter routing, transient LoRA patches, fusion gates, and parameter-space drift.
- The first product is a high-trust diagnostic audit: instrument one agent deployment, measure hidden communication-surface risk, and deliver an executive risk report plus engineering controls.

## Technical Thesis

We study whether TFLOW / Thought Flow-style multi-agent systems expose hidden communication capacity through transient LoRA perturbations. Unlike text-based MAS, where inter-agent messages are visible in the context window, TFLOW compiles sender hidden states into temporary, low-rank receiver weight updates. We introduce a measurement framework for testing whether task-independent signals can survive semantic constraints, implementation noise, and task-conditioned anomaly filters, mapping the resulting capacity–detectability Pareto frontier. Under a perturbation-hidden observer model, text-only and output-only audits cannot directly observe this channel, motivating weight-space telemetry for future MAS safety audits.

## 90-Day Milestone

- Reproduce a TFLOW-style transient LoRA pipeline or faithful toy implementation.
- Collect first task-conditioned perturbation telemetry.
- Produce an initial capacity–detectability curve.
- Demo a telemetry dashboard showing spectrum, fusion-gate, and subspace drift signals.
