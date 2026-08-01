# Perturbation-Hidden Observer Model

A perturbation-hidden observer has access to:

- prompts;
- generated outputs;
- application logs;
- ordinary model-serving metadata.

The observer does not have access to:

- transient LoRA / adapter tensors;
- sender fusion gates;
- receiver-side weight patches;
- hidden-state routing internals.

The defensive goal is to add telemetry that closes this observability gap.
