# Threat Model

The project studies a perturbation-hidden observer model. The observer can inspect prompts, generated outputs, and ordinary application logs, but cannot directly observe transient LoRA tensors, fusion gates, or receiver-side weight patches during active inference.

This model is used to evaluate whether text-only and output-only auditing is sufficient for systems that route inter-agent information through transient adapters or weight updates.
