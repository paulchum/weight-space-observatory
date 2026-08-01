from __future__ import annotations

import torch


def simulate_transient_lora_update(
    num_senders: int = 3,
    target_rank: int = 4,
    dim: int = 256,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simulate fusion of sender updates into a receiver-side LoRA perturbation."""
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
        gate_logits = torch.randn(num_senders, generator=generator)
    else:
        generator = None
        gate_logits = torch.randn(num_senders)

    fusion_weights = torch.nn.functional.softmax(gate_logits, dim=-1)
    fused_delta = torch.zeros((dim, dim))
    for i in range(num_senders):
        left = torch.randn(dim, target_rank, generator=generator)
        right = torch.randn(target_rank, dim, generator=generator)
        fused_delta += fusion_weights[i] * (left @ right)
    return fused_delta, gate_logits, fusion_weights
