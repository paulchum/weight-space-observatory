import torch
import torch.nn.functional as F


def fusion_probabilities(gate_logits: torch.Tensor) -> torch.Tensor:
    """Return normalized sender-fusion probabilities from raw gate logits."""
    if gate_logits.ndim != 1:
        raise ValueError(f"gate_logits must be 1D, got shape {tuple(gate_logits.shape)}")
    return F.softmax(gate_logits.detach().to(dtype=torch.float32), dim=-1)


def calculate_fusion_entropy(gate_logits: torch.Tensor, eps: float = 1e-9) -> float:
    """Compute Shannon entropy of sender-routing fusion gates.

    High entropy indicates distributed sender contribution; anomaly depends on the
    task-conditioned baseline.
    """
    probs = fusion_probabilities(gate_logits)
    entropy = -(probs * torch.log(probs + eps)).sum(dim=-1)
    return float(entropy.item())
