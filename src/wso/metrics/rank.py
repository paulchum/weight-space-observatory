import torch


def numerical_rank(singular_values: torch.Tensor, tol: float = 1e-6) -> int:
    """Count singular values above a numerical tolerance."""
    return int((singular_values.detach().to(dtype=torch.float32) > tol).sum().item())
