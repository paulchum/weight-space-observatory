import torch


def compute_full_spectrum(delta_w: torch.Tensor) -> torch.Tensor:
    """Return all singular values of a transient perturbation."""
    if delta_w.ndim != 2:
        raise ValueError(f"delta_w must be 2D, got shape {tuple(delta_w.shape)}")
    return torch.linalg.svdvals(delta_w.detach().to(dtype=torch.float32))


def compute_spectral_signature(delta_w: torch.Tensor, top_k: int = 8) -> torch.Tensor:
    """Return the top-k singular values of a transient perturbation."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    s = compute_full_spectrum(delta_w)
    if s.numel() < top_k:
        padding = torch.zeros(top_k - s.numel(), dtype=s.dtype, device=s.device)
        s = torch.cat([s, padding])
    return s[:top_k]


def calculate_spectral_gap(singular_values: torch.Tensor, k: int = 4) -> float:
    """Return sigma_k - sigma_{k+1}; small gaps imply unstable subspace measurement."""
    if k <= 0:
        raise ValueError("k must be positive")
    if singular_values.numel() <= k:
        return 0.0
    return float((singular_values[k - 1] - singular_values[k]).item())


def frobenius_norm(delta_w: torch.Tensor) -> float:
    """Return the Frobenius norm of a transient perturbation."""
    if delta_w.ndim != 2:
        raise ValueError(f"delta_w must be 2D, got shape {tuple(delta_w.shape)}")
    return float(torch.linalg.norm(delta_w.detach().to(dtype=torch.float32), ord="fro").item())


def effective_rank_from_singular_values(singular_values: torch.Tensor, eps: float = 1e-12) -> float:
    """Return entropy-based effective rank from a singular-value vector."""
    s = singular_values.detach().to(dtype=torch.float32)
    total = s.sum()
    if float(total.item()) <= eps:
        return 0.0
    p = s / (total + eps)
    entropy = -(p * torch.log(p + eps)).sum()
    return float(torch.exp(entropy).item())
