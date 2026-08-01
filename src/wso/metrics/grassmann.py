import torch


def get_principal_subspace(delta_w: torch.Tensor, rank: int = 4) -> torch.Tensor:
    """Extract an orthonormal basis for the leading left singular subspace."""
    if delta_w.ndim != 2:
        raise ValueError(f"delta_w must be 2D, got shape {tuple(delta_w.shape)}")
    max_rank = min(delta_w.shape)
    if rank <= 0 or rank > max_rank:
        raise ValueError(f"rank must be in [1, {max_rank}], got {rank}")
    u, _, _ = torch.linalg.svd(delta_w.detach().to(dtype=torch.float32), full_matrices=False)
    return u[:, :rank]


def principal_angles(u1: torch.Tensor, u2: torch.Tensor) -> torch.Tensor:
    """Return principal angles between two orthonormal subspace bases."""
    if u1.shape != u2.shape:
        raise ValueError(f"subspaces must have same shape, got {u1.shape} and {u2.shape}")
    s = torch.linalg.svdvals(u1.T @ u2)
    s = torch.clamp(s, 0.0, 1.0)
    # Identical subspaces may produce singular values infinitesimally below 1
    # due to floating-point SVD roundoff; snap near-boundary values to avoid
    # reporting artificial nonzero principal angles.
    s = torch.where(s > 1.0 - 1e-6, torch.ones_like(s), s)
    s = torch.where(s < 1e-12, torch.zeros_like(s), s)
    return torch.acos(s)


def grassmann_geodesic_distance(u1: torch.Tensor, u2: torch.Tensor) -> float:
    """Geodesic Grassmann distance: L2 norm of principal angles."""
    angles = principal_angles(u1, u2)
    return float(torch.linalg.norm(angles).item())


def grassmann_chordal_distance(u1: torch.Tensor, u2: torch.Tensor) -> float:
    """Chordal Grassmann distance: L2 norm of sin(principal angles)."""
    angles = principal_angles(u1, u2)
    return float(torch.linalg.norm(torch.sin(angles)).item())
