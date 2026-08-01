import pytest
import torch

from wso.metrics.grassmann import (
    get_principal_subspace,
    grassmann_chordal_distance,
    grassmann_geodesic_distance,
)


def test_same_subspace_zero_distance() -> None:
    x = torch.randn(16, 16)
    u = get_principal_subspace(x, rank=4)
    assert grassmann_geodesic_distance(u, u) == pytest.approx(0.0, abs=1e-4)
    assert grassmann_chordal_distance(u, u) == pytest.approx(0.0, abs=1e-4)


def test_invalid_rank_raises() -> None:
    x = torch.randn(3, 3)
    with pytest.raises(ValueError):
        get_principal_subspace(x, rank=4)
