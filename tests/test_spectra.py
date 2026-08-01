import pytest
import torch

from wso.metrics.spectra import (
    calculate_spectral_gap,
    compute_full_spectrum,
    compute_spectral_signature,
    effective_rank_from_singular_values,
    frobenius_norm,
)


def test_spectral_signature_shape() -> None:
    x = torch.randn(8, 6)
    s = compute_spectral_signature(x, top_k=10)
    assert s.shape == (10,)


def test_spectral_gap_known_values() -> None:
    s = torch.tensor([5.0, 3.0, 2.0, 1.0, 0.25])
    assert calculate_spectral_gap(s, k=4) == pytest.approx(0.75)


def test_full_spectrum_and_norm() -> None:
    x = torch.eye(4)
    s = compute_full_spectrum(x)
    assert torch.allclose(s, torch.ones(4))
    assert frobenius_norm(x) == pytest.approx(2.0)


def test_effective_rank_zero_vector() -> None:
    assert effective_rank_from_singular_values(torch.zeros(4)) == 0.0
