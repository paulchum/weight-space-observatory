"""Defensive Davis-Kahan-style rotation bound helper.

This is a simple scalar helper for measurement planning, not a formal proof system.
"""

from __future__ import annotations


def sin_theta_bound(noise_norm: float, spectral_gap: float) -> float:
    """Return a conservative sin(theta) bound of noise_norm / spectral_gap."""
    if spectral_gap <= 0:
        return float("inf")
    return noise_norm / spectral_gap
