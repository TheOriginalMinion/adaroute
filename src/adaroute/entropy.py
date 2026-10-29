"""Threshold-calibrated decision-entropy functions for skewed probabilistic streams."""

from __future__ import annotations
from typing import Literal
import numpy as np
from scipy.special import expit, logit


def shannon_entropy(probs: np.ndarray | list[float]) -> np.ndarray:
    """Compute standard binary Shannon entropy H(p) in [0, 1] bits (peaks at p = 0.50)."""
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def adaroute_decision_entropy(
    probs: np.ndarray | list[float],
    tau_star: float,
    scale: float = 0.55,
    mode: Literal["logit", "piecewise"] = "logit",
) -> np.ndarray:
    """Compute threshold-calibrated binary decision entropy H_{tau*}(p) in [0, 1] bits.

    Reaches global maximum 1.0 bit exactly when p == tau_star, and decays toward
    0.0 bits when p << tau_star (confident rule-out) or p >> tau_star (confident rule-in).
    """
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    tau = float(np.clip(tau_star, 1e-6, 1.0 - 1e-6))
    if scale <= 0:
        raise ValueError(f"scale must be strictly positive, got {scale}.")

    if mode == "logit":
        p_tilde = expit((logit(p) - logit(tau)) / float(scale))
    elif mode == "piecewise":
        p_tilde = np.where(
            p <= tau,
            0.5 * (p / tau),
            0.5 + 0.5 * ((p - tau) / (1.0 - tau)),
        )
    else:
        raise ValueError(f"Unsupported mode: {mode!r}. Expected 'logit' or 'piecewise'.")

    p_tilde = np.clip(p_tilde, 1e-9, 1.0 - 1e-9)
    return -(p_tilde * np.log2(p_tilde) + (1.0 - p_tilde) * np.log2(1.0 - p_tilde))
