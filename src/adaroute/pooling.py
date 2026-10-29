"""Precision-weighted logarithmic opinion pooling for Virtual Evidence fusion."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import numpy as np
from scipy.special import expit, logit


def log_linear_opinion_pool(
    p_sys1: np.ndarray | Sequence[float],
    p_sys2: np.ndarray | Sequence[float],
    gamma: np.ndarray | Sequence[float] | float,
    logit_shift: float = 0.0,
) -> np.ndarray:
    """Convex log-linear opinion pooling in log-odds space."""
    l1 = logit(np.clip(np.asarray(p_sys1, dtype=np.float64), 1e-6, 1.0 - 1e-6))
    l2 = logit(np.clip(np.asarray(p_sys2, dtype=np.float64), 1e-6, 1.0 - 1e-6)) + float(
        logit_shift
    )
    g = np.clip(np.asarray(gamma, dtype=np.float64), 0.0, 1.0)
    return expit((1.0 - g) * l1 + g * l2)


@dataclass
class LogLinearOpinionPooler:
    """Tier-aware logarithmic opinion pooler with threshold alignment shift."""

    tier_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "SKIP_SYSTEM1_CONFIDENT": 0.80,
            "LOW": 0.82,
            "MEDIUM": 0.92,
            "HIGH": 1.00,
        }
    )
    thought_bonus_weight: float = 0.005
    reference_thought_tokens: float = 700.0
    logit_shift: float = 0.0

    def fit_threshold_shift(self, tau_sys1: float, tau_sys2: float) -> "LogLinearOpinionPooler":
        """Compute log-odds alignment shift between System 1 and System 2 thresholds."""
        t1 = float(np.clip(tau_sys1, 1e-6, 1.0 - 1e-6))
        t2 = float(np.clip(tau_sys2, 1e-6, 1.0 - 1e-6))
        self.logit_shift = float(logit(t1) - logit(t2))
        return self

    def fuse(
        self,
        p_sys1: Sequence[float],
        p_sys2: Sequence[float],
        tiers: Sequence[str],
        thought_tokens: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """Fuse System 1 and System 2 probabilities across routed tiers."""
        tiers_arr = np.asarray(tiers, dtype=object)
        base_g = np.array(
            [self.tier_weights.get(str(t), 0.80) for t in tiers_arr], dtype=np.float64
        )
        if thought_tokens is not None:
            th = np.maximum(0.0, np.asarray(thought_tokens, dtype=np.float64))
            bonus = self.thought_bonus_weight * np.minimum(
                1.0, np.log1p(th) / np.log1p(self.reference_thought_tokens)
            )
            bonus[tiers_arr == "SKIP_SYSTEM1_CONFIDENT"] = 0.0
            gamma = np.clip(base_g + bonus, 0.0, 1.0)
        else:
            gamma = np.clip(base_g, 0.0, 1.0)
        return log_linear_opinion_pool(p_sys1, p_sys2, gamma, logit_shift=self.logit_shift)
