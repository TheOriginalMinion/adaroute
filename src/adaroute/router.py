"""Multi-tier epistemic compute router for test-time LLM deliberation."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Sequence
import numpy as np
from sklearn.metrics import roc_curve
from .entropy import adaroute_decision_entropy


@dataclass(frozen=True)
class RoutingSummary:
    """Summary telemetry of tier allocations and estimated token savings."""
    total_records: int
    tier_counts: Dict[str, int]
    tier_fractions: Dict[str, float]
    skip_fraction: float
    estimated_token_savings_fraction: float


@dataclass
class AdaRouter:
    """Adaptive Decision-Entropy Allocation Router for tiered test-time LLM budgets."""

    tau_star: Optional[float] = None
    scale: float = 0.55
    mode: Literal["logit", "piecewise"] = "logit"
    delta_low: float = 0.35
    delta_med: float = 0.72
    delta_high: float = 0.85
    min_missing_for_high: int = 2

    def fit_youden_threshold(
        self, y_true: Sequence[int], sys1_probs: Sequence[float]
    ) -> "AdaRouter":
        """Calibrate tau_star on training data by maximizing Youden's J statistic."""
        fpr, tpr, thresholds = roc_curve(np.asarray(y_true), np.asarray(sys1_probs))
        self.tau_star = float(thresholds[np.argmax(tpr - fpr)])
        return self

    def compute_entropy(self, sys1_probs: Sequence[float]) -> np.ndarray:
        """Compute H_{tau*}(p) for an array of System 1 probabilities."""
        if self.tau_star is None:
            raise RuntimeError("tau_star is not set. Call fit_youden_threshold() or pass tau_star.")
        return adaroute_decision_entropy(
            sys1_probs, self.tau_star, scale=self.scale, mode=self.mode
        )

    def route(
        self,
        sys1_probs: Sequence[float],
        missing_counts: Optional[Sequence[int]] = None,
        high_risk_flags: Optional[Sequence[bool]] = None,
    ) -> np.ndarray:
        """Assign each record to SKIP_SYSTEM1_CONFIDENT, LOW, MEDIUM, or HIGH."""
        ent = self.compute_entropy(sys1_probs)
        n = len(ent)
        miss = (
            np.zeros(n, dtype=int)
            if missing_counts is None
            else np.asarray(missing_counts, dtype=int)
        )
        flags = (
            np.zeros(n, dtype=bool)
            if high_risk_flags is None
            else np.asarray(high_risk_flags, dtype=bool)
        )

        tiers = np.full(n, "LOW", dtype=object)
        tiers[ent < self.delta_low] = "SKIP_SYSTEM1_CONFIDENT"
        tiers[ent >= self.delta_med] = "MEDIUM"
        high_mask = (ent >= self.delta_high) & ((miss >= self.min_missing_for_high) | flags)
        tiers[high_mask] = "HIGH"
        return tiers

    def summarize(
        self,
        tiers: Sequence[str],
        tier_token_costs: Optional[Dict[str, float]] = None,
    ) -> RoutingSummary:
        """Summarize tier distribution and compute savings relative to uniform HIGH."""
        costs = tier_token_costs or {
            "SKIP_SYSTEM1_CONFIDENT": 0.0,
            "LOW": 1127.0,
            "MEDIUM": 1130.0,
            "HIGH": 1834.0,
        }
        arr = np.asarray(tiers, dtype=object)
        n = len(arr)
        counts = {
            k: int(np.sum(arr == k))
            for k in ("SKIP_SYSTEM1_CONFIDENT", "LOW", "MEDIUM", "HIGH")
        }
        fracs = {k: (v / n if n > 0 else 0.0) for k, v in counts.items()}
        mean_tokens = sum(fracs[k] * costs.get(k, 0.0) for k in counts)
        high_cost = costs.get("HIGH", 1834.0)
        saved = 1.0 - (mean_tokens / high_cost) if high_cost > 0 else 0.0
        return RoutingSummary(
            total_records=n,
            tier_counts=counts,
            tier_fractions=fracs,
            skip_fraction=fracs["SKIP_SYSTEM1_CONFIDENT"],
            estimated_token_savings_fraction=float(saved),
        )
