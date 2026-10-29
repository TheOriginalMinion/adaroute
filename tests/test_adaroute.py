"""Unit test suite for AdaRoute decision entropy, routing, pooling, and MAR memoization."""

import numpy as np
import pytest
from adaroute import (
    AdaRouter,
    LogLinearOpinionPooler,
    MemoizedVariableElimination,
    adaroute_decision_entropy,
    log_linear_opinion_pool,
    shannon_entropy,
)


def test_decision_entropy_peaks_at_tau_star():
    tau = 0.0267
    probs = np.array([0.001, tau, 0.50, 0.95])
    ent_logit = adaroute_decision_entropy(probs, tau_star=tau, scale=0.55, mode="logit")
    ent_pw = adaroute_decision_entropy(probs, tau_star=tau, mode="piecewise")

    # At p == tau_star, threshold-calibrated entropy must equal 1.0 bit
    assert np.isclose(ent_logit[1], 1.0, atol=1e-6)
    assert np.isclose(ent_pw[1], 1.0, atol=1e-6)

    # Confident low-risk (p << tau*) and overt high-risk (p >> tau*) decay toward 0
    assert ent_logit[0] < 0.05
    assert ent_logit[2] < 0.05
    assert ent_logit[3] < 0.001

    # Contrast with uncalibrated Shannon entropy, which under-scores p = tau_star
    raw_shannon = shannon_entropy(probs)
    assert raw_shannon[1] < 0.20
    assert np.isclose(raw_shannon[2], 1.0, atol=1e-6)


def test_adarouter_fit_and_route():
    rng = np.random.default_rng(42)
    y_true = np.array([0] * 970 + [1] * 30)
    probs = np.concatenate([rng.uniform(0.001, 0.03, 970), rng.uniform(0.02, 0.40, 30)])

    router = AdaRouter(scale=0.55, delta_low=0.35, delta_med=0.72, delta_high=0.85)
    router.fit_youden_threshold(y_true, probs)
    assert router.tau_star is not None
    assert 0.01 <= router.tau_star <= 0.10

    test_probs = np.array([0.001, router.tau_star, router.tau_star, 0.018])
    missing = np.array([0, 3, 0, 1])
    tiers = router.route(test_probs, missing_counts=missing)

    assert tiers[0] == "SKIP_SYSTEM1_CONFIDENT"
    assert tiers[1] == "HIGH"
    assert tiers[2] == "MEDIUM"

    summary = router.summarize(tiers)
    assert summary.total_records == 4
    assert summary.tier_counts["SKIP_SYSTEM1_CONFIDENT"] == 1
    assert summary.tier_counts["HIGH"] == 1
    assert 0.0 < summary.estimated_token_savings_fraction < 1.0


def test_log_linear_opinion_pooler():
    pooler = LogLinearOpinionPooler().fit_threshold_shift(tau_sys1=0.0267, tau_sys2=0.0332)
    assert pooler.logit_shift < 0.0

    p1 = np.array([0.005, 0.0267, 0.0267])
    p2 = np.array([0.005, 0.0100, 0.1500])
    tiers = ["SKIP_SYSTEM1_CONFIDENT", "LOW", "HIGH"]
    thoughts = [0, 0, 1540]

    fused = pooler.fuse(p1, p2, tiers=tiers, thought_tokens=thoughts)
    assert fused.shape == (3,)
    assert fused[2] > p1[2]
    assert fused[1] < p1[1]


def test_memoized_variable_elimination():
    call_count = 0

    def mock_ve(obs):
        nonlocal call_count
        call_count += 1
        return 0.15 if obs.get("lactate") == "High" else 0.02

    mve = MemoizedVariableElimination(
        query_fn=mock_ve,
        feature_names=["age", "lactate"],
        missingness_map={"lactate": "lactate_measured"},
    )
    records = [
        {"age": "Over_65", "lactate": "Normal", "lactate_measured": False},
        {"age": "Over_65", "lactate": "High", "lactate_measured": False},
        {"age": "Over_65", "lactate": "High", "lactate_measured": True},
    ]
    preds = mve.predict_proba(records)
    # First two records have lactate_measured=False, so both marginalize over lactate and hit cache!
    assert np.isclose(preds[0], preds[1])
    assert preds[2] == 0.15
    assert mve.cache_size == 2
    assert call_count == 2
