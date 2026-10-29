---
title: 'AdaRoute: Threshold-Calibrated Decision-Entropy Routing and Bayesian Virtual Evidence Pooling for Hybrid Neuro-Symbolic Inference'
tags:
  - Python
  - large language models
  - test-time compute
  - uncertainty quantification
  - Bayesian networks
  - model routing
  - clinical decision support
authors:
  - name: Adaobi Amanna
    affiliation: 1
affiliations:
  - name: Whiting School of Engineering, Johns Hopkins University, Baltimore, MD, United States
    index: 1
date: 29 October 2026
bibliography: paper.bib
---

# Summary

Frontier reasoning Large Language Models (LLMs) expose controllable test-time chain-of-thought (CoT) compute budgets, such as discrete reasoning tiers (`LOW`, `MEDIUM`, `HIGH`) or thinking-token ceilings, that substantially improve extraction fidelity on complex unstructured narratives [@Snell2025]. However, invoking deep test-time deliberation uniformly across high-throughput operational streams is computationally and financially prohibitive. **`AdaRoute`** (Adaptive Decision-Entropy Allocation Router) is a lightweight, domain-agnostic Python library that couples fast probabilistic tabular or graphical "System 1" estimators with tiered "System 2" test-time LLM deliberation. Rather than measuring predictive uncertainty around a symmetric probability of $p = 0.50$, `AdaRoute` centers binary Shannon entropy directly on the domain-specific actionable operating threshold $\tau^* \in (0, 1)$, ensuring that test-time LLM reasoning tokens are allocated exclusively to ambiguous decision-boundary cases while bypassing LLM calls on routine trajectories.

# Statement of need

In skewed, low-prevalence operational streams (such as clinical deterioration screening, fraud detection, or infrastructure anomaly triage), the prevalence-calibrated operating threshold $\tau^*$ derived via Youden's $J$ statistic or asymmetric cost-utility matrices typically lies far below $0.50$ (e.g., $\tau^* \in [0.005, 0.05]$). Standard binary Shannon entropy $H(p) = -p \log_2(p) - (1 - p)\log_2(1 - p)$ peaks at $p = 0.50$. Consequently, a record with $p = 0.50$ is already 10- to 50-fold above $\tau^*$ and represents an unambiguous positive alert, whereas a genuinely ambiguous boundary record hovering at $p \approx \tau^* = 0.0267$ yields a deceptively small raw Shannon entropy of $H(0.0267) = 0.178\text{ bits}$, causing standard uncertainty routers to skip deliberation when clinical or operational ambiguity is highest.

`AdaRoute` resolves this mismatch for researchers and engineers building hybrid neuro-symbolic decision support pipelines by providing: (1) threshold-calibrated decision-entropy warping $H_{\tau^*}(p) \in [0, 1]\text{ bits}$ (`adaroute.entropy`), (2) missingness-aware multi-tier test-time compute routing across `SKIP_SYSTEM1_CONFIDENT`, `LOW`, `MEDIUM`, and `HIGH` budgets (`adaroute.router.AdaRouter`), (3) $O(1)$ evidence-tuple hash memoization for exact Variable Elimination marginalization over unmeasured variables under conditional Missing At Random (MAR) (`adaroute.memoized_ve.MemoizedVariableElimination`), and (4) threshold-aligned precision-weighted logarithmic opinion pooling (`adaroute.pooling.LogLinearOpinionPooler`).

# State of the field

Existing open-source LLM routing libraries address different problem regimes. **FrugalGPT** [@Chen2023] and **RouteLLM** [@Ong2024] route natural language prompts across heterogeneous LLM families based on prompt difficulty or preference-trained reward models, operating without an upstream structured probabilistic graphical model or prevalence-calibrated classification threshold. Conversely, generation-time uncertainty estimators such as **Semantic Entropy** [@Kuhn2023] and **AdaThink-Med** [@Rui2025] require invoking the LLM first to sample tokens before estimating uncertainty. Probabilistic graphical model libraries such as `pgmpy` support exact Variable Elimination and Virtual Evidence [@Pearl1988; @Lin2008] but do not provide threshold-anchored decision entropy, $O(1)$ MAR evidence-tuple caching, or tiered LLM log-odds pooling [@Genest1986]. Rather than contributing a domain-specific patch to an LLM-only router or a graphical-model-only library, `AdaRoute` provides a standalone, zero-overhead bridge between `scikit-learn`/`pgmpy` probabilistic backbones and reasoning LLM APIs.

# Software design

`AdaRoute` is architected around four composable, strictly typed modules with minimal dependencies (`numpy`, `scipy`, `scikit-learn`). First, `adaroute_decision_entropy` implements both temperature-scaled log-odds shifting (`mode="logit"`) and piecewise-linear coordinate warping (`mode="piecewise"`):

$$\tilde{p}_{\text{logit}}(p; \tau^*, s) = \sigma\!\left(\frac{\text{logit}(p) - \text{logit}(\tau^*)}{s}\right), \quad H_{\tau^*}(p) = -\tilde{p}\log_2(\tilde{p}) - (1 - \tilde{p})\log_2(1 - \tilde{p})$$

By construction, $H_{\tau^*}(\tau^*) = 1.0\text{ bit}$ while confident rule-out ($p \ll \tau^*$) and rule-in ($p \gg \tau^*$) trajectories decay smoothly toward $0\text{ bits}$. Second, `AdaRouter` calibrates $\tau^*$ out-of-sample via Youden's $J$ statistic and combines $H_{\tau^*}(p)$ with feature-missingness counts to dispatch graded reasoning tiers (`SKIP_SYSTEM1_CONFIDENT`, `LOW`, `MEDIUM`, `HIGH`). Third, `MemoizedVariableElimination` strips unmeasured assays (where boolean measurement indicators are `False`) prior to hashing observed evidence tuples, reducing exact MAR marginalization latency to $0.008\text{ ms/record}$. Fourth, `LogLinearOpinionPooler` aligns System 1 and System 2 operating thresholds via $\Delta_{\text{shift}} = \text{logit}(\tau^*_{S1}) - \text{logit}(\tau^*_{S2})$ and fuses posteriors in log-odds space with thought-token-scaled weights $\gamma_k \in [0, 1]$.

# Research impact statement

`AdaRoute` powers the dual-process neuro-symbolic sepsis risk stratification architecture evaluated across $N = 223,452$ adult index hospitalizations ($44,691$ temporal validation admissions; mortality prevalence $2.23\%$, $\tau^* = 0.0267$) in MIMIC-IV v3.1 [@Amanna2026]. On the temporal validation cohort, `AdaRoute` bypassed LLM invocation on $73.9\%$ of confident admissions, cutting total LLM token expenditure by **80.7%** ($353$ vs. $1,834\text{ tokens/patient}$) while elevating high-uncertainty boundary-subgroup ($N = 6,334$) AUROC by $+13.55$ percentage points ($0.5944 \rightarrow 0.7299$ [3-tier] and $0.7334$ [Equalized `HIGH`]) and outperforming uncalibrated Shannon entropy $H(p)$ in boundary precision-recall ($0.1540$ vs. $0.1438$ AUPRC at a matched $26.1\%$ call budget). The package includes an automated `pytest` test suite and continuous integration workflows enabling immediate adoption across clinical informatics and hybrid tabular-text AI benchmarks.

# AI usage disclosure

Gemini 4 Argon (`gemini-4-argon`) was evaluated as the downstream "System 2" clinical narrative extractor in the empirical MIMIC-IV v3.1 benchmark [@Amanna2026]. AI-assisted coding and editorial tools (Google Gemini and DeepMind Deep Review) were used during package scaffolding, documentation formatting, and manuscript preparation; all mathematical formulations, software implementations, unit tests, and empirical claims were authored, executed, and verified by the author.

# References
