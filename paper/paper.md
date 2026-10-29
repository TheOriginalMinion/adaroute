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

Frontier reasoning Large Language Models (LLMs) expose controllable test-time chain-of-thought (CoT) compute budgets, such as discrete reasoning tiers (`LOW`, `MEDIUM`, `HIGH`) or thinking-token ceilings, that substantially improve extraction fidelity on complex unstructured narratives [@Snell2025]. However, invoking deep test-time deliberation uniformly across high-throughput operational streams is computationally and financially prohibitive. **`AdaRoute`** (Adaptive Decision-Entropy Allocation Router) is a lightweight, domain-agnostic Python library that couples fast probabilistic tabular or graphical "System 1" estimators with tiered "System 2" test-time LLM deliberation [@Kahneman2011]. Rather than measuring predictive uncertainty around a symmetric probability of $p = 0.50$, `AdaRoute` centers binary Shannon entropy directly on the domain-specific actionable operating threshold $\tau^{\ast} \in (0, 1)$, ensuring that test-time LLM reasoning tokens are allocated exclusively to ambiguous decision-boundary cases while bypassing LLM calls on routine trajectories.

# Statement of need

In skewed, low-prevalence operational streams (such as clinical deterioration screening, financial fraud detection, or cloud infrastructure anomaly triage), the prevalence-calibrated operating threshold $\tau^{\ast}$ derived via Youden's $J$ statistic or asymmetric cost-utility matrices typically lies far below $0.50$ (for example, $\tau^{\ast} \in [0.005, 0.05]$). Standard binary Shannon entropy $H(p) = -p \log_2(p) - (1 - p)\log_2(1 - p)$ peaks at $p = 0.50$. Consequently, a record with $p = 0.50$ is already 10- to 50-fold above $\tau^{\ast}$ and represents an unambiguous positive alert, whereas a genuinely ambiguous boundary record hovering at $p \approx \tau^{\ast} = 0.0267$ yields a deceptively small raw Shannon entropy of $H(0.0267) = 0.178\text{ bits}$, causing standard uncertainty routers to skip deliberation when clinical or operational ambiguity is highest.

`AdaRoute` resolves this mismatch for researchers and systems engineers building hybrid neuro-symbolic decision support pipelines by providing four integrated capabilities: (1) threshold-calibrated decision-entropy warping $H_{\tau^{\ast}}(p) \in [0, 1]\text{ bits}$ (`adaroute.entropy`), (2) missingness-aware multi-tier test-time compute routing across `SKIP_SYSTEM1_CONFIDENT`, `LOW`, `MEDIUM`, and `HIGH` budgets (`adaroute.router.AdaRouter`), (3) $O(1)$ evidence-tuple hash memoization for exact Variable Elimination marginalization over unmeasured variables under conditional Missing At Random (MAR) (`adaroute.memoized_ve.MemoizedVariableElimination`), and (4) threshold-aligned precision-weighted logarithmic opinion pooling (`adaroute.pooling.LogLinearOpinionPooler`).

# State of the field

Existing open-source LLM routing libraries address different problem regimes. **FrugalGPT** [@Chen2023] and **RouteLLM** [@Ong2024] route natural language prompts across heterogeneous LLM families based on prompt difficulty or preference-trained reward models, operating without an upstream structured probabilistic graphical model or prevalence-calibrated classification threshold. Conversely, generation-time uncertainty estimators such as **Semantic Entropy** [@Kuhn2023] and **AdaThink-Med** [@Rui2025] require invoking the LLM first to sample tokens before estimating uncertainty. Probabilistic graphical model libraries such as `pgmpy` [@Ankan2015] support exact Variable Elimination and Virtual Evidence [@Pearl1988; @Lin2008] but do not provide threshold-anchored decision entropy, $O(1)$ MAR evidence-tuple caching, or tiered LLM log-odds pooling [@Genest1986]. Similarly, standard tabular machine learning toolkits (`scikit-learn` [@Pedregosa2011]) and conformal abstention frameworks [@Shashikumar2021] defer uncertain predictions to human review rather than dispatching graded test-time LLM reasoning to resolve boundary ambiguity. Rather than contributing a domain-specific patch to an LLM-only router or a graphical-model-only library, `AdaRoute` provides a standalone, zero-overhead bridge between `scikit-learn`/`pgmpy` probabilistic backbones and reasoning LLM APIs.

# Software design

`AdaRoute` is architected around four composable, strictly typed modules with minimal dependencies (`numpy`, `scipy`, `scikit-learn`), summarized in Table 1.

| Module | Primary Class / Function | Mathematical Role & Runtime Complexity |
| :--- | :--- | :--- |
| `adaroute.entropy` | `adaroute_decision_entropy` | Maps $p \in (0, 1)$ relative to $\tau^{\ast}$ into $H_{\tau^{\ast}}(p) \in [0, 1]\text{ bits}$ via log-odds temperature scaling or piecewise-linear warping; vectorized $O(N)$. |
| `adaroute.router` | `AdaRouter` | Learns $\tau^{\ast}$ out-of-sample via Youden's $J$ statistic and dispatches records across `SKIP_SYSTEM1_CONFIDENT`, `LOW`, `MEDIUM`, and `HIGH` tiers using $H_{\tau^{\ast}}(p)$ and feature missingness. |
| `adaroute.memoized_ve` | `MemoizedVariableElimination` | Strips unmeasured variables (`measured == False`) and caches exact Bayesian network Variable Elimination posteriors by canonical observed-state tuple; $O(1)$ amortized lookup ($0.008\text{ ms/record}$). |
| `adaroute.pooling` | `LogLinearOpinionPooler` | Aligns System 1 and System 2 log-odds thresholds ($\Delta_{\text{shift}}$) and fuses posteriors via thought-token-scaled logarithmic opinion pooling. |

Table 1: Modular software architecture of the `adaroute` Python package.

First, `adaroute_decision_entropy` implements both temperature-scaled log-odds shifting (`mode="logit"`) and piecewise-linear coordinate warping (`mode="piecewise"`):

$$
\tilde{p}_{\text{logit}}(p; \tau^{\ast}, s) = \sigma\left(\frac{\text{logit}(p) - \text{logit}(\tau^{\ast})}{s}\right), \quad H_{\tau^{\ast}}(p) = -\tilde{p}\log_2(\tilde{p}) - (1 - \tilde{p})\log_2(1 - \tilde{p})
$$

By construction, $H_{\tau^{\ast}}(\tau^{\ast}) = 1.0\text{ bit}$ while confident rule-out ($p \ll \tau^{\ast}$) and rule-in ($p \gg \tau^{\ast}$) trajectories decay smoothly toward $0\text{ bits}$. Second, `AdaRouter` calibrates $\tau^{\ast}$ out-of-sample via Youden's $J$ statistic and combines $H_{\tau^{\ast}}(p)$ with feature-missingness counts $M_{\text{miss}}$ to dispatch graded reasoning tiers (`SKIP_SYSTEM1_CONFIDENT`, `LOW`, `MEDIUM`, `HIGH`). Third, `MemoizedVariableElimination` strips unmeasured assays (where boolean measurement indicators are `False`) prior to hashing observed evidence tuples, reducing exact MAR marginalization latency across discrete Bayesian networks to $0.008\text{ ms/record}$. Fourth, `LogLinearOpinionPooler` aligns System 1 and System 2 operating thresholds via $\Delta_{\text{shift}} = \text{logit}(\tau_{S1}^{\ast}) - \text{logit}(\tau_{S2}^{\ast})$ and fuses posteriors in log-odds space with thought-token-scaled weights $\gamma_k \in [0, 1]$:

$$
\text{logit}\left(P_{\text{fused}}^{(k)}(x)\right) = (1 - \gamma_k)\,\text{logit}\left(P_{\text{Sys1}}(x)\right) + \gamma_k\left(\text{logit}\left(P_{\text{VE}}(x, \hat{z}_k)\right) + \Delta_{\text{shift}}\right)
$$

A minimal reproducible workflow requires fewer than 15 lines of Python:

```python
import numpy as np
from adaroute import AdaRouter, LogLinearOpinionPooler, adaroute_decision_entropy

# 1. Compute threshold-calibrated decision entropy H_{tau*}(p) around tau* = 0.0267
probs = np.array([0.002, 0.0267, 0.45])
ent = adaroute_decision_entropy(probs, tau_star=0.0267, scale=0.55, mode="logit")

# 2. Route records across SKIP_SYSTEM1_CONFIDENT, LOW, MEDIUM, and HIGH
router = AdaRouter(tau_star=0.0267, scale=0.55)
tiers = router.route(probs, missing_counts=[0, 2, 0])

# 3. Fuse System 1 and System 2 posteriors via precision-weighted log-linear pooling
pooler = LogLinearOpinionPooler().fit_threshold_shift(tau_sys1=0.0267, tau_sys2=0.0332)
fused_probs = pooler.fuse(probs, [0.002, 0.140, 0.45], tiers=tiers, thought_tokens=[0, 1540, 0])
```

# Research impact statement

`AdaRoute` powers the dual-process neuro-symbolic sepsis risk stratification architecture evaluated across $N = 223,452$ adult index hospitalizations ($44,691$ temporal validation admissions; mortality prevalence $2.23\%$, $\tau^{\ast} = 0.0267$) in MIMIC-IV v3.1 [@Amanna2026]. As summarized in Table 2, `AdaRoute` bypassed LLM invocation on $73.9\%$ of confident admissions, cutting total LLM token expenditure by **80.7%** ($353$ vs. $1,834\text{ tokens/patient}$) while elevating high-uncertainty boundary-subgroup ($N = 6,334$) AUROC by $+13.55$ percentage points ($0.5944 \rightarrow 0.7299$ [3-tier] and $0.7334$ [Equalized `HIGH`]) and outperforming uncalibrated Shannon entropy $H(p)$ in boundary precision-recall ($0.1540$ vs. $0.1438$ AUPRC at a matched $26.1\%$ call budget).

| Routing / Inference Policy | Full Cohort AUROC ($N = 44,691$) | Full Cohort AUPRC | Boundary AUROC ($N = 6,334$) | Boundary AUPRC ($N = 6,334$) | Mean LLM Tokens / Patient (Saved) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| System 1 DBN Alone (`0%` LLM Calls) | 0.9094 | 0.1638 | 0.5944 | 0.0383 | 0 (100.0%) |
| Uncalibrated Shannon $H(p)$ (`26.1% HIGH`) | 0.9356 | 0.4337 | 0.7299 | 0.1438 | 479 (73.9%) |
| **`AdaRoute` $H_{\tau^{\ast}}(p)$ (Proposed 3-Tier)** | **0.9339** | **0.4085** | **0.7299** | **0.1479** | **353 (80.7%)** |
| **`AdaRoute` $H_{\tau^{\ast}}(p)$ (Equalized `26.1% HIGH`)** | **0.9341** | **0.4108** | **0.7334** | **0.1540** | **479 (73.9%)** |
| Uniform System 2 `HIGH` (`100%` LLM Calls) | 0.9402 | 0.4359 | 0.7334 | 0.1540 | 1,834 (0.0%) |

Table 2: Empirical benchmark of `AdaRoute` routing policies on the MIMIC-IV v3.1 temporal validation cohort [@Amanna2026].

The package includes an automated `pytest` test suite (`10/10` unit tests passing), an interactive Google Colab quickstart notebook (`examples/adaroute_colab_quickstart.ipynb`), and continuous integration workflows enabling immediate adoption across clinical informatics, cloud reliability engineering, and hybrid tabular-text AI benchmarks.

# AI usage disclosure

Gemini 4 Argon (`gemini-4-argon`) was evaluated as the downstream "System 2" clinical narrative extractor in the empirical MIMIC-IV v3.1 benchmark [@Amanna2026]. AI-assisted coding and editorial tools (Google Gemini and DeepMind Deep Review) were used during package scaffolding, documentation formatting, and manuscript preparation; all mathematical formulations, software implementations, unit tests, and empirical claims were authored, executed, and verified by the author.

# References
