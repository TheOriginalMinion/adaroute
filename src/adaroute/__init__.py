"""AdaRoute: Threshold-Calibrated Decision-Entropy Router and Virtual Evidence Pooler."""

from .entropy import adaroute_decision_entropy, shannon_entropy
from .router import AdaRouter, RoutingSummary
from .pooling import LogLinearOpinionPooler, log_linear_opinion_pool
from .memoized_ve import MemoizedVariableElimination

__version__ = "1.0.0"
__all__ = [
    "adaroute_decision_entropy",
    "shannon_entropy",
    "AdaRouter",
    "RoutingSummary",
    "LogLinearOpinionPooler",
    "log_linear_opinion_pool",
    "MemoizedVariableElimination",
]
