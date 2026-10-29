"""O(1) evidence-tuple memoization wrapper for discrete Bayesian network inference."""

from __future__ import annotations
from typing import Any, Callable, Dict, Hashable, Iterable, Mapping, Optional, Tuple
import numpy as np


class MemoizedVariableElimination:
    """Memoizes exact Variable Elimination queries over discrete observed evidence tuples.

    Unmeasured variables (marked as None or omitted by boolean measurement indicators)
    are excluded from the evidence dictionary so exact Variable Elimination sums over
    their latent state space under conditional Missing At Random (MAR).
    """

    def __init__(
        self,
        query_fn: Callable[[Dict[str, Any]], float],
        feature_names: Iterable[str],
        missingness_map: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.query_fn = query_fn
        self.feature_names = tuple(feature_names)
        self.missingness_map = dict(missingness_map or {})
        self._cache: Dict[Tuple[Tuple[str, Hashable], ...], float] = {}

    @property
    def cache_size(self) -> int:
        """Number of unique observed evidence configurations cached."""
        return len(self._cache)

    def query_single(self, row: Mapping[str, Any]) -> float:
        """Evaluate posterior probability for a single record with MAR marginalization."""
        observed: Dict[str, Any] = {}
        for col in self.feature_names:
            meas_col = self.missingness_map.get(col)
            if meas_col is not None and not bool(row.get(meas_col, True)):
                continue
            val = row.get(col)
            if val is not None:
                observed[col] = val

        key = tuple(sorted(observed.items()))
        if key not in self._cache:
            self._cache[key] = float(self.query_fn(observed))
        return self._cache[key]

    def predict_proba(self, records: Iterable[Mapping[str, Any]]) -> np.ndarray:
        """Evaluate posterior probabilities across a sequence of records."""
        return np.array([self.query_single(r) for r in records], dtype=np.float64)
