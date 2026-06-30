"""
Adaptation module — TRAP formula: f'(x; g(f(x), θ))

Split-conformal prediction sets layered on top of EDCR.
When EDCR cannot correct (error_flag=True, no unique correction),
the system returns a calibrated prediction set or ABSTAINS.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class AdaptationOutput:
    prediction_sets: list[set[int]]      # (N,) conformal prediction sets
    abstain: np.ndarray                   # (N,) bool — True if |set| > 1
    coverages: dict[str, float]          # nominal and empirical coverage
    single_preds: np.ndarray             # (N,) argmax when not abstaining


class ConformalAdapter:
    """
    Split-conformal prediction sets (RAPS-style) using MAPIE.

    Fits on a calibration split, then wraps any classifier's probabilities.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self._quantile: Optional[float] = None
        self._fitted = False

    def fit(self, proba_cal: np.ndarray, labels_cal: np.ndarray) -> 'ConformalAdapter':
        """
        Compute the conformal quantile from calibration probabilities.

        Args:
            proba_cal:  (N_cal, K) softmax probabilities
            labels_cal: (N_cal,)   true labels
        """
        N = len(labels_cal)
        # Non-conformity score: 1 - p(true_class)
        scores = 1 - proba_cal[np.arange(N), labels_cal]
        # Bonferroni-adjusted quantile
        level = np.ceil((N + 1) * (1 - self.alpha)) / N
        level = min(level, 1.0)
        self._quantile = float(np.quantile(scores, level))
        self._fitted = True
        return self

    def predict(self, proba_test: np.ndarray) -> AdaptationOutput:
        """
        Args:
            proba_test: (N, K) softmax probabilities

        Returns:
            AdaptationOutput with prediction sets and abstention flags
        """
        if not self._fitted:
            raise RuntimeError('Call fit() before predict().')

        q = self._quantile
        pred_sets = []
        for i in range(len(proba_test)):
            # Include class k if 1 - p_k <= q  ↔  p_k >= 1 - q
            s = {k for k, p in enumerate(proba_test[i]) if p >= 1 - q}
            if not s:  # edge case: include argmax
                s = {int(proba_test[i].argmax())}
            pred_sets.append(s)

        abstain = np.array([len(s) > 1 for s in pred_sets])
        single_preds = np.array([
            next(iter(s)) if len(s) == 1 else int(proba_test[i].argmax())
            for i, s in enumerate(pred_sets)
        ])

        return AdaptationOutput(
            prediction_sets=pred_sets,
            abstain=abstain,
            coverages={'nominal': 1 - self.alpha},
            single_preds=single_preds,
        )

    def selective_accuracy(
        self,
        proba_test: np.ndarray,
        labels_test: np.ndarray,
    ) -> dict[str, float]:
        """Report selective accuracy at the fitted coverage level."""
        out = self.predict(proba_test)
        retained = ~out.abstain
        emp_coverage = retained.mean()
        if retained.sum() == 0:
            sel_acc = float('nan')
        else:
            sel_acc = (out.single_preds[retained] == labels_test[retained]).mean()
        return {
            'empirical_coverage': float(emp_coverage),
            'selective_accuracy': float(sel_acc),
            'abstention_rate': float(out.abstain.mean()),
        }
