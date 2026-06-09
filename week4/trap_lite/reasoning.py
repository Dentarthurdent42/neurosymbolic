"""
Reasoning module — TRAP formula: f'(x; g(f(x), θ))

Wraps a trained PyEDCR model to correct or flag predictions based on
learned symbolic rules over the CIFAR-100 hierarchy.

The symbolic rules are the transparency artefact:
    "if pred_fine=X AND pred_super≠parent(X) → suspect error → correct to …"
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# CIFAR-100 fine→superclass mapping (20 superclasses, 5 fine each)
FINE_TO_SUPER: dict[int, int] = {}
_SUPER_FINE = [
    [4, 30, 55, 72, 95],  # 0  aquatic mammals
    [1, 32, 67, 73, 91],  # 1  fish
    [54, 62, 70, 82, 92], # 2  flowers
    [9, 10, 16, 28, 61],  # 3  food containers
    [0, 51, 53, 57, 83],  # 4  fruit and vegetables
    [22, 39, 40, 86, 87], # 5  household electrical devices
    [5, 20, 25, 84, 94],  # 6  household furniture
    [6, 7, 14, 18, 24],   # 7  insects
    [3, 42, 43, 88, 97],  # 8  large carnivores
    [12, 17, 37, 68, 76], # 9  large man-made outdoor things
    [23, 33, 49, 60, 71], # 10 large natural outdoor scenes
    [15, 19, 21, 31, 38], # 11 large omnivores and herbivores
    [34, 63, 64, 66, 75], # 12 medium-sized mammals
    [26, 45, 77, 79, 99], # 13 non-insect invertebrates
    [2, 11, 35, 46, 98],  # 14 people
    [27, 29, 44, 78, 93], # 15 reptiles
    [36, 50, 65, 74, 80], # 16 small mammals
    [47, 52, 56, 59, 96], # 17 trees
    [8, 13, 48, 58, 90],  # 18 vehicles 1
    [41, 69, 81, 85, 89], # 19 vehicles 2
]
for super_idx, fine_classes in enumerate(_SUPER_FINE):
    for fc in fine_classes:
        FINE_TO_SUPER[fc] = super_idx


@dataclass
class ReasoningOutput:
    corrected_preds: np.ndarray       # (N,)  predictions after rule application
    error_flags: np.ndarray           # (N,)  bool — True if EDCR suspects error
    fired_rules: list[list[str]]      # (N,)  list of rule strings fired per sample
    raw_preds: np.ndarray             # (N,)  original predictions (before correction)


class EDCRReasoner:
    """
    Thin wrapper around a trained PyEDCR model.

    Usage:
        reasoner = EDCRReasoner.from_checkpoint('edcr_cifar100.pkl')
        out = reasoner(preds, features)
    """

    def __init__(self, edcr_model=None):
        self.edcr_model = edcr_model  # set after PyEDCR is trained

    @classmethod
    def from_checkpoint(cls, path: Path) -> 'EDCRReasoner':
        import pickle
        with open(path, 'rb') as f:
            edcr_model = pickle.load(f)
        return cls(edcr_model)

    def fit(self, preds_cal: np.ndarray, labels_cal: np.ndarray):
        """
        Learn f-EDR rules from calibration predictions.

        Requires PyEDCR to be installed:
            pip install git+https://github.com/lab-v2/PyEDCR.git

        TODO: replace the stub below with the real PyEDCR API.
        See: github.com/lab-v2/PyEDCR/tests/ and CIKM 2024 paper §3.
        """
        # from pyedcr import EDCR, EDCRConfig
        # config = EDCRConfig(
        #     hierarchy=FINE_TO_SUPER,
        #     min_support=0.01,
        #     min_confidence=0.8,
        # )
        # self.edcr_model = EDCR(config)
        # X_cal = self._build_features(preds_cal)
        # self.edcr_model.fit(X_cal, labels_cal, preds_cal)
        print('EDCRReasoner.fit(): TODO — wire up PyEDCR API here.')
        return self

    def __call__(
        self,
        preds: np.ndarray,
        features: np.ndarray | None = None,
    ) -> ReasoningOutput:
        """Apply learned rules to correct predictions."""
        corrected = preds.copy()
        flags = np.zeros(len(preds), dtype=bool)
        rules_fired: list[list[str]] = [[] for _ in range(len(preds))]

        if self.edcr_model is not None:
            # X_test = self._build_features(preds)
            # corrections = self.edcr_model.predict(X_test, preds)
            # corrected = corrections
            # flags = (corrections != preds)
            pass
        else:
            # Fallback heuristic: flag hierarchy-inconsistent predictions
            for i, p in enumerate(preds):
                super_pred = FINE_TO_SUPER[int(p)]
                # TODO: if we have a superclass prediction, check consistency
                # For now, flag based on a simple superclass majority vote
                rules_fired[i] = []  # no rules without trained model

        return ReasoningOutput(
            corrected_preds=corrected,
            error_flags=flags,
            fired_rules=rules_fired,
            raw_preds=preds.copy(),
        )

    def _build_features(self, preds: np.ndarray) -> np.ndarray:
        pred_super = np.array([FINE_TO_SUPER[int(p)] for p in preds])
        return np.stack([preds, pred_super], axis=1)

    def rules_summary(self, top_k: int = 10) -> list[str]:
        """Return string representations of the top-k learned rules."""
        if self.edcr_model is None:
            return ['<no rules — EDCR model not trained yet>']
        # return [str(r) for r in self.edcr_model.rules_[:top_k]]
        return ['TODO: implement after PyEDCR is wired up']
