"""
Calibrateur de Platt appliqué à la MARGE brute de XGBoost (log-odds),
et non à predict_proba.

Pourquoi : appliquer une sigmoïde à une probabilité déjà bornée écrase le bas
de l'échelle (90 % des transactions collées au même plancher). Sur la marge,
la résolution est conservée.

IMPORTANT : ce module doit être importable par train.py ET par API_Score.py
(mettre calibration.py dans le même dossier), sinon joblib ne pourra pas
recharger l'artefact.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class MarginPlattCalibrator:
    def __init__(self, xgb_model, C: float = 1e6):
        self.xgb_model = xgb_model
        self.C = C
        self.classes_ = np.array([0, 1])

    def _margin(self, X) -> np.ndarray:
        m = self.xgb_model.predict(X, output_margin=True)
        return np.clip(np.asarray(m, dtype=float), -30.0, 30.0).reshape(-1, 1)

    def fit(self, X, y):
        self.lr_ = LogisticRegression(C=self.C, solver="lbfgs", max_iter=1000)
        self.lr_.fit(self._margin(X), np.asarray(y).astype(int))
        return self

    def predict_proba(self, X) -> np.ndarray:
        p = self.lr_.predict_proba(self._margin(X))[:, 1]
        return np.column_stack([1.0 - p, p])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)