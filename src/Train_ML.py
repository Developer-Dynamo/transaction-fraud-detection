"""
FraudGuard ML — Entraînement du modèle (XGBoost) — v5 (final)
==========================================================================
Lancement :
    python train.py --data data/fraudTrain_clean.csv

Prérequis : calibration.py dans le même dossier (et importable par
API_Score.py).

CHANGEMENTS v5 (par rapport à v4)
  1. CALIBRATION : Platt sur la marge brute XGBoost (calibration.py) au lieu
     d'une sigmoïde sur predict_proba -> fin de l'écrasement des probabilités
     au plancher (~90 % des transactions).
  2. scale_pos_weight = (n_neg/n_pos) ** --spw-power (défaut 0.5, racine).
  3. t_low ancré par défaut sur le quantile 99.5 % des probas LÉGIT
     (0.95 donnait 47 revues / 1000 tx pour très peu de fraudes en plus).
  4. SCORE_P_FLOOR dynamique (score_calibration["p_floor"]) : calé sur le
     minimum réel des probas calibrées. L'API doit lire cette clé.
  5. Grille de seuils géométrique (résolution sous 0.01).
  6. IsolationForest évalué (ROC-AUC sur le test) : à vous de décider
     s'il entre dans la décision.
  7. Rapport reports/missed_in_allow.csv : fraudes tombées en ALLOW, avec le
     rang de leur proba brute parmi les légitimes (calibration ou vraie
     invisibilité ?).
  8. Affichage des quantiles en :.1% ; messages de parité sans bruit quand
     la colonne n'existe pas.
==========================================================================
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier

from calibration import MarginPlattCalibrator

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = ROOT / "data" / "fraudTrain_clean.csv"
MODEL_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODEL_PATH = MODEL_DIR / "fraud_xgb_model.joblib"

RANDOM_STATE = 42

# ----------------------------------------------------------------------
# Coûts métier et tolérances
# ----------------------------------------------------------------------
DEFAULT_COST_FALSE_NEGATIVE = 200.0
DEFAULT_COST_FALSE_POSITIVE = 10.0
DEFAULT_COST_TOLERANCE = 0.05

DEFAULT_IMPORTANCE_THRESHOLD = 0.95
DEFAULT_MIN_FEATURES = 8
DEFAULT_MAX_FEATURES = 20

HISTORY_DEPTH_FEATURES = ["month", "tx_rank", "time_since_first_tx", "is_first_tx"]
WEAK_OR_LEAKY_FEATURES = ["merchant_freq", "category_share_card"]
SERVING_UNRELIABLE_FEATURES = [
    "dist_from_prev_merch_km", "speed_kmh",
    "card_mean_dist_prev_km", "dist_to_card_mean",
    "nb_prev_same_merchant", "is_new_merchant",
]
FORCED_PERSONAL_FEATURES = [
    "amt_zscore_card", "amt_to_card_mean", "category_share_card", "hour_share_card",
]


def _env_score(name: str, default: float) -> float:
    try:
        return min(max(float(os.getenv(name, str(default))), 0.0), 100.0)
    except (TypeError, ValueError):
        return default


ALLOW_MAX_SCORE = _env_score("FRAUD_ALLOW_MAX_SCORE", 20.0)
BLOCK_MIN_SCORE = _env_score("FRAUD_BLOCK_MIN_SCORE", 80.0)
if not (0.0 < ALLOW_MAX_SCORE < BLOCK_MIN_SCORE < 100.0):
    ALLOW_MAX_SCORE, BLOCK_MIN_SCORE = 20.0, 80.0

BAND_TARGET_RECALL = 0.98
BAND_TARGET_PRECISION = 0.95
BAND_MAX_REVIEW_SHARE = 0.05
SCORE_P_FLOOR = 1e-6  # valeur par défaut ; remplacée par bands["p_floor"]

ARTIFACT_REQUIRED_KEYS = [
    "xgb_model", "xgb_model_raw_uncalibrated", "threshold", "threshold_f1_reference",
    "columns", "all_candidate_columns", "category_column", "category_map",
    "serving_priors", "score_calibration", "feature_stats", "feature_importance_share",
    "model_version",
]


def false_negative_cost_by_amount(
    amt, base_cost: float = DEFAULT_COST_FALSE_NEGATIVE,
    amt_multiplier: float = 0.5, min_cost: float = 50.0, max_cost: float = 2000.0,
) -> np.ndarray:
    """cost(amt) = clip(base_cost + amt_multiplier * amt, min_cost, max_cost)."""
    amt = np.asarray(amt, dtype=float)
    return np.clip(base_cost + amt_multiplier * amt, min_cost, max_cost)


# ----------------------------------------------------------------------
# Colonnes
# ----------------------------------------------------------------------
CANDIDATE_NUMERIC_FEATURES = [
    "amt", "amt_zscore_card", "amt_to_card_mean", "amt_to_card_max",
    "is_new_max_amt", "amt_vs_category_z", "is_new_category", "hour",
    "time_since_last_tx", "nb_tx_1h", "sum_amt_1h", "nb_tx_24h", "sum_amt_24h",
    "amt_share_of_24h", "tx_accel_1h_vs_24h", "hour_share_card", "age",
    "dist_home_merch_km",
]
CATEGORY_COLUMN = "category_code"
ALL_CANDIDATE_COLUMNS = CANDIDATE_NUMERIC_FEATURES + [CATEGORY_COLUMN]
REQUIRED_RAW_COLUMNS = set(CANDIDATE_NUMERIC_FEATURES) | {
    "cc_num", "unix_time", "category", "is_fraud",
}


# ----------------------------------------------------------------------
# Chargement
# ----------------------------------------------------------------------
def load_dataset(data_path: Path, required: set | None = None) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {data_path}")
    df = pd.read_csv(data_path)
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
    missing = (REQUIRED_RAW_COLUMNS if required is None else required) - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {data_path.name} : {sorted(missing)}")
    print(f"[train] {len(df):,} transactions chargées, "
          f"{df['is_fraud'].sum():,} fraudes ({df['is_fraud'].mean() * 100:.3f} %)")
    return df


def build_category_mapping(df: pd.DataFrame) -> dict:
    categories = sorted(df["category"].dropna().unique().tolist())
    return {cat: code for code, cat in enumerate(categories)}


def build_features(df: pd.DataFrame, category_map: dict, columns: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in [c for c in columns if c != CATEGORY_COLUMN]:
        out[col] = df[col].astype(float).replace([np.inf, -np.inf], np.nan)
    if CATEGORY_COLUMN in columns:
        out[CATEGORY_COLUMN] = df["category"].map(category_map).astype(float)
    return out[columns]


# ----------------------------------------------------------------------
# serving_priors (parité train -> serving)
# ----------------------------------------------------------------------
def build_serving_priors(df: pd.DataFrame, category_map: dict) -> dict:
    priors: dict = {}

    cat_stats = df.groupby("category")["amt"].agg(["mean", "std"])
    priors["category_amt_stats"] = {
        cat: {"mean": float(r["mean"]), "std": float(r["std"]) if pd.notna(r["std"]) else 0.0}
        for cat, r in cat_stats.iterrows()
    }
    mean_s = df["category"].map(lambda c: priors["category_amt_stats"].get(c, {}).get("mean"))
    std_s = df["category"].map(lambda c: priors["category_amt_stats"].get(c, {}).get("std"))
    z_corr = ratio_corr = float("nan")
    if "amt_vs_category_z" in df.columns:
        z_corr = float(((df["amt"] - mean_s) / std_s.replace(0, np.nan)).corr(df["amt_vs_category_z"]))
    if "amt_to_category_mean" in df.columns:
        ratio_corr = float((df["amt"] / mean_s.replace(0, np.nan)).corr(df["amt_to_category_mean"]))
    priors["category_stats_parity"] = {"z_corr": z_corr, "ratio_corr": ratio_corr}
    msg = f"[priors] Parité stats catégorie : z_corr={z_corr:.4f}"
    if not np.isnan(ratio_corr):
        msg += f", ratio_corr={ratio_corr:.4f}"
    print(msg + " (l'API n'active ces priors qu'au-dessus de 0.98)")

    if "merchant" in df.columns:
        freq = df["merchant"].value_counts() / len(df)
        priors["merchant_freq_map"] = {str(k): float(v) for k, v in freq.items()}
        priors["merchant_freq_default"] = float(freq.median())
        print(f"[priors] merchant_freq_map : {len(priors['merchant_freq_map']):,} marchand(s), "
              f"défaut = médiane = {priors['merchant_freq_default']:.6f}")

    d = df.sort_values(["cc_num", "unix_time"]).copy()
    raw_delta_s = d.groupby("cc_num")["unix_time"].diff()
    is_first = d["is_first_tx"] == 1 if "is_first_tx" in d.columns else raw_delta_s.isna()

    unit_last = 60.0
    if "time_since_last_tx" in d.columns:
        valid = (~is_first) & raw_delta_s.notna() & (d["time_since_last_tx"] > 0)
        if valid.sum() >= 100:
            unit_last = float((raw_delta_s[valid] / d.loc[valid, "time_since_last_tx"]).median())
    unit_first = 3600.0
    detected_first = False
    if "time_since_first_tx" in d.columns:
        first_seen = d.groupby("cc_num")["unix_time"].transform("min")
        valid_f = (~is_first) & (d["time_since_first_tx"] > 0)
        if valid_f.sum() >= 100:
            unit_first = float(((d["unix_time"] - first_seen)[valid_f] / d.loc[valid_f, "time_since_first_tx"]).median())
            detected_first = True
    priors["time_unit_seconds"] = {"time_since_last_tx": unit_last, "time_since_first_tx": unit_first}
    print(f"[priors] Unité time_since_last_tx = {unit_last:.4g} s/unité (détectée)"
          + (f", time_since_first_tx = {unit_first:.4g} s/unité" if detected_first
             else " ; time_since_first_tx : colonne absente, défaut non utilisé"))

    if "is_first_tx" in df.columns:
        first_rows = df[df["is_first_tx"] == 1]
        candidate_defaults = [
            "time_since_last_tx", "log_time_since_last_tx", "time_since_first_tx",
            "dist_from_prev_merch_km", "speed_kmh",
            "card_mean_amt_prev", "card_std_amt_prev", "card_max_amt_prev",
            "amt_zscore_card", "amt_to_card_mean", "amt_to_card_max", "is_new_max_amt",
            "card_mean_dist_prev_km", "dist_to_card_mean",
            "nb_prev_same_merchant", "is_new_merchant", "is_new_category",
            "category_share_card", "hour_share_card",
        ]
        first_defaults = {}
        for col in candidate_defaults:
            if col in first_rows.columns and len(first_rows) > 0:
                val = first_rows[col].median()
                if pd.notna(val):
                    first_defaults[col] = float(val)
        priors["first_tx_defaults"] = first_defaults
        print(f"[priors] first_tx_defaults appris sur {len(first_rows):,} transaction(s).")
    return priors


# ----------------------------------------------------------------------
# Sélection des features
# ----------------------------------------------------------------------
def select_important_features(
    xgb_model, columns, importance_threshold=DEFAULT_IMPORTANCE_THRESHOLD,
    min_features=DEFAULT_MIN_FEATURES, max_features=DEFAULT_MAX_FEATURES,
    n_features_override=None, forced=None,
):
    gain_scores = xgb_model.get_booster().get_score(importance_type="gain")
    importance_df = pd.DataFrame(
        {"feature": columns, "gain": [gain_scores.get(c, 0.0) for c in columns]}
    ).sort_values("gain", ascending=False).reset_index(drop=True)
    total_gain = importance_df["gain"].sum()
    importance_df["gain_share"] = importance_df["gain"] / total_gain if total_gain > 0 else 0.0
    importance_df["cumulative_gain_share"] = importance_df["gain_share"].cumsum()

    if n_features_override is not None:
        n_keep = max(1, min(n_features_override, len(columns)))
    else:
        n_keep = int((importance_df["cumulative_gain_share"] < importance_threshold).sum()) + 1
        n_keep = max(min_features, min(n_keep, max_features, len(columns)))
    selected = importance_df.head(n_keep)["feature"].tolist()

    forced_added = []
    for name in (forced or []):
        if name in columns and name not in selected:
            selected.append(name)
            forced_added.append(name)
    return selected, importance_df, forced_added


def build_behavior_report(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    fraud = df.loc[df["is_fraud"] == 1]
    normal = df.loc[df["is_fraud"] == 0]
    rows = []
    for col in features:
        if col == CATEGORY_COLUMN:
            continue
        f_vals = pd.to_numeric(fraud[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        n_vals = pd.to_numeric(normal[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        f_mean, n_mean = f_vals.mean(), n_vals.mean()
        f_std, n_std = f_vals.std(ddof=0), n_vals.std(ddof=0)
        pooled = np.sqrt((f_std ** 2 + n_std ** 2) / 2.0)
        cohen_d = (f_mean - n_mean) / pooled if pooled > 0 else 0.0
        rel = (f_mean - n_mean) / abs(n_mean) * 100.0 if pd.notna(n_mean) and n_mean != 0 else np.nan
        rows.append({
            "feature": col,
            "mean_fraud": round(float(f_mean), 4) if pd.notna(f_mean) else None,
            "mean_normal": round(float(n_mean), 4) if pd.notna(n_mean) else None,
            "median_fraud": round(float(f_vals.median()), 4) if pd.notna(f_vals.median()) else None,
            "median_normal": round(float(n_vals.median()), 4) if pd.notna(n_vals.median()) else None,
            "relative_diff_pct": round(float(rel), 1) if pd.notna(rel) else None,
            "cohens_d": round(float(cohen_d), 3),
        })
    return pd.DataFrame(rows).sort_values(
        "cohens_d", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------
# Seuil de décision par coût métier
# ----------------------------------------------------------------------
def default_threshold_grid() -> np.ndarray:
    """Grille géométrique sous 0.01 (probabilités très écrasées), linéaire au-dessus."""
    fine = np.geomspace(1e-5, 0.01, 200)
    coarse = np.round(np.arange(0.01, 1.00, 0.01), 2)
    return np.unique(np.concatenate([fine, coarse]))


def compute_threshold_sweep(
    y_true, scores, amt, thresholds=None,
    cost_false_negative=DEFAULT_COST_FALSE_NEGATIVE,
    cost_false_positive=DEFAULT_COST_FALSE_POSITIVE,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = default_threshold_grid()
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    amt = np.asarray(amt, dtype=float)
    cost_fn_arr = np.broadcast_to(np.asarray(cost_false_negative, dtype=float), y_true.shape)
    cost_fp_arr = np.broadcast_to(np.asarray(cost_false_positive, dtype=float), y_true.shape)

    rows = []
    for t in thresholds:
        y_pred = (scores >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        fn_mask = (y_true == 1) & (y_pred == 0)
        fp_mask = (y_true == 0) & (y_pred == 1)
        rows.append({
            "threshold": float(t),
            "n_flagged": int(tp + fp),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "amount_flagged": round(float(amt[y_pred == 1].sum()), 2),
            "fraud_amount_caught": round(float(amt[(y_true == 1) & (y_pred == 1)].sum()), 2),
            "estimated_cost": round(float(cost_fn_arr[fn_mask].sum() + cost_fp_arr[fp_mask].sum()), 2),
        })
    return pd.DataFrame(rows)


def best_f1_threshold(y_true, scores) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    f1_scores = np.divide(2 * precisions * recalls, precisions + recalls,
                          out=np.zeros_like(precisions), where=(precisions + recalls) != 0)
    if len(thresholds) == 0:
        return 0.5
    return float(thresholds[int(np.argmax(f1_scores[:-1]))])


def evaluate(y_true, scores, threshold, amt=None):
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    result = {
        "pr_auc": round(float(average_precision_score(y_true, scores)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, scores)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "threshold": round(float(threshold), 6),
        "n_test": int(len(y_true)),
        "confusion_matrix": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
    }
    if amt is not None:
        result["fraud_amount_caught"] = round(
            float(amt[(np.asarray(y_true) == 1) & (y_pred == 1)].sum()), 2)
    return result, y_pred


# ----------------------------------------------------------------------
# Qualité des features
# ----------------------------------------------------------------------
def population_stability_index(expected, actual, bins: int = 10) -> float:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) < 50 or len(actual) < 50:
        return float("nan")
    edges = np.unique(np.quantile(expected, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < bins // 2:
        rest = expected[expected > edges[0]]
        if len(rest) >= 50:
            edges = np.unique(np.concatenate([edges, np.quantile(rest, np.linspace(0.0, 1.0, bins))]))
    if len(edges) < 3:
        vals = np.unique(expected)
        if len(vals) > 50:
            return float("nan")
        e = np.array([(expected == v).mean() for v in vals])
        a = np.array([(actual == v).mean() for v in vals])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        e = np.histogram(expected, bins=edges)[0] / len(expected)
        a = np.histogram(actual, bins=edges)[0] / len(actual)
    e = np.clip(e, 1e-4, None)
    a = np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def build_feature_quality_report(df_train, df_test, category_map, columns,
                                 importance_df, selected, forced) -> pd.DataFrame:
    X_tr = build_features(df_train, category_map, columns)
    X_te = build_features(df_test, category_map, columns)
    y_tr = df_train["is_fraud"].astype(int).to_numpy()
    gain_share = importance_df.set_index("feature")["gain_share"]

    rows = []
    for col in columns:
        s_tr = X_tr[col]
        n_unique = int(s_tr.nunique(dropna=True))
        miss_pct = float(s_tr.isna().mean() * 100.0)
        zero_pct = float((s_tr == 0).mean() * 100.0)
        auc = 0.5
        if n_unique > 1 and y_tr.min() != y_tr.max():
            med = s_tr.median()
            auc = float(roc_auc_score(y_tr, s_tr.fillna(med if pd.notna(med) else 0.0)))
        sep_auc = max(auc, 1.0 - auc)
        psi = population_stability_index(s_tr.to_numpy(), X_te[col].to_numpy())
        share = float(gain_share.get(col, 0.0))
        if n_unique <= 1:
            verdict = "constante : inutile"
        elif pd.notna(psi) and psi >= 0.25:
            verdict = "instable dans le temps (PSI >= 0.25)"
        elif miss_pct > 30:
            verdict = "trop de valeurs manquantes"
        elif sep_auc < 0.55 and share < 0.005:
            verdict = "peu informative"
        elif pd.notna(psi) and psi >= 0.10:
            verdict = "dérive modérée (PSI >= 0.10)"
        else:
            verdict = "OK"
        rows.append({
            "feature": col, "selected": col in selected, "forced": col in forced,
            "gain_share": round(share, 5), "univariate_auc": round(sep_auc, 4),
            "psi_train_vs_test": round(psi, 4) if pd.notna(psi) else None,
            "missing_pct": round(miss_pct, 2), "zero_pct": round(zero_pct, 2),
            "n_unique": n_unique, "verdict": verdict,
        })
    return pd.DataFrame(rows).sort_values("gain_share", ascending=False).reset_index(drop=True)


def audit_precomputed_features(df: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    if "merchant" in df.columns and "merchant_freq" in df.columns:
        n_vals = df.groupby("merchant")["merchant_freq"].nunique()
        if len(n_vals) and float(n_vals.median()) <= 1.0:
            warnings.append("merchant_freq est constante par marchand : calculée sur TOUT "
                            "le fichier (test compris). Fuite légère.")
    if "is_first_tx" in df.columns and "cc_num" in df.columns:
        bad = float((df.groupby("cc_num")["is_first_tx"].sum() != 1).mean())
        if bad > 0.02:
            warnings.append(f"is_first_tx : {bad:.1%} des cartes n'ont pas exactement "
                            "une première transaction.")
    for w in warnings:
        print(f"[audit] ATTENTION : {w}")
    if not warnings:
        print("[audit] Colonnes pré-calculées : aucun signal de fuite détecté.")
    return warnings


def build_feature_stats(X: pd.DataFrame) -> dict:
    stats: dict = {}
    qs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    for col in X.columns:
        s = X[col].replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            continue
        q = s.quantile(qs).to_numpy()
        stats[col] = {
            "min": float(s.min()), "p01": float(q[0]), "p05": float(q[1]),
            "p25": float(q[2]), "p50": float(q[3]), "p75": float(q[4]),
            "p95": float(q[5]), "p99": float(q[6]), "max": float(s.max()),
            "n_unique": int(min(s.nunique(), 1000)),
            "nan_rate": float(X[col].isna().mean()),
        }
    return stats


# ----------------------------------------------------------------------
# Score de risque 0-100 (échelle log) et plages de décision
# ----------------------------------------------------------------------
def risk_score_from_proba(proba, t_low, t_high, allow_max=ALLOW_MAX_SCORE,
                          block_min=BLOCK_MIN_SCORE, p_floor=SCORE_P_FLOOR):
    """p_floor -> 0, t_low -> allow_max, t_high -> block_min, 1 -> 100 (échelle log).
    L'API doit utiliser la MÊME formule et lire score_calibration["p_floor"]."""
    p = np.clip(np.asarray(proba, dtype=float), p_floor, 1.0)
    xp = np.log([p_floor, t_low, t_high, 1.0])
    return np.interp(np.log(p), xp, [0.0, allow_max, block_min, 100.0])


def compute_score_bands(sweep, f1_reference, cost_threshold,
                        target_recall=BAND_TARGET_RECALL,
                        target_precision=BAND_TARGET_PRECISION,
                        max_review_share=BAND_MAX_REVIEW_SHARE,
                        t_low_quantile=None, legit_proba=None) -> dict:
    s = sweep.sort_values("threshold").reset_index(drop=True)
    n_total = int(s.loc[0, ["tp", "fp", "fn", "tn"]].sum())
    use_quantile = t_low_quantile is not None and legit_proba is not None and len(legit_proba) > 0

    def flagged(t: float) -> int:
        return int(s.loc[(s["threshold"] - t).abs().idxmin(), "n_flagged"])

    def pick(recall_goal: float):
        if use_quantile:
            t_low = float(np.quantile(legit_proba, t_low_quantile))
            src_low = f"quantile {t_low_quantile:.1%} des probas LÉGIT (validation)"
        else:
            ok_recall = s[s["recall"] >= recall_goal]
            if len(ok_recall):
                t_low, src_low = float(ok_recall["threshold"].max()), f"rappel >= {recall_goal:.0%}"
            else:
                t_low, src_low = float(cost_threshold), "repli : seuil de coût métier"
        t_low = float(min(max(t_low, 1e-7), 0.98))
        ok_prec = s[(s["precision"] >= target_precision) & (s["threshold"] > t_low)]
        if len(ok_prec):
            t_high, src_high = float(ok_prec["threshold"].min()), f"précision >= {target_precision:.0%}"
        else:
            t_high, src_high = float(f1_reference), "repli : seuil du meilleur F1"
        if not (t_high > t_low):
            t_high, src_high = float(min(0.99, t_low + 0.5 * (1.0 - t_low))), "repli : milieu de [t_low, 1]"
        t_high = float(min(t_high, 0.999))
        share = max(flagged(t_low) - flagged(t_high), 0) / max(n_total, 1)
        return t_low, t_high, src_low, src_high, share

    if use_quantile:
        t_low, t_high, src_low, src_high, share = pick(target_recall)
        goal = target_recall
    else:
        goals = sorted({g for g in [target_recall, 0.97, 0.96, 0.95, 0.93, 0.90, 0.85]
                        if g <= target_recall}, reverse=True)
        for goal in goals:
            t_low, t_high, src_low, src_high, share = pick(goal)
            if share <= max_review_share:
                break
    if share > max_review_share:
        print(f"[bands] ATTENTION : revue manuelle = {share:.1%} des transactions de "
              f"validation (objectif {max_review_share:.0%}).")

    return {
        "t_low": t_low, "t_high": t_high, "source_low": src_low, "source_high": src_high,
        "recall_used": float(goal), "review_share_validation": round(float(share), 5),
        "allow_max_score": ALLOW_MAX_SCORE, "block_min_score": BLOCK_MIN_SCORE,
        "target_recall": target_recall, "target_precision": target_precision,
        "max_review_share": max_review_share,
        "t_low_quantile": t_low_quantile if use_quantile else None,
        "scale": "log", "p_floor": SCORE_P_FLOOR,
    }


def _scores(proba, bands: dict) -> np.ndarray:
    return np.round(risk_score_from_proba(
        proba, bands["t_low"], bands["t_high"],
        p_floor=bands.get("p_floor", SCORE_P_FLOOR)), 2)


def evaluate_bands(y_true, proba, amt, bands: dict) -> dict:
    y = np.asarray(y_true).astype(int)
    amt = np.asarray(amt, dtype=float)
    score = _scores(proba, bands)
    allow = score < ALLOW_MAX_SCORE
    block = score >= BLOCK_MIN_SCORE
    review = ~allow & ~block
    n_fraud = int(y.sum())
    n = lambda m: int(m.sum())  # noqa: E731
    fa, fr, fb = n(allow & (y == 1)), n(review & (y == 1)), n(block & (y == 1))
    return {
        "n": int(len(y)), "n_fraud": n_fraud,
        "share_allow_pct": round(float(allow.mean() * 100), 3),
        "share_review_pct": round(float(review.mean() * 100), 3),
        "share_block_pct": round(float(block.mean() * 100), 3),
        "fraud_in_allow": fa, "fraud_in_review": fr, "fraud_in_block": fb,
        "recall_block_pct": round(100.0 * fb / max(n_fraud, 1), 2),
        "recall_block_or_review_pct": round(100.0 * (fb + fr) / max(n_fraud, 1), 2),
        "precision_block_pct": round(100.0 * fb / max(n(block), 1), 2),
        "precision_review_pct": round(100.0 * fr / max(n(review), 1), 2),
        "review_per_1000_tx": round(float(review.mean() * 1000), 2),
        "fraud_amount_missed_in_allow": round(float(amt[allow & (y == 1)].sum()), 2),
    }


def metrics_by_period(y_true, proba, amt, bands: dict, n_chunks: int = 4) -> pd.DataFrame:
    y = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    amt = np.asarray(amt, dtype=float)
    rows = []
    for i, idx in enumerate(np.array_split(np.arange(len(y)), n_chunks), start=1):
        if len(idx) == 0:
            continue
        yy, pp, aa = y[idx], proba[idx], amt[idx]
        row = {"period": i, "n": int(len(idx)), "n_fraud": int(yy.sum())}
        if yy.min() != yy.max():
            row["pr_auc"] = round(float(average_precision_score(yy, pp)), 4)
            row["roc_auc"] = round(float(roc_auc_score(yy, pp)), 4)
        else:
            row["pr_auc"] = row["roc_auc"] = None
        b = evaluate_bands(yy, pp, aa, bands)
        row.update({
            "fraud_in_allow": b["fraud_in_allow"],
            "recall_block_or_review_pct": b["recall_block_or_review_pct"],
            "precision_block_pct": b["precision_block_pct"],
            "review_per_1000_tx": b["review_per_1000_tx"],
        })
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Vérification de l'artefact
# ----------------------------------------------------------------------
def verify_artifact(path: Path) -> None:
    print(f"[verify] Rechargement de {path} pour vérification...")
    art = joblib.load(path)
    if not isinstance(art, dict):
        raise RuntimeError(f"L'artefact {path} n'est pas un dict (type={type(art)})")
    missing = [k for k in ARTIFACT_REQUIRED_KEYS if k not in art]
    if missing:
        raise RuntimeError(f"Artefact incomplet : clés manquantes {missing}")
    cat_map = art.get("category_map") or {}
    if not isinstance(cat_map, dict) or not cat_map:
        raise RuntimeError("category_map vide ou non-dict")
    cat_stats = (art.get("serving_priors") or {}).get("category_amt_stats") or {}
    if not cat_stats:
        raise RuntimeError("serving_priors.category_amt_stats vide")
    cal = art.get("score_calibration") or {}
    t_low, t_high, p_floor = cal.get("t_low"), cal.get("t_high"), cal.get("p_floor")
    if not all(isinstance(v, (int, float)) for v in (t_low, t_high, p_floor)):
        raise RuntimeError(f"t_low / t_high / p_floor absents ou non numériques : {cal}")
    if not (0.0 < float(p_floor) < float(t_low) < float(t_high) < 1.0):
        raise RuntimeError(f"Calibration incohérente : p_floor={p_floor}, t_low={t_low}, t_high={t_high}")
    if not hasattr(art["xgb_model"], "predict_proba"):
        raise RuntimeError("xgb_model sans predict_proba")
    print(f"[verify] OK — {len(art)} clés. category_map={len(cat_map)}, "
          f"p_floor={p_floor:.2e}, t_low={t_low:.5f}, t_high={t_high:.4f}, "
          f"iso_model={'oui' if art.get('iso_model') is not None else 'non'}, "
          f"features={len(art.get('columns', []))}.")


# ----------------------------------------------------------------------
# Entraînement
# ----------------------------------------------------------------------
def main(
    data_path: Path,
    cost_false_negative=DEFAULT_COST_FALSE_NEGATIVE,
    cost_false_positive=DEFAULT_COST_FALSE_POSITIVE,
    use_amount_dependent_cost=True,
    cost_fn_amt_multiplier=0.5, cost_fn_min=50.0, cost_fn_max=2000.0,
    importance_threshold=DEFAULT_IMPORTANCE_THRESHOLD,
    min_features=DEFAULT_MIN_FEATURES, max_features=DEFAULT_MAX_FEATURES,
    n_features_override=None, cost_tolerance=DEFAULT_COST_TOLERANCE,
    include_history_depth=False, include_weak=False, include_unreliable=False,
    force_personal=True,
    band_recall=BAND_TARGET_RECALL, band_precision=BAND_TARGET_PRECISION,
    max_review_share=BAND_MAX_REVIEW_SHARE,
    t_low_quantile: float | None = 0.995,
    spw_power: float = 0.5,
) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    excluded_features = [] if include_history_depth else [
        c for c in HISTORY_DEPTH_FEATURES if c in CANDIDATE_NUMERIC_FEATURES]
    if not include_weak:
        excluded_features += [c for c in WEAK_OR_LEAKY_FEATURES if c in CANDIDATE_NUMERIC_FEATURES]
    if not include_unreliable:
        excluded_features += [c for c in SERVING_UNRELIABLE_FEATURES
                              if c in CANDIDATE_NUMERIC_FEATURES and c not in excluded_features]
    candidate_columns = [c for c in CANDIDATE_NUMERIC_FEATURES if c not in excluded_features] + [CATEGORY_COLUMN]
    forced_features = [f for f in (FORCED_PERSONAL_FEATURES if force_personal else [])
                       if f not in excluded_features]
    if excluded_features:
        print(f"[train] Features écartées : {excluded_features}")

    required = set(c for c in candidate_columns if c != CATEGORY_COLUMN) | {
        "cc_num", "unix_time", "category", "is_fraud"}
    df = load_dataset(data_path, required=required)
    data_audit = audit_precomputed_features(df)

    # --- Split chronologique ---------------------------------------------
    df = df.sort_values("unix_time").reset_index(drop=True)
    n = len(df)
    train_end, val_calib_end, val_end = int(n * 0.70), int(n * 0.775), int(n * 0.85)
    df_train = df.iloc[:train_end]
    df_val_calib = df.iloc[train_end:val_calib_end]
    df_val_thresh = df.iloc[val_calib_end:val_end]
    df_test = df.iloc[val_end:]
    print(f"[train] split chronologique : train={len(df_train):,}, val_calib={len(df_val_calib):,}, "
          f"val_thresh={len(df_val_thresh):,}, test={len(df_test):,}")

    category_map = build_category_mapping(df_train)
    print(f"[category_map] {len(category_map)} catégories : {category_map}")
    serving_priors = build_serving_priors(df_train, category_map)

    y_train = df_train["is_fraud"].astype(int).to_numpy()
    y_val_calib = df_val_calib["is_fraud"].astype(int).to_numpy()
    y_val_thresh = df_val_thresh["is_fraud"].astype(int).to_numpy()
    y_test = df_test["is_fraud"].astype(int).to_numpy()
    amt_test = df_test["amt"].to_numpy()

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = (n_neg / max(n_pos, 1)) ** spw_power
    print(f"[train] scale_pos_weight = {scale_pos_weight:.1f} "
          f"(ratio brut {n_neg / max(n_pos, 1):.1f} ^ {spw_power:g} ; {n_neg:,} légit / {n_pos:,} fraude)")

    xgb_hyperparams = dict(n_estimators=400, max_depth=6, learning_rate=0.05,
                           subsample=0.9, colsample_bytree=0.9)

    # --- Étape 1 : éclaireur ---------------------------------------------
    print(f"[train] Étape 1/2 : éclaireur sur {len(candidate_columns)} features candidates...")
    X_train_full = build_features(df_train, category_map, candidate_columns)
    scout_model = XGBClassifier(**xgb_hyperparams, scale_pos_weight=scale_pos_weight,
                                eval_metric="aucpr", random_state=RANDOM_STATE, n_jobs=-1)
    scout_model.fit(X_train_full, y_train)

    selected_features, importance_full_df, forced_added = select_important_features(
        scout_model, candidate_columns, importance_threshold=importance_threshold,
        min_features=min_features, max_features=max_features,
        n_features_override=n_features_override, forced=forced_features)
    importance_full_df.to_csv(REPORTS_DIR / "feature_importance_full.csv", index=False)
    print(f"[train] {len(selected_features)}/{len(candidate_columns)} features retenues "
          f"(gain cumulé >= {importance_threshold:.0%}) :")
    for f in selected_features:
        share = importance_full_df.loc[importance_full_df["feature"] == f, "gain_share"].iloc[0]
        tag = "  (forcée : comportement personnel de la carte)" if f in forced_added else ""
        print(f"    - {f:<28} gain_share={share:.3f}{tag}")

    quality_report = build_feature_quality_report(
        df_train, df_test, category_map, candidate_columns,
        importance_full_df, selected_features, forced_added)
    quality_report.to_csv(REPORTS_DIR / "feature_quality.csv", index=False)
    n_bad = int((quality_report["selected"] & (quality_report["verdict"] != "OK")).sum())
    print(f"[train] Qualité des features -> {REPORTS_DIR / 'feature_quality.csv'} "
          f"({n_bad} feature(s) retenue(s) avec réserve).")

    behavior_report = build_behavior_report(df, selected_features)
    behavior_report.to_csv(REPORTS_DIR / "feature_behavior.csv", index=False)
    print("\n[train] Comportement fraude vs normal (top 10) :")
    print(behavior_report.head(10).to_string(index=False))
    print(f"[train] Rapport complet -> {REPORTS_DIR / 'feature_behavior.csv'}\n")

    # --- Étape 2 : modèle final + calibration Platt sur la marge -----------
    print(f"[train] Étape 2/2 : modèle final sur {len(selected_features)} features...")
    X_train = build_features(df_train, category_map, selected_features)
    X_val_calib = build_features(df_val_calib, category_map, selected_features)
    X_val_thresh = build_features(df_val_thresh, category_map, selected_features)
    X_test = build_features(df_test, category_map, selected_features)

    xgb_model = XGBClassifier(**xgb_hyperparams, scale_pos_weight=scale_pos_weight,
                              eval_metric="aucpr", random_state=RANDOM_STATE, n_jobs=-1)
    xgb_model.fit(X_train, y_train)

    print("[train] Calibration (Platt sur la marge brute, sur val_calib)...")
    calibrated_xgb = MarginPlattCalibrator(xgb_model).fit(X_val_calib, y_val_calib)

    proba_val_thresh = calibrated_xgb.predict_proba(X_val_thresh)[:, 1]
    proba_test = calibrated_xgb.predict_proba(X_test)[:, 1]
    proba_test_uncalibrated = xgb_model.predict_proba(X_test)[:, 1]
    print(f"[train] Probas calibrées (test) : min={proba_test.min():.2e}, "
          f"p50={np.median(proba_test):.2e}, p99={np.quantile(proba_test, 0.99):.2e}, "
          f"max={proba_test.max():.3f}")

    # --- IsolationForest (signal non supervisé, évalué) ---------------------
    print("[train] Entraînement IsolationForest (anomaly_score) sur X_train...")
    iso_model = IsolationForest(n_estimators=200, contamination="auto",
                                random_state=RANDOM_STATE, n_jobs=-1)
    iso_model.fit(X_train.fillna(0.0))
    iso_scores_train = iso_model.score_samples(X_train.fillna(0.0))
    iso_score_quantiles = {
        "p05": float(np.percentile(iso_scores_train, 5)),
        "p50": float(np.percentile(iso_scores_train, 50)),
        "p95": float(np.percentile(iso_scores_train, 95)),
    }
    iso_test_auc = float(roc_auc_score(y_test, -iso_model.score_samples(X_test.fillna(0.0))))
    print(f"[iso] Quantiles score_samples (train) : p05={iso_score_quantiles['p05']:.4f}, "
          f"p50={iso_score_quantiles['p50']:.4f}, p95={iso_score_quantiles['p95']:.4f}")
    print(f"[iso] ROC-AUC seul sur le test = {iso_test_auc:.4f} "
          f"(XGBoost : voir ci-dessous ; l'IF n'entre PAS dans la décision)")

    # --- Seuil de décision (coût métier) ----------------------------------
    cost_fn_val = (false_negative_cost_by_amount(
        df_val_thresh["amt"].to_numpy(), base_cost=cost_false_negative,
        amt_multiplier=cost_fn_amt_multiplier, min_cost=cost_fn_min, max_cost=cost_fn_max,
    ) if use_amount_dependent_cost else cost_false_negative)
    threshold_f1_reference = best_f1_threshold(y_val_thresh, proba_val_thresh)
    sweep_val_df = compute_threshold_sweep(
        y_val_thresh, proba_val_thresh, df_val_thresh["amt"].to_numpy(),
        cost_false_negative=cost_fn_val, cost_false_positive=cost_false_positive)
    min_cost = float(sweep_val_df["estimated_cost"].min())
    min_cost_row = sweep_val_df.loc[sweep_val_df["estimated_cost"].idxmin()]
    near_optimal = sweep_val_df.loc[sweep_val_df["estimated_cost"] <= min_cost * (1.0 + cost_tolerance)]
    best_row = near_optimal.loc[near_optimal["f1_score"].idxmax()]
    threshold = float(best_row["threshold"])
    print(f"[train] Coût minimal (val_thresh) = {min_cost:.1f} au seuil {min_cost_row['threshold']:.5f} "
          f"— seuil RETENU (meilleur F1 parmi coût <= min x {1 + cost_tolerance:.2f}) : "
          f"{threshold:.5f} (F1={best_row['f1_score']:.3f}) ; référence F1 seul : "
          f"{threshold_f1_reference:.5f}.")
    print(f"[train] Seuil théorique de Bayes ≈ C_FP/(C_FP+C_FN) = "
          f"{cost_false_positive / (cost_false_positive + cost_false_negative):.4f} "
          f"(à comparer au seuil retenu : écart faible = bonne calibration).")
    sweep_val_df.to_csv(REPORTS_DIR / "threshold_sweep_validation.csv", index=False)

    legit_proba_val = proba_val_thresh[y_val_thresh == 0]
    use_q = bool(t_low_quantile and t_low_quantile > 0)
    score_bands = compute_score_bands(
        sweep_val_df, threshold_f1_reference, threshold,
        target_recall=band_recall, target_precision=band_precision,
        max_review_share=max_review_share,
        t_low_quantile=t_low_quantile if use_q else None,
        legit_proba=legit_proba_val if use_q else None)

    # p_floor dynamique : sous le minimum réel des probas, et nettement sous t_low
    p_floor = float(max(1e-9, min(proba_val_thresh.min() * 0.5, score_bands["t_low"] / 10.0)))
    score_bands["p_floor"] = p_floor
    print(f"[train] Score 0-100 : ALLOW < {ALLOW_MAX_SCORE:g} <=> proba < "
          f"{score_bands['t_low']:.5f} ({score_bands['source_low']}) ; "
          f"BLOCK >= {BLOCK_MIN_SCORE:g} <=> proba >= {score_bands['t_high']:.4f} "
          f"({score_bands['source_high']}) ; p_floor={p_floor:.2e}.")

    # --- Évaluation sur le TEST -------------------------------------------
    metrics, y_pred = evaluate(y_test, proba_test, threshold, amt_test)
    cost_fn_test = (false_negative_cost_by_amount(
        amt_test, base_cost=cost_false_negative, amt_multiplier=cost_fn_amt_multiplier,
        min_cost=cost_fn_min, max_cost=cost_fn_max,
    ) if use_amount_dependent_cost else cost_false_negative)
    compute_threshold_sweep(
        y_test, proba_test, amt_test,
        cost_false_negative=cost_fn_test, cost_false_positive=cost_false_positive,
    ).to_csv(REPORTS_DIR / "threshold_sweep_test.csv", index=False)

    print(f"[train][TEST]  PR-AUC={metrics['pr_auc']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}  "
          f"F1={metrics['f1_score']:.3f}  precision={metrics['precision']:.3f}  "
          f"recall={metrics['recall']:.3f}  seuil={metrics['threshold']:.5f}")

    val_band_stats = evaluate_bands(y_val_thresh, proba_val_thresh, df_val_thresh["amt"].to_numpy(), score_bands)
    test_band_stats = evaluate_bands(y_test, proba_test, amt_test, score_bands)
    score_bands["validation"] = val_band_stats
    score_bands["test"] = test_band_stats
    print("[train][TEST] Plages ALLOW / MANUAL_REVIEW / BLOCK : "
          f"{test_band_stats['share_allow_pct']:.2f} % / {test_band_stats['share_review_pct']:.2f} % / "
          f"{test_band_stats['share_block_pct']:.2f} % ; "
          f"fraudes en ALLOW = {test_band_stats['fraud_in_allow']}/{test_band_stats['n_fraud']} "
          f"({test_band_stats['fraud_amount_missed_in_allow']:.0f} $ manqués) ; "
          f"rappel BLOCK = {test_band_stats['recall_block_pct']:.1f} % ; "
          f"précision BLOCK = {test_band_stats['precision_block_pct']:.1f} % ; "
          f"revue manuelle = {test_band_stats['review_per_1000_tx']:.2f} pour 1000 tx.")

    # --- Rapport des fraudes manquées (ALLOW) -------------------------------
    score_test = _scores(proba_test, score_bands)
    missed_mask = (y_test == 1) & (score_test < ALLOW_MAX_SCORE)
    if missed_mask.any():
        legit_raw = proba_test_uncalibrated[y_test == 0]
        missed = df_test.loc[missed_mask, ["amt", "category", "hour", "nb_tx_24h",
                                           "sum_amt_24h", "amt_vs_category_z"]].copy()
        missed["proba_calibree"] = proba_test[missed_mask]
        missed["proba_brute"] = proba_test_uncalibrated[missed_mask]
        missed["rang_brut_vs_legit_pct"] = [round(float((legit_raw < p).mean() * 100), 2)
                                            for p in missed["proba_brute"]]
        missed = missed.sort_values("amt")
        missed.to_csv(REPORTS_DIR / "missed_in_allow.csv")
        print(f"[train][TEST] {len(missed)} fraude(s) en ALLOW -> reports/missed_in_allow.csv "
              f"(rang brut médian vs légitimes = {missed['rang_brut_vs_legit_pct'].median():.1f} %)")

    period_df = metrics_by_period(y_test, proba_test, amt_test, score_bands)
    period_df.to_csv(REPORTS_DIR / "metrics_by_period.csv", index=False)
    print("[train][TEST] Stabilité dans le temps (4 tranches chronologiques) :")
    print(period_df.to_string(index=False))

    all_metrics = {
        "test": metrics, "score_bands": score_bands,
        "metrics_by_period": period_df.to_dict(orient="records"),
        "selected_features": selected_features, "forced_features": forced_added,
        "excluded_features": excluded_features, "data_audit": data_audit,
        "n_candidate_features": len(candidate_columns),
        "iso_score_quantiles": iso_score_quantiles, "iso_test_roc_auc": iso_test_auc,
        "calibration_method": "platt_on_margin", "spw_power": spw_power,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # --- Sauvegarde de l'artefact -----------------------------------------
    model_version = f"xgb-fraudTrain_features-{datetime.now().strftime('%Y%m%d-%H%M')}"
    feature_stats = build_feature_stats(X_train)
    gain_final = xgb_model.get_booster().get_score(importance_type="gain")
    total_gain_final = sum(gain_final.get(c, 0.0) for c in selected_features) or 1.0
    feature_importance_share = {c: float(gain_final.get(c, 0.0) / total_gain_final)
                                for c in selected_features}

    artifact = {
        "xgb_model": calibrated_xgb,
        "xgb_model_raw_uncalibrated": xgb_model,
        "calibration_method": "platt_on_margin",
        "threshold": threshold,
        "threshold_f1_reference": threshold_f1_reference,
        "cost_false_negative": cost_false_negative,
        "cost_false_positive": cost_false_positive,
        "columns": selected_features,
        "all_candidate_columns": candidate_columns,
        "category_column": CATEGORY_COLUMN,
        "category_map": category_map,
        "serving_priors": serving_priors,
        "score_calibration": score_bands,
        "feature_stats": feature_stats,
        "feature_importance_share": feature_importance_share,
        "iso_model": iso_model,
        "iso_score_quantiles": iso_score_quantiles,
        "forced_features": forced_added,
        "metrics": all_metrics,
        "model_version": model_version,
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"[train] Modèle sauvegardé -> {MODEL_PATH} ({model_version})")
    verify_artifact(MODEL_PATH)

    # --- Rapports graphiques ------------------------------------------------
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
    plt.title("Matrice de confusion — XGBoost (test)")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=140)
    plt.close()

    plt.figure()
    fpr, tpr, _ = roc_curve(y_test, proba_test)
    plt.plot(fpr, tpr, label=f"XGBoost (AUC = {roc_auc_score(y_test, proba_test):.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Taux de faux positifs")
    plt.ylabel("Taux de vrais positifs")
    plt.title("Courbe ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "roc_curve.png", dpi=140)
    plt.close()

    importances = pd.Series(xgb_model.feature_importances_, index=selected_features).sort_values()
    plt.figure(figsize=(7, max(4, 0.35 * len(selected_features))))
    importances.plot(kind="barh")
    plt.title("Importance des variables retenues — modèle final")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "feature_importance_selected.png", dpi=140)
    plt.close()

    plt.figure(figsize=(6, 6))
    fr_raw, mp_raw = calibration_curve(y_test, proba_test_uncalibrated, n_bins=10, strategy="quantile")
    fr_cal, mp_cal = calibration_curve(y_test, proba_test, n_bins=10, strategy="quantile")
    plt.plot(mp_raw, fr_raw, "s-", label="XGBoost brut (non calibré)")
    plt.plot(mp_cal, fr_cal, "o-", label="XGBoost calibré (Platt sur marge)")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Calibration parfaite")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Probabilité prédite moyenne (par bin, échelle log)")
    plt.ylabel("Fréquence réelle de fraude (par bin, échelle log)")
    plt.title("Courbe de calibration (test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "calibration_curve.png", dpi=140)
    plt.close()

    top_behavior = behavior_report.head(8)
    if not top_behavior.empty:
        fig, ax = plt.subplots(figsize=(8, max(4, 0.5 * len(top_behavior))))
        ratio = top_behavior["mean_fraud"] / top_behavior["mean_normal"].replace(0, np.nan)
        ax.barh(np.arange(len(top_behavior)), ratio.fillna(0), color="#c0392b")
        ax.axvline(1.0, color="gray", linestyle="--", label="Niveau transaction normale")
        ax.set_yticks(np.arange(len(top_behavior)))
        ax.set_yticklabels(top_behavior["feature"])
        ax.set_xlabel("Moyenne fraude / moyenne normale (x fois)")
        ax.set_title("Comportement fraude vs normal — top features")
        ax.legend()
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "fraud_vs_normal_behavior.png", dpi=140)
        plt.close()

    precision_curve, recall_curve, _ = precision_recall_curve(y_test, proba_test)
    plt.figure(figsize=(7, 5))
    plt.plot(recall_curve, precision_curve, label="XGBoost")
    plt.xlabel("Rappel")
    plt.ylabel("Précision")
    plt.title("Courbe Precision–Recall (test)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "precision_recall_curve.png", dpi=140)
    plt.close()

    print("[train] Rapports générés dans reports/")
    print(f"[train] Terminé — artefact vérifié, prêt pour API_Score.py : {MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement FraudGuard v5 (XGBoost)")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--cost-fn", type=float, default=DEFAULT_COST_FALSE_NEGATIVE)
    parser.add_argument("--cost-fp", type=float, default=DEFAULT_COST_FALSE_POSITIVE)
    parser.add_argument("--flat-cost-fn", action="store_true")
    parser.add_argument("--cost-fn-amt-multiplier", type=float, default=0.5)
    parser.add_argument("--cost-fn-min", type=float, default=50.0)
    parser.add_argument("--cost-fn-max", type=float, default=2000.0)
    parser.add_argument("--importance-threshold", type=float, default=DEFAULT_IMPORTANCE_THRESHOLD)
    parser.add_argument("--min-features", type=int, default=DEFAULT_MIN_FEATURES)
    parser.add_argument("--max-features", type=int, default=DEFAULT_MAX_FEATURES)
    parser.add_argument("--n-features", type=int, default=None)
    parser.add_argument("--cost-tolerance", type=float, default=DEFAULT_COST_TOLERANCE)
    parser.add_argument("--include-history-depth-features", action="store_true")
    parser.add_argument("--include-weak-features", action="store_true")
    parser.add_argument("--include-unreliable-features", action="store_true")
    parser.add_argument("--no-force-personal", action="store_true")
    parser.add_argument("--band-recall", type=float, default=BAND_TARGET_RECALL)
    parser.add_argument("--band-precision", type=float, default=BAND_TARGET_PRECISION)
    parser.add_argument("--max-review-share", type=float, default=BAND_MAX_REVIEW_SHARE)
    parser.add_argument("--t-low-quantile", type=float, default=0.995,
                        help="Quantile des probas LÉGIT (validation) ancrant t_low. "
                             "0 = règle historique (rappel cible). Défaut : 0.995.")
    parser.add_argument("--spw-power", type=float, default=0.5,
                        help="scale_pos_weight = (n_neg/n_pos)**power. 1.0 = ancien "
                             "comportement, 0.5 = racine (défaut), 0 = pas de pondération.")
    args = parser.parse_args()
    main(
        Path(args.data),
        cost_false_negative=args.cost_fn, cost_false_positive=args.cost_fp,
        use_amount_dependent_cost=not args.flat_cost_fn,
        cost_fn_amt_multiplier=args.cost_fn_amt_multiplier,
        cost_fn_min=args.cost_fn_min, cost_fn_max=args.cost_fn_max,
        importance_threshold=args.importance_threshold,
        min_features=args.min_features, max_features=args.max_features,
        n_features_override=args.n_features, cost_tolerance=args.cost_tolerance,
        include_history_depth=args.include_history_depth_features,
        include_weak=args.include_weak_features,
        include_unreliable=args.include_unreliable_features,
        force_personal=not args.no_force_personal,
        band_recall=args.band_recall, band_precision=args.band_precision,
        max_review_share=args.max_review_share,
        t_low_quantile=(args.t_low_quantile if args.t_low_quantile > 0 else None),
        spw_power=args.spw_power,
    )