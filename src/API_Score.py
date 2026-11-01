#!/usr/bin/env python3
"""
FraudGuard ML — Scoring Engine (API FastAPI) — aligné sur train2.py (v3)
Sert le modèle XGBoost (calibré) entraîné par train2.py (ou train1.py) et journalise chaque
décision dans SQLite (fraud.db). Le dashboard (app.py) lit cette MÊME base
SQLite en lecture seule.

==========================================================================
ALIGNEMENT AVEC train1.py
==========================================================================
train1.py entraîne d'abord un modèle "éclaireur" sur 46 features numériques
candidates + category_code, puis ne conserve automatiquement qu'un
SOUS-ENSEMBLE de ces features pour le modèle final (sélection par gain
cumulé). La liste retenue dépend donc du dataset et change d'un
entraînement à l'autre.

Conséquences pour ce script :

1. Il sait calculer les 46 features candidates (SUPPORTED_NUMERIC_FEATURES)
   + category_code, puis extrait à la volée uniquement les colonnes
   `artifact["columns"]` retenues par train1.py, dans l'ORDRE de
   l'artefact. Aucune liste de features n'est codée en dur pour le modèle.

2. Clés de l'artefact lues (voir joblib.dump dans train1.py) :
       xgb_model, threshold, threshold_f1_reference, columns,
       all_candidate_columns, category_column, category_map, model_version.
   Les anciennes clés (threshold_xgb, numeric_columns, ...) n'existent plus.
   Le fichier modèle par défaut est models/fraud_xgb_model.joblib.

3. Un garde-fou au démarrage refuse de démarrer si le modèle demande une
   feature que ce script ne sait pas calculer (au lieu de planter à la
   première requête).

4. Les features dérivées sont TOUJOURS calculées côté serveur à partir de
   l'historique SQLite, de façon STRICTEMENT CAUSALE (jamais la transaction
   en cours ni le futur). L'appelant n'envoie que des données brutes.

5. Les valeurs ±inf sont converties en NaN comme dans build_features
   (train1.py). category_code inconnu -> NaN (jamais -1).

==========================================================================
COMPATIBILITÉ AVEC LE DASHBOARD (appp.py)
==========================================================================
6. init_db() crée ET migre transactions_scored (ALTER TABLE ADD COLUMN pour
   toute colonne manquante d'une base créée par une ancienne version), crée
   feedback_history et failed_transactions (lue par le /health du dashboard).
   is_weekend et is_night sont maintenant journalisés.

7. POST /api/v1/feedback : reçoit le clic analyste du dashboard
   ({"transaction_id": ..., "feedback": "true_positive"|"false_positive"})
   et l'écrit dans feedback_history (label TP/FP). Ce script reste le SEUL
   écrivain de fraud.db ; le ré-entraînement à partir de ces labels n'est
   PAS automatisé ici.

==========================================================================
PARITÉ AVEC L'ENTRAÎNEMENT (serving_priors, produits par train1.py v2.1)
==========================================================================
8. Si l'artefact contient "serving_priors", ce script les utilise au lieu de
   valeurs supposées :
     - valeur des features d'historique pour la 1ère transaction d'une carte
       (FIRST_TX_DEFAULTS) ;
     - unité de time_since_last_tx / time_since_first_tx ;
     - moyenne/écart-type de montant par catégorie (amt_vs_category_z,
       amt_to_category_mean), UNIQUEMENT si train1.py a vérifié qu'ils
       reproduisent les colonnes du CSV (corrélation >= 0.98), sinon repli sur
       la base SQLite ;
     - fréquence de chaque marchand (merchant_freq), marchand inconnu -> médiane.
   Sans "serving_priors" (ancien modèle), l'ancien comportement est conservé.

9. Décision par PLAGES DE SCORE 0-100 (voir point 14) : ALLOW < 20, BLOCK >= 80,
   MANUAL_REVIEW entre les deux. FRAUD_REVIEW_RATIO n'existe plus.
   Les erreurs de traitement sont journalisées dans failed_transactions.

==========================================================================
VILLES DU MONDE ENTIER (générateur worldwide) — corrections v2.2
==========================================================================
Le générateur envoie désormais des cartes domiciliées partout dans le monde
(et plus seulement en France / Tunisie). Trois hypothèses implicites de ce
script devenaient fausses ; elles sont corrigées :

10. HEURE LOCALE. Les horodatages arrivent en UTC, mais hour / day_of_week /
    month / is_weekend / is_night / hour_sin / hour_cos / hour_share_card
    décrivent le rythme de vie du porteur (comme dans le CSV d'entraînement,
    en heure locale). Un achat à 14h UTC à Tokyo est un achat de 23h : il est
    donc calculé avec l'heure locale approchée du domicile (fuseau « solaire »
    déduit de la longitude, 15° = 1 h ; pas de gestion des heures d'été).
    Les fenêtres de vélocité et les écarts de temps restent en UTC absolu.
    Désactivable : FRAUD_LOCAL_TIME=0.

11. CANAL ONLINE. La position d'un marchand en ligne (siège d'Amazon à
    Seattle...) n'est PAS l'endroit où se trouve le porteur. Pour ces
    transactions : distance domicile-marchand = 0 (on retient le domicile) ;
    vitesse / distance au marchand précédent = 0 ; et pour une transaction en
    point de vente, la vitesse est mesurée depuis la dernière transaction
    PHYSIQUE (pas depuis un achat en ligne). Sans cela, chaque achat en ligne
    produisait un « voyage impossible » de 10 000 km en quelques minutes.
    Désactivable : FRAUD_ONLINE_USES_HOME=0.

12. ROBUSTESSE. haversine_km est borné (erreurs d'arrondi près des antipodes
    -> arcsin(NaN)) ; latitude/longitude sont validées (±90 / ±180) ;
    l'horodatage journalisé est normalisé en UTC (format SQLite) pour que
    les filtres de fenêtre du dashboard restent corrects.

==========================================================================
MÉTRIQUES COMPORTEMENTALES CALCULÉES PAR L'API (v2.3)
==========================================================================
13. Le bloc « Métriques Comportementales » du dashboard (vélocité 24 h,
    écart montant z-score, écart relatif vs médiane, intervalle
    inter-transactions, affinité marchand / catégorie habituelle) est
    désormais calculé ICI, à l'enrichissement de chaque transaction, à
    partir de l'HISTORIQUE COMPLET de la carte : les lignes de History.csv
    + toutes les transactions déjà journalisées de cette carte (causal :
    strictement antérieures à la transaction en cours).

    - Le résultat est stocké en JSON dans transactions_scored.behavioral_metrics
      (colonne migrée automatiquement) et renvoyé dans la réponse de /score
      (clé "behavioral_metrics"). Le dashboard n'a qu'à l'afficher.
    - Au démarrage, History.csv est importé dans fraud.db (is_history = 1,
      INSERT OR IGNORE : sans doublon, sans effet si seed_history.py l'a déjà
      rejoué). Ces lignes servent de mémoire de la carte mais restent
      invisibles dans le dashboard (vue transactions_scored_live).
      Fichier cherché dans : FRAUD_HISTORY_CSV, data/History.csv, data/sample/History.csv,
      ./History.csv, dossier du script. Désactivable : FRAUD_IMPORT_HISTORY=0.
    - Ces métriques d'AFFICHAGE sont indépendantes des 46 features du modèle
      (elles ne changent ni la prédiction ni le seuil). Le z-score du montant
      utilise l'écart-type de POPULATION (la feature amt_zscore_card du modèle, elle,
      utilise l'écart-type échantillon ddof=1, comme le CSV d'entraînement).

==========================================================================
SCORE DE RISQUE, EXPLICATION ET QUALITÉ DES FEATURES (v3)
==========================================================================
14. SCORE 0-100. La probabilité calibrée du modèle est très écrasée vers 0
    (fraude ~0,6 %) : y appliquer « ALLOW < 20 / BLOCK >= 80 » telle quelle
    laisserait passer presque toutes les fraudes. Le score est donc une
    transformation MONOTONE de cette probabilité (échelle logarithmique),
    ancrée sur deux seuils appris à l'entraînement (artefact
    "score_calibration") : t_low -> 20 et t_high -> 80. Un artefact ancien,
    sans "score_calibration", utilise à la place le seuil de coût métier
    (-> 20) et le seuil du meilleur F1 (-> 80).
    - transactions_scored.fraud_probability contient désormais score/100 (le
      dashboard l'affiche déjà comme « Risk Score ») ; la probabilité brute du
      modèle est dans model_probability et dans la réponse de /score.
    - FRAUD_SCORE_SCALING=0 revient à score = probabilité x 100.
    - FRAUD_ALLOW_MAX_SCORE (20) et FRAUD_BLOCK_MIN_SCORE (80) règlent les plages.
    - FRAUD_THRESHOLD n'influence plus la décision (il ne sert qu'à l'affichage).
15. EXPLICATION PAR TRANSACTION (SHAP). Les contributions du booster brut
    (pred_contribs) sont stockées dans transactions_scored.top_factors (JSON) et
    résumées en français dans llm_explanation (« Analyse Fraude » du dashboard).
16. QUALITÉ DES FEATURES PAR TRANSACTION. Chaque feature du modèle est comparée
    aux quantiles d'entraînement (artefact "feature_stats", produit par
    train2.py) : hors plage = valeur que le modèle n'a jamais vue. La fiabilité
    (0-100 %) pondère ces écarts par l'importance de chaque feature. Elle est
    ajoutée aux lignes du bloc « Métriques Comportementales » du dashboard.
"""
import hashlib
import json
import logging
import math
import os
import sqlite3
import statistics
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple
from calibration import MarginPlattCalibrator
import joblib
import numpy as np
import pandas as pd
import uvicorn
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraudguard.scoring")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path(os.getenv("FRAUD_DATA_DIR", "data")).resolve()
DB_PATH = DATA_DIR / "fraud.db"
MODEL_DIR = Path(os.getenv("FRAUD_MODEL_DIR", "models")).resolve()
# Nom de fichier identique à MODEL_PATH de train1.py.
MODEL_PATH = Path(
    os.getenv("FRAUD_MODEL_PATH", str(MODEL_DIR / "fraud_xgb_model.joblib"))
).resolve()

# Les 46 features numériques candidates de train1.py (CANDIDATE_NUMERIC_FEATURES).
# Ce script doit savoir toutes les calculer, car le sous-ensemble retenu par
# la sélection automatique n'est connu qu'au chargement de l'artefact.
SUPPORTED_NUMERIC_FEATURES = [
    "hour", "day_of_week", "month", "is_weekend", "is_night",
    "hour_sin", "hour_cos", "age",
    "amt", "log_amt", "is_round_amt",
    "amt_vs_category_z", "amt_to_category_mean",
    "dist_home_merch_km",
    "tx_rank", "is_first_tx",
    "time_since_last_tx", "log_time_since_last_tx", "time_since_first_tx",
    "dist_from_prev_merch_km", "speed_kmh",
    "card_mean_amt_prev", "card_std_amt_prev", "card_max_amt_prev",
    "amt_zscore_card", "amt_to_card_mean", "amt_to_card_max", "is_new_max_amt",
    "card_mean_dist_prev_km", "dist_to_card_mean",
    "nb_prev_same_merchant", "is_new_merchant", "is_new_category",
    "category_share_card", "hour_share_card", "merchant_freq",
    "nb_tx_1h", "sum_amt_1h", "nb_tx_6h", "sum_amt_6h",
    "nb_tx_24h", "sum_amt_24h", "nb_tx_7d", "sum_amt_7d",
    "amt_share_of_24h", "tx_accel_1h_vs_24h",
]

# Valeurs des features dépendant de l'historique carte pour la TOUTE PREMIÈRE
# transaction d'une carte (aucun historique). À vérifier/ajuster selon la
# convention réellement utilisée pour générer fraudTrain_features.csv.
FIRST_TX_DEFAULTS = {
    "time_since_last_tx": 0.0,
    "log_time_since_last_tx": 0.0,
    "time_since_first_tx": 0.0,
    "tx_rank": 1.0,
    "is_first_tx": 1.0,
    "dist_from_prev_merch_km": 0.0,
    "speed_kmh": 0.0,
    "card_mean_amt_prev": 0.0,
    "card_std_amt_prev": 0.0,
    "card_max_amt_prev": 0.0,
    "amt_zscore_card": 0.0,
    "amt_to_card_mean": 1.0,
    "amt_to_card_max": 1.0,
    "is_new_max_amt": 1.0,
    "card_mean_dist_prev_km": 0.0,
    "dist_to_card_mean": 0.0,
    "nb_prev_same_merchant": 0.0,
    "is_new_merchant": 1.0,
    "is_new_category": 1.0,
    "category_share_card": 0.0,
    "hour_share_card": 0.0,
}

# (La tranche « revue manuelle » est désormais définie par les plages de score
# 0-100 : voir « Score de risque » plus bas et le point 14 du docstring.)


def _utc_now_str() -> str:
    """Horodatage UTC au format SQLite (comparable à datetime('now', ...) :
    le format ISO avec 'T' fausserait les filtres de fenêtre du dashboard)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _valid_unit(value, default: float) -> float:
    return float(value) if _finite(value) and value > 0 else float(default)


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def to_utc_datetime(values):
    """Convertit en datetimes UTC en tolérant des formats MIXTES (avec/sans
    microsecondes, 'T' ou espace). Sans cela, pandas >= 2 infère le format de la
    1re valeur et transforme les autres en NaT : elles disparaîtraient
    silencieusement de l'historique."""
    try:
        return pd.to_datetime(values, utc=True, errors="coerce", format="ISO8601")
    except (TypeError, ValueError):
        return pd.to_datetime(values, utc=True, errors="coerce")


# Voir points 10 et 11 du docstring.
LOCAL_TIME_FROM_LONGITUDE = _env_flag("FRAUD_LOCAL_TIME", True)
ONLINE_USES_HOME_LOCATION = _env_flag("FRAUD_ONLINE_USES_HOME", True)


def utc_offset_hours(longitude) -> int:
    """Décalage horaire approximatif (en heures) du domicile, déduit de sa
    longitude : fuseau « solaire » (15° = 1 h), borné à [-12, +14]. Suffisant
    pour distinguer jour et nuit ; ignore fuseaux politiques et heure d'été."""
    if not LOCAL_TIME_FROM_LONGITUDE or not _finite(longitude):
        return 0
    return int(max(-12, min(14, round(longitude / 15.0))))


def is_online_channel(channel) -> bool:
    """True si la transaction doit être traitée comme un achat en ligne
    (position du marchand = siège, pas la position du porteur)."""
    return ONLINE_USES_HOME_LOCATION and str(channel or "").strip().lower() == "online"


# ---------------------------------------------------------------------------
# Score de risque 0-100 et plages de décision (point 14)
# ---------------------------------------------------------------------------
def _env_score(name: str, default: float) -> float:
    try:
        return min(max(float(os.getenv(name, str(default))), 0.0), 100.0)
    except ValueError:
        return default


ALLOW_MAX_SCORE = _env_score("FRAUD_ALLOW_MAX_SCORE", 20.0)   # score < 20  -> ALLOW
BLOCK_MIN_SCORE = _env_score("FRAUD_BLOCK_MIN_SCORE", 80.0)   # score >= 80 -> BLOCK
if not (0.0 < ALLOW_MAX_SCORE < BLOCK_MIN_SCORE < 100.0):
    ALLOW_MAX_SCORE, BLOCK_MIN_SCORE = 20.0, 80.0             # configuration incohérente
SCORE_SCALING = _env_flag("FRAUD_SCORE_SCALING", True)
SCORE_P_FLOOR = 1e-6          # identique à train2.py

SCORE_KNOTS = None            # (log p aux 4 nœuds) ; None = score = probabilité x 100
SCORE_KNOTS_INFO = "désactivé (score = probabilité x 100)"

def risk_score_from_proba(proba, t_low, t_high, allow_max=ALLOW_MAX_SCORE,
                         block_min=BLOCK_MIN_SCORE, p_floor=SCORE_P_FLOOR):
    """حساب درجة الخطر (0-100) بدقة باستخدام p_floor الديناميكي و t_low و t_high"""
    p = np.clip(np.asarray(proba, dtype=float), p_floor, 1.0)
    xp = np.log([p_floor, t_low, t_high, 1.0])
    return float(np.interp(np.log(p), xp, [0.0, allow_max, block_min, 100.0]))


def decide(score: float) -> str:
    """ALLOW < ALLOW_MAX_SCORE <= MANUAL_REVIEW < BLOCK_MIN_SCORE <= BLOCK."""
    if score >= BLOCK_MIN_SCORE:
        return "BLOCK"
    if score < ALLOW_MAX_SCORE:
        return "ALLOW"
    return "MANUAL_REVIEW"


DECISION_FR = {"BLOCK": "BLOQUÉE", "MANUAL_REVIEW": "REVUE MANUELLE", "ALLOW": "AUTORISÉE"}

# ---------------------------------------------------------------------------
# Libellés et formatage des features (explications lisibles)
# ---------------------------------------------------------------------------
FEATURE_LABELS = {
    "hour": "Heure de la transaction", "day_of_week": "Jour de la semaine", "month": "Mois",
    "is_weekend": "Week-end", "is_night": "Transaction de nuit",
    "hour_sin": "Heure (cycle sin)", "hour_cos": "Heure (cycle cos)", "age": "Âge du titulaire",
    "amt": "Montant", "log_amt": "Montant (log)", "is_round_amt": "Montant rond",
    "amt_vs_category_z": "Montant vs norme de la catégorie",
    "amt_to_category_mean": "Montant / moyenne de la catégorie",
    "dist_home_merch_km": "Distance domicile-marchand", "tx_rank": "Rang de la transaction",
    "is_first_tx": "Première transaction",
    "time_since_last_tx": "Délai depuis la dernière transaction",
    "log_time_since_last_tx": "Délai depuis la dernière transaction (log)",
    "time_since_first_tx": "Ancienneté de la carte",
    "dist_from_prev_merch_km": "Distance depuis le marchand précédent",
    "speed_kmh": "Vitesse de déplacement implicite",
    "card_mean_amt_prev": "Montant moyen habituel de la carte",
    "card_std_amt_prev": "Variabilité habituelle des montants",
    "card_max_amt_prev": "Montant maximal habituel de la carte",
    "amt_zscore_card": "Montant vs habitude de la carte (z-score)",
    "amt_to_card_mean": "Montant / moyenne de la carte",
    "amt_to_card_max": "Montant / maximum habituel de la carte",
    "is_new_max_amt": "Nouveau montant record pour la carte",
    "card_mean_dist_prev_km": "Distance habituelle de la carte",
    "dist_to_card_mean": "Écart à la distance habituelle",
    "nb_prev_same_merchant": "Achats précédents chez ce marchand",
    "is_new_merchant": "Nouveau marchand", "is_new_category": "Nouvelle catégorie pour la carte",
    "category_share_card": "Affinité de la carte pour cette catégorie",
    "hour_share_card": "Affinité de la carte pour cette heure",
    "merchant_freq": "Popularité du marchand",
    "nb_tx_1h": "Transactions sur 1 h", "sum_amt_1h": "Somme dépensée sur 1 h",
    "nb_tx_6h": "Transactions sur 6 h", "sum_amt_6h": "Somme dépensée sur 6 h",
    "nb_tx_24h": "Transactions sur 24 h", "sum_amt_24h": "Somme dépensée sur 24 h",
    "nb_tx_7d": "Transactions sur 7 jours", "sum_amt_7d": "Somme dépensée sur 7 jours",
    "amt_share_of_24h": "Part du montant dans les dépenses de 24 h",
    "tx_accel_1h_vs_24h": "Accélération 1 h vs 24 h",
    "category_code": "Catégorie du marchand",
}


def format_feature_value(name: str, v) -> str:
    """Valeur d'une feature, formatée pour un analyste (jamais d'exception)."""
    try:
        if v is None or not math.isfinite(float(v)):
            return "n/d"
        v = float(v)
        if name == "category_code":
            return CATEGORY_BY_CODE.get(int(v), "inconnue")
        if name.startswith("is_"):
            return "oui" if v >= 0.5 else "non"
        if name == "hour":
            return f"{int(v)} h"
        if name in ("amt", "card_mean_amt_prev", "card_std_amt_prev", "card_max_amt_prev") or name.startswith("sum_amt_"):
            return f"{v:,.2f} €".replace(",", " ")
        if name == "log_amt":
            return f"{math.expm1(v):,.2f} €".replace(",", " ")
        if name in ("amt_vs_category_z", "amt_zscore_card"):
            return f"{v:+.2f} σ"
        if name in ("amt_to_category_mean", "amt_to_card_mean", "amt_to_card_max"):
            return f"×{v:.2f}"
        if name == "speed_kmh":
            return f"{v:,.0f} km/h".replace(",", " ")
        if name.endswith("_km"):
            return f"{v:,.0f} km".replace(",", " ")
        if name == "time_since_last_tx":
            return _bm_fmt_duration(v * TIME_UNIT_LAST_S)
        if name == "log_time_since_last_tx":
            return _bm_fmt_duration(math.expm1(v) * TIME_UNIT_LAST_S)
        if name == "time_since_first_tx":
            return _bm_fmt_duration(v * TIME_UNIT_FIRST_S)
        if name.endswith("_share_card") or name == "amt_share_of_24h":
            return f"{v * 100:.0f} %"
        if name.startswith("nb_") or name == "tx_rank":
            return f"{v:.0f}"
        if name in ("hour_sin", "hour_cos"):
            return f"{v:+.2f}"
        return f"{v:.3g}"
    except Exception:
        return "n/d"


app = FastAPI(title="FraudGuard ML Scoring Engine")

# ---------------------------------------------------------------------------
# Chargement du modèle au démarrage
# ---------------------------------------------------------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Fichier modèle introuvable : {MODEL_PATH}\n"
        "Lancez d'abord l'entraînement : python -m train2 --data data/fraudTrain_features.csv"
    )

artifact = joblib.load(MODEL_PATH)

# XGBoost calibré (sigmoid) : SEUL modèle de décision (cf. train1.py).
xgb_model = artifact["xgb_model"]
threshold = float(artifact["threshold"])
threshold_f1_reference = artifact.get("threshold_f1_reference")

# Permet de FORCER un seuil de décision différent de celui figé dans
# l'artefact (ex. pour du réglage/test), sans avoir à ré-entraîner le
# modèle. Non définie par défaut : le seuil de l'artefact est conservé.
_threshold_override = os.getenv("FRAUD_THRESHOLD")
if _threshold_override:
    try:
        _new_threshold = float(_threshold_override)
        if not (0.0 < _new_threshold < 1.0):
            raise ValueError("doit être strictement entre 0 et 1")
        logger.warning(
            "Seuil de décision FORCÉ par FRAUD_THRESHOLD=%.4f (au lieu de %.4f dans l'artefact)",
            _new_threshold, threshold,
        )
        threshold = _new_threshold
    except ValueError as e:
        logger.warning(
            "FRAUD_THRESHOLD=%r invalide (%s) : seuil de l'artefact conservé (%.4f)",
            _threshold_override, e, threshold,
        )
category_map = artifact["category_map"]
category_column = artifact["category_column"]      # "category_code"
feature_columns = artifact["columns"]              # features RETENUES, dans l'ordre d'entraînement
model_version = artifact.get("model_version", "v2.0")

# ---- Priors d'entraînement (train1.py v2.1) : voir point 8 du docstring ----
serving_priors = artifact.get("serving_priors") or {}
PRIOR_MIN_CORR = 0.98
CATEGORY_AMT_STATS = serving_priors.get("category_amt_stats") or {}
_parity = serving_priors.get("category_stats_parity") or {}
USE_PRIOR_CATEGORY_Z = bool(CATEGORY_AMT_STATS) and _finite(_parity.get("z_corr")) and _parity["z_corr"] >= PRIOR_MIN_CORR
USE_PRIOR_CATEGORY_RATIO = bool(CATEGORY_AMT_STATS) and _finite(_parity.get("ratio_corr")) and _parity["ratio_corr"] >= PRIOR_MIN_CORR
MERCHANT_FREQ_MAP = serving_priors.get("merchant_freq_map")          # None si absent
MERCHANT_FREQ_DEFAULT = float(serving_priors.get("merchant_freq_default", 0.0))
_units = serving_priors.get("time_unit_seconds") or {}
TIME_UNIT_LAST_S = _valid_unit(_units.get("time_since_last_tx"), 60.0)      # défaut : minutes
TIME_UNIT_FIRST_S = _valid_unit(_units.get("time_since_first_tx"), 3600.0)  # défaut : heures
_first_defaults = serving_priors.get("first_tx_defaults") or {}
for _name, _value in _first_defaults.items():
    if _name in FIRST_TX_DEFAULTS:
        FIRST_TX_DEFAULTS[_name] = float(_value)
FIRST_TX_DEFAULTS["is_first_tx"] = 1.0    # vrai par définition
FIRST_TX_DEFAULTS["tx_rank"] = 0.0

# ---- Score 0-100 : nœuds de l'échelle (point 14) ----
def _init_score_scaling() -> None:
    global SCORE_KNOTS, SCORE_KNOTS_INFO, SCORE_P_FLOOR
    if not SCORE_SCALING:
        SCORE_KNOTS, SCORE_KNOTS_INFO = None, "désactivé (FRAUD_SCORE_SCALING=0) : score = probabilité x 100"
        return
    cal = artifact.get("score_calibration") or {}
    # Plancher de proba appris à l'entraînement (ex. 1.62e-09) : sans lui, les
    # scores < 20 seraient faussés (plancher codé en dur à 1e-6).
    p_floor = cal.get("p_floor", artifact.get("p_floor"))
    if _finite(p_floor) and 0.0 < p_floor < 1e-3:
        SCORE_P_FLOOR = float(p_floor)
        logger.info("Plancher de probabilité du score (p_floor) lu dans l'artefact : %.3g", SCORE_P_FLOOR)
    else:
        logger.warning("p_floor absent de l'artefact : valeur par défaut %.3g conservée.", SCORE_P_FLOOR)
    t_low, t_high = cal.get("t_low"), cal.get("t_high")
    source = "artefact (score_calibration de train2.py)"
    if not (_finite(t_low) and _finite(t_high) and SCORE_P_FLOOR * 10 < t_low < t_high < 1.0):
        t_low, t_high = artifact.get("threshold"), artifact.get("threshold_f1_reference")
        source = ("REPLI : seuil de coût métier -> ALLOW_MAX, seuil du meilleur F1 -> BLOCK_MIN "
                  "(ré-entraînez avec train2.py pour des seuils dédiés)")
    if not (_finite(t_low) and _finite(t_high) and SCORE_P_FLOOR * 10 < t_low < t_high < 1.0):
        SCORE_KNOTS, SCORE_KNOTS_INFO = None, "seuils invalides dans l'artefact : score = probabilité x 100"
        return
    SCORE_KNOTS = [math.log(SCORE_P_FLOOR), math.log(t_low), math.log(t_high), 0.0]
    SCORE_KNOTS_INFO = (f"proba {t_low:.4f} -> score {ALLOW_MAX_SCORE:g} ; proba {t_high:.4f} -> "
                        f"score {BLOCK_MIN_SCORE:g} ; source : {source}")


_init_score_scaling()

# ---- SHAP, qualité des features, catégories (points 15 et 16) ----
RAW_BOOSTER = None
_raw_model = artifact.get("xgb_model_raw_uncalibrated")
if _raw_model is not None:
    try:
        RAW_BOOSTER = _raw_model.get_booster()
    except Exception:
        logger.exception("Booster brut illisible : pas d'explication SHAP par transaction.")
FEATURE_STATS = artifact.get("feature_stats") or {}
FEATURE_IMPORTANCE = artifact.get("feature_importance_share") or {}
CATEGORY_BY_CODE = {int(code): str(cat) for cat, code in category_map.items()}

# Garde-fou : le modèle ne doit demander que des features calculables ici.
_unsupported = [
    c for c in feature_columns
    if c != category_column and c not in SUPPORTED_NUMERIC_FEATURES
]
if _unsupported:
    raise RuntimeError(
        f"Le modèle demande des features que ce script ne sait pas calculer : {_unsupported}. "
        "Ajoutez-les dans compute_behavioral_features et SUPPORTED_NUMERIC_FEATURES."
    )

logger.info("Modèle chargé (%s), %d features retenues : %s",
            model_version, len(feature_columns), feature_columns)
logger.info("Seuil de décision (coût métier) : %.4f", threshold)
if threshold_f1_reference is not None:
    logger.info("Seuil de référence (meilleur F1) : %.4f", float(threshold_f1_reference))
logger.info("Chaque transaction scorée sera journalisée dans : %s (lu par app.py)", DB_PATH)
if serving_priors:
    logger.info(
        "Priors d'entraînement actifs : stats catégorie (z=%s, ratio=%s), fréquences marchand=%s (%d), "
        "unités temps (last=%.4g s, first=%.4g s), valeurs 1ère transaction=%s",
        USE_PRIOR_CATEGORY_Z, USE_PRIOR_CATEGORY_RATIO, MERCHANT_FREQ_MAP is not None,
        len(MERCHANT_FREQ_MAP or {}), TIME_UNIT_LAST_S, TIME_UNIT_FIRST_S, bool(_first_defaults),
    )
else:
    logger.warning("Aucun 'serving_priors' dans l'artefact (modèle entraîné avec un ancien train1.py) : "
                   "repli sur les statistiques de la base SQLite et sur des conventions supposées.")
logger.info("Décision par plages de score : ALLOW < %.0f | MANUAL_REVIEW [%.0f ; %.0f[ | BLOCK >= %.0f",
            ALLOW_MAX_SCORE, ALLOW_MAX_SCORE, BLOCK_MIN_SCORE, BLOCK_MIN_SCORE)
logger.info("Échelle du score : %s", SCORE_KNOTS_INFO)
if _threshold_override:
    logger.warning("FRAUD_THRESHOLD n'influence plus la décision (plages de score 0-100) : ignoré pour BLOCK/ALLOW.")
if RAW_BOOSTER is None:
    logger.warning("Artefact sans 'xgb_model_raw_uncalibrated' : top_factors (SHAP) restera vide.")
if not FEATURE_STATS:
    logger.warning("Artefact sans 'feature_stats' : la fiabilité des features par transaction est "
                   "indisponible (ré-entraînez avec train2.py).")
if "amt_vs_category_z" in feature_columns and not USE_PRIOR_CATEGORY_Z:
    logger.error(
        "amt_vs_category_z (souvent la feature n°1) est calculée depuis fraud.db et NON depuis les "
        "statistiques d'entraînement (parité insuffisante ou absente). Avec des montants simulés, "
        "cette feature sera faussée : vérifiez 'Parité stats catégorie' dans la sortie de train2.py.")


# Colonnes de transactions_scored (hors clé primaire). Sert à CRÉER la table
# et à MIGRER une base existante créée par une ancienne version du script.
TRANSACTIONS_SCORED_COLUMNS = [
    ("trans_date_trans_time", "TEXT"), ("processed_at", "TEXT"), ("channel", "TEXT"),
    ("cc_num", "TEXT"), ("cardholder_name", "TEXT"), ("merchant", "TEXT"),
    ("category", "TEXT"), ("amt", "REAL"), ("city", "TEXT"), ("age", "REAL"),
    ("lat", "REAL"), ("long", "REAL"), ("merch_lat", "REAL"), ("merch_long", "REAL"),
    ("dist_home_merch_km", "REAL"), ("amt_zscore_card", "REAL"),
    ("is_new_merchant", "INTEGER"), ("is_weekend", "INTEGER"), ("is_night", "INTEGER"),
    ("nb_tx_1h", "REAL"), ("nb_tx_24h", "REAL"),
    ("fraud_probability", "REAL"), ("anomaly_score", "REAL"), ("decision", "TEXT"),
    ("top_factors", "TEXT"), ("recommended_actions", "TEXT"), ("llm_explanation", "TEXT"),
    ("model_version", "TEXT"), ("content_hash", "TEXT"),
    # 0 = transaction réelle du flux live, 1 = ligne rejouée depuis
    # History.csv par seed_history.py. Le dashboard (app.py) DOIT filtrer
    # sur is_history = 0 pour ne pas afficher l'historique de warm-up.
    ("is_history", "INTEGER"),
    # Métriques comportementales (JSON) calculées par /score à partir de
    # l'historique de la carte (History.csv + transactions déjà journalisées).
    # Lues telles quelles par le dashboard : voir point 13 du docstring.
    ("behavioral_metrics", "TEXT"),
    # Probabilité de fraude BRUTE du modèle calibré (fraud_probability contient
    # le score de risque / 100 : voir point 14).
    ("model_probability", "REAL"),
]


def init_db() -> None:
    """Initialise (et migre si besoin) la base SQLite lue par le dashboard.

    Le dashboard (appp.py) lit ces colonnes via des alias (voir TX_COLUMNS) :
    dist_home_merch_km -> distance_km, amt_zscore_card -> amt_zscore,
    is_new_merchant -> new_merchant, nb_tx_1h/24h -> txn_count_1h/24h.
    Les colonnes state, city_pop, job et foreign_transaction n'existent plus
    (aucune source de données) : le dashboard les reçoit à NULL.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    with conn:
        columns_ddl = ",\n                ".join(
            f"{name} {sql_type}" for name, sql_type in TRANSACTIONS_SCORED_COLUMNS
        )
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS transactions_scored (
                transaction_id TEXT PRIMARY KEY,
                {columns_ddl}
            )
        """)
        # Migration : une base créée par une ancienne version n'est PAS
        # modifiée par CREATE TABLE IF NOT EXISTS -> on ajoute les colonnes
        # manquantes une à une.
        existing = {row[1] for row in conn.execute("PRAGMA table_info(transactions_scored)")}
        for name, sql_type in TRANSACTIONS_SCORED_COLUMNS:
            if name not in existing:
                conn.execute(f"ALTER TABLE transactions_scored ADD COLUMN {name} {sql_type}")
                logger.info("Migration fraud.db : colonne ajoutée transactions_scored.%s", name)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT, cc_num TEXT, label TEXT,
                analyst TEXT, closing_reason TEXT, timestamp TEXT
            )
        """)
        # Lue par le /health du dashboard (compteur d'échecs pipeline).
        conn.execute("""
            CREATE TABLE IF NOT EXISTS failed_transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT, channel TEXT, raw_payload_json TEXT,
                reason TEXT, attempts INTEGER, replay_count INTEGER,
                created_at TEXT, dead_at TEXT, fingerprint TEXT
            )
        """)
        # Index utilisé par les requêtes d'historique par carte (vélocité,
        # récence, habitudes) : sans lui, chaque /score ferait un scan complet.
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_transactions_scored_cc_num_time
            ON transactions_scored (cc_num, trans_date_trans_time)
        """)
        # Index utilisés par les agrégats GLOBAUX (amt_vs_category_z,
        # amt_to_category_mean, merchant_freq), calculés sur TOUTE la base.
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_transactions_scored_category
            ON transactions_scored (category)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_transactions_scored_merchant
            ON transactions_scored (merchant)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_feedback_history_transaction
            ON feedback_history (transaction_id)
        """)
        # Vue utilisée par le dashboard (dashboard.py, connexion STRICTEMENT
        # mode=ro) : masque les lignes rejouées depuis History.csv par
        # seed_history.py (is_history=1). Créée ici, côté écrivain, car une
        # connexion mode=ro ne peut pas créer de vue. DROP+CREATE (et non
        # IF NOT EXISTS) pour qu'une définition provenant d'une version
        # antérieure de ce script soit toujours remplacée par celle-ci.
        conn.execute("DROP VIEW IF EXISTS transactions_scored_live")
        conn.execute("""
            CREATE VIEW transactions_scored_live AS
            SELECT * FROM transactions_scored
            WHERE is_history IS NULL OR is_history = 0
        """)
    conn.close()


init_db()


# ---------------------------------------------------------------------------
# Schéma de validation de la requête
# ---------------------------------------------------------------------------
class TransactionInput(BaseModel):
    """Valide la charge utile reçue par /score. UNIQUEMENT des données
    BRUTES : toutes les features dérivées sont calculées côté serveur (voir
    compute_behavioral_features), jamais fournies par l'appelant."""

    transaction_id: str
    cc_num: str
    merchant: str
    category: str
    amt: float
    age: float
    lat: float = Field(..., ge=-90, le=90)          # latitude du domicile de la carte
    long: float = Field(..., ge=-180, le=180)       # longitude du domicile de la carte
    merch_lat: float = Field(..., ge=-90, le=90)    # latitude du marchand
    merch_long: float = Field(..., ge=-180, le=180)  # longitude du marchand
    trans_date_trans_time: Optional[str] = None
    channel: str = "online"
    cardholder_name: Optional[str] = None
    city: Optional[str] = None
    is_history: bool = False  # True = rejeu de History.csv (seed_history.py) :
    # sert au calcul des features (vélocité, z-score...) mais NE DOIT PAS
    # apparaître dans le dashboard (voir filtre is_history=0 dans app.py).


# ---------------------------------------------------------------------------
# Géographie
# ---------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    """Distance en km entre deux points GPS. Fonctionne aussi bien sur des
    floats que sur des colonnes pandas (vectorisé via numpy)."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    # Près des antipodes (villes du monde entier), l'arrondi peut donner a > 1
    # -> arcsin(NaN). On borne à [0, 1].
    a = np.clip(a, 0.0, 1.0)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def safe_ratio(num: float, den: float) -> float:
    """num / den, ou NaN si le dénominateur est nul (équivalent d'un ±inf
    converti en NaN par build_features dans train1.py)."""
    return float(num / den) if den > 0 else float("nan")


# ---------------------------------------------------------------------------
# Import de History.csv dans fraud.db (mémoire des cartes) — voir point 13
# ---------------------------------------------------------------------------
HISTORY_REQUIRED_COLUMNS = [
    "transaction_id", "cc_num", "trans_date_trans_time", "amt", "merchant",
    "category", "lat", "long", "merch_lat", "merch_long",
]
HISTORY_INSERT_COLUMNS = [
    "transaction_id", "trans_date_trans_time", "processed_at", "channel",
    "cc_num", "cardholder_name", "merchant", "category", "amt", "city", "age",
    "lat", "long", "merch_lat", "merch_long", "dist_home_merch_km",
    "model_version", "is_history",
]
HISTORY_INSERT_SQL = (
    f"INSERT OR IGNORE INTO transactions_scored ({', '.join(HISTORY_INSERT_COLUMNS)}) "
    f"VALUES ({', '.join('?' for _ in HISTORY_INSERT_COLUMNS)})"
)


def _find_history_csv() -> Optional[Path]:
    candidates = []
    env_path = os.getenv("FRAUD_HISTORY_CSV")
    if env_path:
        candidates.append(Path(env_path))
    candidates += [
        DATA_DIR / "History.csv",
        DATA_DIR / "sample" / "History.csv",
        Path.cwd() / "History.csv",
        Path(__file__).resolve().parent / "History.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return None


def import_history_csv() -> int:
    """Charge History.csv dans transactions_scored avec is_history = 1.

    - INSERT OR IGNORE sur transaction_id : relancer l'API ne crée aucun
      doublon, et une ligne déjà rejouée (seed_history.py) n'est pas écrasée.
    - Ces lignes servent UNIQUEMENT de mémoire de la carte (vélocité, montants
      habituels, rythme, catégories...) ; le dashboard les masque
      (vue transactions_scored_live).
    - Ne lève jamais : un problème d'import ne doit pas empêcher l'API de démarrer.
    Retourne le nombre de lignes réellement insérées."""
    if not _env_flag("FRAUD_IMPORT_HISTORY", True):
        logger.info("Import de History.csv désactivé (FRAUD_IMPORT_HISTORY=0).")
        return 0
    path = _find_history_csv()
    if path is None:
        logger.warning(
            "History.csv introuvable (FRAUD_HISTORY_CSV, %s, %s, ./History.csv) : les métriques "
            "comportementales s'appuieront uniquement sur les transactions déjà journalisées.",
            DATA_DIR / "History.csv", DATA_DIR / "sample" / "History.csv",
        )
        return 0
    try:
        df = pd.read_csv(path, dtype={"transaction_id": str, "cc_num": str})
        missing = [c for c in HISTORY_REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            logger.warning("History.csv ignoré : colonnes manquantes %s", missing)
            return 0

        df["_ts"] = to_utc_datetime(df["trans_date_trans_time"])
        for col in ("amt", "lat", "long", "merch_lat", "merch_long"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=HISTORY_REQUIRED_COLUMNS + ["_ts"])

        processed_at = _utc_now_str()
        rows = []
        for r in df.to_dict("records"):
            channel = str(r.get("channel") or "").strip().lower() or (
                "online" if str(r["category"]).endswith("_net") else "pos"
            )
            eff_lat, eff_long = (
                (r["lat"], r["long"]) if is_online_channel(channel) else (r["merch_lat"], r["merch_long"])
            )
            age = r.get("age")
            city = r.get("city")
            rows.append((
                str(r["transaction_id"]),
                r["_ts"].strftime("%Y-%m-%d %H:%M:%S"),
                processed_at,
                channel,
                str(r["cc_num"]).strip(),
                f"Client {str(r['cc_num']).strip()[-4:]}",
                str(r["merchant"]),
                str(r["category"]),
                float(r["amt"]),
                None if city is None or (isinstance(city, float) and math.isnan(city)) else str(city),
                float(age) if _finite(age) else None,
                float(r["lat"]), float(r["long"]),
                float(r["merch_lat"]), float(r["merch_long"]),
                float(haversine_km(r["lat"], r["long"], eff_lat, eff_long)),
                "history_csv",
                1,
            ))

        conn = sqlite3.connect(DB_PATH, timeout=10)
        try:
            before = conn.total_changes
            with conn:
                conn.executemany(HISTORY_INSERT_SQL, rows)
            inserted = conn.total_changes - before
        finally:
            conn.close()
        logger.info(
            "History.csv (%s) : %d ligne(s) lues, %d nouvelle(s) insérée(s) dans fraud.db (is_history=1), "
            "%d carte(s).", path, len(rows), inserted, df["cc_num"].nunique(),
        )
        return inserted
    except Exception:
        logger.exception("Échec de l'import de History.csv (%s) — l'API démarre sans.", path)
        return 0


import_history_csv()


# ---------------------------------------------------------------------------
# Métriques comportementales d'affichage (bloc du dashboard) — point 13
# ---------------------------------------------------------------------------
# Mêmes règles, mêmes seuils et mêmes libellés que l'ancien calcul du
# dashboard ; la différence : l'historique utilisé est ici COMPLET (History.csv
# + flux live) au lieu du seul flux live.
BM_MIN_HISTORY = 3


def _bm_row(key: str, label: str, value: str, level: str = "", **extra) -> dict:
    row = {"key": key, "label": label, "value": value, "level": level}
    row.update(extra)
    return row


def _bm_round(value, digits: int):
    return None if value is None else round(float(value), digits)


def _bm_fmt_duration(seconds: float) -> str:
    s = max(0, int(round(seconds)))
    if s < 60:
        return f"{s} s"
    if s < 3600:
        return f"{s // 60} min {s % 60:02d} s"
    if s < 86400:
        return f"{s // 3600} h {(s % 3600) // 60:02d} min"
    return f"{s // 86400} j {(s % 86400) // 3600} h"


def _bm_fmt_expected(x: float) -> str:
    if x < 0.01:
        return "<0.01"
    return f"{x:.2f}" if x < 10 else f"{x:.1f}"


def _bm_velocity(now: datetime, ts_list: List[datetime]) -> Tuple[list, dict]:
    """Nb de transactions (incluant l'actuelle) sur 24 h glissantes, comparé au
    rythme habituel de la carte (mesuré sur l'historique antérieur à 24 h pour
    ne pas être contaminé par une rafale en cours)."""
    secs = 86400
    cut = now - timedelta(seconds=secs)
    old = [t for t in ts_list if t < cut]
    rate_h = None
    if len(old) >= 3:
        span_h = max((cut - min(old)).total_seconds() / 3600.0, 24.0)
        rate_h = len(old) / span_h
    count = 1 + sum(1 for t in ts_list if cut < t <= now)
    expected = rate_h * secs / 3600.0 if rate_h is not None else None
    ratio = count / max(expected or 0.0, 1.0)
    if count >= 5 and ratio >= 5:
        level = "danger"
    elif count >= 3 and ratio >= 2.5:
        level = "warn"
    else:
        level = ""
    if expected is None:
        value = f"{count} txn"
    else:
        value = f"{count} txn (attendu ≈ {_bm_fmt_expected(expected)}, ×{ratio:.1f})"
    return [_bm_row("velocity_24h", "Vélocité (24 h)", value, level)], {
        "window": "24 h", "count": count,
        "rate_per_hour": round(count * 3600.0 / secs, 2),
        "baseline_txn_per_hour": _bm_round(rate_h, 4),
        "expected": _bm_round(expected, 4),
        "ratio": round(ratio, 2), "level": level,
    }


def _bm_amount(amt: float, past_amts: List[float]) -> Tuple[list, dict]:
    """Z-score (moyenne / écart-type) et écart relatif à la médiane du montant
    courant vs les montants passés de la carte."""
    n = len(past_amts)
    if not _finite(amt):
        msg = "Montant indisponible"
        return [_bm_row("amt_z", "Écart montant (z-score)", msg),
                _bm_row("amt_rel", "Écart relatif vs médiane", msg)], {"n": n}
    if n < BM_MIN_HISTORY:
        msg = f"Historique insuffisant ({n} txn)"
        return [_bm_row("amt_z", "Écart montant (z-score)", msg),
                _bm_row("amt_rel", "Écart relatif vs médiane", msg)], {"n": n}
    mean = statistics.fmean(past_amts)
    std = statistics.pstdev(past_amts)   # population (affichage) ; la feature modèle amt_zscore_card utilise ddof=1
    med = statistics.median(past_amts)
    z = (amt - mean) / std if std > 1e-9 else None
    rel = (amt - med) / med * 100.0 if med > 1e-9 else None
    if z is None:
        z_row = _bm_row("amt_z", "Écart montant (z-score)",
                        f"n/a (montants passés constants ≈ {mean:.2f} €)",
                        "warn" if abs(amt - mean) > 1e-9 else "")
    else:
        lvl = "danger" if abs(z) > 3 else "warn" if abs(z) > 2 else ""
        z_row = _bm_row("amt_z", "Écart montant (z-score)", f"{z:+.2f} (moy {mean:.2f} €, σ {std:.2f})", lvl)
    if rel is None:
        rel_row = _bm_row("amt_rel", "Écart relatif vs médiane", "n/a (médiane nulle)")
    else:
        lvl = "danger" if rel >= 300 else "warn" if rel >= 100 else ""
        rel_row = _bm_row("amt_rel", "Écart relatif vs médiane", f"{rel:+.0f} % (médiane {med:.2f} €)", lvl)
    return [z_row, rel_row], {
        "n": n, "mean": round(mean, 2), "std": round(std, 2), "median": round(med, 2),
        "zscore": _bm_round(z, 3), "relative_to_median_pct": _bm_round(rel, 1),
    }


def _bm_time_delta(now: datetime, ts_sorted: List[datetime]) -> Tuple[list, dict]:
    """Temps écoulé depuis la dernière activité de la carte (+ rythme médian
    habituel). `ts_sorted` : horodatages passés, ordre croissant."""
    if not ts_sorted:
        return [_bm_row("time_delta", "Intervalle inter-transactions", "Première transaction connue")], {
            "seconds": None, "median_gap_seconds": None}
    delta = max(0.0, (now - ts_sorted[-1]).total_seconds())
    gaps = [(b - a).total_seconds() for a, b in zip(ts_sorted, ts_sorted[1:])]
    med_gap = statistics.median(gaps) if len(gaps) >= 3 else None
    level = "danger" if delta < 60 else "warn" if delta < 300 else ""
    value = _bm_fmt_duration(delta)
    if med_gap is not None:
        value += f" (habituel : {_bm_fmt_duration(med_gap)})"
    return [_bm_row("time_delta", "Intervalle inter-transactions", value, level)], {
        "seconds": round(delta, 1), "median_gap_seconds": _bm_round(med_gap, 1)}


def _bm_affinity(category: str, past_categories: List[str]) -> Tuple[list, dict]:
    """Part de la catégorie marchand courante dans l'historique de la carte.
    Déviation = 1 − affinité."""
    label = "Affinité marchand (catégorie/MCC)"
    norm = lambda v: str(v).strip().lower() if v is not None and pd.notna(v) and str(v).strip() else None
    cur_k = norm(category)
    hist_keys = [k for k in (norm(c) for c in past_categories) if k]
    n = len(hist_keys)
    if cur_k is None:
        return [_bm_row("affinity", label, "Catégorie inconnue")], {"n": n}
    if n < BM_MIN_HISTORY:
        return [_bm_row("affinity", label, f"Historique insuffisant ({n} txn)")], {"n": n}
    counts = Counter(hist_keys)
    same = counts.get(cur_k, 0)
    share = same / n
    dev = 1.0 - share
    top_k, top_n = counts.most_common(1)[0]
    level = "danger" if (same == 0 and n >= 15) else "warn" if share < 0.05 else ""
    rows = [
        _bm_row("affinity", label, f"{share * 100:.0f} % · déviation {dev * 100:.0f} % ({same}/{n})", level),
        # "category" + "suffix" : lus par le JS du dashboard (catDisplay)
        _bm_row("affinity_top", "Catégorie habituelle", f"{top_k} ({top_n / n * 100:.0f} %)", "",
                category=top_k, suffix=f"({top_n / n * 100:.0f} %)"),
    ]
    return rows, {"n": n, "source": "category", "same_count": same,
                  "affinity": round(share, 4), "deviation": round(dev, 4), "top": top_k}


def compute_behavioral_metrics_block(tx: "TransactionInput", now: pd.Timestamp, history: pd.DataFrame) -> dict:
    """Métriques comportementales de la transaction `tx`, calculées sur
    l'historique causal COMPLET de sa carte (`history` = résultat de
    fetch_card_history : History.csv + transactions déjà journalisées,
    strictement antérieures à `now`). Ne lève jamais : en cas d'erreur, un
    bloc « indisponible » est renvoyé pour ne pas interrompre le scoring."""
    try:
        if history is None or history.empty:
            ts_sorted: List[datetime] = []
            past_amts: List[float] = []
            past_categories: List[str] = []
        else:
            ts_sorted = [t.to_pydatetime() for t in history["trans_date_trans_time"]]  # déjà triés (ASC)
            past_amts = [float(a) for a in history["amt"] if _finite(a)]
            past_categories = list(history["category"])
        now_dt = now.to_pydatetime()

        vel_rows, vel = _bm_velocity(now_dt, ts_sorted)
        amt_rows, amt = _bm_amount(tx.amt, past_amts)
        td_rows, td = _bm_time_delta(now_dt, ts_sorted)
        af_rows, af = _bm_affinity(tx.category, past_categories)
        return {
            "version": 1,
            "origin": "scoring_api",
            "transaction_id": tx.transaction_id,
            "reference_time": now.isoformat(),
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "history_count": len(ts_sorted),
            "velocity": vel, "amount": amt, "time_delta": td, "affinity": af,
            "rows": vel_rows + amt_rows + td_rows + af_rows,
        }
    except Exception:
        logger.exception("Calcul des métriques comportementales impossible pour %s", tx.transaction_id)
        return {
            "version": 1, "origin": "scoring_api", "transaction_id": tx.transaction_id,
            "history_count": 0,
            "rows": [_bm_row("error", "Métriques comportementales", "Calcul indisponible", "warn")],
        }


# ---------------------------------------------------------------------------
# Historique carte + agrégats globaux
# ---------------------------------------------------------------------------
def fetch_card_history(cc_num: str, before_time: pd.Timestamp, conn: sqlite3.Connection) -> pd.DataFrame:
    """Historique de LA MÊME carte, STRICTEMENT antérieur à before_time.
    Ne retourne jamais la transaction en cours (pas encore insérée) ni
    aucune transaction future."""
    query = """
        SELECT trans_date_trans_time, amt, category, merchant, channel,
               lat, long, merch_lat, merch_long
        FROM transactions_scored
        WHERE cc_num = ?
        ORDER BY trans_date_trans_time ASC
    """
    hist = pd.read_sql_query(query, conn, params=(cc_num,))
    if hist.empty:
        return hist
    hist["trans_date_trans_time"] = to_utc_datetime(hist["trans_date_trans_time"])
    hist = hist.dropna(subset=["trans_date_trans_time"])
    hist = hist[hist["trans_date_trans_time"] < before_time]
    hist = hist.sort_values("trans_date_trans_time").reset_index(drop=True)
    if not hist.empty:
        # Achat en ligne : la position du marchand n'est pas celle du porteur
        # -> on retient le domicile (distance domicile-marchand = 0).
        # Ancien enregistrement sans canal (NULL) : traité comme physique.
        if ONLINE_USES_HOME_LOCATION:
            hist["is_online"] = (
                hist["channel"].fillna("").astype(str).str.strip().str.lower().eq("online")
            )
        else:
            hist["is_online"] = False
        eff_lat = np.where(hist["is_online"], hist["lat"], hist["merch_lat"])
        eff_long = np.where(hist["is_online"], hist["long"], hist["merch_long"])
        hist["dist_home_merch_km"] = haversine_km(hist["lat"], hist["long"], eff_lat, eff_long)
    return hist


def _category_features_from_db(category: str, amt: float, conn: sqlite3.Connection) -> Tuple[float, float]:
    """Repli : (amt_vs_category_z, amt_to_category_mean) calculés sur la base
    SQLite (toutes cartes confondues). Écart-type via E[X^2] - E[X]^2.
    Sans historique suffisant : valeurs neutres (0.0, 1.0)."""
    row = conn.execute(
        "SELECT COUNT(*), AVG(amt), AVG(amt * amt) FROM transactions_scored WHERE category = ?",
        (category,),
    ).fetchone()
    n, mean_amt, mean_amt_sq = row
    if not n or mean_amt is None:
        return 0.0, 1.0

    ratio = float(amt / mean_amt) if mean_amt > 0 else 1.0
    if n < 2:
        return 0.0, ratio

    variance = max(mean_amt_sq - mean_amt ** 2, 0.0)
    std = math.sqrt(variance)
    z = float((amt - mean_amt) / std) if std > 0 else 0.0
    return z, ratio


def compute_category_features(category: str, amt: float, conn: sqlite3.Connection) -> Tuple[float, float]:
    """Retourne (amt_vs_category_z, amt_to_category_mean). Utilise les
    statistiques de l'ENTRAÎNEMENT quand train1.py a vérifié qu'elles
    reproduisent le CSV (référence stable, aucun démarrage à froid), sinon
    celles de la base SQLite."""
    z = ratio = None
    stats = CATEGORY_AMT_STATS.get(category)
    if stats:
        mean_c, std_c = stats["mean"], stats["std"]
        if USE_PRIOR_CATEGORY_Z:
            z = float((amt - mean_c) / std_c) if std_c and std_c > 0 else 0.0
        if USE_PRIOR_CATEGORY_RATIO:
            ratio = float(amt / mean_c) if mean_c and mean_c > 0 else 1.0
    if z is None or ratio is None:
        z_db, ratio_db = _category_features_from_db(category, amt, conn)
        z = z_db if z is None else z
        ratio = ratio_db if ratio is None else ratio
    return z, ratio


def compute_merchant_freq(merchant: str, conn: sqlite3.Connection) -> float:
    """Fréquence globale du marchand. Utilise la table apprise à
    l'entraînement (marchand inconnu -> médiane, valeur neutre) ; à défaut,
    nb de tx sur ce marchand / nb total de tx journalisées en base."""
    if MERCHANT_FREQ_MAP is not None:
        return float(MERCHANT_FREQ_MAP.get(merchant, MERCHANT_FREQ_DEFAULT))
    total = conn.execute("SELECT COUNT(*) FROM transactions_scored").fetchone()[0]
    if not total:
        return 0.0
    n_merchant = conn.execute(
        "SELECT COUNT(*) FROM transactions_scored WHERE merchant = ?", (merchant,)
    ).fetchone()[0]
    return float(n_merchant / total)


# ---------------------------------------------------------------------------
# Calcul des 46 features candidates (voir CANDIDATE_NUMERIC_FEATURES, train1.py)
# ---------------------------------------------------------------------------
def compute_behavioral_features(tx: TransactionInput, conn: sqlite3.Connection) -> dict:
    """Calcule, au moment du scoring, les mêmes features que celles
    pré-calculées dans fraudTrain_features.csv — mais causalement, en
    n'utilisant que l'historique de fraud.db déjà journalisé avant l'instant
    de cette transaction (carte courante + agrégats globaux)."""
    if tx.trans_date_trans_time:
        now = to_utc_datetime(tx.trans_date_trans_time)
    else:
        now = pd.NaT
    if pd.isna(now):
        now = pd.Timestamp.now(tz="UTC")

    features: dict = {}

    # Horodatage normalisé en UTC (format SQLite, comparable à
    # datetime('now', ...)) : c'est lui qui est journalisé dans /score.
    features["trans_time_utc"] = now.strftime("%Y-%m-%d %H:%M:%S") + (
        f".{now.microsecond:06d}" if now.microsecond else ""
    )

    # --- Contexte temporel (pas d'historique nécessaire) ---
    # Heure LOCALE approchée du domicile (voir point 10 du docstring) : les
    # features hour / is_night / ... décrivent le rythme de vie du porteur.
    # `now` (UTC) reste utilisé pour les écarts de temps et les fenêtres.
    tz_offset = utc_offset_hours(tx.long)
    local_now = now + pd.Timedelta(hours=tz_offset)
    # hour_sin / hour_cos dérivent de `hour` (entier 0-23), comme la feature
    # `hour` elle-même dans le CSV d'entraînement.
    hour = int(local_now.hour)
    dow = int(local_now.dayofweek)
    features["hour"] = float(hour)
    features["day_of_week"] = float(dow)
    features["month"] = float(local_now.month)
    features["is_weekend"] = float(dow >= 5)
    features["is_night"] = float(hour >= 22 or hour < 6)
    features["hour_sin"] = float(np.sin(2 * np.pi * hour / 24))
    features["hour_cos"] = float(np.cos(2 * np.pi * hour / 24))
    features["age"] = float(tx.age)

    # --- Montant ---
    features["amt"] = float(tx.amt)
    features["log_amt"] = float(np.log1p(max(tx.amt, 0.0)))
    features["is_round_amt"] = float(abs(tx.amt - round(tx.amt / 10) * 10) < 1e-6)
    z_cat, ratio_cat = compute_category_features(tx.category, tx.amt, conn)
    features["amt_vs_category_z"] = z_cat
    features["amt_to_category_mean"] = ratio_cat

    # --- Géographie : distance domicile <-> marchand, toujours calculable ---
    # Achat en ligne : la position du marchand (siège) n'est pas celle du
    # porteur -> on retient le domicile (distance 0). Voir point 11.
    online_now = is_online_channel(tx.channel)
    eff_lat, eff_long = (tx.lat, tx.long) if online_now else (tx.merch_lat, tx.merch_long)
    dist_home_merch = float(haversine_km(tx.lat, tx.long, eff_lat, eff_long))
    features["dist_home_merch_km"] = dist_home_merch

    # --- Fréquence marchand (agrégat global) ---
    features["merchant_freq"] = compute_merchant_freq(tx.merchant, conn)

    # --- Historique de LA CARTE, strictement antérieur à `now` ---
    history = fetch_card_history(tx.cc_num, now, conn)

    if history.empty:
        # Première transaction connue de cette carte : pas d'historique.
        features.update(FIRST_TX_DEFAULTS)
    else:
        # Montants de la carte
        amt_hist = history["amt"]
        card_mean_amt = float(amt_hist.mean())
        # ddof=1 (écart-type échantillon) : c'est ce que contient le CSV
        # d'entraînement (ex. [7.27, 52.94] -> 32.29) ; 0.0 s'il n'y a qu'1 point.
        card_std_amt = float(amt_hist.std(ddof=1)) if len(amt_hist) > 1 else 0.0
        card_max_amt = float(amt_hist.max())
        features["card_mean_amt_prev"] = card_mean_amt
        features["card_std_amt_prev"] = card_std_amt
        features["card_max_amt_prev"] = card_max_amt
        features["amt_zscore_card"] = (
            (tx.amt - card_mean_amt) / card_std_amt if card_std_amt > 0 else 0.0
        )
        features["amt_to_card_mean"] = safe_ratio(tx.amt, card_mean_amt)
        features["amt_to_card_max"] = safe_ratio(tx.amt, card_max_amt)
        features["is_new_max_amt"] = float(tx.amt > card_max_amt)

        # Distances habituelles de la carte
        dist_hist = history["dist_home_merch_km"]
        card_mean_dist = float(dist_hist.mean())
        features["card_mean_dist_prev_km"] = card_mean_dist
        # RATIO dist / distance moyenne (comme le CSV d'entraînement), pas une différence.
        features["dist_to_card_mean"] = safe_ratio(dist_home_merch, card_mean_dist)

        # Récence / rang
        last_row = history.iloc[-1]
        minutes_since_last = max(
            (now - last_row["trans_date_trans_time"]).total_seconds() / 60.0, 0.0
        )
        # Unité apprise sur le CSV (TIME_UNIT_LAST_S secondes par unité).
        time_since_last = minutes_since_last * 60.0 / TIME_UNIT_LAST_S
        features["time_since_last_tx"] = float(time_since_last)
        features["log_time_since_last_tx"] = float(np.log1p(time_since_last))
        features["is_first_tx"] = 0.0
        first_time = history["trans_date_trans_time"].iloc[0]
        features["time_since_first_tx"] = float((now - first_time).total_seconds() / TIME_UNIT_FIRST_S)
        features["tx_rank"] = float(len(history))   # 0-based, comme le CSV

        # Distance / vitesse depuis le marchand précédent.
        # Seules les transactions PHYSIQUES localisent le porteur : un achat en
        # ligne n'a ni distance ni vitesse, et une transaction physique est
        # comparée à la dernière transaction physique (pas à un achat en ligne
        # dont la « position » est le siège du marchand). Voir point 11.
        dist_prev_merch = 0.0
        speed_kmh = 0.0
        if not online_now:
            physical = history[~history["is_online"]]
            if not physical.empty:
                last_phys = physical.iloc[-1]
                dist_prev_merch = float(haversine_km(
                    last_phys["merch_lat"], last_phys["merch_long"], tx.merch_lat, tx.merch_long
                ))
                hours_since_phys = max(
                    (now - last_phys["trans_date_trans_time"]).total_seconds() / 3600.0, 0.0
                )
                speed_kmh = dist_prev_merch / hours_since_phys if hours_since_phys > 0 else 0.0
        features["dist_from_prev_merch_km"] = dist_prev_merch
        features["speed_kmh"] = float(speed_kmh)

        # Habitudes marchand / catégorie / heure
        nb_same_merchant = float((history["merchant"] == tx.merchant).sum())
        features["nb_prev_same_merchant"] = nb_same_merchant
        features["is_new_merchant"] = float(nb_same_merchant == 0)
        features["is_new_category"] = float(not (history["category"] == tx.category).any())
        features["category_share_card"] = float((history["category"] == tx.category).mean())
        # Même décalage que pour `hour` (même porteur = même domicile).
        hist_hours = (history["trans_date_trans_time"] + pd.Timedelta(hours=tz_offset)).dt.hour
        features["hour_share_card"] = float((hist_hours == hour).mean())

    # --- Vélocité multi-fenêtres : nb de tx / somme des montants de la
    # carte dans les fenêtres glissantes causales (jamais la tx courante) ---
    for label, delta in [
        ("1h", pd.Timedelta(hours=1)), ("6h", pd.Timedelta(hours=6)),
        ("24h", pd.Timedelta(hours=24)), ("7d", pd.Timedelta(days=7)),
    ]:
        if history.empty:
            nb, total_amt = 0.0, 0.0
        else:
            window = history[history["trans_date_trans_time"] >= now - delta]
            nb, total_amt = float(len(window)), float(window["amt"].sum())
        features[f"nb_tx_{label}"] = nb
        features[f"sum_amt_{label}"] = total_amt

    denom_24h = features["sum_amt_24h"] + tx.amt
    features["amt_share_of_24h"] = float(tx.amt / denom_24h) if denom_24h > 0 else 0.0
    # Même définition que le CSV d'entraînement : taux 24h incluant la tx courante
    # (ex. nb_1h=1, nb_24h=2 -> 1 / (3/24) = 8.0).
    rate_24h_per_hour = (features["nb_tx_24h"] + 1.0) / 24.0
    features["tx_accel_1h_vs_24h"] = float(features["nb_tx_1h"] / rate_24h_per_hour)

    # --- Métriques comportementales d'AFFICHAGE (bloc du dashboard) : calculées
    # sur le MÊME historique complet de la carte (History.csv + flux), puis
    # journalisées en JSON. N'entrent pas dans le modèle. Voir point 13.
    bm = compute_behavioral_metrics_block(tx, now, history)
    features["behavioral_metrics"] = bm

    return features


# ---------------------------------------------------------------------------
# Préparation des features
# ---------------------------------------------------------------------------
def preprocess_transaction(
    tx: TransactionInput, conn: sqlite3.Connection
) -> Tuple[pd.DataFrame, dict]:
    """Reproduit build_features de train1.py : calcule toutes les features
    candidates + category_code, puis ne garde que `feature_columns` (les
    features retenues par la sélection automatique) dans l'ORDRE exact
    chargé depuis l'artefact du modèle, jamais un ordre codé en dur ici."""
    raw = compute_behavioral_features(tx, conn)

    if tx.category not in category_map:
        logger.warning(
            "Catégorie inconnue reçue (%s) — encodée en NaN, hors distribution d'entraînement",
            tx.category,
        )
    raw[category_column] = category_map.get(tx.category, np.nan)

    # `behavioral_metrics` (dict, affichage dashboard) n'est pas une feature.
    out = pd.DataFrame([{k: v for k, v in raw.items() if k != "behavioral_metrics"}])
    missing = [c for c in feature_columns if c not in out.columns]
    if missing:
        raise ValueError(f"Colonnes de features manquantes après préparation : {missing}")

    # Même nettoyage que build_features (train1.py) : float + ±inf -> NaN.
    X = out[feature_columns].astype(float).replace([np.inf, -np.inf], np.nan)
    return X, raw


# ---------------------------------------------------------------------------
# Explication (SHAP), qualité des features et texte d'analyse — points 15 et 16
# ---------------------------------------------------------------------------
def compute_shap_factors(X: pd.DataFrame, top_n: int = 6) -> list:
    """Contributions SHAP du booster BRUT (log-odds) pour cette transaction.
    Retourne les `top_n` features les plus influentes. Ne lève jamais."""
    if RAW_BOOSTER is None:
        return []
    try:
        dm = xgb.DMatrix(X[feature_columns], feature_names=list(feature_columns))
        contribs = np.asarray(RAW_BOOSTER.predict(dm, pred_contribs=True))[0]
        vals = contribs[:-1].astype(float)
        total_abs = float(np.abs(vals).sum()) or 1.0
        factors = []
        for i in np.argsort(-np.abs(vals))[:top_n]:
            c = float(vals[i])
            if abs(c) < 1e-9:
                continue
            name = feature_columns[i]
            raw_v = X.iloc[0][name]
            factors.append({
                "feature": name,
                "label": FEATURE_LABELS.get(name, name),
                "value": float(raw_v) if _finite(float(raw_v)) else None,
                "display": format_feature_value(name, raw_v),
                "impact": round(c, 4),
                "direction": "up" if c > 0 else "down",
                "share": round(abs(c) / total_abs, 3),
            })
        return factors
    except Exception:
        logger.exception("Calcul SHAP impossible : explication vide pour cette transaction.")
        return []


def assess_feature_quality(X: pd.DataFrame) -> dict:
    """Compare chaque feature du modèle à la plage vue à l'entraînement.
    Une feature est signalée si sa valeur est < min ou > max de l'entraînement
    (le modèle n'a jamais vu ça) ou manquante de façon inhabituelle. Une valeur
    simplement élevée mais déjà vue n'est PAS signalée : les vraies fraudes
    vivent dans les queues de distribution.
    La fiabilité = 1 - somme(importance des features signalées)."""
    base = {"available": False, "reliability": None, "n_features": len(feature_columns),
            "n_flagged": 0, "flagged": []}
    if not FEATURE_STATS:
        return base
    try:
        flagged, penalty = [], 0.0
        for name in feature_columns:
            raw_v = X.iloc[0][name]
            st = FEATURE_STATS.get(name) or {}
            imp = float(FEATURE_IMPORTANCE.get(name, 0.0))
            status, weight = None, 0.0
            try:
                v = float(raw_v)
            except (TypeError, ValueError):
                v = float("nan")
            if not math.isfinite(v):
                if st.get("nan_rate", 0.0) <= 0.05:          # une valeur manquante inhabituelle
                    status, weight = "valeur manquante", 0.5
            elif st.get("n_unique", 0) > 2:
                if v < st["min"] or v > st["max"]:
                    status, weight = "hors plage d'entraînement", 1.0
            if status:
                penalty += imp * weight
                flagged.append({
                    "feature": name, "label": FEATURE_LABELS.get(name, name),
                    "value": v if math.isfinite(v) else None,
                    "display": format_feature_value(name, raw_v),
                    "status": status, "importance": round(imp, 4),
                    "train_range": [st.get("min"), st.get("max")] if st else None,
                })
        flagged.sort(key=lambda f: -f["importance"])
        return {"available": True, "reliability": round(max(0.0, 1.0 - penalty), 4),
                "n_features": len(feature_columns), "n_flagged": len(flagged), "flagged": flagged[:8]}
    except Exception:
        logger.exception("Évaluation de la qualité des features impossible.")
        return base


def build_model_rows(risk_score: float, p_model: float, decision: str, factors: list, quality: dict) -> list:
    """Lignes ajoutées au bloc « Métriques Comportementales » du dashboard."""
    level = {"BLOCK": "danger", "MANUAL_REVIEW": "warn", "ALLOW": "safe"}.get(decision, "")
    rows = [_bm_row("model_score", "Score de risque (modèle)",
                    f"{risk_score:.0f}/100 · probabilité modèle {p_model * 100:.2f} %", level)]
    if factors:
        txt = " · ".join(
            f"{'↑' if f['direction'] == 'up' else '↓'} {f['label']} ({f['display']})" for f in factors[:3])
        rows.append(_bm_row("model_factors", "Facteurs principaux du score", txt))
    if quality.get("available"):
        rel = quality["reliability"]
        lvl = "safe" if rel >= 0.9 else "warn" if rel >= 0.7 else "danger"
        rows.append(_bm_row(
            "feature_quality", "Fiabilité des features",
            f"{rel * 100:.0f} % · {quality['n_flagged']}/{quality['n_features']} hors plage d'entraînement", lvl))
        if quality["flagged"]:
            txt = " · ".join(f"{f['label']} ({f['display']})" for f in quality["flagged"][:3])
            rows.append(_bm_row("feature_quality_detail", "Features hors plage", txt,
                                "danger" if rel < 0.7 else "warn"))
    else:
        rows.append(_bm_row("feature_quality", "Fiabilité des features",
                            "Indisponible (ré-entraînez avec train2.py)"))
    return rows


def build_explanation(risk_score: float, p_model: float, decision: str, factors: list, quality: dict) -> str:
    """Texte d'analyse en français (affiché dans « Analyse Fraude » du dashboard)."""
    parts = [f"Score de risque {risk_score:.0f}/100 ({DECISION_FR.get(decision, decision)}) ; "
             f"probabilité de fraude estimée par le modèle : {p_model * 100:.2f} %."]
    ups = [f for f in factors if f["direction"] == "up"][:3]
    downs = [f for f in factors if f["direction"] == "down"][:2]
    if ups:
        parts.append("Facteurs qui augmentent le risque : "
                     + " ; ".join(f"{f['label']} ({f['display']})" for f in ups) + ".")
    if downs:
        parts.append("Facteurs qui le réduisent : "
                     + " ; ".join(f"{f['label']} ({f['display']})" for f in downs) + ".")
    if quality.get("available"):
        rel = quality["reliability"]
        if quality["n_flagged"]:
            names = ", ".join(f"{f['label']} ({f['display']})" for f in quality["flagged"][:3])
            txt = (f"Fiabilité des features : {rel * 100:.0f} % — valeurs hors de la plage vue à "
                   f"l'entraînement : {names}.")
            if rel < 0.7:
                txt += " Le score est à interpréter avec prudence."
            parts.append(txt)
        else:
            parts.append(f"Fiabilité des features : {rel * 100:.0f} % "
                         "(toutes dans la plage d'entraînement).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": model_version,
        "n_features": len(feature_columns),
        "threshold": round(threshold, 4),
        "decision_bands": {"allow_below": ALLOW_MAX_SCORE, "block_from": BLOCK_MIN_SCORE},
        "score_scale": SCORE_KNOTS_INFO,
        "shap_available": RAW_BOOSTER is not None,
        "feature_quality_available": bool(FEATURE_STATS),
    }


def record_failed_transaction(tx: "TransactionInput", reason: str) -> None:
    """Journalise un échec de traitement dans failed_transactions (panneau
    « échecs pipeline » du dashboard). Ne lève JAMAIS : un problème de journal
    ne doit pas masquer l'erreur d'origine."""
    try:
        payload = tx.model_dump() if hasattr(tx, "model_dump") else tx.dict()
        raw = json.dumps(payload, default=str, ensure_ascii=False)
        now = _utc_now_str()
        conn = sqlite3.connect(DB_PATH, timeout=10)
        try:
            with conn:
                conn.execute(
                    "INSERT INTO failed_transactions (transaction_id, channel, raw_payload_json, reason, "
                    "attempts, replay_count, created_at, dead_at, fingerprint) VALUES (?,?,?,?,?,?,?,?,?)",
                    (tx.transaction_id, tx.channel, raw, (reason or "")[:500], 1, 0, now, now,
                     hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]),
                )
        finally:
            conn.close()
    except Exception:
        logger.exception("Impossible de journaliser l'échec de %s", getattr(tx, "transaction_id", "?"))


INSERT_COLUMNS = [
    "transaction_id", "trans_date_trans_time", "processed_at", "channel",
    "cc_num", "cardholder_name", "merchant", "category", "amt", "city", "age",
    "lat", "long", "merch_lat", "merch_long",
    "dist_home_merch_km", "amt_zscore_card", "is_new_merchant",
    "is_weekend", "is_night",
    "nb_tx_1h", "nb_tx_24h",
    "fraud_probability", "anomaly_score", "decision", "model_version",
    "is_history", "behavioral_metrics",
    "top_factors", "llm_explanation", "model_probability",
]
INSERT_SQL = (
    f"INSERT OR REPLACE INTO transactions_scored ({', '.join(INSERT_COLUMNS)}) "
    f"VALUES ({', '.join('?' for _ in INSERT_COLUMNS)})"
)


@app.post("/score")
def score_transaction(tx: TransactionInput):
    conn = sqlite3.connect(DB_PATH, timeout=10)
    try:
        try:
            X, raw_features = preprocess_transaction(tx, conn)

            # Prédiction XGBoost calibré : SEUL modèle de décision. La décision
            # se lit sur le score de risque 0-100 (point 14), pas sur la
            # probabilité brute.
            model_prob = float(xgb_model.predict_proba(X)[:, 1][0])

            score_calib = artifact.get("score_calibration", {})
            p_floor = score_calib.get("p_floor", SCORE_P_FLOOR)
            t_low = score_calib.get("t_low", 0.001)
            t_high = score_calib.get("t_high", 0.85)
            risk_score = round(risk_score_from_proba(
                proba=model_prob,
                t_low=t_low,
                t_high=t_high,
                p_floor=p_floor
                ), 2)
            decision = decide(risk_score)
            fraud_prob = round(risk_score / 100.0, 4)   # valeur stockée : le dashboard l'affiche en « Risk Score »

            # Explication SHAP, qualité des features et lignes du dashboard.
            # Ne doit JAMAIS faire échouer le scoring.
            factors, quality, explanation = [], {"available": False}, ""
            try:
                factors = compute_shap_factors(X)
                quality = assess_feature_quality(X)
                explanation = build_explanation(risk_score, model_prob, decision, factors, quality)
                bm = raw_features["behavioral_metrics"]
                bm.setdefault("rows", []).extend(build_model_rows(risk_score, model_prob, decision, factors, quality))
                bm["model"] = {
                    "risk_score": risk_score, "model_probability": round(model_prob, 6),
                    "decision": decision, "factors": factors, "feature_quality": quality,
                }
            except Exception:
                logger.exception("Enrichissement (explication / qualité) ignoré pour %s", tx.transaction_id)

            processed_at = _utc_now_str()
            cardholder_name = tx.cardholder_name or f"Client {tx.cc_num[-4:]}"

            # Sauvegarde dans SQLite pour app.py — DOIT arriver après le
            # calcul des features (qui dépendent de l'historique déjà en
            # base), jamais avant.
            with conn:
                conn.execute(INSERT_SQL, (
                    tx.transaction_id,
                    raw_features["trans_time_utc"],
                    processed_at,
                    tx.channel,
                    tx.cc_num,
                    cardholder_name,
                    tx.merchant,
                    tx.category,
                    tx.amt,
                    tx.city,
                    tx.age,
                    tx.lat,
                    tx.long,
                    tx.merch_lat,
                    tx.merch_long,
                    raw_features["dist_home_merch_km"],
                    raw_features["amt_zscore_card"],
                    int(raw_features["is_new_merchant"]),
                    int(raw_features["is_weekend"]),
                    int(raw_features["is_night"]),
                    raw_features["nb_tx_1h"],
                    raw_features["nb_tx_24h"],
                    fraud_prob,
                    None,  # anomaly_score : pas de modèle non supervisé pour l'instant
                    decision,
                    model_version,
                    int(tx.is_history),
                    json.dumps(raw_features["behavioral_metrics"], ensure_ascii=False),
                    json.dumps(factors, ensure_ascii=False),
                    explanation,
                    model_prob,
                ))

            return {
                "transaction_id": tx.transaction_id,
                "fraud_probability": fraud_prob,          # = risk_score / 100
                "risk_score": risk_score,
                "model_probability": round(model_prob, 6),
                "decision": decision,
                "decision_bands": {"allow_below": ALLOW_MAX_SCORE, "block_from": BLOCK_MIN_SCORE},
                "threshold": round(threshold, 4),
                "top_factors": factors,
                "feature_quality": quality,
                "explanation": explanation,
                "behavioral_metrics": raw_features["behavioral_metrics"],
            }

        except HTTPException:
            raise
        except (KeyError, ValueError) as e:
            logger.warning("Requête /score invalide : %s", e)
            record_failed_transaction(tx, f"invalid: {e}")
            raise HTTPException(status_code=400, detail=f"Donnée invalide ou manquante : {e}")
        except Exception:
            logger.exception("Erreur interne lors du scoring de %s", tx.transaction_id)
            record_failed_transaction(tx, "internal_error")
            raise HTTPException(status_code=500, detail="Erreur interne du moteur de scoring")
    finally:
        conn.close()


class FeedbackInput(BaseModel):
    """Corps envoyé par le dashboard (sendModelFeedback) : le libellé est
    'true_positive' (fraude confirmée) ou 'false_positive' (légitime)."""

    transaction_id: str
    feedback: str
    analyst: Optional[str] = None          # injecté par le proxy du dashboard
    closing_reason: Optional[str] = None


FEEDBACK_LABELS = {
    "true_positive": "TP", "false_positive": "FP", "TP": "TP", "FP": "FP",
}


@app.post("/api/v1/feedback")
def record_feedback(fb: FeedbackInput):
    """Enregistre le verdict d'un analyste dans feedback_history (lu par le
    dashboard : verdict par carte, KPI précision/rappel). Ce script reste le
    seul écrivain de fraud.db."""
    label = FEEDBACK_LABELS.get(fb.feedback.strip()) or FEEDBACK_LABELS.get(fb.feedback.strip().upper())
    if label is None:
        raise HTTPException(
            status_code=400,
            detail="feedback doit valoir 'true_positive' ou 'false_positive'",
        )

    conn = sqlite3.connect(DB_PATH, timeout=10)
    try:
        row = conn.execute(
            "SELECT cc_num FROM transactions_scored WHERE transaction_id = ?",
            (fb.transaction_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Transaction introuvable")
        with conn:
            conn.execute(
                "INSERT INTO feedback_history "
                "(transaction_id, cc_num, label, analyst, closing_reason, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    fb.transaction_id, row[0], label,
                    fb.analyst or "analyst",
                    fb.closing_reason or "analyst_feedback",
                    _utc_now_str(),
                ),
            )
        logger.info("Feedback %s enregistré pour %s", label, fb.transaction_id)
        return {"status": "success", "transaction_id": fb.transaction_id, "label": label}
    except HTTPException:
        raise
    except sqlite3.Error:
        logger.exception("Erreur SQLite lors de l'enregistrement du feedback")
        raise HTTPException(status_code=500, detail="Erreur interne lors de l'enregistrement du feedback")
    finally:
        conn.close()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)