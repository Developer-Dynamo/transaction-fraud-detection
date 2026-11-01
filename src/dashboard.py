#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dashboard Service — FraudGuard ML Transaction Viewer (lecture seule)
======================================================================
Service FastAPI INDÉPENDANT du moteur de scoring FraudGuard ML (agent
de scoring temps réel : XGBoost + IsolationForest).

Rôle :
  - Lit la MÊME base SQLite que l'agent de scoring (fraud.db), en connexion
    STRICTEMENT LECTURE SEULE (mode=ro), jamais d'écriture directe.
  - Sert le dashboard HTML/JS + la page de login.
  - Expose ses propres routes /dashboard/transactions, /stats/*, /debug/* —
    copies en lecture seule des fonctions équivalentes de l'agent de scoring
    (aucune dépendance ML : pas de xgboost/scikit-learn/pandas ici).
  - Ne fait AUCUNE écriture en base. Le clic Fraude confirmée / Transaction
    légitime (POST /api/v1/feedback) est transmis (proxy HTTP) tel quel à
    l'agent de scoring, qui reste le SEUL écrivain de la base et le seul à
    ré-entraîner / recalibrer les modèles.

Pourquoi ce découplage :
  - L'agent de scoring embarque XGBoost + IsolationForest + pipeline de
    features temps réel — lourd et sujet à redémarrages (déploiements,
    ré-entraînement périodique, crash sur une dépendance externe).
  - Le dashboard doit rester disponible pour les analystes fraude même si
    l'agent de scoring redémarre.
  - Un seul écrivain sur la base SQLite = zéro risque de "database is locked"
    causé par ce script (mode WAL, déjà activé côté agent).

Horodatage :
  - Toutes les écritures de ce service (sessions d'authentification,
    tentatives de connexion, création de comptes) utilisent UTC via
    datetime.now(timezone.utc) / time.time(). C'est cohérent avec les
    colonnes processed_at / trans_date_trans_time de fraud.db, qui sont
    elles-mêmes écrites en UTC par l'agent de scoring, et avec les
    fonctions SQL datetime('now', ...) de SQLite (UTC par défaut).
    Aucune conversion de fuseau horaire n'est appliquée : tout reste en UTC
    de bout en bout, y compris pour l'affichage (le formatage en heure
    locale, si besoin, doit être fait côté client dans le JS, pas ici).

Prérequis :
  - L'agent de scoring doit avoir déjà tourné au moins une fois (pour créer
    fraud.db et activer PRAGMA journal_mode=WAL).
  - Même variable d'environnement FRAUD_DATA_DIR que l'agent, pour pointer
    vers le même fichier de base.

Schéma attendu de fraud.db (colonnes SQL explicites — contrairement à
l'ancien dashboard SOC qui stockait un blob JSON par alerte) :

  CREATE TABLE transactions_scored (      -- créée / migrée par score1.py (init_db)
      transaction_id          TEXT PRIMARY KEY,
      trans_date_trans_time   TEXT,      -- horodatage de la transaction (UTC)
      processed_at            TEXT,      -- horodatage du scoring (UTC)
      channel                 TEXT,      -- 'online' | 'pos'
      cc_num                  TEXT,      -- numéro de carte
      cardholder_name         TEXT,
      merchant                TEXT,
      category                TEXT,
      amt                     REAL,
      city                    TEXT,      -- ville du domicile de la carte
      age                     REAL,
      lat                     REAL, long REAL,           -- position titulaire (domicile)
      merch_lat               REAL, merch_long REAL,     -- position marchand
      dist_home_merch_km      REAL,      -- exposé par l'API sous le nom "distance_km"
      amt_zscore_card         REAL,      -- exposé sous le nom "amt_zscore"
      is_new_merchant         INTEGER,   -- exposé sous le nom "new_merchant"
      is_weekend              INTEGER,   -- 0/1
      is_night                INTEGER,   -- 0/1
      nb_tx_1h                REAL,      -- exposé sous le nom "txn_count_1h"
      nb_tx_24h               REAL,      -- exposé sous le nom "txn_count_24h"
      fraud_probability       REAL,      -- sortie XGBoost calibré, 0.0–1.0
      anomaly_score           REAL,      -- NULL (pas de modèle non supervisé)
      decision                TEXT,      -- 'BLOCK' | 'ALLOW' (MANUAL_REVIEW non produit)
      top_factors             TEXT,      -- NULL (pas d'explicabilité SHAP pour l'instant)
      recommended_actions     TEXT,      -- NULL
      llm_explanation         TEXT,      -- NULL
      model_version           TEXT,
      content_hash            TEXT,
      is_history              INTEGER,   -- 1 = ligne de History.csv (mémoire de carte, masquée ici)
      behavioral_metrics      TEXT       -- JSON calculé par l'API de scoring (bloc « Métriques Comportementales »)
  );
  -- Anciennes colonnes supprimées (aucune source de données) : state, city_pop,
  -- job, foreign_transaction. Le dashboard les renvoie à NULL via TX_COLUMNS.

  CREATE TABLE feedback_history (
      id              INTEGER PRIMARY KEY AUTOINCREMENT,
      transaction_id  TEXT,
      cc_num          TEXT,
      label           TEXT,     -- 'TP' (fraude confirmée) | 'FP' (transaction légitime)
      analyst         TEXT,
      closing_reason  TEXT,
      timestamp       TEXT
  );

  CREATE TABLE failed_transactions (       -- ex "dead_letters" : échecs pipeline
      id                INTEGER PRIMARY KEY AUTOINCREMENT,
      transaction_id    TEXT,
      channel           TEXT,
      raw_payload_json  TEXT,
      reason            TEXT,
      attempts          INTEGER,
      replay_count      INTEGER,
      created_at        TEXT,
      dead_at           TEXT,
      fingerprint       TEXT
  );

Variables d'environnement :
  - FRAUD_DATA_DIR : dossier contenant fraud.db (doit être IDENTIQUE à celui
    de l'agent de scoring). Défaut : "data" (relatif au cwd).
  - DASHBOARD_PORT : port d'écoute de ce service. Défaut : 8002.
  - AGENT_API_URL  : URL de base de l'agent de scoring, utilisée UNIQUEMENT
    pour transmettre (proxy) le POST /api/v1/feedback (écriture).
    Défaut : http://localhost:8001.

Dépendances : fastapi, uvicorn, httpx. Aucune dépendance ML (pas de xgboost,
scikit-learn, pandas requis dans CE service).
"""

import os
import re
import json
import time
import sqlite3
import logging
import asyncio
import threading
import unicodedata
import math
import statistics
from collections import Counter
from functools import lru_cache
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

# ============================================================================
# Logging
# ============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, LOG_LEVEL),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fraudguard_dashboard_service")

# ============================================================================
# Configuration
# ============================================================================
# IMPORTANT : même valeur que côté agent de scoring (FRAUD_DATA_DIR), pour
# pointer vers exactement le même fichier fraud.db.
DATA_DIR = Path(os.getenv("FRAUD_DATA_DIR", "data")).resolve()
DB_PATH = DATA_DIR / "fraud.db"

DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8002"))

# URL de l'agent de scoring — utilisée UNIQUEMENT pour proxifier les écritures
# (feedback HITL). Ce service ne ré-entraîne jamais lui-même les modèles.
AGENT_API_URL = os.getenv("AGENT_API_URL", "http://localhost:8001").rstrip("/")

if not DB_PATH.exists():
    logger.warning(
        f"Base introuvable à {DB_PATH}. Vérifiez que FRAUD_DATA_DIR pointe vers "
        f"le même dossier que l'agent de scoring, et que celui-ci a déjà tourné "
        f"au moins une fois (init_db())."
    )

logger.info(f"FraudGuard dashboard — DB (lecture seule)={DB_PATH} | AGENT_API_URL={AGENT_API_URL}")

# ============================================================================
# Connexion SQLite en LECTURE SEULE
# ============================================================================
def _ro_connect(retries: int = 5, delay: float = 0.15) -> sqlite3.Connection:
    """
    Ouvre une connexion SQLite strictement lecture seule (mode=ro au niveau du
    driver, pas juste par convention). Échoue explicitement si le fichier
    n'existe pas encore. Retry léger en cas de contention transitoire
    (checkpoint WAL en cours côté agent de scoring).
    """
    uri = f"file:{DB_PATH}?mode=ro"
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=5)
            conn.execute("PRAGMA busy_timeout=5000")
            return conn
        except sqlite3.OperationalError as e:
            last_exc = e
            time.sleep(delay)
    logger.error(f"Impossible d'ouvrir la base en lecture seule après {retries} tentatives : {last_exc}")
    raise last_exc

def safe_db_call(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except sqlite3.Error as e:
        logger.error(f"SQLite error (lecture): {e}")
        raise

def _parse_ts(value: Optional[str]) -> Optional[datetime]:
    """Parse un timestamp ISO (avec ou sans timezone) en datetime naïf UTC. Retourne None si invalide."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except (ValueError, TypeError):
        return None

def _clean_merchant_name(raw: Optional[str]) -> Optional[str]:
    """Nettoie un nom de marchand brut issu du dataset ML (préfixe 'fraud_'
    laissé par la génération synthétique des données d'entraînement) pour un
    affichage professionnel dans le dashboard."""
    if not raw:
        return raw
    name = str(raw).strip()
    if name.lower().startswith("fraud_"):
        name = name[len("fraud_"):]
    # Les noms sont généralement déjà séparés par des espaces (ex: "Kiehn Inc")
    # côté dataset synthétique ; on ne réinjecte pas d'espaces artificiels dans
    # des noms en camelCase pour ne pas casser de vraies marques (TotalEnergies).
    return name.strip() or raw


def _latest_feedback_map(conn: sqlite3.Connection, transaction_ids: List[str]) -> Dict[str, str]:
    """Retourne, pour chaque transaction_id fourni, le dernier label analyste
    ('TP' = fraude confirmée / 'FP' = légitime) enregistré dans
    feedback_history, s'il existe. Utilisé pour distinguer la Décision
    automatique du modèle de la Qualification réelle par un analyste."""
    ids = [t for t in (transaction_ids or []) if t]
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    try:
        rows = conn.execute(f"""
            SELECT fh.transaction_id, fh.label
            FROM feedback_history fh
            INNER JOIN (
                SELECT transaction_id, MAX(id) AS max_id
                FROM feedback_history
                WHERE transaction_id IN ({placeholders})
                GROUP BY transaction_id
            ) latest ON latest.transaction_id = fh.transaction_id AND latest.max_id = fh.id
        """, ids).fetchall()
    except sqlite3.OperationalError:
        # Table absente / pas encore créée côté agent : pas de statut analyste.
        return {}
    return {tid: label for tid, label in rows}


def _mask_card(cc_num: Optional[str]) -> str:
    """Masque un numéro de carte pour l'affichage (garde les 4 derniers chiffres)."""
    if not cc_num:
        return "•••• ••••"
    digits = "".join(ch for ch in str(cc_num) if ch.isdigit())
    if len(digits) < 4:
        return "•••• " + digits
    return "•••• •••• •••• " + digits[-4:]

TX_COLUMNS = """
    transaction_id, trans_date_trans_time, processed_at, channel, cc_num,
    cardholder_name, merchant, category, amt, city, NULL AS state, lat, long,
    merch_lat, merch_long, dist_home_merch_km AS distance_km, NULL AS city_pop,
    NULL AS job, age, is_weekend,
    is_night, nb_tx_1h AS txn_count_1h, nb_tx_24h AS txn_count_24h,
    amt_zscore_card AS amt_zscore, is_new_merchant AS new_merchant,
    NULL AS foreign_transaction, fraud_probability, anomaly_score, decision,
    top_factors, recommended_actions, llm_explanation, model_version,
    content_hash
"""
# Les colonnes SQL de score1.py (dist_home_merch_km, nb_tx_1h, ...) sont exposées
# sous les anciens noms d'API (distance_km, txn_count_1h, ...) grâce aux alias
# ci-dessus : _row_to_transaction() et le JS du front restent inchangés.
# state / city_pop / job / foreign_transaction n'existent plus côté agent de
# scoring : ils sont renvoyés à NULL (affichés vides / False).

# ============================================================================
# Explication XAI en langage naturel
# ----------------------------------------------------------------------------
# L'agent de scoring écrit llm_explanation sous forme de liste technique
# (« Score de risque 11/100 (AUTORISÉE) ; probabilité ... : 0.00 %. Facteurs ... »).
# Le dashboard la réécrit en UN paragraphe fluide, sans « ; », sans « : »,
# sans tirets ni « -- », à partir de top_factors (si présents) ou, à défaut,
# en relisant le texte brut de l'agent.
# ============================================================================
_XAI_DECISION_TXT = {
    "ALLOW": "autorisée",
    "MANUAL_REVIEW": "placée en revue manuelle par un analyste",
    "BLOCK": "bloquée",
}

def _xai_num(txt) -> str:
    """Virgule décimale à la française et signe moins écrit en toutes lettres."""
    txt = re.sub(r"(?<=\d)\.(?=\d)", ",", str(txt))
    txt = re.sub(r"(?<![\w])[-−–]\s?(?=\d)", "moins ", txt)
    return txt

def _xai_float(txt) -> Optional[float]:
    """Premier nombre contenu dans un affichage (« +13.94 σ », « 2 394.70 € », « 98 % »)."""
    t = re.sub(r"(?<=\d)\s(?=\d)", "", str(txt or ""))
    m = re.search(r"([+\-−–]?)\s?(\d+(?:[.,]\d+)?)", t)
    if not m:
        return None
    v = float(m.group(2).replace(",", "."))
    return -v if m.group(1) in ("-", "−", "–") else v

def _xai_join(items: List[str]) -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " et " + items[-1]

def _xai_parse_raw(raw: str) -> List[Dict[str, Any]]:
    """Relit l'ancien texte technique de l'agent (si top_factors est vide) et
    renvoie la liste de facteurs {label, display, direction}."""
    out: List[Dict[str, Any]] = []
    if not raw:
        return out
    def _items(title, direction):
        mm = re.search(title + r"\s*:\s*(.*?)\.\s+(?=Facteurs|Fiabilit|$)", raw + " ", flags=re.S)
        if not mm:
            return
        for part in re.split(r"\s;\s", mm.group(1)):
            part = part.strip().rstrip(".")
            pm = re.match(r"^(.*?)\s*\((.*)\)$", part)
            if pm:
                out.append({"label": pm.group(1).strip(), "display": pm.group(2).strip(), "direction": direction})
            elif part:
                out.append({"label": part, "display": "", "direction": direction})
    _items(r"Facteurs qui augmentent le risque", "up")
    _items(r"Facteurs qui le r[ée]duisent", "down")
    return out

def _xai_window(low: str) -> str:
    """Fenêtre temporelle citée dans le libellé d'un facteur (« sur 1 h », « sur 24 h », « sur 7 jours »)."""
    if "7 jours" in low or "7 j" in low:
        return "les 7 derniers jours"
    if "6 h" in low:
        return "les 6 dernières heures"
    if "1 h" in low:
        return "la dernière heure"
    return "les dernières 24 heures"

def _xai_during(win: str) -> str:
    """« les dernières 24 heures » -> « au cours des dernières 24 heures »."""
    return "au cours " + ("des " + win[4:] if win.startswith("les ") else "de " + win)

def _xai_clause(label: str, display: str, direction: str = "") -> str:
    """Traduit un facteur technique en une proposition simple, compréhensible
    par un non spécialiste (commence par un sujet, en minuscule)."""
    lab = (label or "").strip()
    low = lab.lower()
    disp = _xai_num(display).strip() if display else ""
    val = _xai_float(display)

    if "σ" in (display or "") or "norme" in low or "écart" in low:
        if val is None:
            return "le montant s'écarte de ce qui est habituel pour ce type d'achat"
        if val >= 2:
            return ("le montant est bien plus élevé que d'habitude pour ce type d'achat, "
                    f"avec un écart d'environ {round(abs(val))} fois la variation normale")
        if val <= -2:
            return "le montant est nettement plus faible que d'habitude pour ce type d'achat"
        return "le montant reste conforme à ce qui est habituel pour ce type d'achat"
    if "part du montant" in low or "part de" in low:
        # 100 % = aucune autre dépense sur 24 h : on le dit clairement au lieu d'un
        # « 100 % de toutes les dépenses » qui contredit « 0,00 € dépensé ».
        if val is not None and val >= 99.5:
            return "aucune autre dépense n'avait été enregistrée sur cette carte au cours des dernières 24 heures"
        return (f"cette transaction représente {disp} des dépenses de la carte sur les dernières 24 heures, elle comprise"
                if disp else "cette transaction pèse lourd dans les dépenses récentes de la carte")
    if "somme" in low or "dépensé" in low or "depense" in low:
        win = _xai_window(low)
        if val is not None and val <= 0:
            return f"aucune autre dépense n'avait été enregistrée sur cette carte {_xai_during(win)}"
        return (f"avant cette transaction, la carte avait dépensé {disp} au total sur {win}"
                if disp else f"la carte avait déjà beaucoup dépensé sur {win}")
    if low.startswith("transactions sur"):
        win = _xai_window(low)
        if val is not None and val <= 0:
            return f"aucune autre transaction n'avait été enregistrée sur cette carte {_xai_during(win)}"
        if val is not None and val == 1:
            return f"une seule autre transaction avait été enregistrée sur cette carte {_xai_during(win)}"
        if val is not None:
            return f"{int(val)} autres transactions avaient été enregistrées sur cette carte {_xai_during(win)}"
        return "le rythme des transactions récentes a été pris en compte"
    if "affinité" in low and "heure" in low:
        return f"la carte est habituellement utilisée à cette heure, pour {disp} de ses achats" if disp else "la carte est habituellement utilisée à cette heure"
    if "affinité" in low:
        return f"ce type de marchand correspond à {disp} des achats habituels de la carte" if disp else "ce type de marchand est familier pour cette carte"
    if "heure" in low:
        return f"la transaction a eu lieu vers {disp}" if disp else "l'heure de la transaction a été prise en compte"
    if low == "montant":
        if disp and (direction or "").lower() == "up":
            return f"le montant de {disp} est élevé par rapport aux achats habituels"
        if disp and (direction or "").lower() == "down":
            return f"le montant de {disp} reste modéré"
        return "le montant de la transaction a été pris en compte"
    if "montant" in low:
        return f"le montant de la transaction est de {disp}" if disp else "le montant de la transaction a été pris en compte"
    if "vélocité" in low or "velocite" in low or "nombre" in low or "txn" in low:
        return f"le nombre de transactions récentes sur cette carte est de {disp}" if disp else "le rythme des transactions récentes a été pris en compte"
    if "distance" in low:
        return f"le lieu d'achat se trouve à {disp} du domicile du titulaire" if disp else "la distance avec le domicile a été prise en compte"
    if "nuit" in low:
        return "la transaction a eu lieu de nuit"
    if "nouveau" in low and "marchand" in low:
        return "ce marchand était nouveau pour cette carte"
    base = lab[:1].lower() + lab[1:]
    return f"{base} vaut {disp}" if disp else base

_XAI_STEPS = ["D'abord", "Ensuite", "Enfin", "Aussi"]

def _xai_block(clauses: List[str], singular: str, plural: str) -> str:
    """singular / plural : début de phrase, ex. « Un point a rassuré le modèle »."""
    if not clauses:
        return ""
    if len(clauses) == 1:
        return f"{singular}, car {clauses[0]}."
    parts = [plural + "."]
    for k, c in enumerate(clauses):
        parts.append(f"{_XAI_STEPS[min(k, len(_XAI_STEPS) - 1)]}, {c}.")
    return " ".join(parts)

def _xai_sanitize(txt: str) -> str:
    """Garde-fou : aucun « ; », « : », « -- » ni tiret de liste dans le texte final."""
    txt = txt.replace("--", ", ").replace("—", ", ").replace("–", ", ")
    txt = re.sub(r"\s+-\s+", ", ", txt)
    txt = txt.replace(";", ",").replace(":", ",")
    txt = re.sub(r"\s*,\s*,+", ",", txt)
    txt = re.sub(r"\s{2,}", " ", txt)
    return txt.strip()

def build_natural_explanation(tx: Dict[str, Any]) -> str:
    """Explication simple pour l'analyste : le verdict, puis UNIQUEMENT les
    facteurs les plus importants, traduits en français courant. Ni score, ni
    probabilité, ni indicateur technique."""
    raw = tx.get("llm_explanation") or ""
    decision = (tx.get("decision") or "").upper()
    dec_txt = _XAI_DECISION_TXT.get(decision)

    factors = [f for f in (tx.get("top_factors") or []) if isinstance(f, dict)]
    if not factors:
        factors = _xai_parse_raw(raw)
    if not factors and not dec_txt:
        return "Aucune analyse disponible pour cette transaction."

    # Seuls les facteurs les plus influents : tri par impact absolu quand il existe
    def _imp(f):
        try:
            return abs(float(f.get("impact")))
        except (TypeError, ValueError):
            return 0.0
    if any(_imp(f) for f in factors):
        factors = sorted(factors, key=_imp, reverse=True)
    factors = factors[:4]

    up, down, seen = [], [], set()
    has_z = any("σ" in str(g.get("display") or "") for g in factors)
    for f in factors:
        if has_z and (f.get("label") or "").strip().lower() == "montant":
            continue            # déjà couvert par « le montant est bien plus élevé que d'habitude »
        c = _xai_clause(f.get("label") or f.get("feature") or "", f.get("display") or "",
                        f.get("direction") or "")
        if not c or c in seen:      # deux facteurs peuvent dire la même chose
            continue
        seen.add(c)
        (up if (f.get("direction") or "").lower() == "up" else down).append(c)
    up, down = up[:3], down[:3]

    # Phrase d'ouverture : la transaction et ce qui lui est arrivé
    head = "Cette transaction"
    amt = tx.get("amt")
    try:
        if amt:
            head += " de " + _xai_num(f"{float(amt):,.2f}".replace(",", " ")) + " €"
    except (TypeError, ValueError):
        pass
    if tx.get("merchant"):
        head += f" chez {tx['merchant']}"
    parts = [head + (f" a été {dec_txt}." if dec_txt else " a été analysée par le modèle.")]

    if up:
        parts.append(_xai_block(up,
            "Un point a attiré l'attention du modèle",
            "Plusieurs points ont attiré l'attention du modèle"))
    if down:
        if up:
            parts.append(_xai_block(down,
                "À l'inverse, un point a rassuré le modèle",
                "À l'inverse, plusieurs points ont rassuré le modèle"))
        else:
            parts.append(_xai_block(down,
                "Un point a rassuré le modèle",
                "Le modèle s'est appuyé sur plusieurs éléments rassurants"))
    if not up and not down:
        parts.append("Aucun élément particulier ne ressort de cette analyse.")

    if decision == "ALLOW":
        parts.append("En résumé, "
            + ("malgré quelques signaux à surveiller, l'ensemble reste cohérent avec les habitudes de la carte et aucune action n'est nécessaire."
               if up else "rien d'anormal n'a été détecté et cette transaction correspond aux habitudes de la carte."))
    elif decision == "MANUAL_REVIEW":
        parts.append("En résumé, ces signaux justifient une vérification par un analyste avant de valider ou de refuser la transaction.")
    elif decision == "BLOCK":
        parts.append("En résumé, ces signaux sont trop inhabituels pour cette carte, c'est pourquoi la transaction a été bloquée par précaution.")

    return _xai_sanitize(" ".join(parts))

# Lignes de « Métriques Comportementales » à ne plus afficher (déjà couvertes
# par l'analyse IA ou sans intérêt pour l'analyste).
_BM_HIDDEN_RE = re.compile(r"score de risque|facteurs? principa|verdict|feature", re.I)

def _bm_strip_hidden(payload):
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        payload["rows"] = [r for r in payload["rows"]
                           if not _BM_HIDDEN_RE.search(f"{r.get('key','')} {r.get('label','')}")]
    return payload

def _row_to_transaction(row) -> Dict[str, Any]:
    (transaction_id, trans_dt, processed_at, channel, cc_num, cardholder_name,
     merchant, category, amt, city, state, lat, lng, merch_lat, merch_long,
     distance_km, city_pop, job, age, is_weekend, is_night, txn_1h, txn_24h,
     amt_z, new_merchant, foreign_tx, fraud_prob, anomaly_score, decision,
     top_factors, recommended_actions, llm_explanation, model_version,
     content_hash) = row
    try:
        top_factors = json.loads(top_factors) if top_factors else []
    except Exception:
        top_factors = []
    try:
        recommended_actions = json.loads(recommended_actions) if recommended_actions else []
    except Exception:
        recommended_actions = []
    _mc = _snap_tx_city(merch_lat, merch_long)   # (nom, lat, long) du lieu de transaction
    _tx = {
        "transaction_id": transaction_id,
        "trans_date_trans_time": trans_dt,
        "processed_at": processed_at,
        "channel": channel,
        "cc_num": cc_num,
        "cc_num_masked": _mask_card(cc_num),
        "cardholder_name": cardholder_name,
        "merchant": _clean_merchant_name(merchant),
        "category": category,
        "amt": amt or 0,
        "city": city,                                   # ville de naissance / domicile (fixe)
        "country": _geo_country(city),
        "merch_city": _mc[0] if _mc else None,          # lieu de transaction (aléatoire, monde entier)
        "merch_country": _geo_country(_mc[0]) if _mc else None,
        "state": state,
        "lat": lat, "long": lng,
        "merch_lat": merch_lat, "merch_long": merch_long,
        "distance_km": distance_km,
        "city_pop": city_pop,
        "job": job,
        "age": age,
        "is_weekend": bool(is_weekend),
        "is_night": bool(is_night),
        "txn_count_1h": txn_1h,
        "txn_count_24h": txn_24h,
        "amt_zscore": amt_z,
        "new_merchant": bool(new_merchant),
        "foreign_transaction": bool(foreign_tx),
        "fraud_probability": fraud_prob or 0,
        "anomaly_score": anomaly_score or 0,
        "decision": (decision or "unknown"),
        "top_factors": top_factors,
        "recommended_actions": recommended_actions,
        "llm_explanation": llm_explanation or "",
        "model_version": model_version,
        "content_hash": content_hash,
        "analyst_status": None,  # rempli par get_transactions_for_dashboard / get_scored_transaction
    }
    try:
        _tx["llm_explanation"] = build_natural_explanation(_tx)
    except Exception:
        pass
    return _tx

# ============================================================================
# Requêtes — transactions scorées / échecs pipeline (copies read-only de
# l'agent de scoring FraudGuard ML)
# ============================================================================
def get_scored_transactions(limit: int = 20, channel: str = None, hours: int = None) -> List[Dict]:
    def _q():
        conn = _ro_connect()
        try:
            if hours is None:
                query = "SELECT transaction_id, channel, cardholder_name, processed_at, content_hash FROM transactions_scored_live WHERE 1=1"
                params = []
            else:
                query = "SELECT transaction_id, channel, cardholder_name, processed_at, content_hash FROM transactions_scored_live WHERE processed_at >= datetime('now', ?)"
                params = [f'-{hours} hours']
            if channel:
                query += " AND channel = ?"
                params.append(channel)
            query += " ORDER BY processed_at DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(query, params).fetchall()
            return [{"transaction_id": r[0], "channel": r[1], "cardholder_name": r[2],
                     "processed_at": r[3], "content_hash": r[4][:12]+"..." if r[4] else ""} for r in rows]
        finally:
            conn.close()
    return safe_db_call(_q)

def get_scored_transaction(transaction_id: str) -> Optional[Dict]:
    def _q():
        conn = _ro_connect()
        try:
            cur = conn.execute(f"SELECT {TX_COLUMNS} FROM transactions_scored_live WHERE transaction_id=?", (transaction_id,))
            row = cur.fetchone()
            fb_map = _latest_feedback_map(conn, [transaction_id]) if row else {}
        finally:
            conn.close()
        if not row:
            return None
        tx = _row_to_transaction(row)
        tx["analyst_status"] = fb_map.get(transaction_id)
        return tx
    return safe_db_call(_q)

def get_related_transactions(cc_num: str, exclude_transaction_id: str = None, limit: int = 20) -> List[Dict]:
    """Historique récent des transactions de la MÊME carte (remplace les 'événements
    associés' du dashboard SOC) — utile pour repérer un pattern de vélocité anormale."""
    if not cc_num:
        return []
    def _q():
        conn = _ro_connect()
        try:
            query = f"SELECT {TX_COLUMNS} FROM transactions_scored_live WHERE cc_num = ?"
            params = [cc_num]
            if exclude_transaction_id:
                query += " AND transaction_id != ?"
                params.append(exclude_transaction_id)
            query += " ORDER BY trans_date_trans_time DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.close()
        return [_row_to_transaction(r) for r in rows]
    return safe_db_call(_q)

def get_card_verdict(cc_num: str) -> Dict[str, Any]:
    """
    Verdict de fraude d'une carte bancaire, basé sur l'historique RÉEL des
    fraudes confirmées par les analystes pour cette carte.

    Source de la donnée : feedback_history (alimentée par l'agent de scoring
    à chaque clic Fraude confirmée / Transaction légitime — colonnes cc_num
    et label déjà présentes, label ∈ {"TP","FP"}). C'est le signal le plus
    fiable disponible : une confirmation humaine réelle, pas une probabilité
    modèle.

    Règle de décision (seuils à ajuster avec l'équipe fraude selon le retour
    terrain) :
      - >= 2 fraudes confirmées distinctes -> "Carte compromise confirmée" (danger)
      - 1 fraude confirmée                 -> "Suspecte - à surveiller"    (warn)
      - 0 fraude mais du feedback existant  -> "Pas d'indicateur"          (safe)
      - Aucun feedback pour cette carte     -> "Pas d'historique"          (unknown)
    """
    if not cc_num:
        return {"cc_num": cc_num, "tp_count": 0, "fp_count": 0, "total_feedback": 0,
                "verdict": "Pas d'historique", "level": "unknown"}

    def _q():
        conn = _ro_connect()
        try:
            rows = conn.execute(
                "SELECT label FROM feedback_history WHERE cc_num = ?",
                (cc_num,)
            ).fetchall()
        finally:
            conn.close()
        return rows

    rows = safe_db_call(_q)
    tp_count = sum(1 for r in rows if (r[0] or "").upper() == "TP")
    fp_count = sum(1 for r in rows if (r[0] or "").upper() == "FP")
    total = len(rows)

    if tp_count >= 2:
        verdict, level = "Carte compromise confirmée", "danger"
    elif tp_count == 1:
        verdict, level = "Suspecte - à surveiller", "warn"
    elif total > 0:
        verdict, level = "Pas d'indicateur", "safe"
    else:
        verdict, level = "Pas d'historique", "unknown"

    return {
        "cc_num": cc_num,
        "tp_count": tp_count,
        "fp_count": fp_count,
        "total_feedback": total,
        "verdict": verdict,
        "level": level,
    }

# ============================================================================
# Métriques comportementales — SOURCE PRINCIPALE : la colonne `behavioral_metrics`
# (JSON) écrite par l'API de scoring (API_Score2.py) au moment de
# l'enrichissement de chaque transaction. L'API les calcule sur l'historique
# COMPLET de la carte : lignes de History.csv (is_history = 1) + transactions
# live déjà journalisées, strictement antérieures à la transaction. Le dashboard
# se contente de les afficher (aucun recalcul, donc cohérence garantie avec l'API).
#
# REPLI (transactions scorées avant cette version, colonne NULL, ou API non
# mise à jour) : recalcul à la volée avec les MÊMES règles, mais sur l'historique
# de la table de base `transactions_scored` (qui inclut les lignes History.csv),
# et non plus sur la vue `transactions_scored_live` qui les masquait — c'était la
# cause du « Historique insuffisant (0 txn) » systématique.
# ============================================================================
_BM_HISTORY_LIMIT = 20000

# Colonne OPTIONNELLE de transactions_scored, détectée dynamiquement
# (PRAGMA table_info) : code MCC réel. Sinon la catégorie marchand `category` est utilisée.
_BM_MCC_COL = "mcc"

def _bm_num(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None

def _bm_fmt_duration(seconds):
    s = max(0, int(round(seconds)))
    if s < 60:
        return f"{s} s"
    if s < 3600:
        return f"{s // 60} min {s % 60:02d} s"
    if s < 86400:
        return f"{s // 3600} h {(s % 3600) // 60:02d} min"
    return f"{s // 86400} j {(s % 86400) // 3600} h"

def _bm_fmt_expected(x):
    if x < 0.01:
        return "<0.01"
    return f"{x:.2f}" if x < 10 else f"{x:.1f}"

def _bm_row(key, label, value, level=""):
    return {"key": key, "label": label, "value": value, "level": level}

def _bm_velocity(cur_ts, hist):
    """Nombre de transactions (incluant l'actuelle) sur la fenêtre glissante de 24 h, comparé
    au rythme habituel de la carte (calculé sur l'historique antérieur à 24 h pour ne pas être
    contaminé par une éventuelle rafale en cours)."""
    ts_list = [h["ts"] for h in hist]
    secs = 86400
    cut = cur_ts - timedelta(seconds=secs)
    old = [t for t in ts_list if t < cut]
    rate_h = None
    if len(old) >= 3:
        span_h = max((cut - min(old)).total_seconds() / 3600.0, 24.0)
        rate_h = len(old) / span_h
    count = 1 + sum(1 for t in ts_list if cut < t <= cur_ts)
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
    row = _bm_row("velocity_24h", "Vélocité (24 h)", value, level)
    return [row], {"window": "24 h", "count": count,
                   "rate_per_hour": round(count * 3600.0 / secs, 2),
                   "baseline_txn_per_hour": None if rate_h is None else round(rate_h, 4),
                   "expected": None if expected is None else round(expected, 4),
                   "ratio": round(ratio, 2), "level": level}

def _bm_amount(cur_amt, hist):
    """Z-score (moyenne/écart-type) et écart relatif à la médiane du montant courant
    vs l'historique des montants de la carte."""
    amt = _bm_num(cur_amt)
    amts = [a for a in (_bm_num(h.get("amt")) for h in hist) if a is not None]
    n = len(amts)
    if amt is None:
        return ([_bm_row("amt_z", "Écart montant (z-score)", "Montant indisponible"),
                 _bm_row("amt_rel", "Écart relatif vs médiane", "Montant indisponible")], {"n": n})
    if n < 3:
        msg = f"Historique insuffisant ({n} txn)"
        return ([_bm_row("amt_z", "Écart montant (z-score)", msg),
                 _bm_row("amt_rel", "Écart relatif vs médiane", msg)], {"n": n})
    mean = statistics.fmean(amts)
    std = statistics.stdev(amts)
    med = statistics.median(amts)
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
    return [z_row, rel_row], {"n": n, "mean": round(mean, 2), "std": round(std, 2), "median": round(med, 2),
                              "zscore": None if z is None else round(z, 3),
                              "relative_to_median_pct": None if rel is None else round(rel, 1)}

def _bm_time_delta(cur_ts, hist):
    """Temps écoulé depuis la dernière activité de la carte (+ rythme médian habituel)."""
    if not hist:
        return ([_bm_row("time_delta", "Intervalle inter-transactions", "Première transaction connue")],
                {"seconds": None})
    delta = max(0.0, (cur_ts - hist[0]["ts"]).total_seconds())
    ts_sorted = sorted(h["ts"] for h in hist)
    gaps = [(b - a).total_seconds() for a, b in zip(ts_sorted, ts_sorted[1:])]
    med_gap = statistics.median(gaps) if len(gaps) >= 3 else None
    level = "danger" if delta < 60 else "warn" if delta < 300 else ""
    value = _bm_fmt_duration(delta)
    if med_gap is not None:
        value += f" (habituel : {_bm_fmt_duration(med_gap)})"
    return ([_bm_row("time_delta", "Intervalle inter-transactions", value, level)],
            {"seconds": round(delta, 1), "median_gap_seconds": None if med_gap is None else round(med_gap, 1)})

def _bm_affinity(cur_d, hist, mcc_col):
    """Part de la catégorie marchand (ou MCC réel si la base l'expose) de la transaction
    courante dans l'historique de la carte. Déviation = 1 − affinité."""
    use_mcc = bool(mcc_col) and cur_d.get(mcc_col) not in (None, "")
    key = mcc_col if use_mcc else "category"
    label = "Affinité marchand (catégorie/MCC)"
    def norm(v):
        return str(v).strip().lower() if v not in (None, "") else None
    cur_k = norm(cur_d.get(key))
    hist_keys = [k for k in (norm(h.get(key)) for h in hist) if k]
    n = len(hist_keys)
    if cur_k is None:
        return [_bm_row("affinity", label, "Catégorie inconnue")], {"n": n}
    if n < 3:
        return [_bm_row("affinity", label, f"Historique insuffisant ({n} txn)")], {"n": n}
    counts = Counter(hist_keys)
    same = counts.get(cur_k, 0)
    share = same / n
    dev = 1.0 - share
    top_k, top_n = counts.most_common(1)[0]
    level = "danger" if (same == 0 and n >= 15) else "warn" if share < 0.05 else ""
    rows = [
        _bm_row("affinity", label, f"{share * 100:.0f} % · déviation {dev * 100:.0f} % ({same}/{n})", level),
        {"key": "affinity_top", "label": "Catégorie habituelle", "value": f"{top_k} ({top_n / n * 100:.0f} %)",
         "level": "", "category": top_k, "suffix": f"({top_n / n * 100:.0f} %)"},
    ]
    return rows, {"n": n, "source": "mcc" if use_mcc else "category", "same_count": same,
                  "affinity": round(share, 4), "deviation": round(dev, 4), "top": top_k}

def _bm_load_stored(conn: sqlite3.Connection, transaction_id: str) -> Optional[Dict[str, Any]]:
    """Métriques écrites par l'API de scoring pour cette transaction, ou None
    (colonne absente, valeur NULL, JSON invalide ou sans lignes affichables)."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(transactions_scored)").fetchall()}
    if "behavioral_metrics" not in cols:
        return None
    row = conn.execute("SELECT behavioral_metrics FROM transactions_scored_live WHERE transaction_id=?",
                       (transaction_id,)).fetchone()
    if not row or not row[0]:
        return None
    try:
        payload = json.loads(row[0])
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list) or not payload["rows"]:
        return None
    payload["transaction_id"] = transaction_id
    payload["origin"] = payload.get("origin") or "scoring_api"
    return payload

def compute_behavioral_metrics(transaction_id: str) -> Optional[Dict[str, Any]]:
    """Métriques comportementales d'une transaction (vélocité 24 h, déviation du montant,
    intervalle inter-transactions, affinité marchand) prêtes à afficher dans le bloc
    « Métriques Comportementales » (liste `rows`).

    1) Valeurs calculées par l'API de scoring lors de l'enrichissement (colonne
       `behavioral_metrics`), basées sur History.csv + flux live.
    2) Sinon, recalcul de repli avec les mêmes règles sur l'historique complet de la carte."""
    def _q():
        conn = _ro_connect()
        try:
            stored = _bm_load_stored(conn, transaction_id)
            if stored is not None:
                return "stored", stored
            cols = {r[1] for r in conn.execute("PRAGMA table_info(transactions_scored)").fetchall()}
            mcc_col = _BM_MCC_COL if _BM_MCC_COL in cols else None
            select_cols = ["transaction_id", "cc_num", "trans_date_trans_time", "processed_at", "amt", "category"]
            if mcc_col:
                select_cols.append(mcc_col)
            sql_cols = ", ".join(select_cols)  # noms issus de listes fixes : aucune entrée utilisateur
            row = conn.execute(f"SELECT {sql_cols} FROM transactions_scored_live WHERE transaction_id=?",
                               (transaction_id,)).fetchone()
            if not row:
                return None
            cur_d = dict(zip(select_cols, row))
            hist_rows = []
            if cur_d.get("cc_num"):
                # Table de BASE (et non la vue *_live) : inclut les lignes de History.csv.
                hist_rows = conn.execute(
                    f"SELECT {sql_cols} FROM transactions_scored WHERE cc_num=? AND transaction_id != ? "
                    f"ORDER BY trans_date_trans_time DESC LIMIT ?",
                    (cur_d["cc_num"], transaction_id, _BM_HISTORY_LIMIT)).fetchall()
        finally:
            conn.close()
        return "computed", (mcc_col, cur_d, [dict(zip(select_cols, r)) for r in hist_rows])

    fetched = safe_db_call(_q)
    if fetched is None:
        return None
    kind, data = fetched
    if kind == "stored":
        return data
    mcc_col, cur_d, hist_raw = data

    cur_ts = _parse_ts(cur_d.get("trans_date_trans_time")) or _parse_ts(cur_d.get("processed_at"))
    if cur_ts is None:
        return {"transaction_id": transaction_id, "history_count": 0,
                "rows": [_bm_row("error", "Métriques comportementales", "Horodatage invalide", "warn")]}

    hist = []
    for h in hist_raw:
        ts = _parse_ts(h.get("trans_date_trans_time")) or _parse_ts(h.get("processed_at"))
        if ts is not None and ts <= cur_ts:      # uniquement le passé de la carte
            h["ts"] = ts
            hist.append(h)
    hist.sort(key=lambda h: h["ts"], reverse=True)

    vel_rows, vel = _bm_velocity(cur_ts, hist)
    amt_rows, amt = _bm_amount(cur_d.get("amt"), hist)
    td_rows, td = _bm_time_delta(cur_ts, hist)
    af_rows, af = _bm_affinity(cur_d, hist, mcc_col)

    return {
        "transaction_id": transaction_id,
        "origin": "dashboard_fallback",
        "reference_time": cur_ts.isoformat(),
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "history_count": len(hist),
        "velocity": vel, "amount": amt, "time_delta": td, "affinity": af,
        "rows": vel_rows + amt_rows + td_rows + af_rows,
    }

def search_scored_transactions(transaction_id: str = None, channel: str = None, cardholder_name: str = None, limit: int = 100) -> List[Dict]:
    def _s():
        conn = _ro_connect()
        try:
            query = "SELECT transaction_id, channel, cardholder_name, processed_at, content_hash FROM transactions_scored_live WHERE 1=1"
            params = []
            if transaction_id:
                query += " AND transaction_id LIKE ?"
                params.append(f"%{transaction_id}%")
            if channel:
                query += " AND channel = ?"
                params.append(channel)
            if cardholder_name:
                query += " AND cardholder_name LIKE ?"
                params.append(f"%{cardholder_name}%")
            query += " ORDER BY processed_at DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(query, params).fetchall()
            return [{"transaction_id": r[0], "channel": r[1], "cardholder_name": r[2], "processed_at": r[3],
                     "content_hash": r[4][:12]+"..." if r[4] else ""} for r in rows]
        finally:
            conn.close()
    return safe_db_call(_s)

def get_recent_failed_transactions(limit: int = 10) -> List[Dict]:
    def _q():
        conn = _ro_connect()
        try:
            cur = conn.execute("SELECT id, transaction_id, channel, reason, attempts, replay_count, created_at, dead_at, raw_payload_json, fingerprint FROM failed_transactions ORDER BY dead_at DESC LIMIT ?", (limit,))
            rows = cur.fetchall()
        finally:
            conn.close()
        result = []
        for row in rows:
            try:
                raw = json.loads(row[8])
            except Exception:
                raw = {"error": "Invalid JSON", "raw": (row[8] or "")[:500]}
            result.append({"id": row[0], "transaction_id": row[1], "channel": row[2], "reason": row[3], "attempts": row[4],
                           "replay_count": row[5], "created_at": row[6], "dead_at": row[7], "fingerprint": row[9], "raw_payload": raw})
        return result
    return safe_db_call(_q)

def search_failed_transactions(transaction_id: Optional[str] = None, channel: Optional[str] = None, reason: Optional[str] = None, limit: int = 50) -> List[Dict]:
    def _s():
        conn = _ro_connect()
        try:
            query = "SELECT id, transaction_id, channel, reason, attempts, replay_count, created_at, dead_at, raw_payload_json, fingerprint FROM failed_transactions WHERE 1=1"
            params = []
            if transaction_id:
                query += " AND transaction_id LIKE ?"
                params.append(f"%{transaction_id}%")
            if channel:
                query += " AND channel = ?"
                params.append(channel)
            if reason:
                query += " AND reason = ?"
                params.append(reason)
            query += " ORDER BY dead_at DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.close()
        result = []
        for row in rows:
            try:
                raw = json.loads(row[8])
            except Exception:
                raw = {"error": "Invalid JSON", "raw": (row[8] or "")[:500]}
            result.append({"id": row[0], "transaction_id": row[1], "channel": row[2], "reason": row[3], "attempts": row[4],
                           "replay_count": row[5], "created_at": row[6], "dead_at": row[7], "fingerprint": row[9], "raw_payload": raw})
        return result
    return safe_db_call(_s)

def get_failed_transaction_count() -> int:
    def _q():
        conn = _ro_connect()
        try:
            return conn.execute("SELECT COUNT(*) FROM failed_transactions").fetchone()[0]
        finally:
            conn.close()
    return safe_db_call(_q)

# ============================================================================
# Statistiques pour le dashboard — SQL direct (colonnes explicites, plus de
# parsing JSON par ligne comme dans l'ancien dashboard SOC)
# ============================================================================
def get_daily_decision_stats() -> Dict[str, Any]:
    conn = _ro_connect()
    try:
        rows = conn.execute("""
            SELECT DATE(processed_at) as day, decision, COUNT(*) as n
            FROM transactions_scored_live
            WHERE decision IN ('BLOCK','MANUAL_REVIEW','ALLOW')
            GROUP BY day, decision ORDER BY day ASC
        """).fetchall()
    finally:
        conn.close()
    daily: Dict[str, Dict[str, int]] = {}
    for day, decision, n in rows:
        daily.setdefault(day, {"BLOCK": 0, "MANUAL_REVIEW": 0, "ALLOW": 0})[decision] = n
    days = sorted(daily.keys())
    return {
        "days": days,
        "block": [daily[d]["BLOCK"] for d in days],
        "manual_review": [daily[d]["MANUAL_REVIEW"] for d in days],
        "allow": [daily[d]["ALLOW"] for d in days],
    }

def get_channel_distribution() -> Dict[str, int]:
    conn = _ro_connect()
    try:
        rows = conn.execute("SELECT channel, COUNT(*) FROM transactions_scored_live GROUP BY channel").fetchall()
    finally:
        conn.close()
    result = {"online": 0, "pos": 0}
    for ch, cnt in rows:
        key = (ch or "").lower()
        if key in result:
            result[key] = cnt
    return result

def get_daily_stats() -> Dict[str, Any]:
    conn = _ro_connect()
    try:
        rows = conn.execute("""
            SELECT channel, DATE(processed_at) as day, COUNT(*) as count
            FROM transactions_scored_live GROUP BY channel, day ORDER BY day ASC
        """).fetchall()
    finally:
        conn.close()
    stats = {"online": [], "pos": []}
    all_days = set()
    for channel, day, count in rows:
        all_days.add(day)
        if channel in stats:
            stats[channel].append({"day": day, "count": count})
    days = sorted(list(all_days))
    return {"days": days, "stats": stats}

def get_category_distribution() -> Dict[str, Any]:
    """Répartition des catégories marchandes par canal (équivalent 'sévérité')."""
    conn = _ro_connect()
    try:
        rows = conn.execute("""
            SELECT channel, category, COUNT(*) as n
            FROM transactions_scored_live GROUP BY channel, category
        """).fetchall()
    finally:
        conn.close()
    dist = {"online": {}, "pos": {}}
    for channel, category, n in rows:
        if channel in dist:
            dist[channel][category or "N/A"] = n
    return dist

def get_top_fraud_categories(limit: int = 5) -> Dict[str, Any]:
    """Top catégories marchandes parmi les transactions signalées comme
    risquées (décision BLOCK ou MANUAL_REVIEW), triées par volume décroissant.
    """
    conn = _ro_connect()
    try:
        rows = conn.execute("""
            SELECT category, COUNT(*) as n
            FROM transactions_scored_live
            WHERE decision IN ('BLOCK','MANUAL_REVIEW')
            GROUP BY category ORDER BY n DESC
        """).fetchall()
    finally:
        conn.close()
    total = sum(n for _, n in rows) or 1
    top = rows[:limit]
    return {
        "categories": [
            {"category": cat or "N/A", "count": n, "pct": round(n / total * 100, 1)}
            for cat, n in top
        ],
        "total_flagged": total,
    }

# Annuaire intégré des villes, regroupé par PAYS (code ISO 3166-1 alpha-2), au format
# « nom ; latitude ; longitude » séparé par « | ». Il sert à deux choses :
#   1. afficher le DRAPEAU du pays devant chaque ville dans le tableau (colonne
#      « Lieu de transaction ») et dans l'info-bulle de la carte ;
#   2. placer sur la carte une ville qui n'a aucune coordonnée GPS exploitable en
#      base (les coordonnées de fraud.db restent prioritaires).
# Les noms sont comparés sans accents ni casse, en français comme en anglais
# (« Le Caire » / « Cairo », « Vienne » / « Vienna »…). Pour ajouter une ville :
# ajouter « nom;lat;lon » à la ligne du bon pays (nouveau pays : ajouter son
# drapeau et son nom dans FLAGS côté JavaScript).
_GEO_CITIES_BY_COUNTRY: Dict[str, str] = {
    "TN": "tunis;36.81;10.18|ariana;36.86;10.19|ben arous;36.75;10.23|manouba;36.81;10.10|la marsa;36.88;10.33|carthage;36.85;10.33|sfax;34.74;10.76|sousse;35.83;10.64|kairouan;35.68;10.10|bizerte;37.27;9.87|gabes;33.88;10.10|gafsa;34.43;8.78|monastir;35.78;10.83|kasserine;35.17;8.83|medenine;33.35;10.50|nabeul;36.45;10.74|hammamet;36.40;10.61|tataouine;32.93;10.45|beja;36.73;9.18|jendouba;36.50;8.78|mahdia;35.50;11.06|sidi bouzid;35.04;9.48|siliana;36.08;9.37|le kef;36.17;8.71|kef;36.17;8.71|tozeur;33.92;8.13|kebili;33.70;8.97|zaghouan;36.40;10.14|djerba;33.81;10.86|houmt souk;33.88;10.86",
    "FR": "paris;48.86;2.35|marseille;43.30;5.37|lyon;45.76;4.84|toulouse;43.60;1.44|nice;43.70;7.27|nantes;47.22;-1.55|strasbourg;48.57;7.75|montpellier;43.61;3.88|bordeaux;44.84;-0.58|lille;50.63;3.06|rennes;48.11;-1.68|reims;49.26;4.03|le havre;49.49;0.11|grenoble;45.19;5.72|dijon;47.32;5.04|angers;47.47;-0.55|toulon;43.12;5.93|nimes;43.84;4.36|brest;48.39;-4.49|tours;47.39;0.69|saint etienne;45.44;4.39|clermont ferrand;45.78;3.09|le mans;48.01;0.20|amiens;49.89;2.30|limoges;45.83;1.26|metz;49.12;6.18|besancon;47.24;6.02|orleans;47.90;1.91|rouen;49.44;1.10|caen;49.18;-0.37|nancy;48.69;6.18|perpignan;42.70;2.90|avignon;43.95;4.81|cannes;43.55;7.01|ajaccio;41.92;8.74",
    "GB": "london;51.51;-0.13|londres;51.51;-0.13|manchester;53.48;-2.24|birmingham;52.49;-1.89|edinburgh;55.95;-3.19|edimbourg;55.95;-3.19|glasgow;55.86;-4.25|liverpool;53.41;-2.98|leeds;53.80;-1.55|bristol;51.45;-2.59|cardiff;51.48;-3.18|belfast;54.60;-5.93",
    "IE": "dublin;53.35;-6.26|cork;51.90;-8.47",
    "DE": "berlin;52.52;13.40|frankfurt;50.11;8.68|francfort;50.11;8.68|munich;48.14;11.58|hamburg;53.55;9.99|hambourg;53.55;9.99|cologne;50.94;6.96|stuttgart;48.78;9.18|dusseldorf;51.23;6.78|dresden;51.05;13.74|dresde;51.05;13.74|leipzig;51.34;12.37|hanover;52.37;9.73|hanovre;52.37;9.73|nuremberg;49.45;11.08|nurnberg;49.45;11.08|bonn;50.73;7.10",
    "ES": "madrid;40.42;-3.70|barcelona;41.39;2.17|barcelone;41.39;2.17|valencia;39.47;-0.38|seville;37.39;-5.98|malaga;36.72;-4.42|bilbao;43.26;-2.93|zaragoza;41.65;-0.88|saragosse;41.65;-0.88|palma;39.57;2.65",
    "PT": "lisbon;38.72;-9.14|lisbonne;38.72;-9.14|porto;41.15;-8.61|faro;37.02;-7.93",
    "IT": "rome;41.90;12.50|milan;45.46;9.19|naples;40.85;14.27|turin;45.07;7.69|florence;43.77;11.26|venice;45.44;12.32|venise;45.44;12.32|bologna;44.49;11.34|bologne;44.49;11.34|genoa;44.41;8.93|genes;44.41;8.93|palermo;38.12;13.36|palerme;38.12;13.36",
    "NL": "amsterdam;52.37;4.90|rotterdam;51.92;4.48|the hague;52.08;4.31|la haye;52.08;4.31|utrecht;52.09;5.12",
    "BE": "brussels;50.85;4.35|bruxelles;50.85;4.35|antwerp;51.22;4.40|anvers;51.22;4.40|ghent;51.05;3.72|gand;51.05;3.72",
    "LU": "luxembourg;49.61;6.13",
    "AT": "vienna;48.21;16.37|vienne;48.21;16.37|salzburg;47.81;13.04|salzbourg;47.81;13.04|graz;47.07;15.44",
    "CH": "zurich;47.38;8.54|geneva;46.20;6.14|geneve;46.20;6.14|bern;46.95;7.45|berne;46.95;7.45|basel;47.56;7.59|bale;47.56;7.59|lausanne;46.52;6.63",
    "SE": "stockholm;59.33;18.07|gothenburg;57.71;11.97|goteborg;57.71;11.97|malmo;55.60;13.00",
    "NO": "oslo;59.91;10.75|bergen;60.39;5.32",
    "DK": "copenhagen;55.68;12.57|copenhague;55.68;12.57|aarhus;56.16;10.20",
    "FI": "helsinki;60.17;24.94",
    "IS": "reykjavik;64.15;-21.94",
    "PL": "warsaw;52.23;21.01|varsovie;52.23;21.01|krakow;50.06;19.94|cracovie;50.06;19.94|gdansk;54.35;18.65|wroclaw;51.11;17.04",
    "CZ": "prague;50.08;14.44",
    "SK": "bratislava;48.15;17.11",
    "HU": "budapest;47.50;19.04",
    "RO": "bucharest;44.43;26.10|bucarest;44.43;26.10",
    "BG": "sofia;42.70;23.32",
    "GR": "athens;37.98;23.73|athenes;37.98;23.73|thessaloniki;40.64;22.94|salonique;40.64;22.94",
    "RS": "belgrade;44.79;20.45",
    "HR": "zagreb;45.81;15.98",
    "SI": "ljubljana;46.06;14.51",
    "TR": "istanbul;41.01;28.98|ankara;39.93;32.86|izmir;38.42;27.14",
    "RU": "moscow;55.76;37.62|moscou;55.76;37.62|saint petersburg;59.93;30.34|saint petersbourg;59.93;30.34",
    "UA": "kyiv;50.45;30.52|kiev;50.45;30.52",
    "EG": "cairo;30.04;31.24|le caire;30.04;31.24|alexandria;31.20;29.92|alexandrie;31.20;29.92|giza;30.01;31.21|gizeh;30.01;31.21|sharm el sheikh;27.91;34.33|hurghada;27.26;33.81|luxor;25.69;32.64|louxor;25.69;32.64",
    "DZ": "algiers;36.75;3.06|alger;36.75;3.06|oran;35.70;-0.63|constantine;36.37;6.61|annaba;36.90;7.77",
    "MA": "casablanca;33.57;-7.59|rabat;34.02;-6.84|marrakech;31.63;-8.01|tangier;35.76;-5.83|tanger;35.76;-5.83|fes;34.03;-5.00|fez;34.03;-5.00|agadir;30.43;-9.60",
    "LY": "tripoli;32.89;13.19|benghazi;32.12;20.09",
    "NG": "lagos;6.52;3.38|abuja;9.08;7.40",
    "KE": "nairobi;-1.29;36.82",
    "ZA": "johannesburg;-26.20;28.05|cape town;-33.92;18.42|le cap;-33.92;18.42|durban;-29.86;31.02|pretoria;-25.75;28.19",
    "GH": "accra;5.60;-0.19",
    "SN": "dakar;14.72;-17.47",
    "ET": "addis ababa;9.03;38.74|addis abeba;9.03;38.74",
    "CI": "abidjan;5.36;-4.01",
    "CD": "kinshasa;-4.44;15.27",
    "CM": "douala;4.05;9.77|yaounde;3.85;11.50",
    "AE": "dubai;25.20;55.27|abu dhabi;24.45;54.38|abou dhabi;24.45;54.38",
    "SA": "riyadh;24.71;46.68|riyad;24.71;46.68|jeddah;21.49;39.19|djeddah;21.49;39.19|mecca;21.42;39.83|la mecque;21.42;39.83|medina;24.47;39.61|medine;24.47;39.61",
    "QA": "doha;25.29;51.53",
    "KW": "kuwait city;29.38;47.99|koweit;29.38;47.99",
    "BH": "manama;26.23;50.59",
    "OM": "muscat;23.59;58.41|mascate;23.59;58.41",
    "LB": "beirut;33.89;35.50|beyrouth;33.89;35.50",
    "JO": "amman;31.95;35.93",
    "IQ": "baghdad;33.31;44.36|bagdad;33.31;44.36",
    "IR": "tehran;35.69;51.39|teheran;35.69;51.39",
    "IL": "tel aviv;32.09;34.78|jerusalem;31.77;35.21",
    "SY": "damascus;33.51;36.29|damas;33.51;36.29",
    "PK": "karachi;24.86;67.01|lahore;31.55;74.34|islamabad;33.68;73.05",
    "IN": "delhi;28.61;77.21|new delhi;28.61;77.21|mumbai;19.08;72.88|bombay;19.08;72.88|bangalore;12.97;77.59|bengaluru;12.97;77.59|chennai;13.08;80.27|kolkata;22.57;88.36|calcutta;22.57;88.36|hyderabad;17.39;78.49",
    "BD": "dhaka;23.81;90.41|dacca;23.81;90.41",
    "TH": "bangkok;13.76;100.50",
    "SG": "singapore;1.35;103.82|singapour;1.35;103.82",
    "MY": "kuala lumpur;3.14;101.69",
    "ID": "jakarta;-6.21;106.85",
    "PH": "manila;14.60;120.98|manille;14.60;120.98",
    "VN": "hanoi;21.03;105.85|ho chi minh city;10.82;106.63|ho chi minh ville;10.82;106.63|saigon;10.82;106.63",
    "HK": "hong kong;22.32;114.17",
    "CN": "shanghai;31.23;121.47|beijing;39.90;116.41|pekin;39.90;116.41|shenzhen;22.54;114.06|guangzhou;23.13;113.26|canton;23.13;113.26|chengdu;30.57;104.07",
    "TW": "taipei;25.03;121.57",
    "KR": "seoul;37.57;126.98|busan;35.18;129.08",
    "JP": "tokyo;35.68;139.65|osaka;34.69;135.50|kyoto;35.01;135.77|yokohama;35.44;139.64|nagoya;35.18;136.91|sapporo;43.06;141.35",
    "AU": "sydney;-33.87;151.21|melbourne;-37.81;144.96|brisbane;-27.47;153.03|perth;-31.95;115.86|adelaide;-34.93;138.60|canberra;-35.28;149.13",
    "NZ": "auckland;-36.85;174.76|wellington;-41.29;174.78",
    "US": "new york;40.71;-74.01|new york city;40.71;-74.01|nyc;40.71;-74.01|los angeles;34.05;-118.24|chicago;41.88;-87.63|houston;29.76;-95.37|phoenix;33.45;-112.07|philadelphia;39.95;-75.17|philadelphie;39.95;-75.17|san antonio;29.42;-98.49|san diego;32.72;-117.16|dallas;32.78;-96.80|san jose;37.34;-121.89|austin;30.27;-97.74|san francisco;37.77;-122.42|seattle;47.61;-122.33|denver;39.74;-104.99|washington;38.91;-77.04|washington dc;38.91;-77.04|boston;42.36;-71.06|miami;25.76;-80.19|atlanta;33.75;-84.39|las vegas;36.17;-115.14|detroit;42.33;-83.05|minneapolis;44.98;-93.27|portland;45.52;-122.68|new orleans;29.95;-90.07|la nouvelle orleans;29.95;-90.07|nashville;36.16;-86.78|orlando;28.54;-81.38|charlotte;35.23;-80.84|columbus;39.96;-83.00|indianapolis;39.77;-86.16|jacksonville;30.33;-81.66|fort worth;32.76;-97.33|el paso;31.76;-106.49|memphis;35.15;-90.05|baltimore;39.29;-76.61|milwaukee;43.04;-87.91|albuquerque;35.08;-106.65|tucson;32.22;-110.97|fresno;36.74;-119.79|sacramento;38.58;-121.49|kansas city;39.10;-94.58|cleveland;41.50;-81.69|pittsburgh;40.44;-79.99|cincinnati;39.10;-84.51|saint louis;38.63;-90.20|st louis;38.63;-90.20|tampa;27.95;-82.46|honolulu;21.31;-157.86|salt lake city;40.76;-111.89|oakland;37.80;-122.27|raleigh;35.78;-78.64",
    "CA": "toronto;43.65;-79.38|montreal;45.50;-73.57|vancouver;49.28;-123.12|ottawa;45.42;-75.70|calgary;51.05;-114.07|edmonton;53.55;-113.49|quebec;46.81;-71.21|winnipeg;49.90;-97.14",
    "MX": "mexico;19.43;-99.13|mexico city;19.43;-99.13|mexico df;19.43;-99.13|ciudad de mexico;19.43;-99.13|guadalajara;20.66;-103.35|monterrey;25.69;-100.32|cancun;21.16;-86.85",
    "BR": "sao paulo;-23.55;-46.63|rio de janeiro;-22.91;-43.17|brasilia;-15.79;-47.88|salvador;-12.97;-38.50",
    "AR": "buenos aires;-34.60;-58.38",
    "CL": "santiago;-33.45;-70.67",
    "CO": "bogota;4.71;-74.07|medellin;6.24;-75.58",
    "PE": "lima;-12.05;-77.04",
    "VE": "caracas;10.48;-66.90",
    "CU": "la havane;23.11;-82.37|havana;23.11;-82.37",
    "PA": "panama;8.98;-79.52|panama city;8.98;-79.52",
    "UY": "montevideo;-34.90;-56.16",
    "EC": "quito;-0.18;-78.47",
}

def _geo_norm(name: Any) -> str:
    """Normalise un nom de ville : sans accents, minuscules, sans ponctuation."""
    txt = unicodedata.normalize("NFKD", str(name or ""))
    txt = "".join(ch for ch in txt if not unicodedata.combining(ch)).lower()
    for ch in "-_'’.,()":
        txt = txt.replace(ch, " ")
    return " ".join(txt.split())

_GEO_GAZETTEER: Dict[str, tuple] = {}
for _cc, _blob in _GEO_CITIES_BY_COUNTRY.items():
    for _entry in _blob.split("|"):
        _parts = _entry.strip().split(";")
        if len(_parts) == 3:
            try:
                _GEO_GAZETTEER[_geo_norm(_parts[0])] = (float(_parts[1]), float(_parts[2]), _cc)
            except ValueError:
                pass

def _geo_country(city: Any) -> Optional[str]:
    """Code pays (ISO alpha-2) d'une ville connue de l'annuaire, sinon None."""
    entry = _GEO_GAZETTEER.get(_geo_norm(city))
    return entry[2] if entry else None

# ── LIEU DE TRANSACTION ─────────────────────────────────────────────────────
# Modèle de données : la ville de naissance/domicile du titulaire est FIXE (champs
# city / lat / long) ; le lieu où la transaction a lieu (merch_lat / merch_long) est
# aléatoire parmi toutes les villes du monde, sans lien avec le client. La base ne
# stocke que des coordonnées pour ce lieu : on retrouve le NOM de la ville en
# associant les coordonnées à la ville la plus proche (à ≤ 60 km) de world_cities.py
# (mêmes noms que l'ingestion) puis, à défaut, de l'annuaire intégré ci-dessus.
try:
    from world_cities import WORLD_CITIES as _TX_WORLD_CITIES
except Exception:                      # world_cities.py absent : repli sur l'annuaire seul
    _TX_WORLD_CITIES = []
_TX_SNAP_MAX_KM = 60.0
_TX_FALLBACK_POINTS: List[tuple] = []
_seen_pos = set()
for _k, (_la, _lo, _cc) in _GEO_GAZETTEER.items():       # une entrée par position (1er alias)
    if (_la, _lo) not in _seen_pos:
        _seen_pos.add((_la, _lo))
        _TX_FALLBACK_POINTS.append((_k.title(), _la, _lo))

def _geo_hav_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * math.asin(math.sqrt(a))

@lru_cache(maxsize=16384)
def _snap_tx_city_cached(lat: float, lon: float) -> Optional[tuple]:
    for points in (_TX_WORLD_CITIES, _TX_FALLBACK_POINTS):
        best = None
        for name, la, lo in points:
            d = _geo_hav_km(lat, lon, la, lo)
            if best is None or d < best[3]:
                best = (name, la, lo, d)
        if best is not None and best[3] <= _TX_SNAP_MAX_KM:
            return best[:3]
    return None

def _snap_tx_city(lat: Any, lon: Any) -> Optional[tuple]:
    """(nom, lat_centre, long_centre) de la ville de transaction la plus proche,
    ou None si coordonnées absentes/invalides/(0,0) ou aucune ville à ≤ 60 km."""
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180) or (lat == 0 and lon == 0):
        return None
    return _snap_tx_city_cached(round(lat, 2), round(lon, 2))

# Coordonnée valable : numérique, dans les bornes GPS et différente de (0, 0).
_GEO_VALID_MERCH_SQL = (
    "(CAST(merch_lat AS REAL) BETWEEN -90 AND 90 AND CAST(merch_long AS REAL) BETWEEN -180 AND 180 "
    "AND NOT (CAST(merch_lat AS REAL) = 0 AND CAST(merch_long AS REAL) = 0))"
)

def get_geo_distribution() -> Dict[str, Any]:
    """Agrégation géographique des transactions par LIEU DE TRANSACTION (ville où
    la transaction a eu lieu : merch_lat/merch_long), pour la carte mondiale temps
    réel — et NON par domicile du titulaire (city/lat/long, fixe par carte).

    Chaque ville regroupe : volume total de transactions, nombre de transactions
    bloquées (décision BLOCK uniquement, la revue manuelle n'est PAS comptée comme fraude) et taux de fraude (%). La base est d'abord
    agrégée par cases de 0,1° (léger bruit autour du centre-ville), puis chaque case
    est rattachée à sa ville la plus proche (_snap_tx_city) et les cases d'une même
    ville sont fusionnées ; le marqueur est placé au centre de la ville.
    Le bloc « diagnostic » permet à l'interface d'expliquer pourquoi la carte est vide.
    """
    conn = _ro_connect()
    try:
        rows = conn.execute(f"""
            SELECT
                ROUND(CAST(merch_lat AS REAL), 1),
                ROUND(CAST(merch_long AS REAL), 1),
                AVG(CAST(merch_lat AS REAL)),
                AVG(CAST(merch_long AS REAL)),
                COUNT(*),
                SUM(CASE WHEN decision = 'BLOCK' THEN 1 ELSE 0 END)
            FROM transactions_scored_live
            WHERE merch_lat IS NOT NULL AND merch_long IS NOT NULL AND {_GEO_VALID_MERCH_SQL}
            GROUP BY 1, 2
        """).fetchall()
        diag_row = conn.execute(f"""
            SELECT
                COUNT(*),
                SUM(CASE WHEN merch_lat IS NOT NULL AND merch_long IS NOT NULL
                          AND {_GEO_VALID_MERCH_SQL} THEN 1 ELSE 0 END)
            FROM transactions_scored_live
        """).fetchone()
    finally:
        conn.close()

    merged: Dict[str, Dict[str, Any]] = {}
    unlocated: Dict[str, int] = {}
    for _blat, _blon, avg_lat, avg_lng, total, flagged in rows:
        if not total:
            continue
        snap = _snap_tx_city(avg_lat, avg_lng)
        if snap is None:
            label = f"{avg_lat:.1f}, {avg_lng:.1f}"
            unlocated[label] = unlocated.get(label, 0) + int(total)
            continue
        name, clat, clng = snap
        c = merged.setdefault(name, {"lat": clat, "long": clng, "total": 0, "fraud": 0})
        c["total"] += int(total)
        c["fraud"] += int(flagged or 0)

    cities = [{
        "city": name,
        "state": "",
        "lat": round(c["lat"], 4),
        "long": round(c["long"], 4),
        "total": c["total"],
        "fraud": c["fraud"],
        "rate": round((c["fraud"] / c["total"]) * 100, 1),
        "source": "transaction",
        "country": _geo_country(name),
    } for name, c in merged.items()]
    # Tri par taux de fraude décroissant (villes les plus à risque en premier)
    cities.sort(key=lambda c: (c["rate"], c["total"]), reverse=True)
    top_unlocated = [n for n, _ in sorted(unlocated.items(), key=lambda kv: kv[1], reverse=True)[:8]]
    n_rows = int((diag_row[0] or 0) if diag_row else 0)
    n_coords = int((diag_row[1] or 0) if diag_row else 0)
    return {
        "cities": cities,
        "total_cities": len(cities),
        "diagnostic": {
            "rows": n_rows,
            "with_city": n_coords,     # lieu de transaction connu (nom déduit des coordonnées)
            "with_coords": n_coords,
            "unlocated": top_unlocated,
        },
    }

def get_decision_distribution() -> Dict[str, int]:
    conn = _ro_connect()
    try:
        rows = conn.execute("SELECT decision, COUNT(*) FROM transactions_scored_live GROUP BY decision").fetchall()
    finally:
        conn.close()
    dist = {"block": 0, "manual_review": 0, "allow": 0, "unknown": 0}
    key_map = {"BLOCK": "block", "MANUAL_REVIEW": "manual_review", "ALLOW": "allow"}
    for decision, cnt in rows:
        dist[key_map.get(decision, "unknown")] = dist.get(key_map.get(decision, "unknown"), 0) + cnt
    return dist

def get_decision_distribution_by_channel() -> Dict[str, Dict[str, int]]:
    conn = _ro_connect()
    try:
        rows = conn.execute("SELECT channel, decision, COUNT(*) FROM transactions_scored_live GROUP BY channel, decision").fetchall()
    finally:
        conn.close()
    result = {
        "online": {"block": 0, "manual_review": 0, "allow": 0, "unknown": 0},
        "pos": {"block": 0, "manual_review": 0, "allow": 0, "unknown": 0},
    }
    key_map = {"BLOCK": "block", "MANUAL_REVIEW": "manual_review", "ALLOW": "allow"}
    for channel, decision, cnt in rows:
        ch_key = (channel or "").lower()
        if ch_key not in result:
            continue
        dec_key = key_map.get(decision, "unknown")
        result[ch_key][dec_key] = result[ch_key].get(dec_key, 0) + cnt
    return result

def get_fraud_score_distribution() -> Dict[str, Any]:
    """Distribution du score de fraude XGBoost (0–100, = fraud_probability*100).

    Inclut aussi les seuils RÉELS de décision (pas des valeurs arbitraires) :
    le score minimal observé pour chaque décision escaladée (MANUAL_REVIEW,
    BLOCK), déduit directement des transactions déjà scorées par l'agent.
    """
    conn = _ro_connect()
    try:
        rows = conn.execute("SELECT fraud_probability FROM transactions_scored_live").fetchall()
        thresh_rows = conn.execute("""
            SELECT decision, MIN(fraud_probability) FROM transactions_scored_live
            WHERE decision IN ('MANUAL_REVIEW','BLOCK') GROUP BY decision
        """).fetchall()
    finally:
        conn.close()
    counts = [0] * 101
    dist = {"0-25": 0, "26-50": 0, "51-75": 0, "76-100": 0}
    for (fp,) in rows:
        try:
            score = int(round(float(fp or 0) * 100))
            score = max(0, min(100, score))
            counts[score] += 1
            if score <= 25:
                dist["0-25"] += 1
            elif score <= 50:
                dist["26-50"] += 1
            elif score <= 75:
                dist["51-75"] += 1
            else:
                dist["76-100"] += 1
        except Exception:
            pass
    thresholds = {"review": None, "block": None}
    for decision, min_fp in thresh_rows:
        if min_fp is None:
            continue
        score = round(float(min_fp) * 100, 1)
        if decision == "MANUAL_REVIEW":
            thresholds["review"] = score
        elif decision == "BLOCK":
            thresholds["block"] = score
    return {"scores": list(range(101)), "counts": counts, "buckets": dist, "thresholds": thresholds}

def get_kpi_metrics(window_days: int = 1) -> Dict[str, Any]:
    """KPIs fraude (Precision/Recall/Réduction friction client) — lecture seule.

    Definitions (adaptées de l'évaluation SOC au contexte fraude bancaire) :
      1. Precision/Recall calculés sur la fenêtre glissante [now - window_days, now],
         en comparant la décision du modèle (BLOCK / ALLOW) au dernier verdict
         analyste enregistré par transaction (feedback dédoublonné, un seul
         verdict retenu par transaction_id).
           - decision=BLOCK  & label=TP (fraude confirmée)   -> TP
           - decision=BLOCK  & label=FP (transaction légitime)-> FP (friction client)
           - decision=ALLOW  & label=TP (fraude confirmée)   -> FN (fraude passée au travers)
      2. "Réduction friction client" mesure le taux de transactions légitimes
         (label=FP) parmi les transactions BLOQUÉES par le modèle — c.-à-d. le
         taux de faux blocages réellement subi par les clients. Une baisse de
         ce taux = moins de clients légitimes bloqués à tort.
    """
    conn = _ro_connect()
    try:
        conn.row_factory = sqlite3.Row
        # Dédoublonnage : un seul feedback (le plus récent) retenu par transaction_id.
        fb_rows = conn.execute("""
            SELECT transaction_id, label, timestamp
            FROM feedback_history
            WHERE rowid IN (SELECT MAX(rowid) FROM feedback_history GROUP BY transaction_id)
        """).fetchall()
        tx_rows = conn.execute("SELECT transaction_id, channel, processed_at, decision FROM transactions_scored_live").fetchall()
    finally:
        conn.close()

    tx_by_id: Dict[str, Dict[str, Any]] = {}
    for r in tx_rows:
        tx_by_id[r["transaction_id"]] = {
            "decision": (r["decision"] or "").upper(),
            "processed_at": _parse_ts(r["processed_at"]),
        }

    feedback_by_id: Dict[str, Dict[str, Any]] = {}
    for fb in fb_rows:
        feedback_by_id[fb["transaction_id"]] = {
            "label": (fb["label"] or "").upper(),
            "timestamp": _parse_ts(fb["timestamp"]),
        }

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    win_curr_start = now - timedelta(days=window_days)
    win_prev_start = now - timedelta(days=2 * window_days)

    # 1. Precision / Recall — restreints à la fenêtre courante, feedback dédoublonné.
    tp = fp = fn = 0
    matched = 0

    for transaction_id, fb in feedback_by_id.items():
        info = tx_by_id.get(transaction_id)
        if not info:
            continue

        processed_at = info["processed_at"]
        if processed_at is None or not (win_curr_start <= processed_at <= now):
            continue

        matched += 1
        decision = info["decision"]
        label = fb["label"]

        if decision == "BLOCK" and label == "TP":
            tp += 1
        elif decision == "BLOCK" and label == "FP":
            fp += 1
        elif decision == "ALLOW" and label == "TP":
            fn += 1

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else None
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else None

    # 2. Taux de transactions légitimes parmi les BLOQUÉES (friction client réelle).
    def _blocked_fp_rate_in_window(start: datetime, end: datetime) -> Optional[float]:
        total_blocked = blocked_fp = 0
        for transaction_id, info in tx_by_id.items():
            ts = info["processed_at"]
            if ts is None or not (start <= ts < end):
                continue
            if info["decision"] == "BLOCK":
                total_blocked += 1
                fb_info = feedback_by_id.get(transaction_id)
                if fb_info and fb_info["label"] == "FP":
                    blocked_fp += 1
        return (blocked_fp / total_blocked * 100.0) if total_blocked > 0 else None

    fp_rate_curr = _blocked_fp_rate_in_window(win_curr_start, now)
    fp_rate_prev = _blocked_fp_rate_in_window(win_prev_start, win_curr_start)

    if fp_rate_curr is not None and fp_rate_prev is not None and fp_rate_prev > 0:
        fp_reduction_pct = ((fp_rate_prev - fp_rate_curr) / fp_rate_prev) * 100.0
    else:
        fp_reduction_pct = None

    return {
        "precision": precision,
        "recall": recall,
        "fp_reduction_pct": fp_reduction_pct,
        "fp_rate_current_pct": fp_rate_curr,
        "fp_rate_previous_pct": fp_rate_prev,
        "samples": {
            "feedback_total": len(fb_rows),
            "feedback_matched": matched,
            "tp": tp, "fp": fp, "fn": fn,
            "total_scored": len(tx_by_id),
        },
        "window_days": window_days,
    }

def get_transactions_for_dashboard(channel: str = None, limit: int = 200) -> List[Dict]:
    conn = _ro_connect()
    try:
        query = f"SELECT {TX_COLUMNS} FROM transactions_scored_live WHERE 1=1"
        params = []
        if channel:
            query += " AND channel = ?"
            params.append(channel)
        query += " ORDER BY processed_at DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        fb_map = _latest_feedback_map(conn, [r[0] for r in rows])
    finally:
        conn.close()
    result = []
    for row in rows:
        try:
            tx = _row_to_transaction(row)
            tx["analyst_status"] = fb_map.get(tx["transaction_id"])
            result.append(tx)
        except Exception:
            pass
    return result

# ============================================================================
# Dashboard HTML
# ============================================================================
DASHBOARD_HTML = r'''<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FraudGuard ML — Détection de Fraude Bancaire Temps Réel</title>
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Cdefs%3E%3ClinearGradient id='g' x1='0' y1='0' x2='1' y2='1'%3E%3Cstop offset='0' stop-color='%23ffb52e'/%3E%3Cstop offset='1' stop-color='%23e67e00'/%3E%3C/linearGradient%3E%3C/defs%3E%3Cpath d='M16 2 L28 6.5 V15 C28 22.5 22.8 27.8 16 30 C9.2 27.8 4 22.5 4 15 V6.5 Z' fill='url%28%23g%29'/%3E%3Ccircle cx='14.5' cy='14' r='5' fill='none' stroke='%230c1220' stroke-width='2.6'/%3E%3Cpath d='M18.3 17.8 L23 22.5' stroke='%230c1220' stroke-width='3' stroke-linecap='round'/%3E%3C/svg%3E">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root {
  --bg0: #03060f;
  --bg1: #060c1a;
  --bg2: #090f1f;
  --bg3: #0c1526;
  --bg4: #0f192e;
  --c1: #00d4ff;
  --c2: #bf5fff;
  --c3: #00ff9d;
  --c4: #ff4d6d;
  --c5: #ffaa00;
  --c6: #4dffdb;
  --c1d: rgba(0,212,255,.12);
  --c2d: rgba(191,95,255,.12);
  --c3d: rgba(0,255,157,.10);
  --c4d: rgba(255,77,109,.12);
  --c5d: rgba(255,170,0,.10);
  --c6d: rgba(77,255,219,.08);
  --border: rgba(0,212,255,.08);
  --border2: rgba(0,212,255,.22);
  --txt1: #e8f4ff;
  --txt2: #c7d6e6;
  --txt3: #a0aec0;
  --radius: 10px;
  --radius2: 14px;
  --glow1: 0 0 20px rgba(0,212,255,.18), 0 0 60px rgba(0,212,255,.06);
  --glow2: 0 0 20px rgba(191,95,255,.18), 0 0 60px rgba(191,95,255,.06);
  --glow3: 0 0 20px rgba(0,255,157,.14), 0 0 60px rgba(0,255,157,.05);
  --glow4: 0 0 20px rgba(255,77,109,.18), 0 0 60px rgba(255,77,109,.06);
  --glow5: 0 0 20px rgba(255,170,0,.14), 0 0 60px rgba(255,170,0,.05);
  --tr: all .18s cubic-bezier(.4,0,.2,1);
}
*{margin:0;padding:0;box-sizing:border-box}
html{scroll-behavior:smooth}
body{
  background:var(--bg0);
  color:var(--txt1);
  font-family:'Calisto MT',Georgia,'Times New Roman',serif;
  min-height:100vh;
  overflow-x:hidden;
  background-image:
    radial-gradient(ellipse 80% 50% at 50% -20%, rgba(0,212,255,.07) 0%, transparent 60%),
    radial-gradient(ellipse 60% 40% at 100% 80%, rgba(191,95,255,.05) 0%, transparent 50%);
}
body::before{
  content:'';position:fixed;inset:0;pointer-events:none;z-index:999;
  background:repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,.03) 2px, rgba(0,0,0,.03) 4px);
}

/* ──────────── TOPBAR ──────────── */
.topbar{
  display:flex;align-items:center;justify-content:space-between;
  padding:14px 32px;
  background:linear-gradient(180deg, rgba(6,12,26,.98) 0%, rgba(3,6,15,.94) 100%);
  border-bottom:1px solid var(--border);
  position:sticky;top:0;z-index:100;
  backdrop-filter:blur(24px);
  box-shadow:0 1px 0 rgba(0,212,255,.06), 0 4px 32px rgba(0,0,0,.5);
}
.logo-wrap{display:flex;align-items:center;gap:14px}
.logo-hex{
  width:42px;height:42px;border-radius:10px;
  background:linear-gradient(135deg,var(--c1),var(--c2));
  display:flex;align-items:center;justify-content:center;font-weight:900;color:#03060f;font-size:1.1rem;
  box-shadow:var(--glow1);
}
.logo-name{font-size:1.02rem;font-weight:800;letter-spacing:.02em}
.logo-sub{font-size:.68rem;color:var(--txt3);letter-spacing:.04em;margin-top:1px}
.topbar-right{display:flex;align-items:center;gap:16px}
.rt-pill{
  display:flex;align-items:center;gap:8px;padding:6px 14px;border-radius:20px;
  background:rgba(6,12,26,.88);border:1px solid rgba(0,255,157,.28);
  font-size:.66rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--txt2);
  box-shadow:0 0 14px rgba(0,255,157,.08);
}
.rt-pill .rt-label{color:var(--txt2)}
.rt-pill .rt-time{font-family:'Calisto MT',Georgia,'Times New Roman',serif;font-weight:700;letter-spacing:.04em;color:#fff;text-transform:none;font-variant-numeric:tabular-nums}
.live-dot{width:6px;height:6px;border-radius:50%;background:var(--c3);box-shadow:0 0 8px var(--c3);animation:pulse 1.8s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
.btn-logout{
  padding:7px 14px;border-radius:8px;border:1px solid rgba(255,77,109,.25);
  background:rgba(255,77,109,.08);color:var(--c4);cursor:pointer;font-size:.72rem;font-weight:700;
  font-family:inherit;transition:var(--tr);
}
.btn-logout:hover{background:rgba(255,77,109,.18);box-shadow:var(--glow4)}

.main{padding:22px 32px 60px;max-width:1600px;margin:0 auto}
.section-title{
  font-size:.72rem;font-weight:700;color:var(--txt3);text-transform:uppercase;letter-spacing:.12em;
  margin:26px 0 12px;display:flex;align-items:center;gap:8px;
}
.section-title::before{content:'';width:3px;height:14px;background:var(--c1);box-shadow:0 0 8px var(--c1);border-radius:2px}

/* ──────────── KPI GRID ──────────── */
.kpi-grid{display:grid;grid-template-columns:repeat(6,1fr);gap:14px}
.kpi-grid-3{grid-template-columns:repeat(3,1fr)}
.kpi{
  background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius2);
  padding:16px 18px;position:relative;overflow:hidden;transition:var(--tr);
}
.kpi::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--kc,var(--c1)),transparent)}
.kpi:hover{border-color:var(--border2);transform:translateY(-1px)}
.kpi-label{font-size:.66rem;color:var(--txt3);text-transform:uppercase;letter-spacing:.07em;font-weight:700}
.kpi-val{font-size:1.7rem;font-weight:800;margin-top:6px;color:var(--kc,var(--c1));text-shadow:0 0 12px var(--kc,var(--c1))33}
.kpi-sub{font-size:.68rem;color:var(--txt3);margin-top:3px}
.kpi.k1{--kc:var(--c1)} .kpi.k2{--kc:var(--c5)} .kpi.k3{--kc:var(--c2)}
.kpi.k4{--kc:var(--c4)} .kpi.k5{--kc:var(--c5)} .kpi.k6{--kc:var(--c3)}
.kpi.k7{--kc:var(--c1)} .kpi.k8{--kc:var(--c2)} .kpi.k9{--kc:var(--c3)}
.kpi-target{display:flex;align-items:center;gap:6px;font-size:.65rem;color:var(--txt3);margin-top:8px}
.tgt-badge{width:7px;height:7px;border-radius:50%;display:inline-block}
.tgt-badge.ok{background:var(--c3);box-shadow:0 0 6px var(--c3)}
.tgt-badge.warn{background:var(--c5);box-shadow:0 0 6px var(--c5)}
.tgt-badge.bad{background:var(--c4);box-shadow:0 0 6px var(--c4)}
.tgt-badge.na{background:var(--txt3)}

/* ──────────── CHARTS ──────────── */
.charts-row{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px}
.charts-row > .chart-box{min-width:0}
.chart-box.span2{grid-column:span 2}
.chart-box{
  background:var(--bg1);border:1px solid var(--border);border-radius:var(--radius2);
  padding:18px 20px;transition:var(--tr);position:relative;overflow:hidden;
}
.chart-box::before{
  content:'';position:absolute;top:0;left:20px;right:20px;height:1px;
  background:linear-gradient(90deg,transparent,var(--c1),transparent);opacity:.3;
}
.chart-box:hover{border-color:var(--border2)}
.chart-hdr{font-size:.68rem;font-weight:700;color:var(--txt2);text-transform:uppercase;letter-spacing:.08em;margin-bottom:14px;
  display:flex;align-items:center;gap:8px}
.chart-hdr::before{content:'';width:8px;height:8px;border-radius:2px;background:var(--c1);box-shadow:var(--glow1)}
.chart-canvas-wrap{position:relative;height:280px}

/* ── Graphiques cliquables : clic = plein écran ── */
.chart-box.chart-clickable{cursor:zoom-in}
.chart-box.chart-clickable:hover{border-color:var(--c1);box-shadow:0 0 0 1px rgba(0,212,255,.25),0 0 22px rgba(0,212,255,.08)}
.chart-box.chart-clickable .chart-hdr::after{content:'⤢';margin-left:auto;font-size:.95rem;font-weight:600;text-transform:none;color:var(--txt3,#4a6a88);opacity:.8}
.chart-box.chart-full{
  position:fixed;inset:0;z-index:9500;margin:0;border-radius:0;border:none;
  padding:26px 44px 30px;background:var(--bg0);cursor:default;
  display:flex;flex-direction:column;overflow:auto;animation:chartFullIn .18s ease-out;
}
.chart-box.chart-full::before{left:44px;right:44px}
.chart-box.chart-full:hover{box-shadow:none;border-color:transparent}
.chart-box.chart-full .chart-hdr{font-size:1.05rem;margin-bottom:14px;padding-right:64px}
.chart-box.chart-full .chart-hdr::after{display:none}
.chart-box.chart-full .chart-canvas-wrap{flex:1 1 auto;height:auto;min-height:0}
.chart-full-close{
  position:absolute;top:18px;right:28px;z-index:2;display:flex;align-items:center;justify-content:center;
  width:40px;height:40px;padding:0;line-height:1;
  background:rgba(0,212,255,.08);border:1px solid var(--border2);color:var(--txt2);
  border-radius:10px;font-size:1.15rem;font-weight:600;cursor:pointer;font-family:inherit;
}
.chart-full-close:hover{background:rgba(255,45,120,.15);border-color:#ff2d78;color:#fff}
body.chart-full-open{overflow:hidden}
@keyframes chartFullIn{from{opacity:0;transform:scale(.985)}to{opacity:1;transform:none}}
canvas{width:100%!important}

/* ──────────── LISTE CATÉGORIES DE FRAUDE ──────────── */
.cat-list{display:flex;flex-direction:column;gap:12px;min-height:280px;justify-content:center}
.cat-row{display:flex;align-items:center;gap:8px}
.cat-name{flex:0 0 88px;font-size:.7rem;line-height:1.25;color:var(--txt2);white-space:normal;overflow:hidden;word-break:break-word}
.cat-bar-track{flex:1;min-width:0;height:9px;background:rgba(255,255,255,.05);border-radius:5px;overflow:hidden}
.cat-bar-fill{height:100%;border-radius:5px;box-shadow:0 0 6px currentColor}
.cat-pct{flex:0 0 42px;text-align:right;font-size:.74rem;font-weight:700;color:var(--txt1)}

/* ──────────── TOOLBAR ──────────── */
.toolbar{
  display:flex;align-items:center;gap:8px;flex-wrap:wrap;
  background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius2);
  padding:12px 16px;margin-bottom:12px;
}
.toolbar-spaced{margin-top:22px}
.tb-label{font-size:.72rem;font-weight:700;color:var(--txt2);margin-right:4px}
.divider{width:1px;height:20px;background:var(--border2)}
.chip{
  padding:6px 13px;border-radius:20px;border:1px solid var(--border);background:transparent;
  color:var(--txt2);cursor:pointer;font-size:.72rem;font-weight:600;font-family:inherit;transition:var(--tr);
}
.chip:hover{border-color:var(--c1);color:var(--c1)}
.chip.on{background:var(--c1);color:#03060f;border-color:var(--c1);font-weight:700;box-shadow:var(--glow1)}
.chip.esc.on{background:var(--c4);border-color:var(--c4);box-shadow:var(--glow4)}
.chip.rev.on{background:var(--c5);border-color:var(--c5);box-shadow:var(--glow5)}
.chip.cls.on{background:var(--c3);border-color:var(--c3);color:#03060f;box-shadow:var(--glow3)}
.search-wrap{display:flex;align-items:center;gap:8px;background:var(--bg3);border:1px solid var(--border);border-radius:20px;padding:6px 14px;flex:1;min-width:180px;max-width:280px}
.search-wrap svg{width:14px;height:14px;color:var(--txt3);flex-shrink:0}
.search-wrap input{background:transparent;border:none;outline:none;color:var(--txt1);font-size:.75rem;font-family:inherit;width:100%}

/* ──────────── TABLE ──────────── */
.table-wrap{background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius2);overflow:hidden}
.tscroll{overflow-x:auto}
table{width:100%;border-collapse:collapse;min-width:1000px}
col.col-id    { width: 150px; }
col.col-amt   { width: 110px; }
col.col-rule  { width: 240px; }

col.col-card  { width: 150px; }
col.col-city  { width: 175px; }
col.col-risk  { width: 150px; }

col.col-dec   { width: 130px; }
col.col-date  { width: 150px; }


thead{background:rgba(0,212,255,.03)}
th{
  padding:11px 13px;text-align:left;font-size:.63rem;font-weight:700;
  color:var(--txt3);text-transform:uppercase;letter-spacing:.1em;
  border-bottom:1px solid var(--border);white-space:nowrap;user-select:none;
  overflow:hidden;text-overflow:ellipsis;
}
td{
  padding:10px 13px;font-size:.78rem;
  border-bottom:1px solid rgba(0,212,255,.04);
  vertical-align:middle;
  overflow:hidden;
  text-overflow:ellipsis;
  white-space:nowrap;
}
tbody tr{cursor:pointer;transition:background .15s;}
tbody tr:hover td{background:rgba(0,212,255,.04);}
tbody tr.row-esc:hover td{background:rgba(255,77,109,.04)!important}
tbody tr.row-rev:hover td{background:rgba(255,170,0,.04)!important}
tbody tr.row-cls:hover td{background:rgba(0,255,157,.03)!important}
tbody tr:last-child td{border-bottom:none}
.cell-id{font-family:'Calisto MT',Georgia,'Times New Roman',serif;font-size:.7rem;color:var(--c1);font-weight:600;
  text-shadow:0 0 8px rgba(0,212,255,.4);display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.cell-rule{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--txt1)}
.cell-ip{font-family:'Calisto MT',Georgia,'Times New Roman',serif;font-size:.73rem;color:var(--txt2);display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.cell-city{font-family:'Calisto MT',Georgia,'Times New Roman',serif;font-size:.75rem;color:var(--txt2);display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.flag{display:inline-block;box-sizing:border-box;width:18px;height:12px;margin-right:7px;vertical-align:-1px;border-radius:2px;object-fit:cover;box-shadow:0 0 0 1px rgba(255,255,255,.14);flex:none}
.flag-none{box-shadow:none;border:1px dashed rgba(255,255,255,.18)}
.cell-date{font-size:.68rem;color:var(--txt3);white-space:nowrap;font-family:'Calisto MT',Georgia,'Times New Roman',serif}

/* ──────────── BADGES ──────────── */
.badge{display:inline-flex;align-items:center;gap:4px;padding:3px 10px;border-radius:20px;font-size:.65rem;font-weight:700;white-space:nowrap;letter-spacing:.03em}
.badge-fs{background:rgba(255,170,0,.1);color:var(--c5);border:1px solid rgba(255,170,0,.25);box-shadow:0 0 8px rgba(255,170,0,.08)}
.badge-qr{background:rgba(191,95,255,.1);color:var(--c2);border:1px solid rgba(191,95,255,.25);box-shadow:0 0 8px rgba(191,95,255,.08)}

.vbadge{display:inline-flex;align-items:center;gap:5px;padding:3px 11px;border-radius:20px;font-size:.68rem;font-weight:700;letter-spacing:.02em}
.vb-tp{background:rgba(255,77,109,.1);color:var(--c4);border:1px solid rgba(255,77,109,.28);box-shadow:0 0 10px rgba(255,77,109,.1)}
.vb-fp{background:rgba(0,255,157,.08);color:var(--c3);border:1px solid rgba(0,255,157,.22);box-shadow:0 0 10px rgba(0,255,157,.07)}
.vb-sus{background:rgba(255,170,0,.08);color:var(--c5);border:1px solid rgba(255,170,0,.22);box-shadow:0 0 10px rgba(255,170,0,.07)}
.vb-unk{background:rgba(139,172,200,.06);color:var(--txt3);border:1px solid var(--border)}

.dbadge{display:inline-flex;align-items:center;gap:4px;padding:4px 12px;border-radius:6px;font-size:.68rem;font-weight:800;letter-spacing:.05em;text-transform:uppercase}
.db-esc{background:rgba(255,77,109,.12);color:var(--c4);border:1px solid rgba(255,77,109,.28);box-shadow:0 0 12px rgba(255,77,109,.12)}
.db-rev{background:rgba(255,170,0,.1);color:var(--c5);border:1px solid rgba(255,170,0,.28);box-shadow:0 0 12px rgba(255,170,0,.1)}
.db-cls{background:rgba(0,255,157,.08);color:var(--c3);border:1px solid rgba(0,255,157,.22);box-shadow:0 0 12px rgba(0,255,157,.07)}
.db-unk{background:rgba(139,172,200,.06);color:var(--txt3);border:1px solid var(--border)}

/* ──────────── SCORE BAR ──────────── */
.sbar{display:flex;align-items:center;gap:8px}
.strack{flex:1;height:4px;background:rgba(255,255,255,.06);border-radius:3px;overflow:hidden;position:relative}
.sfill{height:100%;border-radius:3px;transition:width .4s ease;box-shadow:0 0 6px currentColor}
.slabel{font-family:'Calisto MT',Georgia,'Times New Roman',serif;font-size:.76rem;font-weight:700;min-width:26px;text-align:right}


/* ──────────── MONTANT ──────────── */
.cell-amt{font-family:'Calisto MT',Georgia,'Times New Roman',serif;font-size:.8rem;font-weight:700;color:var(--txt1)}
.cell-amt.warn{color:var(--c5)}
.cell-amt.danger{color:var(--c4)}

/* ──────────── ID TRANSACTION + COPIE ──────────── */
.idwrap{display:inline-flex;align-items:center;gap:6px}
.copy-btn{
  opacity:0;flex-shrink:0;width:20px;height:20px;border-radius:5px;border:1px solid var(--border2);
  background:var(--bg3);color:var(--txt2);cursor:pointer;display:inline-flex;align-items:center;justify-content:center;
  font-size:.62rem;transition:var(--tr);
}
tbody tr:hover .copy-btn{opacity:1}
.copy-btn:hover{color:var(--c1);border-color:var(--c1);box-shadow:var(--glow1)}
.copy-btn.copied{color:var(--c3);border-color:var(--c3)}

/* ──────────── CATÉGORIE ──────────── */
.cat-ico{margin-right:3px}

/* ──────────── STATUT ANALYSTE (badges) ──────────── */
.vb-pend{background:rgba(139,172,200,.07);color:var(--txt2);border:1px solid var(--border2)}


/* ──────────── PAGINATION ──────────── */
.pagbar{display:flex;align-items:center;justify-content:space-between;padding:12px 18px;border-top:1px solid var(--border)}
.paginfo{font-size:.72rem;color:var(--txt3);font-family:'Calisto MT',Georgia,'Times New Roman',serif}
.pagbtns{display:flex;gap:5px}
.pbtn{
  padding:5px 11px;border-radius:6px;border:1px solid var(--border);background:transparent;
  color:var(--txt2);cursor:pointer;font-size:.72rem;font-family:inherit;transition:var(--tr);
}
.pbtn:hover{border-color:var(--c1);color:var(--c1);background:var(--c1d)}
.pbtn.on{background:var(--c1);color:#03060f;border-color:var(--c1);font-weight:700;box-shadow:var(--glow1)}
.pbtn:disabled{opacity:.25;cursor:not-allowed}

/* ──────────── MODAL ──────────── */
.overlay{
  display:none;position:fixed;inset:0;
  background:rgba(3,6,15,.88);z-index:1000;
  backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
  align-items:flex-start;justify-content:center;padding:28px 16px;overflow-y:auto;
}
.overlay.open{display:flex}
.modal{
  background:var(--bg2);border:1px solid var(--border);
  border-radius:var(--radius2);width:100%;max-width:1240px;
  box-shadow:0 30px 100px rgba(0,0,0,.6), var(--glow1);
  animation:mIn .22s ease-out;
}
@keyframes mIn{from{opacity:0;transform:translateY(18px) scale(.98)}to{opacity:1;transform:translateY(0) scale(1)}}
.m-top{
  display:flex;align-items:flex-start;justify-content:flex-end;
  padding:20px 26px;border-bottom:1px solid var(--border);
  background:linear-gradient(90deg,rgba(0,212,255,.04),transparent);
}
.m-top-l{display:flex;flex-direction:column;gap:8px}
.m-id{font-family:'Calisto MT',Georgia,'Times New Roman',serif;font-size:.82rem;color:var(--c1);font-weight:600;text-shadow:0 0 10px rgba(0,212,255,.4)}
.m-badges{display:flex;gap:7px;flex-wrap:wrap}
.m-close{
  width:32px;height:32px;border-radius:7px;
  background:rgba(255,77,109,.08);border:1px solid rgba(255,77,109,.2);
  color:var(--c4);cursor:pointer;font-size:.95rem;
  display:flex;align-items:center;justify-content:center;transition:var(--tr);
}
.m-close:hover{background:rgba(255,77,109,.18);box-shadow:var(--glow4)}
.m-body{padding:22px 26px}

/* Scoring hero */
.hero{
  display:flex;align-items:center;gap:18px;
  padding:16px 20px;
  background:var(--bg3);border:1px solid var(--border);border-radius:var(--radius2);
  margin-bottom:18px;
  position:relative;overflow:hidden;
}
.hero::before{
  content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,var(--c1),transparent);opacity:.4;
}
.score-ring{
  width:66px;height:66px;border-radius:50%;
  display:flex;align-items:center;justify-content:center;
  font-size:1.3rem;font-weight:900;
  border:3px solid var(--ring-color,var(--c1));
  color:var(--ring-color,var(--c1));
  box-shadow:0 0 20px var(--ring-color,var(--c1)),inset 0 0 20px rgba(0,212,255,.05);
  flex-shrink:0;
  text-shadow:0 0 12px var(--ring-color,var(--c1));
}
.hero-info{flex:1;min-width:0}
.hero-decision{font-weight:700;font-size:.9rem}
.hero-sub{font-size:.72rem;color:var(--txt2);margin-top:3px}
.hero-factors{font-size:.68rem;color:var(--txt3);margin-top:4px}
.actions-box{
  margin-top:10px;padding:10px 14px;
  background:rgba(255,170,0,.04);border:1px solid rgba(255,170,0,.15);
  border-radius:var(--radius);
}
.actions-title{font-size:.63rem;font-weight:700;color:var(--c5);text-transform:uppercase;letter-spacing:.08em;margin-bottom:5px}
.actions-list{list-style:none}
.actions-list li{font-size:.7rem;color:var(--txt2);padding:2px 0;display:flex;gap:6px}
.actions-list li::before{content:'›';color:var(--c5)}

/* Explanation */
.explain{
  background:var(--bg3);border-left:3px solid var(--c1);
  padding:14px 18px;margin-bottom:18px;border-radius:0 var(--radius) var(--radius) 0;
  box-shadow:inset 0 0 20px rgba(0,212,255,.03);
}
.explain-title{font-size:.63rem;font-weight:700;color:var(--c1);text-transform:uppercase;letter-spacing:.1em;margin-bottom:7px;
  text-shadow:0 0 8px rgba(0,212,255,.4)}
.explain-text{font-size:.82rem;color:var(--txt2);line-height:1.55}

/* Info grids */
.igrid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px}
.ipanel{
  background:var(--bg3);border:1px solid var(--border);border-radius:var(--radius2);
  padding:14px 16px;overflow:hidden;position:relative;
}
.ipanel::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,var(--panel-accent,var(--c1)),transparent);opacity:.35}
.ipanel.pc1{--panel-accent:var(--c1)}
.ipanel.pc2{--panel-accent:var(--c2)}
.ipanel.pc3{--panel-accent:var(--c3)}
.ipanel.pc4{--panel-accent:var(--c5)}
.ipanel.pc5{--panel-accent:var(--c4)}
.ptitle{
  font-size:.63rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;
  margin-bottom:10px;display:flex;align-items:center;gap:7px;
  color:var(--panel-accent,var(--c1));
  text-shadow:0 0 8px var(--panel-accent,var(--c1));
}
.ptitle::before{content:'';width:5px;height:5px;border-radius:50%;background:var(--panel-accent,var(--c1));
  box-shadow:0 0 6px var(--panel-accent,var(--c1))}
.irow{display:flex;justify-content:space-between;align-items:center;padding:4px 0;border-bottom:1px solid rgba(255,255,255,.03)}
.irow:last-child{border-bottom:none}
.ikey{font-size:.84rem;color:#fff;font-weight:500}
.ival{font-size:.72rem;color:var(--txt1);font-weight:500;text-align:right;max-width:58%;word-break:break-all;font-family:'Calisto MT',Georgia,'Times New Roman',serif}
.ival.hi{color:var(--c1);text-shadow:0 0 6px rgba(0,212,255,.3)}
.ival.warn{color:var(--c5);text-shadow:0 0 6px rgba(255,170,0,.3)}
.ival.danger{color:var(--c4);text-shadow:0 0 6px rgba(255,77,109,.3)}
.ival.safe{color:var(--c3);text-shadow:0 0 6px rgba(0,255,157,.3)}
.ival.dim{color:var(--txt3)}

/* JSON toggle */
.jtoggle{
  width:100%;text-align:left;margin-top:10px;
  padding:9px 14px;background:var(--bg3);border:1px solid var(--border);
  border-radius:var(--radius);cursor:pointer;font-size:.75rem;color:var(--txt3);
  font-family:inherit;font-weight:500;transition:var(--tr);
}
.jtoggle:hover{border-color:var(--c1);color:var(--c1);background:var(--c1d)}
.jviewer{
  display:none;margin-top:7px;
  background:var(--bg0);border:1px solid var(--border);border-radius:var(--radius);
  padding:14px;font-family:'Calisto MT',Georgia,'Times New Roman',serif;font-size:.67rem;
  color:var(--c6);overflow:auto;max-height:320px;
  white-space:pre-wrap;word-break:break-all;line-height:1.5;
}
.jviewer.open{display:block}

/* ──────────── UTILS ──────────── */
.empty{text-align:center;padding:44px;color:var(--txt3)}
.spin{display:inline-block;width:18px;height:18px;border:2px solid rgba(0,212,255,.18);
  border-top-color:var(--c1);border-radius:50%;animation:spin .6s linear infinite;box-shadow:var(--glow1)}
@keyframes spin{to{transform:rotate(360deg)}}

/* ──────────── GEO MAP ──────────── */
.geo-row{grid-template-columns:1fr;align-items:stretch}
.geo-map-box{display:flex;flex-direction:column}
.geo-map-wrap{position:relative;width:100%;aspect-ratio:1000/440;max-height:480px;background:#050a14}
.geo-map-wrap svg{position:absolute;inset:0;width:100%;height:100%;display:block;overflow:hidden;touch-action:none}
.geo-map-wrap.zoomed svg{cursor:grab}
.geo-map-wrap.dragging svg{cursor:grabbing}
/* traits d'épaisseur constante quel que soit le zoom */
.geo-land{fill:#16283d;stroke:#233a54;stroke-width:.6;stroke-linejoin:round;vector-effect:non-scaling-stroke}
.geo-borders{fill:none;stroke:rgba(139,172,200,.35);stroke-width:.4;stroke-linejoin:round;vector-effect:non-scaling-stroke}
.geo-marker,.geo-marker-pulse,.geo-flash{vector-effect:non-scaling-stroke}
/* fusion lumineuse : les cercles superposés s'additionnent au lieu de se masquer */
.geo-marker{cursor:pointer;mix-blend-mode:screen;transition:r .8s ease,fill .8s ease,stroke .8s ease,opacity .15s}
.geo-map-wrap.zooming .geo-marker{transition:none}
.geo-marker:hover{opacity:.85}

/* Barre de contrôle de la carte (régions, filtre fraudes, zoom) */
.geo-toolbar{position:absolute;left:12px;right:12px;top:12px;z-index:4;display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap;pointer-events:none}
.geo-toolbar>*{pointer-events:auto}
.geo-tb-group{display:flex;align-items:center;gap:6px;flex-wrap:wrap}
.geo-toolbar .chip:not(.on){background:rgba(6,12,26,.88);backdrop-filter:blur(4px)}
.geo-toolbar .chip{padding:5px 11px;font-size:.68rem}
.geo-zoom-btn{
  width:28px;height:28px;border-radius:8px;border:1px solid var(--border2);background:rgba(6,12,26,.88);
  color:var(--txt1);cursor:pointer;font-size:1rem;font-weight:700;font-family:inherit;line-height:1;
  display:inline-flex;align-items:center;justify-content:center;transition:var(--tr);backdrop-filter:blur(4px);
}
.geo-zoom-btn:hover{border-color:var(--c1);color:var(--c1);box-shadow:var(--glow1)}
.geo-hint{position:absolute;right:12px;bottom:10px;z-index:4;font-size:.6rem;color:var(--txt3);pointer-events:none;letter-spacing:.03em}
.geo-marker-pulse{pointer-events:none;transform-box:fill-box;transform-origin:center;animation:geoPulse 2.2s ease-out infinite}
@keyframes geoPulse{0%{transform:scale(1);opacity:.55}100%{transform:scale(2.8);opacity:0}}
.geo-flash{pointer-events:none;transform-box:fill-box;transform-origin:center;animation:geoFlash 1.4s ease-out forwards}
@keyframes geoFlash{0%{transform:scale(1);opacity:.9}100%{transform:scale(3.4);opacity:0}}

/* Légende posée sur la carte (bas-gauche) */
.geo-legend{position:absolute;left:12px;bottom:12px;z-index:4;display:flex;gap:10px;pointer-events:none}
.geo-legend-box{
  background:rgba(6,12,26,.88);border:1px solid var(--border2);border-radius:10px;
  padding:9px 14px 10px;backdrop-filter:blur(4px);box-shadow:0 4px 18px rgba(0,0,0,.45);
}
.geo-legend-label{display:block;font-size:.6rem;color:var(--txt2);text-transform:uppercase;letter-spacing:.08em;font-weight:700;margin-bottom:8px}
.geo-legend-dots{display:flex;align-items:flex-end;gap:12px}
.geo-legend-item{display:flex;flex-direction:column;align-items:center;gap:4px}
.geo-legend-item i{display:block;border-radius:50%;background:rgba(0,212,255,.55);border:1px solid rgba(0,212,255,.85)}
.geo-legend-item em{font-style:normal;font-size:.56rem;color:var(--txt3)}
.geo-legend-gradient{width:130px;height:8px;border-radius:4px;background:linear-gradient(90deg,#00d4ff,#bf5fff 50%,#ff2d78)}
.geo-legend-scale{display:flex;justify-content:space-between;margin-top:4px;font-size:.56rem;color:var(--txt3)}

.geo-status{position:absolute;left:50%;top:46%;transform:translate(-50%,-50%);z-index:4;display:none;max-width:70%;text-align:center;pointer-events:none;background:rgba(6,12,26,.92);border:1px solid var(--border2);border-radius:12px;padding:14px 20px;font-size:.78rem;line-height:1.6;color:var(--txt2)}
.geo-status b{display:block;color:var(--c1);margin-bottom:4px;font-size:.82rem}
.geo-tooltip{
  position:absolute;pointer-events:none;z-index:5;display:none;
  background:rgba(6,12,26,.97);border:1px solid var(--c1);border-radius:8px;
  padding:9px 13px;font-size:.72rem;color:var(--txt2);line-height:1.55;
  box-shadow:0 6px 22px rgba(0,0,0,.5),0 0 14px rgba(0,212,255,.15);
  white-space:nowrap;transform:translate(-50%,-100%);margin-top:-10px;
}
.geo-tooltip b{color:var(--c1);font-size:.76rem}
@media(max-width:1300px){.kpi-grid{grid-template-columns:repeat(3,1fr)}}
@media(max-width:1000px){.charts-row{grid-template-columns:repeat(2,1fr)}}
@media(max-width:900px){.kpi-grid,.kpi-grid-3{grid-template-columns:repeat(2,1fr)}.igrid{grid-template-columns:1fr}.main{padding:14px}}
@media(max-width:600px){.kpi-grid,.kpi-grid-3{grid-template-columns:1fr}.charts-row{grid-template-columns:1fr}.topbar{padding:10px 14px}}
</style>
</head>
<body>

<!-- TOPBAR -->
<div class="topbar">
  <div class="logo-wrap">
    <div class="logo-hex">FG</div>
    <div>
      <div class="logo-name">FraudGuard ML Dashboard</div>
      <div class="logo-sub">Détection de Fraude Bancaire Temps Réel</div>
    </div>
  </div>
  <div class="topbar-right">
    <div class="rt-pill"><span class="live-dot"></span><span class="rt-label">Temps réel</span><span class="rt-time" id="clk">—</span></div>
    <button class="btn-logout" id="btnLogout" title="Se déconnecter">⏻ Déconnexion</button>
  </div>
</div>

<div class="main">

  <!-- Charts : une seule rangée de 4 blocs -->
  <div class="section-title">Analyse temporelle &amp; distribution</div>
  <div class="charts-row">
    <div class="chart-box">
      <div class="chart-hdr">Principales catégories de fraude</div>
      <div id="catList" class="cat-list"><div class="empty"><div class="spin"></div></div></div>
    </div>
    <div class="chart-box">
      <div class="chart-hdr">Répartition des transactions</div>
      <div class="chart-canvas-wrap"><canvas id="cDec"></canvas></div>
    </div>
    <div class="chart-box chart-clickable" data-fs="cRisk" tabindex="0" role="button" aria-label="Agrandir : Distribution des Risk Scores">
      <div class="chart-hdr">Distribution des Risk Scores (0–100)</div>
      <div class="chart-canvas-wrap"><canvas id="cRisk"></canvas></div>
    </div>
    <div class="chart-box chart-clickable" data-fs="cTrend" tabindex="0" role="button" aria-label="Agrandir : Tendance temporelle">
      <div class="chart-hdr">Tendance temporelle (7 jours)</div>
      <div class="chart-canvas-wrap"><canvas id="cTrend"></canvas></div>
    </div>
  </div>

  <!-- Geo analysis : carte seule -->
  <div class="section-title">Analyse géographique des transactions</div>
  <div class="charts-row geo-row">
    <div class="chart-box geo-map-box">
      <div class="geo-map-wrap">
        <svg id="geoMap" viewBox="0 0 1000 440" preserveAspectRatio="xMidYMid meet"></svg>
        <div class="geo-toolbar">
          <div class="geo-tb-group" id="geoRegions">
            <button class="chip on" data-region="world">Monde</button>
            <button class="chip" data-region="europe">Europe</button>
            <button class="chip" data-region="namerica">Amérique N.</button>
            <button class="chip" data-region="samerica">Amérique S.</button>
            <button class="chip" data-region="africa">Afrique</button>
            <button class="chip" data-region="asia">Asie</button>
            <button class="chip" data-region="oceania">Océanie</button>
          </div>
          <div class="geo-tb-group">
            <button class="chip esc" id="geoFraudOnly" title="Masquer les villes sans aucune fraude">Uniquement les fraudes</button>
            <button class="geo-zoom-btn" id="geoZoomIn" title="Zoom avant">+</button>
            <button class="geo-zoom-btn" id="geoZoomOut" title="Zoom arrière">−</button>
          </div>
        </div>
        <div class="geo-hint">Molette : zoom · Glisser : déplacer</div>
        <div class="geo-tooltip" id="geoTooltip"></div>
        <div class="geo-status" id="geoStatus"></div>
        <div class="geo-legend">
          <div class="geo-legend-box">
            <span class="geo-legend-label">Volume transactions</span>
            <div class="geo-legend-dots" id="geoLegendDots"></div>
          </div>
          <div class="geo-legend-box">
            <span class="geo-legend-label">Taux de fraude (%)</span>
            <div class="geo-legend-gradient"></div>
            <div class="geo-legend-scale"><span>0 %</span><span>100 %</span></div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Toolbar -->
  <div class="toolbar toolbar-spaced">
    <span class="tb-label">Transactions scorées</span>
    <div class="divider"></div>
    <button class="chip esc" data-d="BLOCK">🔴 Bloquée</button>
    <button class="chip rev" data-d="MANUAL_REVIEW">🟡 Revue</button>
    <button class="chip cls" data-d="ALLOW">🟢 Autorisée</button>
    <div class="search-wrap">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>
      <input type="text" id="srch" placeholder="ID, marchand, carte, ville...">
    </div>
  </div>

  <!-- Table -->
  <div class="table-wrap">
    <div class="tscroll">
      <table>
        <colgroup>
          <col class="col-id">
          <col class="col-amt">
          <col class="col-rule">
          <col class="col-card">
          <col class="col-city">
          <col class="col-risk">
          <col class="col-dec">
          <col class="col-date">
        </colgroup>
        <thead>
          <tr>
            <th>ID Transaction</th>
            <th>Montant</th>
            <th>Marchand / Catégorie</th>
            <th>Carte</th>
            <th>Lieu de transaction</th>
            <th>Score Fraude</th>
            <th>Décision</th>
            <th>Traité le</th>
          </tr>
        </thead>
        <tbody id="tbody">
          <tr><td colspan="8"><div class="empty"><div class="spin"></div><br><br>Chargement des transactions...</div></td></tr>
        </tbody>
      </table>
    </div>
    <div class="pagbar">
      <div class="paginfo" id="pInfo">—</div>
      <div class="pagbtns" id="pBtns"></div>
    </div>
  </div>
</div>

<!-- MODAL -->
<div class="overlay" id="ov">
  <div class="modal">
    <div class="m-body" id="mBody"></div>
  </div>
</div>

<script>
// ─── STATE ───
let all=[], filtered=[], srcF='all', decF=null, page=1;
let cDec=null, cRisk=null, cTrend=null;
const PS=25;

// ─── CHART THEME ───
Chart.defaults.color='#4a6a88';
Chart.defaults.borderColor='rgba(0,212,255,.06)';

// Plugin custom : fait clignoter (pulse rouge) le point le plus critique de
// l'histogramme des risk scores, c.-à-d. celui qui combine le risk score le
// plus élevé ET le plus grand nombre de transactions (score*count maximal).
// Remplace l'ancienne ligne verticale statique "SEUIL BLOQUÉ".
const blinkPointPlugin={
  id:'blinkPoint',
  afterDatasetsDraw(chart, args, opts){
    if(!opts || opts.index==null)return;
    const meta=chart.getDatasetMeta(opts.datasetIndex||0);
    const point=meta && meta.data && meta.data[opts.index];
    if(!point)return;
    const {x,y}=point.getProps(['x','y'],true);
    const k=chart._fsScale||1;
    const t=Date.now()/450;
    const pulse=(Math.sin(t)+1)/2; // oscille entre 0 et 1
    const ctx=chart.ctx;
    ctx.save();
    // halo pulsant
    ctx.beginPath();
    ctx.arc(x,y,(7+pulse*7)*k,0,Math.PI*2);
    ctx.fillStyle=`rgba(255,20,60,${0.12+pulse*0.28})`;
    ctx.fill();
    // point central, alterne intensité de rouge
    ctx.beginPath();
    ctx.arc(x,y,4.5*k,0,Math.PI*2);
    ctx.fillStyle=`rgba(255,20,60,${0.55+pulse*0.45})`;
    ctx.fill();
    ctx.lineWidth=1.2;
    ctx.strokeStyle='#03060f';
    ctx.stroke();
    ctx.restore();
    if(!chart._blinkRAF){
      const loop=()=>{
        if(!chart.ctx){chart._blinkRAF=null;return;}
        chart.draw();
        chart._blinkRAF=requestAnimationFrame(loop);
      };
      chart._blinkRAF=requestAnimationFrame(loop);
    }
  },
  afterDestroy(chart){
    if(chart._blinkRAF){cancelAnimationFrame(chart._blinkRAF);chart._blinkRAF=null;}
  }
};
Chart.register(blinkPointPlugin);

// ─── CATÉGORIES (mapping icône + libellé propre) ───
const CATEGORY_MAP={
  'entertainment':{ico:'🎬',label:'Loisirs & Divertissement'},
  'food_dining':{ico:'🍔',label:'Restauration'},
  'gas_transport':{ico:'⛽',label:'Carburant & Transport'},
  'grocery_net':{ico:'🛒',label:'Épicerie en ligne'},
  'grocery_pos':{ico:'🛒',label:'Épicerie'},
  'health_fitness':{ico:'💪',label:'Santé & Fitness'},
  'home':{ico:'🏠',label:'Maison'},
  'kids_pets':{ico:'🐾',label:'Enfants & Animaux'},
  'misc_net':{ico:'🛍️',label:'Achat en ligne divers'},
  'misc_pos':{ico:'🛍️',label:'Achat divers'},
  'personal_care':{ico:'💅',label:'Soins personnels'},
  'shopping_net':{ico:'🛒',label:'Shopping en ligne'},
  'shopping_pos':{ico:'🛒',label:'Shopping'},
  'travel':{ico:'✈️',label:'Voyage'},
};
function catDisplay(cat){
  const c=CATEGORY_MAP[cat];
  if(c)return c.ico+' '+c.label;
  return cat?esc(cat):'—';
}

// ─── UTILS ───
function esc(s){return String(s||'').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]))}
function fmt(d){if(!d)return '—';try{return new Date(d).toLocaleString('fr-FR',{day:'2-digit',month:'2-digit',year:'2-digit',hour:'2-digit',minute:'2-digit'})}catch{return d}}
function scColor(s){if(s>=75)return 'var(--c4)';if(s>=50)return 'var(--c5)';if(s>=25)return '#f6e05e';return 'var(--c3)'}
function shortId(id){
  const s=String(id||'');
  if(s.length<=18)return s;
  return s.slice(0,8)+'…'+s.slice(-5);
}
async function api(u){
  const r=await fetch(u);
  if(r.status===401){window.location.href='/login';throw new Error('Session expirée');}
  if(!r.ok)throw new Error(r.statusText);
  return r.json();
}
function copyId(ev,id){
  ev.stopPropagation();
  const btn=ev.currentTarget;
  navigator.clipboard.writeText(id).then(()=>{
    const old=btn.textContent;
    btn.textContent='✓';btn.classList.add('copied');
    setTimeout(()=>{btn.textContent=old;btn.classList.remove('copied')},1200);
  }).catch(()=>{});
}

// ─── KPI LOAD ───
async function loadData(){
  const a=await api('/dashboard/transactions?limit=5000');
  all=a;
  applyFilters();
}
function setKpi(id,n){document.getElementById(id).textContent=n.toLocaleString('fr-FR')}
function setText(id,t){document.getElementById(id).textContent=t}
function pct(n,tot){return ((n/tot)*100).toFixed(1)+'%'}

// ─── CHARTS ───
async function loadCharts(){
  const [dec,fraudSc,catStats,daily]=await Promise.all([
    api('/stats/decisions'),
    api('/stats/fraud-scores'),
    api('/stats/fraud-categories?limit=5'),
    api('/stats/daily-decisions'),
  ]);

  // ── Répartition des transactions — Bloquées / Revue manuelle / Autorisées ──
  if(cDec)cDec.destroy();
  cDec=new Chart(document.getElementById('cDec').getContext('2d'),{
    type:'doughnut',
    data:{
      labels:['Bloquées','Revue manuelle','Autorisées'],
      datasets:[{
        data:[dec.block||0,dec.manual_review||0,dec.allow||0],
        backgroundColor:['rgba(255,0,0,.75)','rgba(246,224,94,.7)','rgba(24,15,203,.75)'],
        borderColor:['#ff0000','#f6e05e','#180fcb'],
        borderWidth:2,hoverOffset:8,
        hoverBorderColor:['#ff0000','#f6e05e','#180fcb'],
      }]
    },
    options:{
      responsive:true,maintainAspectRatio:false,cutout:'65%',
      plugins:{
        legend:{position:'bottom',labels:{color:'#8bacc8',font:{size:10},padding:8,boxWidth:10}},
        tooltip:{
          backgroundColor:'rgba(9,15,31,.95)',borderColor:'rgba(0,212,255,.2)',borderWidth:1,titleColor:'#00d4ff',bodyColor:'#8bacc8',padding:12,
          callbacks:{
            // Affiche le pourcentage du total (au lieu du nombre brut de transactions)
            label:(item)=>{
              const total=(item.dataset.data||[]).reduce((a,b)=>a+(b||0),0)||1;
              const p=((item.parsed/total)*100).toFixed(1);
              return `${item.label} : ${p}%`;
            }
          }
        }
      }
    }
  });

  // ── Distribution des Risk Scores — courbe reliant les points (score, nb de transactions) ──
  // X = fraud_probability*100 (0 à 100), Y = nombre de transactions ayant ce score.
  const riskScores=(fraudSc&&fraudSc.scores)||Array.from({length:101},(_,i)=>i);
  const riskCounts=(fraudSc&&fraudSc.counts)||new Array(101).fill(0);
  const riskMax=Math.max(4,...riskCounts);

  // ── Point critique à faire clignoter : celui qui combine le risk score le
  // plus élevé ET le plus grand nombre de transactions (score*count maximal).
  // On ignore les scores à 0 (score*0 pondération nulle -> jamais retenus),
  // ce qui écarte naturellement les pics de faible risque au profit du vrai
  // pic dangereux (score élevé + volume élevé).
  let criticalIdx=0, criticalWeight=-1;
  riskCounts.forEach((c,i)=>{
    const score=Number(riskScores[i])||0;
    const weight=score*(c||0);
    if(weight>criticalWeight){criticalWeight=weight;criticalIdx=i;}
  });
  const pointColors=riskCounts.map((_,i)=>i===criticalIdx?'#ff143c':'#00d4ff');
  const pointBorders=riskCounts.map((_,i)=>i===criticalIdx?'#ff143c':'#03060f');
  const pointRadii=riskCounts.map((_,i)=>i===criticalIdx?5:2.5);

  if(cRisk)cRisk.destroy();

  cRisk=new Chart(document.getElementById('cRisk').getContext('2d'),{
    type:'line',
    data:{
      labels:riskScores,
      datasets:[{
        label:'Nombre de transactions',
        data:riskCounts,
        borderColor:'#00d4ff',
        backgroundColor:'rgba(0,212,255,.15)',
        borderWidth:2,
        pointRadius:pointRadii,
        pointHoverRadius:6,
        pointBackgroundColor:pointColors,
        pointBorderColor:pointBorders,
        pointBorderWidth:1,
        tension:.3,
        fill:true,
      }]
    },
    options:{
      responsive:true,maintainAspectRatio:false,
      animation:(fsId==='cRisk')?false:{},   // recréé pendant le plein écran : pas d'animation d'entrée
      interaction:{intersect:false,mode:'index'},
      plugins:{
        legend:{display:false},
        blinkPoint:{index:criticalIdx,datasetIndex:0},
        tooltip:{
          backgroundColor:'rgba(9,15,31,.95)',borderColor:'rgba(0,212,255,.2)',borderWidth:1,
          titleColor:'#00d4ff',bodyColor:'#8bacc8',padding:12,
          callbacks:{
            title:(items)=>'Risk Score : '+items[0].label,
            label:(item)=>(item.parsed.y||0)+' transaction'+(item.parsed.y===1?'':'s')
          }
        }
      },
      scales:{
        y:{beginAtZero:true,suggestedMax:riskMax,ticks:{precision:0,color:'#4a6a88',font:{size:10}},grid:{color:'rgba(0,212,255,.05)'},title:{display:true,text:'Nombre de transactions',color:'#4a6a88',font:{size:10}}},
        x:{
          grid:{display:false},
          ticks:{color:'#4a6a88',font:{size:9},autoSkip:true,maxTicksLimit:11,maxRotation:0},
          title:{display:true,text:'Risk Score',color:'#4a6a88',font:{size:10}}
        }
      }
    }
  });

  // ── Principales catégories de fraude (parmi les transactions BLOCK / MANUAL_REVIEW) ──
  renderFraudCategories((catStats&&catStats.categories)||[]);

  // ── Tendance temporelle — volume total vs décisions à risque (7 derniers jours) ──
  renderTrendChart(daily||{});
  reapplyChartFullscreen();

  // ── Analyse géographique : carte alimentée par son propre flux temps réel (startGeoRealtime) ──
}

// ─────────────────────────────────────────────────────────────────────────
// ─── ANALYSE GÉOGRAPHIQUE : carte mondiale (SVG, sans dépendance externe) ──
// ─────────────────────────────────────────────────────────────────────────
// Tracés géographiques réels (Natural Earth 110m, via world-atlas), projetés
// une fois pour toutes en coordonnées SVG (projection plate carrée identique
// à geoProject() ci-dessous, pour un alignement exact des marqueurs). Aucune
// tuile ni service cartographique externe n'est chargé au runtime — tout est
// embarqué en dur dans la page (utile si le serveur bancaire n'a pas d'accès
// internet sortant).
const GEO_VB_W=1000, GEO_VB_H=440, GEO_LAT_TOP=78, GEO_LAT_BOTTOM=-58;
function geoProject(lon,lat){
  const x=(lon+180)/360*GEO_VB_W;
  const y=(GEO_LAT_TOP-lat)/(GEO_LAT_TOP-GEO_LAT_BOTTOM)*GEO_VB_H;
  return [x,y];
}
const WORLD_LAND_D='M994.8,309L995.5,308.5L996.4,309.4L996,311.1L994.3,311.5L992.7,311.1L992.5,309.7L993.5,308.6ZM1000,305.9L998.2,306.7L996.5,307.4L996.1,306.2L997.5,305.5L998.4,305.3L1000,304.3L1000,304.3L1000,305.9ZM0,304.3L0,304.3L0,304.3L0.6,304.2L0.2,305.7L0,305.9L0,305.9L0,304.3ZM635.8,128.4L636.7,130.8L639.3,131.4L641.2,133.1L645.2,133.6L649.5,132.8L649.8,132L649.3,129.7L649.7,126.3L647.5,125.2L648.2,123L646.4,122.8L647,120.1L649.6,120.9L652,119.9L650,117.9L649.2,116.1L647,116.9L646.7,119.3L645.8,117.2L645.7,116.4L646.4,115L645.8,113.9L642.6,112.8L641.4,109.9L639.8,109.1L639.7,108L642.4,108.3L642.5,106L644.9,105.4L647.3,105.9L647.8,102.8L647.3,100.8L644.6,100.9L642.2,100.1L639,101.6L636.4,102.2L635.1,104.2L632.4,104.7L629.7,108L632.2,111.1L631.9,113.3L635,117.1L636.4,118.8L637.8,121.1L639.1,121.2L640,122.1L637.7,122.4L637.2,124.9L636.7,126L635.7,126.8ZM0,29.2L0,29.2L6.8,31.7L14.1,34.9L13.8,36.9L15.7,37.7L15.1,35.4L22.6,35.9L28.1,38.9L25.3,40.3L20.8,40.6L20.7,43.8L19.6,44.5L17,44.4L14.8,43.3L11.2,42.3L10.5,40.9L7.7,40.4L4.6,40.8L3,39.7L3.6,38.5L0.3,39.2L1.6,40.8L0,42.1L0,42.1L0,29.2ZM1000,42.1L1000,42.1L996.4,43.6L992.8,43.3L995.3,45L997,47.7L998.3,48.6L998.6,49.9L997.9,50.8L992.7,50.1L984.9,52.5L982.4,52.9L978.2,55.2L974.2,57.1L973.1,58.6L969.2,56.4L961.9,58.9L960.7,57.7L958,59.1L954.3,58.7L953.4,60.8L950,63.9L950.1,65.2L953.3,66L952.9,70.7L950.4,70.8L949.2,73.5L950.3,74.9L945.5,76.5L944.5,80.2L940.4,81L939.5,84.3L935.5,87.3L934.5,85.1L933.3,80.4L931.8,73.2L933.1,68.7L935.4,66.8L935.6,65.2L939.9,64.5L944.9,60.5L949.6,57.1L954.6,54.5L956.9,50L953.5,50.3L951.8,52.9L944.8,56.5L942.5,52.5L935.3,53.6L928.4,59L930.7,61L924.5,61.8L920.2,62.2L920.4,59.8L916.1,59.3L912.6,60.9L904.1,60.4L895,61.3L886,67.7L875.3,75.3L879.7,75.7L881.1,77.7L883.8,78.4L885.6,76.8L888.6,77L892.6,80.6L892.7,83.3L890.5,86.6L890.3,90.4L889.1,95.6L884.9,100.3L883.9,102.5L880.2,106.3L876.4,110L874.6,111.9L870.9,113.8L869.2,113.9L867.4,112.3L863.7,114.7L863.3,115.8L862.2,115.6L861,116.7L860.2,117.8L860.3,120.1L858.9,120.8L858.4,121.4L857.3,122.3L855.5,122.9L854.3,123.7L854.2,125.1L853.8,125.5L855,126L856.5,127.4L858.9,131.2L859.6,133.3L859.6,137.1L858.6,138.9L856.1,139.5L853.8,140.8L851.3,141.1L851,139.3L851.5,136.9L850.3,133.5L852.4,133L850.5,130.2L849.1,129.6L848.8,130.2L848,130.5L847.9,129.9L847.2,129.6L846.4,129.1L847.2,127.6L847.8,127.3L847.6,126.7L848.3,124.9L848.1,124.4L846.5,124L845.2,123.2L841.3,124.1L839.3,125.6L836.3,126.5L837.7,125L837.2,123.7L839.4,121.6L837.9,119.9L835.5,121L832.3,123.3L830.6,125.4L827.9,125.5L826.5,127L827.9,129.2L830.2,129.7L830.3,131.2L832.5,132.1L835.6,129.8L838.1,131.1L839.9,131.2L840.3,132.9L836.4,133.8L835.1,135.5L832.4,137.1L831,139.4L834,141.2L835.1,144.4L836.7,147.3L838.6,149.8L838.6,152.2L836.8,153.1L837.5,154.8L839.1,155.8L838.7,158.5L838,161L836.5,161.3L834.4,164.8L832.2,169.1L829.6,172.9L825.8,175.9L821.9,178.6L818.8,179L817.1,180.5L816.1,179.4L814.6,181L810.7,182.6L807.7,183.1L806.8,186.5L805.2,186.7L804.5,184.4L805.2,183.1L801.4,182.1L800.1,182.6L796.4,185.4L794.1,188.4L793.5,190.7L795.6,194.1L798.2,198.3L800.7,200.3L802.4,202.9L803.7,208.9L803.3,214.6L801,216.7L797.8,218.8L795.6,221.5L792.1,224.5L791.1,222.5L791.9,220.3L789.8,218.4L787.5,218L786.4,216.3L785,212.9L782.5,211.4L780.1,211.5L780.5,209L778,209L777.8,212.5L776.3,217.3L775.4,220.1L775.6,222.5L777.4,222.6L778.6,225.5L779.1,228.3L780.6,230.2L782.3,230.5L783.7,232.2L784.4,232.5L786,234.5L787.2,236.6L787.3,238.8L787,240.3L787.3,241.4L787.5,243.3L788.5,244.2L789.6,247.1L789.5,248.2L787.6,248.4L784.9,246L781.6,243.4L781.3,241.8L779.7,239.6L779.3,236.9L778.3,235.2L778.6,232.8L778,231.4L776.9,230.2L776.4,228.6L775,226.8L773.6,225.2L773.2,227.1L772.6,225.3L772.9,223.3L773.8,220.2L773.5,217.8L774.3,215.3L773.4,213.4L773.6,209.9L772.5,208.2L771.6,204.3L771.1,200.3L769.9,197.6L768.1,199.2L764.9,201.5L763.4,201.2L761.6,200.5L762.6,196.5L762,193.4L759.8,189.7L760.2,188.5L758.5,188.1L756.6,185.5L755.8,183.8L755.6,182.1L755.1,180.6L753.9,178.7L751.4,178.6L751.6,179.9L750.8,181.7L749.6,181L749.2,181.6L748.4,181.3L747.3,181L746.9,182.2L745,182.1L741.6,182.8L741.8,185.2L740.3,187.2L736.3,189.3L733.2,193.1L731.1,195.2L728.3,197.3L728.3,198.8L726.9,199.6L724.4,200.7L723.1,200.9L722.3,203.4L722.9,207.6L723,210.3L721.8,213.3L721.8,218.8L720.4,219L719.1,221.5L720,222.5L717.4,223.5L716.5,225.7L715.4,226.6L712.8,223.6L711.5,219L710.4,215.8L709.4,214.2L708,211.1L707.3,207.1L706.8,205.1L704.3,200.6L703.1,194.3L702.3,190.2L702.3,186.3L701.7,183.3L697.7,185.2L695.7,184.8L692.1,180.9L693.5,179.7L692.6,178.4L689.4,175.7L687.3,174.9L686.5,172.6L684.4,170.1L679.2,170.7L674.7,170.8L670.8,171.2L665.6,170.2L662.6,169.5L659.4,169.1L658.3,165.1L656.9,164.5L654.8,165.1L652,166.7L648.6,165.6L645.8,163.1L643.1,162.2L641.3,159.1L639.2,154.8L637.7,155.3L635.9,154.3L634.9,155.5L633.3,155.4L633.8,156.8L633.6,157.5L634.5,160L635.6,162.8L636.9,163.5L637.4,164.6L639.3,166L639.5,167.3L639.2,168.4L639.6,169.5L640.4,170.4L640.7,171.5L641.1,172.3L641,169.9L641.7,168.2L642.5,167.9L643.3,168.9L643.4,170.8L642.7,172.7L643.3,173.9L643.8,173.8L643.9,174.6L646,174.1L648.3,174.2L650,174.3L651.9,172.1L654,170L655.8,168.1L656.6,167L656.9,167.2L656.6,168.6L656.3,169.2L656.7,171.7L657.9,173.9L659.5,175.1L661.5,175.5L663.1,176.1L664.4,178L665.1,179L666.1,179.4L666.1,180.2L665.1,182.1L664.7,183L663.5,184L662.5,186.3L661.2,186.1L660.6,186.9L660.2,188.5L660.5,190.7L660.3,191.1L659,191L657.2,192.3L657,193.8L656.3,194.5L654.6,194.5L653.5,195.3L653.5,196.6L652.2,197.5L650.7,197.2L648.8,198.3L647.5,198.5L645.5,199.3L645,200.8L644.9,201.9L642.1,203.3L637.7,204.8L635.2,207L634,207.2L633.2,207L631.5,208.4L629.8,209L627.4,209.2L626.7,209.4L626.1,210.2L625.4,210.4L625,211.3L623.6,211.2L622.7,211.6L620.8,211.5L620.1,209.6L620.1,207.8L619.7,206.9L619.1,204.5L618.3,203.1L618.9,203L618.6,201.5L619,200.9L618.8,199.5L618.5,198.1L617.6,197.1L617.4,195.8L616,194.7L614.5,191.9L613.7,189.3L611.8,187.1L610.6,186.6L608.7,183.5L608.4,181.2L608.5,179.3L606.9,175.7L605.6,174.5L604.1,173.8L603.2,171.9L603.4,171.2L602.6,169.5L601.8,168.8L600.7,166.4L599,163.8L597.6,161.6L596.2,161.6L596.6,159.8L596.8,158.7L597.1,157.4L597,156.9L596.2,158.2L595.6,160.7L594.9,162.3L594.2,162.9L593.3,161.9L592,160.4L590.1,155.8L589.8,156.1L590.9,159.5L592.6,162.7L594.7,167.8L595.8,169.5L596.7,171.4L599.1,174.9L598.6,175.5L598.7,177.6L601.9,180.5L602.4,181.2L603.3,184.4L602.7,184.9L603.1,188.3L604.1,192.1L605.2,192.9L606.7,194.1L608.3,197.9L609.1,200.8L610.6,202.4L614.4,205.5L615.9,207.3L617.4,209.2L618.3,210.3L619.7,211.3L620.3,212.3L620.2,213.6L618.7,214.4L619.8,215.3L620.8,215.9L621.3,217.2L622.5,218.6L623.9,218.6L626.5,217.7L629.6,217.4L632,216.4L633.4,216.1L634.4,215.5L636,215.4L636.9,215.4L638.1,214.9L639.6,214.6L640.9,213.5L642,213.4L642,214.3L641.8,216.2L641.8,217.9L641.2,219.1L640.4,222.6L639.1,226.2L637.4,230.3L635,235.1L632.6,238.7L629.3,243.1L626.6,245.7L622.4,248.9L619.8,251.4L616.8,255.3L616.1,257L615.5,257.8L613.6,259.1L612.9,260.4L611.8,260.7L611.4,263L610.6,264.3L610,266.4L608.9,267.5L608.9,267.5L607.6,271.5L607.8,273.3L609.6,274.5L609.6,275.3L608.9,277.3L609,278.3L608.9,279.8L609.8,281.8L611,285L612,285.7L612.4,287.2L612.3,290.4L612.7,293.2L612.8,298.3L613.3,299.9L612.4,302.2L611.4,304.4L609.6,306.4L607,307.7L603.9,309.3L600.8,312.7L599.7,313.3L597.8,315.6L596.6,316.4L596.4,318.7L597.7,321.1L598.3,323L598.3,324L598.8,323.8L598.7,327L598.3,328.5L598.9,329.1L598.5,330.4L597.3,331.5L595,332.6L591.7,334.4L590.5,335.6L590.7,336.9L591.4,337.2L591.2,338.9L590.5,341.2L590.2,343.9L589.5,345.4L587.6,347L587,347.5L585.8,349.1L585.1,350.8L583.5,353.1L580.3,356.4L578.4,358.4L576.3,359.9L573.4,361.1L572,361.3L571.6,362.2L569.9,361.7L568.5,362.3L565.5,361.7L563.9,362.1L562.7,361.9L559.8,363.2L557.5,363.7L555.8,364.9L554.5,365L553.3,363.9L552.4,363.8L551.2,362.3L551,362.8L550.7,361.9L550.7,360L549.8,357.9L550.7,357.3L550.6,354.8L548.8,351.8L547.4,349L547.4,349L545.4,344.8L543.3,342.4L542.3,340L541.6,336.9L541,334.5L540,329.5L540,325.7L539.6,323.9L538.5,322.6L537.1,319.9L535.6,316L535,314L532.8,310.8L532.6,308.3L532.3,306.3L532.7,303.5L533.7,300.5L533.8,299.1L534.7,296.2L535.4,294.9L537,292.7L537.9,291.3L538.2,288.9L538,287.1L537.2,285.9L536.4,284L535.8,282L535.9,281.3L536.8,280.1L535.9,276.9L535.4,274.8L534,272.7L534.2,272.1L533.8,271.1L533.1,268.7L530.8,265.2L528,262L526.1,259.3L524.4,255.9L524.5,254.9L525.1,253.8L525.8,251.5L526.4,249.1L525.8,248.6L526.8,245L527.2,242.4L526.1,240.3L524.9,239.7L524.3,238.3L523.6,237.8L523.6,236.9L520.7,238.1L519.7,237.9L518.6,238.6L516.4,238.6L514.9,236.5L514,234.2L512,232.1L509.9,232.1L507.5,232.1L505.2,232.5L502.9,233.2L498.6,235.1L497,236.2L494.5,237.1L492.1,236.2L490.8,236.2L488.9,235.6L487.1,235.6L483.8,236.2L481.9,237.1L479.1,238.3L478.6,238.2L477.8,238.3L475,236.7L472.5,234.3L470.1,232.5L468.2,230.4L467.5,230.2L465.5,228.9L464,227.1L463.5,225.9L463.2,223.5L462,221.6L460.9,220.4L460.2,220L459.5,219.3L459.2,217.9L458.8,217.2L458,216.6L456.5,215.3L455.3,215.1L454.7,214.2L454.7,213.7L453.9,213L453.7,212.3L453.2,209.8L453.6,208.4L452.4,205.8L451,204.7L452.3,204.1L453.6,201.8L454.3,200.1L454,198.4L454.8,196.8L455.1,193.8L454.8,190.6L454.5,189L454.8,187.3L454.1,185.8L452.6,184.4L452.7,183L452.9,181.5L453.9,180.7L454.8,179L454.6,177.9L455.6,175.6L457.2,173.5L458.1,173L458.8,171.1L458.9,169.4L459.9,167.4L461.7,166.2L463.5,162.9L463.6,162.9L464.9,161.6L467.5,161.3L469.7,159.1L471.1,158.2L473.4,155.5L472.7,151.5L473.8,148.7L474.2,147L476,144.8L478.7,143.3L480.8,142L482.7,138.6L483.5,136.7L485.6,136.7L487.2,138L489.9,137.8L492.8,138.5L494,138.6L496.6,136.8L499.6,136.2L501.4,134.9L504.1,133.9L508.8,133.3L513.4,133.1L514.8,133.6L517.4,132.3L520.4,132.3L521.5,133L523.4,132.8L526.4,131.5L528.4,131.9L528.3,133.5L530.6,132.4L530.8,133L529.4,134.6L529.4,136.1L530.4,136.9L530,139.7L528.2,141.3L528.7,143L530.2,143.1L530.9,144.6L531.9,145.1L535.2,146.3L536.3,146L538.7,146.5L542.3,148L543.6,150.8L546.1,151.5L550.1,152.8L553,154.4L554.4,153.6L555.7,152.1L555.1,149.6L555.9,148.1L557.9,146.5L559.8,146.1L563.6,146.8L564.5,148.2L565.6,148.2L566.5,148.8L569.2,149.2L569.9,150.2L573.6,150.2L576.3,151L579,152L580.3,152.5L582.5,151.5L583.6,150.5L586,150.3L588,150.7L588.8,152.3L589.4,151.2L591.6,152L593.8,152.2L595.2,151.3L596,150.3L595.8,150.1L596.5,148.6L597.1,146.1L597.5,145.3L597.6,145.3L598.6,142.7L599.9,140.4L600,140.3L599.7,137.8L600.4,136.5L599.4,135L600.4,133.8L598.8,134.1L596.4,133.3L594.5,135.2L590.3,135.5L588.1,133.8L585.1,133.7L584.4,135L582.5,135.4L579.8,133.7L576.8,133.8L575.1,130.5L573.1,128.7L574.5,126.2L572.7,124.7L575.8,121.6L580.1,121.5L581.2,119L586.5,119.4L589.9,117.3L593.1,116.4L597.7,116.3L602.5,118.6L606.5,119.9L609.8,119.4L612.1,119.7L615.4,118L615.8,116.6L615.1,114.4L613.5,113.2L612,112.8L611,111.8L607.4,109.1L604.3,107.9L601.9,106L603.9,105.5L606.2,102.7L604.6,101.5L608.7,100.1L608.7,99.4L606.2,100L604,100.2L602.1,101.3L599.5,101.4L597.1,102.6L597.3,104.4L597.3,104.7L598.6,105.4L601.5,105.2L600.9,106.4L597.9,107L594.1,108.8L592.6,108.2L593.2,106.7L590.1,105.7L590.6,105.1L593.3,104L592.9,103.6L592.5,103.3L588.2,102.5L588,101.2L585.4,101.6L584.4,103.4L582.2,105.8L582.3,106.6L580.9,107.3L580.1,107L579.3,110.9L577.9,112.3L576.9,114.6L577.8,116.4L578.1,117.7L580.5,118.7L580,119.5L576.7,119.7L575.5,120.7L573.2,122.5L572.3,120.9L572.4,120.3L570.7,120.2L569.2,119.9L565.9,120.7L567.8,122.5L566.4,123.1L564.8,123.1L563.4,121.4L562.9,122.1L563.5,124L564.9,125.6L563.8,126.3L565.4,127.8L566.7,128.7L566.8,130.5L564.2,129.7L565,131.3L563.3,131.7L564.3,134.5L562.5,134.6L560.2,133.2L559.2,130.6L558.7,128.4L557.6,126.9L556.2,125.1L556,124.2L555.5,123.9L555.4,123.2L553.9,122.1L553.7,120.6L553.9,118.4L554.3,117.4L553.8,116.9L553.2,116.6L552.4,115.6L551.2,114.9L548.6,113.7L547,112.6L544.5,111.6L542.2,109.2L542.7,109L541.4,107.6L541.4,106.5L539.6,106L538.8,107.4L537.9,106.3L538,105.2L538.1,105.1L538.7,104.9L536.5,104.4L534.2,105.5L534.4,107.1L534.1,108.1L535,109.7L537.6,111.3L539,114L542.1,116.6L544.2,116.6L544.9,117.3L544.1,118L546.6,119.1L548.7,120.1L551,121.8L551.3,122.4L550.8,123.6L549.3,122L546.9,121.5L545.7,123.6L547.7,124.8L547.4,126.5L546.2,126.7L544.7,129.5L543.6,129.7L543.6,128.7L544.1,127L544.7,126.3L543.7,124.4L542.8,122.8L541.7,122.4L540.8,121L539.1,120.4L537.9,119.1L535.8,118.9L533.6,117.4L531.1,115.3L529.2,113.5L528.3,110.3L527,109.9L524.7,108.8L523.4,109.3L521.8,110.8L520.7,111L518.1,112.8L512.7,111.9L508.6,113L508.3,114.9L508.4,116.8L505.8,119L502.3,119.7L502,120.7L500.3,122.5L499.2,125.2L500.3,127L498.7,128.5L498.1,130.6L496,131.2L494,133.7L490.5,133.7L487.9,133.7L486.1,134.8L485.1,136.1L483.7,135.8L482.7,134.7L481.9,132.8L479.3,132.3L478.2,133.2L476.7,132.7L475.3,133.1L475.7,130.5L475.4,128.6L474.2,128.3L473.5,127L473.8,124.9L474.9,123.7L475.1,122.4L475.6,120.5L475.6,119.1L475,117.9L474.9,116.9L475,114.6L473.9,113.1L477.8,110.8L481.2,111.4L485,111.4L487.9,111.9L490.2,111.8L494.7,111.9L496.2,109.9L496.7,103.5L493.8,100.1L491.8,98.5L487.5,97.2L487.2,94.8L490.8,94.1L495.5,95L494.6,91.3L497.3,92.7L503.7,90.2L504.6,87.5L507,86.9L509.2,86.2L510.6,85.3L513.1,80.6L516.9,79.2L519.2,79.3L519.7,78.6L522,78.5L522.6,79.2L524.4,77.6L523.8,76.4L523.7,74.5L522.6,72.7L522.5,69.4L522.9,68.6L523.7,67.6L526.2,67.4L527.2,66.5L529.4,65.6L529.3,67.2L528.5,68.3L528.8,69.2L530.3,69.7L529.6,70.9L528.8,70.6L526.8,72.9L527.6,74.5L527.6,75.7L530.4,76.5L530.4,77.6L533.2,77L534.8,76.1L537.9,77.4L539.2,78.4L541.1,77.5L545.5,76L549,74.9L551.7,75.4L551.9,76.2L554.6,76.3L555.2,74.8L559.1,73.8L558.5,71.1L558.6,68.6L559.9,66.6L562.6,65.5L564.8,67.9L567,67.9L567.5,65.4L567.9,63.5L566.8,63.9L565.1,62.7L564.8,60.9L568.3,60L571.8,59.5L574.9,60L577.7,59.9L580.9,58.1L578,56.6L572.9,56.9L568,58L563.5,58.7L561.9,57L559.2,55.9L559.8,52.7L558.5,49.8L559.8,47.9L562.3,45.9L568.7,42.4L570.5,41.7L570.3,40.3L566.4,38.8L561.6,39.7L558.9,42L559.4,44L554.9,46.6L549.6,49.3L547.6,53.9L549.5,56.2L552.2,58L549.6,61.6L546.7,62.4L545.7,67.8L544.1,70.8L540.7,70.5L539.2,73.1L536,73.2L535.1,70.2L532.7,66.5L530.6,61.9L528.8,59.9L523.3,63.7L519.6,64.5L515.7,62.8L514.7,59.3L513.9,51.9L516.4,49.8L523.8,47.1L529.2,43.7L534.3,39.2L541,33L545.7,30.5L553.3,26.5L559.4,25.1L564,25.2L568.2,22.5L573.2,22.7L578.2,22L586.9,24.4L583.3,25.3L586.4,27.3L589.3,26.2L593.8,28.1L601.4,28.9L611.9,32.6L614.1,34.1L614.2,36.3L611.2,38L606.6,38.8L594.2,36.4L592.2,36.8L596.7,39.1L596.9,40.6L597.1,44L600.6,44.9L602.8,45.8L603.2,44.2L601.5,42.8L603.3,41.6L610,43.6L612.3,42.8L610.5,40.4L616.9,37.3L619.5,37.5L622.1,38.6L623.7,36.4L621.4,34.4L622.7,32.5L620.7,30.5L628.5,31.5L630.1,33.4L626.5,33.8L626.6,35.6L628.7,36.7L633,36L633.7,33.9L639.5,32.4L649.2,29.6L651.3,29.7L648.6,31.7L652,32L654,30.9L659.2,30.8L663.3,29.5L666.5,31.5L669.7,29.3L666.7,27.4L668.2,26.4L676.4,27.3L680.2,28.4L690.3,32.1L692.2,30.4L689.3,28.6L689.3,28L685.9,27.6L686.8,26.1L685.3,23.6L685.3,22.6L690.4,19.6L692.2,16.7L694.3,16L701.6,16.9L702.2,18.7L699.6,21.3L701.3,22.4L702.2,24.6L701.6,29L704.6,31L703.4,33.2L698,37.8L701.2,38.3L702.3,37.1L705.3,36.3L706.1,34.7L708.5,33.1L706.9,31.3L708.2,29.2L705.1,28.9L704.4,27.1L706.7,23.8L703.1,21.2L708,19L707.4,16.7L708.8,16.6L710.2,18.4L709.1,21.6L712.1,22.2L710.8,19.8L715.5,18.5L721.3,18.4L726.4,20.2L723.9,17.5L723.6,14.1L728.5,13.4L735.2,13.6L741.2,13.1L738.9,11.5L742.1,9.3L745.3,9.2L750.7,7.6L758.1,7.2L759,6.3L766.3,6L768.5,6.7L774.8,5L779.9,5.1L780.7,3.7L783.3,2.3L789.9,1L794.6,2L790.8,2.8L797.1,3.3L797.9,4.9L800.4,4.1L808.5,4.2L814.8,5.8L817,7L816.3,8.6L813.3,9.6L806,11.4L803.9,12.4L807.3,12.8L811.4,13.6L813.9,13L815.4,15.1L816.6,14.3L821,13.7L829.9,14.3L830.6,15.8L842.2,16.3L842.4,13.8L848.3,14.4L852.7,14.3L857.2,16.1L858.5,18.1L856.8,19.5L860.3,22L864.7,23.3L867.4,19.9L871.8,21.4L876.6,20.5L881.9,21.5L884,20.6L888.5,21.1L886.5,18.1L890.2,16.7L915.3,18.8L917.6,20.7L924.9,23.2L936.1,22.5L941.7,23.1L944,24.4L943.6,26.8L947.1,27.7L950.8,27L955.7,27L961,27.6L966.2,27.2L971,30.1L974.5,29.1L972.3,27L973.5,25.6L982.3,26.5L988.1,26.3L996.1,27.8L1000,29.2L1000,42.1ZM108.4,26.8L113.6,27.6L117.9,29.2L120.8,29.4L123.3,28.1L126.6,27.1L130.8,27.5L134.9,26.1L139.5,25.3L141.4,26.6L143.4,25.8L144.1,24.3L146,24.7L150.7,27.6L154.4,25.4L154.8,27.8L158.2,27.3L159.2,26.3L162.6,26.5L166.8,27.9L173.3,29.1L177.2,29.6L179.9,29.4L183.6,31.1L179.7,32.7L184.7,33.4L192.2,33L194.6,32.4L197.6,34.4L200.6,32.7L197.7,31.3L199.5,30.2L202.9,30.1L205.1,29.8L207.4,30.5L210.2,32.3L213.3,32L218.2,33.5L222.5,33L226.5,33.1L226.2,31L228.7,30.5L233,31.6L233,34.6L234.8,32.1L237,32.1L238.2,28.9L235.3,26.9L232,25.6L232.2,22L235.5,19.7L239.2,20.2L242,21.6L245.8,25.3L243.3,26.9L248.5,27.5L248.5,30.8L252.2,28.3L255.5,30.4L254.7,32.8L257.4,34.9L260.3,32.6L262.3,29.8L262.4,26.3L266.4,26.5L270.5,27L274.2,28.6L274.4,30.2L272.3,31.9L274.3,33.7L273.9,35.2L268.5,37.5L264.6,38L261.8,37L260.9,38.6L258.2,41.4L257.4,42.8L254.2,45L250.2,45.2L248,46.6L247.9,48.7L244.6,49.1L241.2,51.7L238.2,55.3L237.1,57.9L237,61.6L241.1,62.2L242.3,65.2L243.6,67.7L247.5,67L252.7,68.4L255.4,69.7L257.4,71.2L260.9,72.1L263.9,73.4L268.4,73.6L271.5,73.9L271,76.7L271.9,80L273.9,83.6L278,86.7L280.2,85.6L281.7,82.3L280.2,77.2L278.3,75.5L282.7,74L285.8,71.7L287.4,69.5L287.2,67.3L285.3,64.5L281.9,62.1L285.2,58.7L284,55.8L283,50.7L285,50L289.7,50.9L292.6,51.2L294.9,50.3L297.5,51.4L300.9,53.3L301.7,54.6L306.7,54.8L306.6,57.5L307.5,61.6L310.1,62.1L312.1,64L316.1,62.2L318.8,58.7L320.6,57.2L322.8,60L326.4,64.2L329.5,68L328.3,70.1L332,71.9L334.5,73.8L339,74.6L340.7,75.6L341.8,78.4L344,78.8L345.1,80L345.3,83.6L343.3,84.9L341.3,86L336.7,87.1L333.2,89.8L328.5,90.3L322.6,89.6L318.4,89.6L315.6,89.8L313.2,92.2L309.7,93.6L305.7,97.9L302.5,100.9L304.8,100.3L309.3,96.1L315.1,93.4L319.3,93.1L321.7,94.7L319.1,96.8L320,100.3L320.9,102.8L324.5,104.4L329.1,103.9L331.9,100.3L332.1,102.6L333.9,103.8L330.4,105.9L324.3,107.8L321.5,109.1L318.4,111.5L316.3,111.2L316.2,108.5L321,105.8L316.6,105.9L313.5,106.3L314,107.4L311,108.9L308.2,110.1L305.2,111L303.8,112.9L303.3,113.7L303.3,115.4L304.2,117.1L305.3,117.2L305,116L305.9,116.7L305.7,117.6L303.8,118.2L302.4,118.1L300.4,118.7L299.2,118.8L297.6,119L295.3,119.9L299.3,119.3L300.2,119.9L296.3,120.9L294.5,120.9L294.6,120.5L293.7,121.4L294.6,121.6L294,123.9L291.9,126.4L291.7,125.5L291.1,125.4L290.2,124.6L290.8,126.3L291.5,126.9L291.5,128.1L290.6,129.4L289.1,131.9L288.8,131.8L289.7,129.6L288.2,128.4L287.9,125.7L287.4,127.1L288,129.1L286.1,128.6L288.1,129.7L288.2,132.8L289,133L289.3,134.1L289.6,137.3L287.9,139.7L285,140.7L283.2,142.6L281.8,142.8L280.4,144L280,145.1L276.9,147.2L275.4,148.7L274.1,150.6L273.6,152.9L274.1,155.2L275.1,157.9L276.3,160.2L276.3,161.6L277.6,165.4L277.5,167.6L277.4,168.8L276.7,170.8L275.9,171.2L274.5,170.8L274.1,169.4L273,168.7L271.6,165.9L270.3,163.4L269.8,162.1L270.4,160L269.6,158.2L267.5,155.5L266.4,155L263.6,156.5L263.1,156.3L261.7,154.8L260,154L256.9,154.4L254.4,154L252.3,154.3L251.1,154.8L251.6,155.6L251.6,156.9L252.2,157.6L251.6,158L250.6,157.5L249.6,158.1L247.6,158.1L245.5,156.3L243.1,156.7L241,156L239.3,156.2L237,157L234.4,159.4L231.7,160.8L230.2,162.3L229.5,163.8L229.5,166L229.6,167.6L230.2,168.7L229.1,171.5L228.6,173.8L228.4,178.2L228.1,179.7L228.6,181.5L229.5,183.1L230,185.6L231.9,188L232.5,189.8L233.6,191.4L236.6,192.3L237.7,193.6L240.1,192.7L242.3,192.4L244.3,191.8L246.1,191.3L247.9,190L248.5,188.1L248.7,185.4L249.2,184.4L251.1,183.6L254,182.8L256.5,182.9L258.2,182.7L258.9,183.3L258.8,184.9L257.3,186.8L256.6,188.8L257.1,189.4L256.7,190.8L256,193.3L255.3,192.4L254.7,192.5L254.7,193L255.3,193L255.2,193.9L254.8,195.3L255,195.8L254.7,196.9L254.9,197.2L254.6,198.9L254,199.7L253.5,199.8L253,201L253.9,201.5L254.1,201.1L254.9,201.5L255.2,201.6L255.8,201L256.6,201L256.9,201.2L257.3,201.1L258.6,201.4L259.9,201.3L260.8,200.9L261.1,200.6L262,200.7L262.7,201L263.4,200.9L263.9,200.6L265.2,201.1L265.6,201.1L266.5,201.7L267.3,202.4L268.3,202.9L269,203.8L268.8,204.1L268.7,204.9L268.9,206.1L268.3,207.2L268,208.5L267.9,209.9L268.1,210.7L268.1,212.2L267.7,212.5L267.4,213.9L267.6,214.7L267.1,215.6L267.2,216.4L267.6,217L268.3,218.7L269.4,220L270.7,221.4L271.7,222.6L271.6,223.3L272.8,223.4L273,223.1L273.8,223.9L275.1,223.7L276.3,222.9L278,222.2L279,221.3L280.5,221.4L280.4,221.8L281.9,221.9L283.2,222.4L284.1,223.4L285.1,224.3L286.6,224.4L288.7,222.1L289.8,221.8L289.8,220.7L290.3,218L291.9,216.5L293.7,216.4L293.9,215.8L296.1,216L298.3,214.4L299.3,213.7L300.7,212.1L301.7,212.3L302.4,213.2L301.9,214.3L301.8,215L300.1,215.4L301.1,216.9L301,218.6L299.8,220.4L300.8,223L302,222.8L302.7,220.5L301.8,219.3L301.7,216.9L305.1,215.5L304.7,214L305.7,213L306.7,215.3L308.7,215.3L310.5,217.1L310.6,218.2L313.1,218.2L316,217.9L317.6,219.3L319.8,219.8L321.3,218.7L321.3,217.9L324.8,217.7L328.1,217.7L325.8,218.6L326.7,220.2L328.9,220.4L331,222L331.5,224.6L332.9,224.5L334,225.3L335.8,226.5L337.5,228.6L337.6,230.2L338.7,230.3L340.2,231.9L341.3,233L344.6,233.7L344.9,233.1L347.1,232.9L350.1,233.7L351.1,234.1L353.1,234.9L356,237.6L356.5,238.9L357.5,238.8L358.1,240.5L359.7,246.2L361.2,246.7L361.3,249L359.2,251.6L360,252.6L364.9,253.1L365,256.4L367.2,254.2L370.6,255.4L375.3,257.4L376.6,259.3L376.2,261.1L379.4,260.1L384.8,261.8L388.9,261.6L393.1,264.3L396.6,268L398.7,268.9L401.1,269L402.1,270L403.1,274.2L403.5,276.1L402.4,281.5L401,283.6L397.1,288.1L395.3,291.7L393.3,294.5L392.6,294.6L391.8,297L392,303L391.2,308L390.9,310.2L390,311.4L389.6,315.8L386.7,320L386.3,323.3L384,324.7L383.4,326.7L380.3,326.7L376,327.9L374,329.3L370.9,330.3L367.6,332.9L365.3,336.1L364.9,338.5L365.3,340.3L364.8,343.5L364.2,345.1L362.3,346.9L359.2,352.6L356.7,355.2L354.8,356.7L353.6,359.8L351.7,361.6L350.5,363.6L347.4,365.4L345.4,364.8L343.8,365.1L341.3,363.7L339.4,363.9L337.7,362.1L337.5,363.7L341,366.5L340.7,368.8L342.4,370.2L342.3,371.7L339.6,375.9L335.5,377.6L329.9,378.3L326.8,378L327.4,379.9L326.9,382.3L327.4,384L325.7,385.1L322.9,385.5L320.2,384.4L319.1,385.2L319.5,388.4L321.4,389.4L322.9,388.4L323.7,390.1L321.2,391.1L318.9,393.1L318.5,396.3L317.9,398.1L315.3,398.1L313.1,399.7L312.3,402.2L315,404.5L317.7,405.2L316.7,408.1L313.4,409.9L311.6,413.7L309.1,415L308,416.5L308.8,419.8L310.7,421.7L309.5,421.6L307.1,421.5L305.7,422.3L303.2,423.5L302.8,426.5L301.6,426.6L298.5,425.5L295.3,423.3L291.8,421.4L290.9,419.4L291.7,417.5L290.3,415.3L290,409.8L291.2,406.7L294.1,404.2L289.9,403.3L292.5,400.4L293.5,395L296.6,396.2L298,389.5L296.1,388.6L295.3,392.7L293.5,392.2L294.4,387.6L295.3,381.6L296.6,379.4L295.8,376.2L295.6,372.6L296.8,372.5L298.5,367.2L300.4,362.1L301.6,357.2L300.9,352.4L301.8,349.7L301.4,345.7L303,341.8L303.5,335.5L304.4,328.8L305.3,321.6L305.1,316.3L304.5,311.7L301.7,309.9L301.5,308.5L296,305.3L291,301.7L288.9,299.7L287.7,297.1L288.2,296.1L285.8,291.9L283.1,285.9L280.5,279.5L279.3,278L278.4,275.6L276.3,273.5L274.3,272.2L275.2,270.8L273.9,267.7L274.7,265.4L276.9,263.4L278.4,260.9L277.8,259.5L276.8,261L275.1,259.6L275.7,258.7L275.2,255.8L276.2,255.3L276.7,253.3L277.7,251.2L277.5,249.9L279.1,249.2L281,247.9L280.6,246.9L281.6,246.6L281.5,245L282.1,243.8L283.5,243.6L284.7,241.6L285.8,239.9L284.7,239.1L285.3,237.2L284.6,234.3L285.2,233.4L284.8,230.7L283.7,229L282.7,228L282.1,226.3L282.8,225.4L282.1,225.2L281.6,224.1L280.2,223.2L279,223.5L278.4,224.6L277.3,225.4L276.7,225.5L276.4,226.2L277.8,227.9L277,228.3L276.6,228.8L275.3,229L274.8,227.1L274.5,227.6L273.6,227.4L273,226.1L271.9,225.9L271.1,225.5L269.9,225.5L269.9,226.2L269.5,225.7L268,225L267.5,224.3L267.8,223.8L267.7,223.1L266.9,222.3L265.8,221.7L264.9,221.2L264.7,220.3L264,219.7L264.1,220.7L263.6,221.4L262.9,220.5L262.1,220.2L261.7,219.6L261.7,218.6L262.1,217.6L261.3,217.1L261.9,216.5L261,215.5L259.7,214.2L259,213.1L257.9,212L256.5,210.6L256.8,210.1L257.2,210.6L257.5,210.3L257,209.3L256.1,209.1L255.8,209.8L254.2,209.8L253.2,209.5L252.1,208.8L250.5,208.6L249.7,207.9L248.3,207.4L246.6,207.3L245.3,206.6L243.8,205.3L240.7,201.8L239.2,200.8L237,199.9L235.4,200.2L233.2,201.4L231.8,201.7L229.8,200.9L227.7,200.2L225.1,198.8L223.1,198.3L219.9,196.8L217.6,195.3L216.9,194.4L215.3,194.2L212.5,193.2L211.3,191.7L208.4,189.9L207,187.8L206.3,186.2L207.2,185.9L206.9,185L207.6,184.2L207.6,183L206.7,181.6L206.4,180.3L205.5,178.7L203,175.5L200.2,172.9L198.9,170.9L196.5,169.6L196,168.8L196.4,166.8L195,166L193.4,164.5L192.7,162.2L191.2,162L189.6,160.3L188.3,158.7L188.1,157.7L186.6,155.2L185.7,152.8L185.7,151.5L183.7,150.2L182.8,150.4L181.2,149.5L180.7,150.8L181.2,152.3L181.5,154.8L182.4,156.1L184.5,158.3L184.9,159.1L185.4,159.3L185.7,160.4L186.2,160.4L186.8,162.5L187.6,163.3L188.2,164.4L190,166.1L190.9,169.1L191.7,170.5L192.5,172L192.6,173.7L194,173.8L195.1,175.3L196.1,176.8L196,177.3L194.9,178.5L194.4,178.5L193.6,176.5L191.8,174.7L189.8,173.1L188.4,172.3L188.5,169.9L188.1,168.2L186.7,167.2L184.8,165.7L184.5,166.2L183.8,165.3L182,164.5L180.4,162.7L180.6,162.4L181.8,162.6L182.8,161.4L182.9,159.9L180.7,157.6L179.1,156.7L178.1,154.7L177.1,152.6L175.8,150L174.6,147.1L174.2,145.4L172.4,143.6L171.1,143.2L170.8,142.3L169.2,142.1L168.2,141.2L165.6,140.9L164.9,140.4L164.6,138.6L161.9,135.4L159.6,130.9L159.7,130.1L158.5,129L156.3,126.3L155.9,123.7L154.5,121.9L155.1,119.2L155,116.5L154.1,114L155.2,110.9L155.5,108L155.8,105.1L155.3,100.7L154.5,98L153.7,96.5L154,95.8L158,96.9L159.5,100L160.2,99.1L159.7,96.5L158.8,93.8L158.4,93.8L153,90.6L151,89.2L146,87.9L144.5,85L144.9,83.1L141.3,81.7L140.8,79.1L137.5,76.7L137.4,75.1L135.9,73.8L133.4,72.8L132.6,70L129.1,67.4L127.6,64.3L124.9,64.1L120.5,64L117.2,63.1L111.5,59.7L108.8,59.1L104,58L100.1,58.2L94.7,56.8L91.4,55.4L88.3,56.1L88.8,58.3L87.3,58.5L84.1,59.2L81.6,60.3L78.6,61L78.2,59.1L79.4,55.9L82.4,54.9L81.6,54.1L78.1,55.9L76.2,58L72.2,60.3L74.2,61.9L71.6,64.2L68.6,65.6L65.8,66.6L65.1,68L60.8,69.7L59.9,71.2L56.7,72.6L54.8,72.3L52.2,73.2L49.3,74.3L47,75.4L42.3,76.3L41.8,75.8L44.9,74.3L47.6,73.3L50.5,71.5L54,71.2L55.4,69.8L59.2,67.9L59.8,67.2L61.9,66.1L62.4,63.6L63.8,61.7L60.6,62.7L59.7,62.2L58.2,63.3L56.4,61.7L55.6,62.9L54.6,61.2L51.8,62.5L50.1,62.5L49.9,60.6L50.4,59.4L48.6,58.3L45,58.9L42.6,57.4L40.7,56.6L40.7,54.8L38.5,53.4L39.6,51.5L41.9,49.7L42.9,48.1L45.1,47.8L47,48.3L49.3,46.8L51.3,47.1L53.4,46.1L52.9,44.6L51.3,44L53.4,42.7L51.7,42.8L48.7,43.5L47.9,44.2L45.7,43.5L41.8,43.9L37.7,43.1L36.5,41.8L33,39.9L36.9,38.5L43.1,37L45.4,37L45,38.6L50.9,38.5L48.6,36.4L45.2,35.2L43.3,33.6L40.6,32.2L36.8,31.2L38.3,29.5L43.3,29.4L46.8,27.9L47.4,26.3L50.3,24.8L53,24.4L58.2,23L60.8,23.2L65.1,21.5L69.3,22.2L71.3,23.6L72.5,23L77.2,23.2L77,23.9L81.3,24.5L84.1,24.2L90,25.2L95.3,25.5L97.4,25.9L101.1,25.4L105.4,26.4ZM266.7,50.3L268.8,48.8L272.6,48.8L272.5,49.5L269.3,51.2L267.3,51.2ZM278.4,16.8L275.3,15.1L275.5,13.9L276.8,13.7L283.2,14.1L287.9,15.8L288.2,16.7L285.2,16.6L282.2,16.6L279.2,17ZM276.9,51.5L278,50.5L279.1,50.6L279.8,51.3L278.7,53L277.5,52.7L276.8,51.7ZM240,9.8L238.5,11L234.4,10.8L231.1,9.9L232.5,8.5L236.5,7.6L239,8.7ZM239.3,1.6L238.1,1.6L232.9,1.4L232.1,0.5L237.7,0.6L239.7,1.2ZM231.2,-2.5L234.6,-1.4L233.8,-0.2L229.7,0.5L227.4,-0.3L226.2,-1.5L226,-2.8L229.6,-2.7ZM255.1,11.7L250.7,11.3L243.3,10.2L242.3,8.5L242,6.8L239.2,5.4L233.4,5L230.2,4L231.3,2.7L237,2.9L240.1,4L245.5,4L247.9,5L247.3,6.2L250.5,7L252.3,7.7L256,7.9L260.1,8.1L264.5,7.4L270.1,7.2L274.6,7.4L277.6,8.6L278.2,10L276.5,10.8L272.4,11.5L268.8,11.1L260.8,11.6ZM190.9,-0.5L194.9,0L193.9,1L188.7,1.9L184.6,0.9L186.9,-0.2ZM191.8,-2.6L195.4,-1.9L192,-1.3L187.4,-1.3L187.4,-1.8L190.3,-2.7ZM345.6,86.3L344.1,88.4L342.2,91.2L344,90.1L345.9,90.8L344.9,91.9L347.4,92.8L348.7,92L351.5,93L350.6,95.4L352.5,94.8L352.9,96.6L353.8,98.6L352.6,101.4L351.3,101.5L349.5,100.9L350.1,98.3L349.3,97.9L346.1,100.7L344.5,100.6L346.4,99L343.7,98.2L340.8,98.4L335.4,98.3L334.9,97.4L336.7,96.2L335.5,95.4L337.8,93.4L340.7,88.3L342.4,86.4L344.8,85.3L346.1,85.4ZM267,41.7L270,42.8L273.2,43.8L273.5,45.4L275.5,45.1L277.5,46.2L275,47.2L270.7,46.4L269.1,45L266.4,46.7L262.4,48.4L261.5,46.5L257.7,46.8L260.1,45.2L260.5,42.6L261.4,39.7L263.4,39.9L264,41.4L265.4,40.9ZM281.2,18.3L283.8,17L290,18.6L293.8,20.2L294.2,21.6L299.3,20.8L302.2,22.9L308.9,24.2L311.3,25.5L314,28.5L308.9,30L315.4,32.1L319.8,32.8L323.8,35.8L328.2,36L327.3,38.3L322.5,42.1L319,40.7L314.7,37.6L311.1,38L310.7,39.8L313.6,41.7L317.4,43.2L318.6,44.1L320.4,47.3L319.4,49.6L315.9,48.7L308.9,46.1L312.9,48.9L315.8,50.9L316.2,52L308.7,50.7L302.7,48.8L299.3,47.2L300.3,46.3L296.2,44.7L292.1,43.1L292.2,44L284.1,44.6L281.8,43.4L283.6,41.1L288.8,41L294.6,40.6L293.6,39.4L294.6,37.8L298.2,34.7L297.4,33.2L296.4,32.1L292.1,30.6L286.5,29.5L288.3,28.6L285.3,26.6L282.9,26.4L280.7,25.3L279.2,26.3L274.2,26.7L264,26L258.2,25L253.7,24.6L251.4,23.4L254.3,21.9L250.3,21.9L249.4,18.6L251.6,15.8L254.4,14.4L261.6,13.6L259.6,15.7L261.7,17.7L264.3,15.1L271.3,13.7L276.1,17.1L275.7,19.2ZM237.5,12.5L243.3,12.6L248.6,13.4L244.4,16.3L241.1,16.9L238.1,19.3L235,19.2L233.2,16.4L233.3,14.8L234.7,13.4ZM158.7,6.1L163.5,3.7L169.2,1.6L173.4,1.6L177.2,1.1L176.8,3.6L174.7,4.8L172.1,4.9L166.9,6.3L162.5,6.8ZM131.4,77.5L134,77.3L133.2,80.9L135.6,83.5L134.5,83.5L132.8,82.1L131.8,80.6L130.4,79.6L129.9,78.1L130.1,77.1ZM207,-4.2L212.4,-3.8L219.9,-2.6L222.1,-1.1L223.1,0.3L218.6,-0.1L214,-1.1L207.8,-1.2L210.5,-2.2L207.2,-3ZM156.9,95.4L155.5,95.9L151,94.4L150.1,93.2L147.6,92.1L147.1,91.2L144.3,90.6L143.2,88.8L143.5,88.1L146.4,88.8L148.1,89.3L150.7,89.6L151.6,90.8L153,92.3L155.8,93.6ZM162.4,11.5L166.4,12.2L173.5,12.3L176.2,13.3L179.1,14.6L175.6,15.5L168.8,17.7L165.4,20L165.4,21.4L158.1,23L156.6,21.5L150.2,19.8L151.4,18.5L153.3,16.1L155.7,14L153,12ZM200.5,7L203,6.4L205.9,6.6L206.4,8.2L204.7,9.7L195.3,10.2L188.3,11.6L184,11.7L183.7,10.6L189.5,9.2L176.9,9.6L173,9L176.8,5.8L179.4,4.9L187.3,6L192.2,7.9L197,8.2L193.1,5.1L195.6,3.9L198.5,4.3L199.4,5.8ZM204.1,15.9L207.2,17.2L209,20.4L209.8,22.7L214.5,24.3L219.5,25.8L219.2,27.2L214.6,27.5L216.4,28.7L215.5,29.9L210.4,29.4L205.7,28.5L202.4,28.7L197.2,29.8L190.2,30.3L185.2,30.6L183.7,29.1L179.9,28.2L177.5,28.6L174.1,26L175.9,25.7L180.2,25.1L184.1,25.3L187.7,24.7L182.4,23.9L176.4,24.2L172.5,24.1L171,22.9L177.5,21.6L173.2,21.7L168.3,20.8L170.7,18.4L172.6,17.1L180,15.2L182.9,15.8L181.5,17.3L187.7,16.3L191.5,18L194.7,16.3L197.2,17.4L199.5,20.5L200.9,19.2L198.9,15.9L201.3,15.4ZM221,17.1L217.9,15L221.2,13.4L224.5,14.1L229.5,13.7L230.2,14.7L227.6,16.2L231.8,17.6L231.3,20.5L226.8,21.8L224.1,21.5L222.2,20.3L215.3,17.8L215.3,16.7ZM203.9,14.2L207.6,14.1L209.7,14.8L207.3,17L202.9,14.7ZM226.4,4.1L228.5,5.6L228.6,7.3L227.3,9.7L222.8,10L219.8,9.5L219.8,7.6L215.3,7.9L215.1,5.4L218.1,5.5L222.3,4.4L226.2,4.6ZM233.3,-8.4L235.2,-9.4L238.1,-9.6L236.8,-10.4L243.3,-10.5L246.9,-8.8L251.5,-8.1L256.1,-7.5L258.3,-5.4L261.6,-4.3L257.8,-3.4L252.7,-0.9L247.8,-0.7L242,-1.1L239,-2.4L239.1,-3.6L241.3,-4.5L236.2,-4.4L233.1,-5.5L231.4,-7ZM245.6,-12.6L249.7,-13.2L253,-13.3L258.4,-13.8L262.5,-15L265.9,-14.9L268.9,-14L271.1,-15.7L274.7,-16.2L279.7,-16.6L288.2,-16.7L289.7,-16.4L297.7,-16.9L303.7,-16.7L309.7,-16.5L317.1,-16.3L323.1,-15.9L328.2,-15L328.1,-14.1L321.3,-12.7L314.6,-12.1L312.1,-11.3L318.1,-11.3L311.6,-9.4L307,-8.5L302.3,-5.8L296.5,-5.3L294.8,-4.6L286.4,-4.3L290.2,-3.9L288.3,-3.3L290.6,-1.7L287.9,-0.6L283.6,0.3L282.3,1.6L278.4,2.6L278.8,3.3L283.6,3.2L283.6,4L276.2,5.9L269,5L260.8,5.5L256.7,5.1L251.4,4.9L251.1,3.4L256.2,2.7L254.8,0.3L256.5,0.1L264,1.5L260.2,-0.6L255.7,-1.2L257.9,-2.5L262.8,-3.2L263.6,-4.4L259.7,-5.6L258.5,-7.3L266.1,-7.1L268.3,-6.8L272.6,-8L266.4,-8.3L256.7,-8.1L251.8,-9.2L249.4,-10.5L246.2,-11.5ZM291.1,34.2L289.3,35.1L286.1,35.3L285.5,33.7L286.6,31.9L289.2,31.4L291.4,32.3L291.4,33.7ZM232.6,27.5L234.3,28.8L232.6,29.9L228.8,28.9L226.6,29.3L222.8,27.8L225.2,26.8L227.2,25.4L230.1,26.3L231.8,26.9ZM320.8,91L321.7,90.7L325.4,91.5L328.2,92.9L328.3,93.5L327,93.5L323.4,92.5ZM322.2,100.2L323.2,101.8L325.2,102.2L327.7,102.1L326.4,103.4L325.4,103.6L321.8,102.3L321.1,101.2ZM68.3,187.4L68.8,187.7L69.3,188.1L70,189.2L69.9,189.4L68.8,190.1L67.9,190.6L67.5,191.2L66.8,190.7L66.9,189.8L66.5,188.6L66.6,188.3L67.1,187.7L66.9,187.1L67.1,186.8L67.3,186.8ZM66.7,185.2L66.5,185.6L65.5,185.8L65,185.1L64.7,184.8L64.7,184.6L65,184.4L66,184.7ZM64.6,183.8L64.5,184.2L63,184.1L63.2,183.7ZM61,182.1L61.3,182.3L62.1,183.4L61.9,183.6L61.7,183.5L60.8,183.4L60.4,182.7L60.3,182.5ZM57.3,180.5L57.4,181.2L57,181.6L56.1,181L56.3,180.7L56.7,180.4ZM37.6,57L39.8,57.3L40.1,58.5L38.4,59L36.5,58.4L34.9,57.5ZM74.4,64.8L76.2,65L77.4,66L75,67.6L72.2,68.8L70.8,68L70.4,66.4L72.9,65.3ZM23,46L24.7,46.6L26.4,46.3L28.7,47.1L31.4,47.6L31.2,47.9L29.1,48.6L27,47.9L25.9,47.3L23.5,47.5L22.8,47.2ZM891.7,260.8L896.5,263L901.6,264.8L903.5,266.5L905.1,268.1L905.5,270L910.1,272L910.8,273.8L908.2,274.1L908.9,276.3L911.3,278.4L913.1,281.8L914.7,281.7L914.6,283.1L916.8,283.7L915.9,284.3L918.9,285.7L918.6,286.6L916.7,286.8L916.1,286L913.7,285.6L910.9,285.1L908.7,283.1L907.1,281.3L905.7,278.5L902.1,277L899.7,278L898,279L898.4,281.4L896.2,282.5L894.6,282L891.8,281.8L889.3,279.2L886.5,278.5L885.8,279.5L882.3,279.6L883.4,276.9L885.2,276L884.5,272.5L883.1,269.8L877.7,267.1L875.5,266.8L871.3,263.8L870.5,265.4L869.4,265.7L868.8,264.5L868.8,263.1L866.6,261.5L869.6,260.3L871.6,260.4L871.4,259.5L867.3,259.5L866.2,257.6L863.7,257L862.6,255.4L866.3,254.6L867.7,253.6L872.2,254.9L872.6,256.1L873.4,261.3L876.3,263.2L878.6,259.8L881.8,257.9L884.2,257.9L886.6,259L888.7,260.1ZM924,264.2L925.1,265.2L925.4,266.9L924.5,267.8L924,265.9L923.3,264.6L922.1,263.6L920.5,262.2L918.5,261.2L919.3,260.4L920.8,261.3L921.7,262.1L922.9,262.8ZM920.3,271.2L918.8,272L917.3,272.8L915.9,272.8L913.6,271.9L912,270.9L912.2,269.9L914.7,270.4L916.2,270.2L916.7,268.6L917.1,268.5L917.3,270.3L918.9,270L919.7,268.9L921.2,267.7L920.9,265.8L922.6,265.8L923.2,266.3L923.1,268.1L922.2,270.1L920.7,270.3ZM929.9,269.6L930.7,270.4L932.1,272.4L933.4,273.5L933,274.4L932.2,274.7L931,273.5L929.8,271.4L929.2,269L929.6,268.7ZM847.5,282.7L845.7,285.2L843.3,285.9L842.9,285.5L843.2,284.4L844.4,282.4L847.1,281.1L847.5,280.4L849.8,279.6L851.8,279.5L852.7,279.1L853.7,279.5L852.7,280.4L849.8,281.8ZM872.8,274.7L872.5,272.2L873,271.1L873.6,270L874.2,270.9L874.2,272.5ZM820.7,234.7L822.8,232.5L824.2,229.9L825.4,229.9L826.8,231.6L826.9,233L828.7,233.9L831.1,234.9L830.9,236.1L829,236.3L829.5,237.9L827.4,239L825.9,241.9L827.9,245L827.4,246.4L830.5,249.4L827.3,249.8L826.3,252L826.4,255L823.8,257.2L823.7,260.4L822.6,265.3L822.2,264.2L819.1,265.6L818,263.7L816,263.5L814.6,262.4L811.3,263.6L810.3,262L808.5,262.2L806.2,261.8L805.7,257.5L804.4,256.6L803,253.8L802.6,251L803,248L804.6,245.9L806.7,247L808.8,246.4L809.4,243.6L810.5,243L813.9,242.3L815.9,239.8L817.2,237.7L818.3,236.5ZM859.4,261.4L862.4,262.4L863.4,264.8L861.1,263.5L858.8,263.2L857.2,263.4L855.3,263.3L855.9,261.6ZM852.4,264.6L850.5,264L850,262.6L852.8,262.5L853.5,263.5ZM855.4,245.3L855.6,247.1L857.2,247.4L857.5,248.7L857.3,251.5L855.9,251.2L855.5,253.2L856.6,254.9L855.8,255.3L854.7,253.2L853.9,249.1L854.4,246.5ZM841.5,249.5L844.7,249.4L847.4,247L847.9,247.8L845.7,251L843.6,251.6L840.9,251L836.3,251.1L833.8,251.6L833.4,254L835.9,256.9L837.4,255.4L842.6,254.3L842.4,255.8L841.2,255.4L840,257.3L837.5,258.5L840.1,262.7L839.6,263.8L842.1,267.5L842.1,269.6L840.6,270.6L839.5,269.4L840.9,266.8L838.2,268.1L837.5,267.2L837.8,265.9L835.8,264L836,260.9L834.2,261.8L834.4,265.6L834.5,270.2L832.8,270.7L831.6,269.8L832.4,266.8L831.9,263.7L830.8,263.6L829.9,261.4L831.1,259.3L831.5,256.7L832.8,251.9L833.4,250.5L835.8,248.1L838,249.1ZM834.1,285.5L830.5,283.3L833.1,282.6L834.5,283.6L835.5,284.6L835.3,285.5ZM837.1,280L838.9,279.7L841.4,278.5L841,280.3L836.8,281.3L833.1,280.9L833.1,279.7L835.3,279ZM828.5,279.4L830.2,279.1L830.9,280.5L827.7,281.2L825.8,281.6L824.3,281.6L825.2,279.7L826.8,279.7L827.5,278.5ZM801.3,273.1L801.7,274.3L807.1,274.6L807.7,273.3L812.8,274.8L813.8,276.9L818,277.5L821.4,279.4L818.2,280.7L815.2,279.4L812.7,279.5L809.8,279.2L807.2,278.6L804,277.4L801.9,277.1L800.8,277.5L795.7,276.1L795.2,274.8L792.7,274.5L794.6,271.4L798,271.6L800.2,272.9ZM789.9,255.9L790.4,258.1L791.4,259.9L793.4,260.2L794.7,262.3L794,266.3L793.9,271.3L790.9,271.4L788.5,268.6L785,266L783.8,264L781.7,261.4L780.3,259L778.2,254.5L775.7,251.8L774.9,249L773.9,246.5L771.4,244.4L769.9,241.6L767.8,239.8L764.9,236.3L764.7,234.6L766.5,234.8L770.8,235.4L773.2,238.5L775.4,240.7L776.9,242.1L779.6,245.6L782.4,245.6L784.7,247.8L786.3,250.5L788.4,252L787.3,254.7L788.9,255.8ZM309.4,422.6L310.4,424.1L311.8,426.6L315.4,428.5L319.3,429.3L318.1,430.9L315.4,431.1L314,430L313.1,431.3L310.7,432.3L309.3,432.2L307.7,431.9L305.7,430.9L302.8,430.5L299.3,428.7L296.4,426.9L292.6,423.3L294.9,424L298.8,426.1L302.5,427.3L303.9,425.8L304.8,423.6L307.4,422.3ZM300.8,194L299,193.4L297.7,193.6L296,193.4L294.7,194L293.2,193L293.4,192L296,192.4L298.1,192.7L299.1,192L297.8,190.6L297.8,189.3L296.1,188.8L296.7,187.9L298.4,188.1L300.8,188.6L301.1,188L303.3,188L305,188.9L305.7,188.8L306.2,189.9L307.7,189.9L307.6,190.8L308.9,191L310.2,192.1L309.2,193.5L307.9,192.8L306.6,192.9L305.7,192.7L305.2,193.3L304.1,193.5L303.7,192.7L302.8,193.2L301.7,195.4L301,194.9ZM760.5,-9.8L766.5,-10.5L771.9,-8.9L778.3,-5.8L777.6,-2.8L771.5,-2.4L763.8,-3.4L759.2,-4.6L757.1,-6.9L753.3,-7.6ZM785.7,-4.1L792.7,-2.3L791.9,-1L776.2,0.3L781.3,-4L783.6,-4.4ZM885.6,6L893,6.2L903,7.9L900.8,10.3L890.6,10.2L886,11L880.5,8.9L882,6.6ZM911.7,8.6L918.7,9.4L915.5,10.7L911,10.4L905.9,9.1L906.5,8.1ZM888.5,15L891.1,13.7L894.6,13.4L898.6,14.6L898.9,15.5L894.7,15.5L889,15.2ZM624.6,-8.4L630,-9L634.2,-9L634.8,-8.1L636.4,-8.9L639,-9.4L643.1,-8.7L642,-8.2L638.3,-7.8L635.8,-7.6L635.4,-7L632.2,-6.5L629.2,-7.3L630.8,-8.3ZM648.6,13.7L655.3,10.9L654.5,9.4L660.7,7.7L669.9,5.7L679.2,5.1L683.9,3.8L689.3,3.4L691.3,4.7L689.4,5.7L679.5,7.3L671.1,8.9L662.4,11.9L658.3,15.1L653.9,18.2L654.5,20.9L659.8,23.6L658.2,23.8L649.1,23.4L648.4,22L643.3,21.1L642.9,19.4L645.8,18.7L645.7,16.9L651.2,14.1ZM897,78.6L897.9,81.7L897.9,84.9L899,88.2L901.8,93.9L897.7,92.8L896,97.5L898.7,100.8L898.6,103.1L896.5,101.1L894.7,103.6L894.2,100.9L894.5,97.8L894.2,94.3L894.8,91.8L894.9,87.5L893.3,84.3L893.6,79.9L896.1,78.4L895,76.9L896.3,76.5ZM1000,23.2L997,23.4L996.5,22.3L1000,21L1000,23.2ZM0,21L0,21L0,21L0.4,20.8L2.7,20.9L6.7,21.8L6.5,22.2L3.6,23L0,23.2L0,23.2L0,21ZM280.6,165.7L281.9,165.4L283.8,165.5L283.8,166.4L280.8,166.9ZM283.9,164.9L286.1,166.3L285.6,168.6L285.1,168.2L285.2,166.5L283.9,165.2ZM282.8,170.8L283.6,170.9L284.6,173.6L284.6,175.5L283.9,175.6L283.2,173.8L282.2,172.8ZM330,420.1L333.3,418.2L335.7,419L337.4,417.7L339.6,419.1L338.8,420.3L335,421.2L333.8,420.1L331.4,421.6ZM542.1,-5.4L543.1,-6.5L547.2,-6.6L550.7,-5.5L559.8,-3.1L552.9,-1.8L551.3,0.6L548.9,1.2L547.6,3.9L544.2,4L538.2,2L540.7,0.9L536.6,-0.1L531.2,-2.8L529,-5.3L536.6,-6.5L538.1,-5.4ZM576.1,-6.7L572,-4.9L564,-4.5L555.8,-5.1L555.3,-6L551.3,-6L548.2,-7.5L556.8,-8.4L560.9,-7.6L563.7,-8.6L570.7,-7.8ZM568.7,0.5L562.5,1.8L557.6,1L559.5,0.2L557.8,-0.8L563.6,-1.5L564.7,-0.3ZM370.1,-15L379.4,-16.9L389.2,-16.8L392.7,-18L402.5,-18.3L424.7,-17.9L442.1,-15.3L437,-14L426.3,-13.9L411.4,-13.6L412.8,-13L422.6,-13.4L431,-12.3L436.4,-13.2L438.7,-12.1L435.6,-10.2L442.7,-11.4L456.2,-12.7L464.5,-12L466.1,-10.6L454.8,-8.3L453.2,-7.6L444.3,-7L450.7,-6.9L447.5,-4.5L445.3,-2.4L445.4,1.2L448.7,3.3L444.3,3.4L439.8,4.4L444.9,6.2L445.6,8.9L442.6,9.2L446.2,12L440,12.2L443.2,13.5L442.3,14.7L438.4,15.2L434.5,15.2L438,17.4L438.1,18.8L432.6,17.5L431.1,18.3L434.9,19.2L438.5,21.1L439.6,23.7L434.6,24.4L432.5,23.1L429,21.3L430,23.4L426.8,25.1L434.1,25.3L437.9,25.5L430.5,28.3L422.9,30.8L414.8,32L411.7,32L408.9,33.2L405,36.6L399,38.9L397.1,39L393.4,39.8L389.4,40.6L387,42.6L387,44.8L385.6,47L381.1,49.6L382.2,52.1L380.9,54.8L379.5,57.9L375.6,58.1L371.5,55.5L365.9,55.5L363.2,53.7L361.4,50.5L356.6,46.5L355.2,44.4L354.8,41.5L350.9,38.5L351.9,36.1L350.1,35L352.8,31.2L357,30L358.1,28.6L358.7,26.1L355.5,27.3L354,27.7L351.5,28.2L348.1,27.1L347.9,24.9L349,23.2L351.6,23.2L357.3,24L352.5,22L350,20.9L347.2,21.3L344.9,20.5L348,17.5L346.3,16.3L344.1,14.1L340.8,10.6L337.2,9.4L337.3,8L329.8,6.1L323.9,5.9L316.5,6L309.7,6.3L306.5,5.2L301.7,3.2L309,2.2L314.5,2L302.7,1.2L296.4,-0.1L296.8,-1.4L307.3,-3L317.5,-4.5L318.5,-5.7L311,-6.9L313.5,-8.1L323.1,-10.4L327.1,-10.7L326,-12.2L332.6,-13.1L341.1,-13.6L349.6,-13.6L352.7,-12.6L360,-14.4L366.7,-13.1L370.6,-12.9L376.3,-11.8L369.7,-13.6ZM691.5,409.7L693.3,410.7L695.9,411.1L696,411.7L695.2,413.2L691,413.4L690.9,411.7L691.3,410.3ZM524.3,114.4L526.1,113.2L526.6,116L525.6,118.5L524.4,117.8L523.7,115.6ZM315.9,192.5L317.3,192.7L317.8,193.4L317.1,194.2L315,194.2L313.4,194.3L313.2,192.9L313.6,192.4ZM284.5,192.5L286.4,192.8L287.9,193.6L288.3,194.5L286.4,194.5L285.5,195.1L284,194.6L282.4,193.4L282.7,192.6L283.9,192.4ZM271.5,177.3L273.9,177.6L276.1,177.6L278.7,178.7L279.8,179.9L282.4,179.5L283.4,180.3L285.7,182.3L287.4,183.7L288.4,183.7L290,184.4L289.8,185.3L291.9,185.4L294,186.7L293.6,187.5L291.8,187.9L289.9,188.1L288,187.8L284,188.1L285.9,186.3L284.7,185.5L283,185.3L282,184.3L281.3,182.5L279.8,182.6L277.2,181.7L276.3,181.1L272.7,180.6L271.8,179.9L272.8,179.1L270.1,179L268.1,180.6L266.9,180.7L266.5,181.5L265.1,181.8L264,181.5L265.4,180.5L266,179.3L267.3,178.6L268.7,178L270.8,177.7ZM637.6,292.7L638.4,294.1L639,296.2L639.5,300.1L640.2,301.6L639.9,303.2L639.4,304.1L638.5,302.2L638,303.2L638.5,305.6L638.3,306.9L637.5,307.7L637.3,310.4L636.2,314.2L634.9,318.7L633.1,324.8L632.1,329.3L630.8,333L628.6,333.8L626.1,335.2L624.5,334.4L622.3,333.2L621.6,331.5L621.4,328.6L620.4,326L620.1,323.7L620.6,321.4L621.9,320.8L621.9,319.7L623.3,317.3L623.5,315.2L622.9,313.7L622.3,311.7L622.1,308.7L623.1,306.9L623.5,304.8L624.8,304.7L626.4,304L627.4,303.4L628.6,303.4L630.2,301.6L632.5,299.6L633.3,297.9L633,296.6L634.1,296.9L635.7,294.7L635.7,292.8L636.7,291.3ZM964.5,303.8L966.2,305.6L965.3,306.1L964.4,304.6ZM963.3,303L962.9,302.2L962.9,299.7L964.2,300.7L964.6,303.3L963.9,302.9ZM573,138.1L572.7,139.1L568.7,139.4L568.7,138.8L565.3,138.2L565.8,136.8L567.4,137.9L569.5,137.7L571.6,138L571.5,138.5ZM482.8,78.1L483.2,80.4L481.1,83.3L476.2,85.2L472.3,84.7L474.5,81.3L473.1,78L476.9,75.5L479,74L481.3,73.9L484.3,75.9ZM960.5,320.6L962.8,322.6L964.2,324L963.2,324.8L961.6,323.9L959.6,322.5L957.9,320.8L956,318.5L955.6,317.4L956.8,317.4L958.4,318.5L959.6,319.6ZM950.3,286.3L951.1,287.4L949.2,287.4L948.1,285.4L949.8,286.1ZM949.1,283.4L948.7,284L946.6,281.2L946.1,279.3L947,279.3L948,281.9ZM946.8,284.3L945.7,284.4L944,284L943.4,283.5L943.6,282.3L945.4,282.8L946.4,283.4ZM943.4,278.3L944.1,279.3L944.2,280L942,278.6L940.5,277.4L939.5,276.4L939.9,276L941.2,276.8ZM936.5,275.1L937.6,276.1L937,276.3L935.8,275.6L934.7,274.2L934.8,273.7ZM991.4,382L990.3,383.7L988.9,385.9L986.8,387.2L986.3,386.4L985.1,385.9L986.7,383.3L985.8,381.5L982.8,380.2L982.9,379L984.9,377.9L985.4,375.4L985.3,373.3L984.1,371.1L984.2,370.6L982.9,369.2L980.7,366.4L979.5,364.1L980.6,363.8L982.1,365.6L984.3,366.4L985,369.3L987,372.7L987.1,370.5L988.4,371.4L988.8,373.9L991,374.9L992.9,375.2L994.5,373.9L995.9,374.3L995.2,377.2L994.4,379.1L992.2,379L991.5,380L991.8,381.4ZM971.3,393.3L973.7,391.6L975.4,389.9L976.6,387.5L977.6,386.7L978,384.9L980,383.4L980.6,384.7L981.2,386.1L983.2,384.8L984,386.1L984,387.5L983,389L981.2,391.4L979.8,392.7L980.8,394.2L978.6,394.3L976.3,395.5L975.5,397.6L973.9,400.9L971.8,402.3L970.4,403.3L967.8,403.2L966,402.1L963,401.9L962.5,400.7L964,398.3L967.5,395.1L969.3,394.5ZM910.2,384.4L911.9,384.6L912.1,388.4L911.2,389.5L910.9,392.2L909.9,391.3L908,393.5L907.4,393.4L905.7,393.2L904,390.5L903.6,388.3L902,385.5L902.1,384L903.9,384.3L906.6,385.4L908.1,385ZM850.4,356.6L847.5,358.2L845.1,359L844.5,360.7L843.5,362L841.1,362.1L839.4,362.4L836.9,361.8L834.9,362.1L833,362.3L831.4,364L830.6,363.9L829.2,364.8L827.8,365.8L825.8,365.7L824,365.7L821,363.6L819.5,363L819.6,361.1L821,360.7L821.4,360L821.3,358.8L821.7,356.5L821.4,354.6L819.9,351.4L819.4,349.5L819.6,347.7L818.4,345.6L818.4,344.6L817.1,343.3L816.8,340.8L815.2,338.2L814.8,336.9L816,338.2L815.1,335.2L816.5,336.2L817.3,337.4L817.3,335.8L815.9,333.2L815.6,332.2L815,331.2L815.3,329.4L815.8,328.6L816.2,327L815.9,325.1L817.1,322.7L817.3,325.2L818.5,323L820.7,321.9L822.1,320.5L824.2,319.3L825.5,319.1L826.2,319.5L828.4,318.3L830.1,317.9L830.5,317.2L831.3,316.9L832.8,317L835.7,316L837.2,314.6L837.9,312.9L839.6,311.2L839.7,309.9L839.8,308.2L841.7,305.4L842.9,308.2L844.1,307.6L843.1,306L843.9,304.5L845.2,305.2L845.5,302.7L847,301.1L847.7,299.8L849.1,299.3L849.1,298.4L850.3,298.8L850.4,298L851.6,297.5L853,297.1L855,298.5L856.6,300.5L858.3,300.5L860.1,300.8L859.5,299L860.8,296.4L862.1,295.6L861.6,294.8L862.8,292.9L864.5,291.8L865.9,292.2L868.3,291.5L868.2,289.9L866.2,288.8L867.7,288.4L869.5,289.2L871,290.5L873.3,291.3L874.1,291L875.8,292L877.4,291.1L878.5,291.3L879.1,290.7L880.4,292.3L879.7,294L878.6,295.4L877.7,295.5L878,296.8L877.2,298.4L876.2,300L876.4,300.9L878.6,302.7L880.7,303.7L882.2,304.8L884.2,306.7L885,306.7L886.4,307.6L886.8,308.6L889.5,309.7L891.3,308.5L891.9,306.8L892.4,305.4L892.8,303.6L893.6,301L893.2,299.5L893.4,298.5L893.1,296.7L893.5,294.2L894,293.6L893.6,292.5L894.2,290.8L894.8,289L894.8,288.1L895.9,286.9L896.7,288.4L896.8,290.5L897.5,290.9L897.7,292.2L898.7,293.9L898.9,295.7L898.8,296.9L899.8,299.4L901.6,298.2L902.5,299.6L903.8,300.8L903.5,302.3L904.1,305L904.5,306.7L905.2,307.1L906,309.8L905.7,311.5L906.6,313.7L909.6,315.4L911.6,316.9L913.5,318.3L913.1,319.1L914.7,321.1L915.8,324.6L916.9,323.9L918,325.3L918.7,324.8L919.2,328.3L921.1,330.2L922.4,331.5L924.6,334.1L925.4,336.7L925.4,338.5L925.3,340.5L926.6,343.3L926.4,346.2L925.9,347.7L925.2,350.5L925.2,352.4L924.7,354.7L923.5,357.7L921.4,359.3L920.4,361.8L919.5,363.4L918.6,366.2L917.6,367.8L916.9,370.2L916.5,372.4L916.7,373.4L915.1,374.6L912,374.7L909.4,376L908.1,377.3L906.4,378.6L904.1,377.2L902.4,376.6L902.9,375L901.3,375.6L898.9,377.9L896.5,377L894.9,376.5L893.3,376.3L890.7,375.4L888.9,373.4L888.3,370.9L887.7,369.3L886.3,368L883.7,367.6L884.6,366L883.9,363.6L882.5,365.8L880.1,366.4L881.5,364.6L882,362.8L883,361.2L882.8,358.8L880.5,361.6L878.8,362.7L877.7,365.2L875.6,363.9L875.7,362.2L873.9,359.8L872.5,358.6L873,357.9L869.4,355.9L867.5,355.8L864.8,354.3L859.8,354.6L856.2,355.7L853.1,356.8ZM727.2,228L726.8,231.4L725.6,232.3L723.2,233L721.9,230.5L721.4,225.8L722.6,220.6L724.6,222.4L725.8,224.6ZM804.1,193.5L801.8,192.5L801.7,189.7L803.1,188.2L806.1,187.3L807.7,187.4L808.4,188.6L807.1,190.1L806.5,191.9ZM838.3,173.4L836.6,178.6L835.4,181.3L833.9,178.5L833.6,176.1L835.3,173L837.5,170.5L838.8,171.5ZM541,128.9L543.1,128.7L542.1,131.2L542.5,132.2L541.9,133.9L539.8,132.7L538.4,132.3L534.5,130.7L534.9,129L538.2,129.3ZM524.2,120L525.6,119L527.2,121.3L526.9,125.6L525.6,125.4L524.5,126.5L523.4,125.6L523.3,121.7L522.7,119.9ZM534.4,70.8L535.2,72.4L533.6,75.1L530.7,73.2L530.3,71.9ZM491.4,79.6L491.4,79.6L491.8,77.7L490,75.7L489.9,75.7L486.5,75.1L485.9,74.2L486.9,72.8L486,71.9L484.5,73.4L484.3,70.3L482.9,68.6L483.9,65.3L486.1,62.7L488.3,62.9L491.7,62.7L488.7,66.2L491.5,65.7L494.6,65.7L493.8,68.4L491.3,71.3L494.2,71.5L494.4,71.8L496.9,75.6L498.8,76.1L500.5,79.8L501.3,81.1L504.7,81.7L504.3,83.8L502.9,84.7L504,86.4L501.5,88.1L497.8,88.1L493.1,89L491.8,88.3L490,89.8L487.4,89.5L485.4,90.7L484,90.1L488,86.7L490.5,86L490.5,86L486.2,85.4L485.4,84.1L488.3,83.1L486.8,81.4L487.3,79.3ZM459.7,37.3L459.1,39.4L462.2,41.6L458.6,44.1L450.6,46.3L448.2,46.9L444.5,46.4L436.8,45.4L439.5,44L433.5,42.4L438.4,41.8L438.3,40.8L432.4,40.1L434.3,38L438.5,37.5L442.8,39.7L447.1,37.9L450.6,38.8L455.1,37.1ZM835.6,211.2L834.2,208.8L836.6,208.9L837.6,210.1L836.8,212.9ZM840.5,220.1L841.2,219.2L841.5,217.1L843,217L842.6,219.1L844.7,216L844.4,219.1L843.4,220.2L842.5,222.2L841.7,223.2L839.9,220.9ZM851,225.1L851.3,227.3L851.5,229.1L850.5,232.1L849.5,228.8L848.2,230.4L849.1,232.8L848.3,234.3L845.1,232.4L844.3,230.1L845.1,228.5L843.4,227L842.5,228.3L841.2,228.2L839.1,230L838.7,229.1L839.8,226.4L841.5,225.4L843,224.2L844,225.7L846.1,224.8L846.6,223.4L848.5,223.3L848.4,220.8L850.6,222.3L850.8,223.9ZM829.2,222.2L825.5,225.3L826.8,223L828.8,221L830.5,218.8L832,215.6L832.5,218.2L830.6,220ZM839.8,193.4L839.4,194.7L840.3,197L839.6,199.7L837.9,200.8L837.5,203.4L838.1,206L839.6,206.4L840.8,206L844.3,207.8L844,209.5L844.9,210.3L844.7,211.8L842.5,210.2L841.5,208.5L840.8,209.7L839,207.8L836.5,208.2L835.1,207.5L835.2,206.2L836.1,205.4L835.3,204.6L834.9,205.8L833.5,203.9L833.1,202.5L833,199.4L834.1,200.5L834.4,195.4L835.3,192.5L837,192.5L838.7,193.4L839.6,192.6ZM839,215.4L838.6,213.9L840.2,214.9L842,214.9L841.9,216.2L840.7,217.6L838.9,218.6L838.8,217.1ZM848.6,213L849.4,216.6L847.3,215.8L847.3,216.8L848,218.8L846.7,219.6L846.6,217.3L845.7,217.1L845.3,215.2L846.9,215.4L846.9,214.2L845.2,211.7L847.8,211.8ZM894.1,125.6L891.6,128.8L891.6,132.2L890.6,134.8L891,136.4L889.6,138.7L886,140.2L881.2,140.4L877.2,144.1L875.3,142.8L875.2,140.4L870.4,141.1L867.1,142.7L863.8,142.7L866.7,145.1L864.8,150.6L863,152L861.7,150.7L862.4,147.8L860.6,146.8L859.5,144.6L862.1,143.6L863.5,141.6L866.3,139.9L868.4,137.7L873.9,136.7L876.9,137.4L879.8,131.7L881.6,133.2L885.7,130L887.3,128.7L889,124.8L888.6,121.1L889.7,119.1L892.7,118.5L894.2,123ZM901.7,110.1L903.7,108.8L904.3,112.4L900.2,113.3L897.7,116.5L893.4,114.3L891.8,117.8L888.8,117.9L888.4,114.6L889.8,112.2L892.7,112L893.5,107.5L894.4,105L897.6,108.4L899.7,109.4ZM867.7,144.1L869.2,142.2L870.8,142.5L872,141.2L874,141.9L874.3,143L872.8,144.9L871.6,143.9L870.2,144.7L869.5,146.5L867.7,145.6ZM364.8,504.9L366.2,504.9L370.4,504.2L374.6,504.9L378,506.3L379.2,508.2L379.5,509.6L379.6,511.3L375.3,512.3L370.8,513.1L365.6,513.9L359.8,514.5L353.2,514.3L349.5,513.2L350,511.9L356,511L358.4,509.9L360.1,508.5L361.4,507.3L363,506.2ZM315.9,512L322.1,512.1L328.1,512.4L330.2,511.1L331.6,510L334.5,511.3L333.7,513L332.9,514.4L327.1,514L320.9,514.2L317.4,513.1L317.4,513ZM294.7,482.9L296.6,482.5L299.8,482.7L300.6,481L300.8,479.8L300.7,477.2L302.3,475.7L304.9,475.2L306.3,476.4L307,477.6L308.2,479.1L309.1,480.5L309.9,481.9L310.2,483.4L309.7,484.6L308.9,485.8L305.7,486.3L302.6,486.9L298.9,486.9L300.3,485.6L297,486L293.9,486.5L291.8,485.5L291.6,484.2ZM215.8,485L217.5,484.4L221,484.8L225.1,485.1L228.1,485.5L231.1,485.1L232.8,487L230.6,486.7L227.2,486.9L223.8,486.7L220,486.9L217.2,486.3ZM159.4,490.7L160,489.6L163.3,490.1L166.9,490.7L170.2,490.1L168.6,491.2L166,492.1L162.2,491.8ZM146.4,490L148.5,489.3L151.2,490.1L155.5,491.4L153.8,491.2L150.2,490.9ZM45.2,506.6L46.9,505.4L52.1,505.9L54.9,506.9L57,508.1L57.8,509.5L52.4,510L48.8,508.9L47.1,507.7L47,507.5ZM0,526.4L0,526.4L0,526.4L0.2,526.5L2.6,524.6L7.6,525.6L7.9,525.5L10.9,524.4L11.3,524.5L11.6,524.5L15.6,525.8L19.1,524.5L19.8,524.3L27.9,523.7L30.6,524.5L31.9,524.9L36,526L43.9,526.8L50.2,527.8L60.9,528.6L68.9,527.7L80.7,528.3L87.4,529.3L94.8,528.4L102.5,527.5L103.1,526L92.1,525.8L83.2,525.1L80.8,523.8L73.4,523.1L73.9,521.7L74.9,520.3L75.9,519.1L75.4,517.8L70.8,516.9L68.6,515.8L64.3,514.7L71.1,514.9L77.5,514.4L81.5,515.5L86.5,514.6L91.1,513.3L93.3,512.3L92.3,510.9L88.7,510.1L84.6,509.1L78.9,508.9L73.9,508.5L68.5,508.1L66.7,506.9L63.1,505.9L61,504.8L60.1,501.1L61.5,501.4L64,502.4L68.5,502.1L72.9,501.7L75.2,503.1L79.6,502.8L83.3,502.1L86.8,501.2L90,500.1L94.2,499.8L94,498.6L93.1,497.4L93.9,496.2L97.5,495.7L99.1,496.7L103.4,496.1L106.6,495.3L110.5,495.2L114.3,494.9L118,494.1L121,493.4L124.4,492.7L126.6,492.9L128.5,493.2L132.6,492.7L136.3,493.3L140.1,493.3L143.8,492.8L147.5,493.1L151.7,493.4L155.5,493.3L159.6,493.4L163.7,493.4L167.5,493.3L170.3,492.4L173.7,491.9L177.2,492.6L180.5,492L183.5,490.8L185.3,491.9L186.3,493L188.1,494.1L190.9,493.1L194.3,494.3L198,494.7L201.2,495.6L205.1,495.4L208.7,494.8L212.9,495L216.6,495.4L220.4,496L221.9,494.6L220.1,493.5L218.7,492.4L215.2,492.1L213.6,490.9L213,489.7L212,487.3L214.1,487.7L217.8,487.9L221.4,487.7L224.6,488.2L227.5,489.2L228.6,490.3L232.4,490.5L236,490.1L239.8,489.4L243.2,489.1L246.1,489.8L249.8,489.6L252.1,487.1L254.4,488.6L257.6,489.1L261.1,488.8L263.4,490.1L267,490.2L270.4,490.6L273.7,491.3L275.9,490.1L277,488.9L279.7,490.2L283.5,489.9L286.4,490.6L288.3,491.7L292,491.3L294.9,490.7L297.7,489.8L301.1,489.4L305,489L308.5,488.6L311.2,487.9L312.9,486.8L313.5,485.5L313.2,484.1L312.3,482.9L311.3,481.6L310.5,480.3L309.8,479.2L309.6,477.9L309.9,476.6L311.2,475.4L312.3,474.1L312.7,472.8L312.2,471.4L311.8,470.2L313.2,468.7L314.7,467.8L316.5,466.6L318.4,465.5L320.6,464.6L321.7,463.2L323.3,462.3L325,461.5L327.7,461.3L329.4,460.3L331.4,459.7L333.6,459.3L335.7,458.4L337.2,457.4L339.4,457.1L341,457.9L340,459L337.2,459.9L336,460.6L333.9,460.1L331.6,460.4L329.7,461.2L327.7,462L326.4,462.9L326,464.2L326.1,465.4L327.4,466.5L325.5,467.3L322.9,467.5L321.4,468.6L319.8,469.6L318,471L317.6,472.2L318.6,473.5L320,474.5L322.3,475.3L324.5,476.3L325.6,477.6L326.2,478.8L327,480.1L328.3,481.1L329.1,482.3L329.5,485.3L330.3,486.5L330.5,487.8L331.4,489.1L331,490.8L329.5,492.1L327.9,493.2L324.2,493.6L322.9,494.8L321.2,495.8L317.1,497.1L313.4,497.6L309.9,498.3L306.1,499L303.9,500.3L299.4,500.4L294.5,500.3L290.1,500.5L285.4,500.5L286.3,501.8L290.6,502.4L293.7,503.3L295.4,504.4L292.3,505.4L287.5,505.1L283.5,505.9L283.4,507.3L283.3,508.5L286.5,509.6L287.1,510.8L290.7,512L296.5,512.5L301.5,513.4L305.5,514.4L310.6,515.4L317.5,515.9L324.3,516.8L329,517.8L334.2,518.9L336.9,520.4L338.3,521.6L341.6,520.4L346.2,519.5L351.1,518.5L356.8,517.7L361.8,516.8L368.7,516.7L375.5,517.2L381.1,517.9L382.9,516.5L386.7,515.6L393.8,515.5L399.3,514.8L404.5,514.1L410.3,513.7L416.4,513.1L420.7,512.3L418.7,511.1L417.5,510L417.5,508.8L412.2,508.9L406.4,509.4L401,509.4L400.2,508.2L400.6,505.8L401.9,505.1L405.8,504.3L410.5,503.6L413.9,502.6L417.3,501.7L419.8,500.4L423.6,499.8L427.3,499.4L429.2,499.1L433.5,499L437.6,498.6L441,497.9L444.4,497.2L447.5,496.4L451.3,495.4L453.8,494.3L456.4,493.4L457.2,492.1L454.3,491.3L455.2,490L457.1,489L460,488.4L463,487.6L465.9,486.6L468,485.3L469.4,483.8L471.4,482.9L474.7,483.1L476.1,484.2L479.4,484.3L479.5,483.1L480.9,481.8L483.9,482.2L484.6,483.4L487.9,483.5L491.5,483L495,482.6L498.2,482.8L499.4,484.1L502.4,483L505.2,482.5L508.4,482L511.5,481.6L514.3,480.8L517.4,480.3L519.8,479.6L521.5,478.5L523.6,479.3L526.5,478.9L528.5,480.4L530,481.5L533.2,480.9L534.5,479.6L537.3,478.7L540.9,478.9L542,480.1L544.3,478.9L547.3,478.5L550.6,478.4L553.5,478.5L556.6,478.9L559.6,479L560.9,480.1L562.7,481.1L565.7,480.5L569,480.4L572.2,480.4L575.3,480.3L578,479.9L581,479.5L583.4,478.6L586,478L588.9,477.7L591,476.8L592.5,475.1L594.1,474L597,474.5L598.1,475.6L600.4,476.4L603.3,476.1L605.3,477.3L607.4,478.1L610.2,477.3L611.2,475.9L613.7,475.4L616.6,474.3L619.3,473.9L622.5,473.2L624.7,472.5L627,471.8L629.2,471.1L631.8,471.4L634.3,470.3L636.1,469.4L638.7,469.5L641,468.7L641.5,467.6L643.9,466.7L646.1,466.1L648.9,465.5L651.5,465.3L653.9,465.5L656.5,465.8L658.8,466.7L659,468.1L661.5,469.2L663.2,470L666.5,470.4L668.3,471.3L670.6,472.2L673.3,472.4L675.5,471.8L677.9,470.4L680.5,471.1L683.3,471.5L685.9,471.9L688.6,472.1L691.4,472.1L693.6,475.5L693.5,476.3L693.2,477.8L690.5,478.6L688.4,479.8L688.7,481.1L691.8,481L691.5,482.3L690.1,483.5L688.7,484.8L690.9,485.8L694.1,486.2L697.3,485.6L698.8,484.3L699.7,483.1L701.3,482.1L703,481.1L703.7,480L705.2,478.4L706.9,478.1L710.1,478L712.8,477.6L715.7,477.1L717,475.8L717.9,474.6L719.8,473.4L722.5,472.6L724.8,472L726.3,470.9L727.9,470.3L729.9,469.8L732.7,470.1L735.2,469.8L737.9,469.4L741,469.6L743,468.7L744.4,466.6L745.4,467.5L746.7,469L749.1,469.6L751.7,469.9L754.4,469.5L757.2,469.7L759.9,469.8L761.6,469.5L763.9,469.7L766.1,470.4L768.6,469.9L771.6,469.9L774.1,469.5L777,469.9L778.8,468.8L780.3,467.8L782.2,466.9L785.6,464.5L787.4,464.9L789.6,465.8L791.4,466.9L794.9,468.9L797.7,469L800.2,469L803.2,468.6L806.2,468.1L808.5,467.3L810.4,466.3L813.5,466.2L815.6,465.5L817.7,466.1L819.2,467.1L821.1,468.1L824.2,468L826.1,468.8L829.4,469.7L832.9,470L835.7,469.7L837.9,468.7L839.8,467.7L842.3,467.5L844.8,467.9L847.7,468.2L850.3,467.7L852.8,467.7L855.2,468L857.8,468.3L860.3,467.8L863.3,467.3L866.1,467.1L869.3,467.1L871.8,466.8L874.3,466.6L875.1,465L875.2,463.6L876.9,464.5L877.4,466L878.3,467.3L879.5,468.4L881.8,469L885,468.8L888.6,468.7L891.1,468.5L894.8,468.5L897.4,468.5L901,468.6L904.1,468.8L906.1,469.9L905.6,471.1L907.3,472L910.3,472.8L913.4,473.6L917,474.2L920.8,474.7L923.6,475.2L926.8,475.2L928.6,474.2L931,475.1L933.1,476.1L935.6,476.8L939,477.1L942.2,477.5L943.5,478.8L946.7,479.6L948.8,480.7L951.9,481.2L955.1,481.1L958.1,481.3L961.4,481.3L964.8,481.5L967.9,482L970.7,482.7L973.6,483.4L975.6,484.3L975.3,485.6L973.8,486.7L972.5,488.2L971.5,489.3L970.2,490.7L966.6,491.2L965,492.3L961.4,493L960.1,494.3L958.2,495.5L956.2,496.5L955.1,497.8L954.4,499L954.1,500.5L954.1,501.7L955.7,503L956.3,504.2L957.6,505.3L962.8,505.7L963.9,507.1L958.9,507.6L954.6,508.3L949.3,508.5L947,510.3L946.5,511.8L945.3,513L943.9,514.2L947.6,515.3L949,516.6L951.4,517.8L954.7,518.9L958.6,519.9L962.8,521L969.1,522L970.6,523.6L978.6,524.3L979.1,524.5L981.2,525.5L988.9,524.6L995.2,525.6L1000,526.4L1000,526.4L1000,543.5L500,543.5L0,543.5ZM590.9,138.7L591.1,138.6L591.5,137.9L593.5,137.9L596,136.9L594.2,138.3L594.4,138.9L594.5,139.2L591.6,140.5L590.2,140.1L589.6,138.8ZM328.7,217.5L330.3,217.1L330.8,217.2L330.7,219.6L328.4,220L327.9,219.7L328.7,218.8Z';
const WORLD_BORDERS_D='M612,285.7L609.8,287.6L606.7,288.9L605.1,288.8L604.1,289.8L602.2,289.9L601.4,290.3L598.1,289.4L596,289.6L595.2,285.2L594.3,283.7L593.7,282.8L591,282.2L589.4,281.2L587.7,280.7L586.5,280.2L585.4,279.3L583.9,275.3L582.3,273.4L581.7,271.6L582,269.9L581.5,266.9L582.6,266.8L583.7,265.6L584.7,263.9L585.4,263.2L585.4,262.2L584.8,261.4L584.6,260.2L585.4,259.8L585.6,257.8L584.5,256L585.5,255.6L588.5,255.7L594.2,255.4L594.6,255.8L604.7,262.4L604.9,264.3L608.9,267.5M452.7,183L452.8,183.1L459,182.8L459.4,181.6L460.5,180.2L461.4,175.7L465.3,172.2L466.6,168.1L467.4,167.9L468.4,165.4L470.7,165L471.7,165.4L473,165.4L473.9,164.7L475.6,164.6L475.5,162.9L475.9,162.9L475.9,163.1L475.9,163.7L475.9,168.6L466.8,168.4L466.8,176.7L464.2,177L463.6,178.7L464.1,183.4L453.2,183.3L452.6,184.4M137.4,75.1L138.9,73.5L138.9,71.4L134.1,69.4L131.3,65.7L129.6,63.4L127,61.9L125.2,60.6L123.7,58.9L120.9,60L118.2,61.8L115.7,59.7L113.8,58.2L111.1,57.3L108.3,57.2L108.4,38.8L108.4,26.8M313.5,106.3L311.7,104.5L311.7,100.1L310.5,99.1L308.6,99.7L307.7,98.8L305.6,101.3L304.7,103.8L303.7,105.3L302.5,105.8L301.7,105.9L301.4,106.7L296.3,106.7L292,106.8L290.8,107.4L287.8,109.7L287.5,109.9L286.6,111.2L284.1,111.2L281.3,111.2L280.1,111.7L280.5,112.4L280.8,113.4L280.7,113.7L277.1,115.3L274.2,115.8L271,117.5L270.3,117.5L269.4,117L269.1,116.5L269.1,116.2L269.7,115.1L271,113.3L271.8,111.4L271.3,108.6L270.7,105.6L267.8,104.1L268.1,103.5L267.7,103.1L267,103.1L266.4,102.6L266.3,101.9L265.7,102.2L265,102.1L265.2,101.8L264.5,101.5L264.2,100.6L262.1,99.6L259.8,98.5L257.1,97.3L254.5,96.1L252,97L251.1,97L247.7,96.2L245.4,96.6L242.8,95.6L239.9,95.1L238,94.9L237.1,94.3L236.6,92.6L235.7,92.6L235.7,93.8L229.9,93.8L220.4,93.8L211,93.8L202.6,93.8L194.3,93.8L186.1,93.8L177.6,93.8L174.9,93.8L166.7,93.8L158.8,93.8M230.2,168.7L229.1,168.8L227.1,168L224.9,167L224.2,165.5L223.6,163.3L221.9,161.4L221,159.5L219.6,157.3L217.6,156L215.3,156.1L213.6,158.6L211.3,157.7L209.8,156.7L209.2,154.9L208.2,153.2L206.6,151.8L205.2,150.8L204.1,149.6L199.3,149.6L199.3,151L197.1,151L191.6,151L185.3,148.7L181.1,147.1L181.3,146.5L177.8,146.8L174.6,147.1M636.4,102.2L635,101.7L635.3,100L633.5,97.9L631.4,98L629.1,95.8L630.7,93.3L629.9,92.7L632.1,89.1L634.9,91L635.3,88.6L641,85.1L645.4,85L651.5,87.3L654.8,88.6L657.7,87.2L662.1,87.1L665.7,88.8L666.5,87.9L670.4,88L671.1,86.5L666.6,84.2L669.2,82.7L668.7,81.8L671.4,80.9L669.4,78.7L670.7,77.6L681,76.5L682.4,75.7L689.4,74.5L691.9,73.2L696.8,73.9L697.7,77.2L700.6,76.4L704.2,77.5L704,79.3L706.6,79.1L713.6,76.1L712.6,77.1L716.1,79.6L722.3,87.8L723.8,86.1L727.6,88L731.6,87.1L733.2,87.7L734.5,89.6L736.4,90.2L737.6,91.6L741.2,91.2L742.7,93.1L740.5,95.3L738.2,95.6L738.1,98.8L736.6,100.3L731.1,99.2L729.1,105L727.6,105.7L722.1,107L724.6,112.6L722.7,113.5L722.9,115.3L721.2,114.9L719.8,113.7L715.7,113.4L711.1,113.3L710.1,113.6L706.1,112.3L704.6,112.9L704.1,114.9L699.6,113.7L697.7,114.2L697.1,115.6L695.5,116.2L691.9,118.5L690.6,120.8L689.6,120.8L688.8,119.3L685.3,119.2L684.7,116.5L683.4,116.5L683.6,113.2L680.3,110.9L675.5,111.1L672.3,111.6L669.6,108.7L667.3,107.5L663,105.1L662.5,104.9L655.4,106.8L655.5,118.7L654,118.9L652.1,116.3L650.2,115.4L647.1,116.1L645.8,117.2M697.1,115.6L697.9,115.9L695.6,118L697.7,119.2L699.6,118.4L702.9,120.1L699.4,122.5L697.3,122.1L696.1,122.2L695.7,121.3L696.3,119.8L692.6,120.6L691.7,122.7L690.4,124.4L688.1,124.3L687.3,125.7L689.4,126.5L690,128.9L688.4,132.2L686.3,131.5L684.8,131.5L684.8,129.5L681.2,128.1L678.2,126.5L676.4,125L673.3,122.8L671.9,119.4L671,118.8L668,119L666.9,118.3L666.6,115.7L662.9,114L660.5,115.9L658.1,117L658.6,118.7L655.5,118.7M891.8,281.8L891.7,271.3L891.7,260.8M847.1,281.1L847.4,281.8L847.5,282.7M804.6,245.9L805.1,248L807,249.9L808.8,249.2L810.5,249.4L812.2,247.8L813.5,247.5L816.1,248.4L818.4,247.7L819.8,243.2L820.9,242.1L821.8,238.4L825,238.4L827.4,239M314,430L312.3,429.9L309.4,429.9L309.4,422.6M309.5,421.6L307,421L300.2,420.6L299.1,418.7L299.1,416.3L297.3,416.5L296.3,415.3L296.1,411.9L298.2,410.5L299.1,408.4L298.8,406.8L300.2,404L301.2,399.8L300.9,397.9L302.2,397.2L301.9,396L300.6,395.4L301.5,394L300.2,392.8L299.6,389.1L300.7,388.4L300.2,384.5L300.9,381.1L301.6,378.3L303.3,377.1L302.5,373.9L302.4,371L304.5,368.8L304.5,366.1L306.1,363L306.1,360L305.4,359.4L304.1,353.8L305.8,350.5L305.5,347.4L306.5,344.4L308.3,341.4L310.3,339.4L309.5,338.1L310,337.1L310,331.7L313,330.1L313.9,326.7L313.6,325.9L315.9,323L319.5,323.8L321.2,326.1L322.3,323.5L325.4,323.6L325.9,324.3L331,329.6L333.3,330.1L336.6,332.5L339.5,333.8L339.9,335.2L337.2,340.1L340,341L343.1,341.5L345.3,341L347.8,338.5L348.3,335.6L349.6,335L351,336.9L351,339.5L348.6,341.2L346.8,342.6L343.6,345.7L339.9,350.1L339.2,352.7L338.5,356L338.5,359.2L337.9,360L337.7,362.1M304.5,311.7L306,310.9L306.7,309.2L308.1,311.4L308.4,313.8L309.9,315.1L309,318.3L310.5,321.9L311.6,326.4L313.6,325.9M585.4,279.3L584.3,279L580.6,279.6L579.8,279.9L579,282L579.6,283.4L579.2,287.3L578.8,290.5L579.6,291.1L581.5,292.3L582.3,291.8L582.5,295.2L580.4,295.2L579.2,293.4L578.2,292.1L576.1,291.6L575.5,289.9L573.8,290.9L571.5,290.5L570.6,289L568.8,288.7L567.5,288.8L567.4,287.8L566.4,287.7L565.2,287.5L563.4,288L562.2,287.9L561.5,288.2L561.7,284.4L560.8,283.2L560.6,281.2L561,279.2L560.4,278L560.4,275.9L557,276L557.2,274.8L555.8,274.8L555.7,275.4L553.9,275.5L553.2,277.4L552.8,278.2L551.3,277.7L550.4,278.2L548.5,278.5L547.5,276.8L546.8,275.7L546,273.8L545.4,271.4L537.2,271.3L536.2,271.7L535.4,271.7L534.2,272.1M533.8,271.1L534.5,270.7L534.6,269.3L535.1,268.5L536.1,267.8L536.8,268.1L537.8,266.9L539.3,266.9L539.5,267.9L540.5,268.4L542.1,266.4L543.8,264.8L544.5,263.8L544.4,261.1L545.6,258L546.8,256.3L548.7,254.8L549,253.7L549.1,252.5L549.5,251.4L549.4,249.6L549.7,246.7L550.3,244.7L551.1,243L551.3,241L551.5,238.8L552.6,237.1L554.1,236.1L556.4,237.2L558.1,238.4L560.2,238.7L562.2,239.3L563.1,237.4L563.4,237.1L564.7,237.4L567.8,235.8L568.9,236.5L569.8,236.4L570.2,235.6L571.3,235.3L573.3,235.7L575.1,235.8L576,235.4L577.7,238.1L579,238.5L579.7,237.9L581,238.2L582.5,237.5L583.2,238.8L585.6,241L585.5,244.8L586.6,245.2L585.7,246.4L584.6,247.2L583.6,248.9L583,250.4L582.8,253L582.2,254.3L582.2,256.7L581.4,257.6L581.3,259.5L580.9,259.8L580.6,261.5L581.3,263L581.5,266.9M615.5,257.8L613.9,255.1L613.8,243.3L616.3,239.7L617,238.7L618.8,238.6L621.3,236.3L624.9,236.2L632.7,226.5L634.7,223.8L635.9,221.8L635.9,220.1L635.9,216.8L635.9,215.5L636,215.4M594.2,255.4L594.1,252L594.9,250.7L596.3,248.5L597.3,246.2L596.1,242.5L595.8,240.9L594.5,238.6L596.2,236.7L598,234.5L599.5,235.1L599.5,236.9L600.4,238L602.4,238L605.9,240.7L606.8,240.7L607.4,240.7L608,241L609.9,241.3L610.7,239.9L613.2,238.6L614.4,239.7L616.3,239.7M606.7,194.1L605.3,196L603.2,196.5L602.4,197.5L602.1,199.6L600.9,204.4L601.2,205.7L600.7,208.5L599.6,211.7L597.9,213.3L596.8,215.7L596.5,217.1L595.2,218L594.3,221.3L594.4,224.3L594.3,221.7L594,221.7L594,220.1L593.7,218.9L592.2,217.7L591.9,215.3L592.2,213L591,212.7L590.8,213.4L589.1,213.6L589.8,214.6L590,216.5L588.5,218.3L587.1,220.6L585.7,220.9L583.3,219.1L582.3,219.7L582,220.7L580.6,221.3L580.5,221.9L577.7,221.9L577.3,221.3L575.3,221.2L574.3,221.7L573.5,221.4L572.1,219.6L571.6,218.7L569.6,219.1L568.9,220.6L568.2,223.5L567.2,224.1L566.4,224.5L568.2,225.7L566.1,224.3L565.2,223.4L565,222.4L565.4,221L565.4,219.7L563.8,217.7L563.5,216.3L563.5,215.5L562.5,214.6L562.5,212.7L561.9,211.4L560.9,211.6L561.2,210.4L561.9,209.1L561.6,207.7L562.5,206.8L562,206L562.7,204L564,201.6L566.4,201.8L566.2,189L566.2,187.6L569.4,187.6L569.4,181.2L580.6,181.2L591.4,181.2L602.4,181.2M563.5,216.3L561.8,216.9L560.3,218.2L558.3,221.7L555.7,223.2L553,223L552.3,223.3L552.5,224.4L551.1,225.6L549.9,226.8L546.4,228.1L545.7,227.3L545.3,227.3L544.7,228.1L542.4,228.3L542.9,227.5L542,225.2L541.6,223.9L540.4,223.3L538.8,221.5L539.4,219.9L540.6,220.3L541.4,220L543,220.1L541.5,217.1L541.6,215L541.4,212.8L540.3,210.8L540.5,209.2L538.8,209.2L538.8,207.1L537.6,205.9L538.8,201.6L542.4,198.6L542.5,194.3L543.6,187.8L544.2,186.4L543,185.3L543,184.3L541.9,183.4L541.3,178.4L544.1,176.6L555.1,182.8L566.2,189M300.8,188.6L301,190.3L300.8,191.6L300.2,192.1L300.9,193.1L300.8,194M559.1,73.8L562,74.4L563.2,74.9L562.9,75.8L563.1,76.6L558,76.6L554.6,76.3M863.3,115.8L862.9,115.2L862.9,113.5L864.3,113.5L864.7,109.6L864,106.9L866.3,105.7L869.7,106.3L871.6,103.1L872.5,99.6L873.6,98.4L875.1,95.5L870.5,96.5L868.1,97.7L863.8,97.7L862.7,94.7L859.4,92.4L854.6,91.4L853.6,88.2L852.6,86.2L851.6,84.8L849.8,81.6L847.4,80.4L843.3,79.4L839.6,79.5L836.1,80.1L833.8,81.7L835.3,82.4L835.4,84.2L833.8,85.3L831.3,88.7L831.4,90.1L827.4,92.2L824.1,90.9L820.8,91.2L819.3,90.1L817.7,89.8L813.6,92.1L809.9,92.6L807.4,93.4L803.9,92.9L801.3,92.9L799.6,91.3L796.9,89.7L794.1,89.3L790.6,89.7L788,90.3L784,88.9L783.5,86.5L780.2,85.7L777.7,85.3L774.6,84L771.7,87.3L772.9,89.2L770.2,91.5L766.1,90.7L763.4,90.5L761.5,89L758.6,89L756.2,88L752,89.5L746.7,92.3L743.8,92.9L742.7,93.1M635,117.1L633.3,118.4L632.8,119.2L631.6,119L629.7,117L628.9,116.9L627.2,116.2L626.3,114.8L623.7,114.2L622,114.7L621.5,114.1L617.8,112.5L613.7,112L611.3,111.4L611,111.8M606.2,100L606.3,98.5L607.7,97.6L610.4,97.4L610.8,96.3L610.2,94.5L611.3,92.8L611.3,91.9L607.2,90.8L605.6,90.9L603.9,89.3L601.7,89.9L598.2,88.7L598.3,88.1L597.3,86.7L595.1,86.5L594.8,85.5L595.5,84.9L593.8,83L590.9,83.3L590,83.2L589.3,83.9L588.3,83.8L587.6,81.7L587,80.6L587.5,80.3L589.7,80.5L590.8,79.7L590,78.9L588.1,78.3L588.3,77.7L587.2,77.1L585.4,75L586,74.1L585.8,72.6L583,71.9L581.6,72.2L581.2,71.4L578.3,70.6L577.4,68.7L577.1,67.1L575.8,66.4L577,65.4L576.2,62.4L578.1,60.5L577.7,59.9M578,56.6L583.9,52.5L586.5,50.6L587.5,49L583.4,46.7L584.6,44.6L582.1,42.2L583.9,39.5L580.7,35.8L583.3,33.3L579,31.2L579.4,28.9L581.7,28.6L586.4,27.3M592.9,103.6L593.6,102.8L595.6,103.5L596.5,103.6L596.8,104.3L597.3,104.4M579.4,28.9L580.6,26.6L577,25.4L572.7,26.4L571.4,28.8L568.7,30.3L565.7,29.5L562.1,29.6L559,27.9L557.3,28.8L555.6,28.9L555.2,31L550,30.5L549.2,32.3L546.6,32.3L544.7,34.6L542,38.2L537.7,42.7L538.7,43.9L537.7,45.1L534.9,45.1L533.1,48.1L533.3,52.4L535.1,54.1L534.2,57.9L531.9,60.1L530.6,61.9M545.4,344.8L546.7,343.2L547.8,344.1L548.3,345.5L549.5,345.7L551.3,346.3L552.8,346.1L555.3,344.4L555.3,332.5L556,333L557.7,336L557.4,338L558,339.1L560,338.8L561.4,337.4L562.7,336.4L563.4,334.9L564.8,334.1L565.9,334.5L567.3,335.4L569.5,335.6L571.3,334.8L571.6,333.8L572.1,332.2L573.6,332L574.4,330.8L575.3,328.6L577.8,326.2L581.8,323.8L582.9,323.9L584.2,324.4L585.2,324L586.6,324.3L588,328.9L588.7,331.2L588.2,334.8L588.4,336L587,335.4L586.2,335.6L586,336.5L585.2,337.8L585.2,338.9L586.9,340.6L588.5,340.3L589.1,338.8L591.2,338.9M580.5,346L579.3,345L578,345.7L576.5,347L575,349L577.1,351.5L578.1,351.2L578.6,350.1L580.1,349.6L580.6,348.6L581.5,347L580.5,346M254.7,192.5L254.2,192.5L253.2,194.5L252.7,194.1L252.4,194.3L252.4,194.7L249.8,194.7L247.2,194.7L247.2,196.5L246,196.5L247,197.6L248,198.4L248.3,199.1L248.8,199.3L248.7,200.4L245.1,200.4L243.8,203L244.2,203.6L243.9,204.4L243.8,205.3M339.9,350.1L341.7,349.8L344.5,352.3L345.6,352.2L348.4,354.2L350.6,356L352.2,358.2L351,359.8L351.7,361.6M348.3,335.6L348.8,333.8L349.2,331.8L349.2,330.1L348.2,329.5L347.1,330L346.1,329.9L345.8,328.6L345.5,325.7L345,324.7L343.1,323.8L342,324.4L339.1,323.8L339.2,319.4L338.4,317.6L339.3,317L339,315.1L339.8,313.7L340.3,311.2L339.6,309.1L338.1,308.2L337.8,307L338.2,305.1L332.9,305L331.8,301.2L332.6,301.1L332.6,299.7L332.1,298.8L331.9,296.9L330.3,296L328.6,296L327.4,295.1L325.5,294.4L324.5,293.2L321.3,292.7L318.3,289.8L318.6,287.6L318.2,286.4L318.5,283.9L314.9,284.5L313.4,285.7L311,287L310.4,288L308.9,288.1L306.9,287.8L305.3,288.3L304,288L304.2,283.1L301.9,285L299.5,284.9L298.4,283.2L296.6,283L297.2,281.6L295.6,279.6L294.5,276.7L295.2,276.1L295.2,274.7L296.9,273.8L296.6,272.1L297.3,270.9L297.5,269.4L300.7,267.2L303,266.6L303.3,266.1L305.9,266.3L307.1,257.4L307.2,256L306.7,254.1L305.5,253L305.5,250.6L307.1,250.1L307.6,250.4L307.7,249.2L306.1,248.8L306.1,246.8L311.5,246.9L312.4,245.8L313.2,246.8L313.7,248.7L314.2,248.3L315.8,250L317.9,249.8L318.5,248.8L320.5,248.1L321.7,247.5L322,246.2L324,245.2L323.8,244.6L321.5,244.3L321.1,242.2L321.2,240.1L320,239.2L320.5,238.9L322.5,239.3L324.7,240.2L325.5,239.4L327.5,238.9L330.6,237.7L331.7,236.4L331.3,235.5L332.7,235.4L333.4,236.1L333,237.5L334,238L334.6,239.5L333.8,240.7L333.4,243.4L334.1,245.1L334.3,246.6L336,248.1L337.4,248.3L337.7,247.6L338.6,247.5L339.8,246.9L340.7,246L342.3,246.3L342.9,246.2L344.5,246.5L344.7,245.8L344.2,245.2L344.5,244.2L345.6,244.5L347,244.2L348.5,244.9L349.8,245.5L350.6,244.7L351.2,244.8L351.6,245.7L352.9,245.5L354,244.3L354.9,241.9L356.5,238.9M338.4,317.6L338.4,316.6L335.8,315L333.2,314.9L328.4,315.9L327,318.7L327,320.5L325.9,324.3M306.7,309.2L308.4,305.7L307.3,303L307.9,301.9L307.4,300.7L308.5,299.1L308.5,296.4L308.7,294.1L309.3,293L306.9,287.8M276.9,263.4L277.3,264.7L276.5,265.5L276.6,266.7L277.7,266.4L278.8,266.8L280,268.4L281.6,267.1L282.1,264.9L283.8,262.1L287.1,260.8L290.2,257.4L291,255.3L290.6,252.8L291.4,252.5L293.2,254.1L294.1,255.6L295.4,256.4L297,259.8L299.1,260.2L300.6,259.4L301.6,259.9L303.3,259.7L305.4,261.2L303.6,264.5L304.5,264.5L305.9,266.3M290.6,252.8L289.4,252.1L288.1,251L287.3,251.5L284.9,251.1L284.3,249.7L283.7,249.7L281,247.9M283.7,229L284,227.4L284.9,227.6L285.4,226.7L284.8,224.8L285.1,224.3M301.9,214.3L300.1,214.8L299.4,216.4L298.3,217.3L297.5,218.5L297.1,220.9L296.4,222.7L297.8,223L298.2,224.4L298.8,225.2L299,226.5L298.7,227.7L298.8,228.3L299.5,228.6L300.1,229.7L303.7,229.4L305.3,229.8L307.3,232.6L308.4,232.3L310.4,232.4L312,232.1L312.9,232.6L312.4,234.4L311.8,235.5L311.6,237.8L312.2,239.9L313,240.9L313,241.6L311.6,243.2L312.6,243.9L313.4,245.1L314.2,248.3M269.5,225.7L269.7,225.1L269.9,224.4L269.8,223.9L270.2,223.5L269.6,223L269.6,221.7L270.7,221.4M261.9,216.5L262.3,216.1L264.2,216.9L264.8,216.5L265.7,216.8L266.1,217.4L267,217.6L267.6,217M257.5,210.3L258.3,210.2L258.7,209.5L259.1,209.4L259,207.9L259.7,207.8L260.2,207.8L260.8,206.9L261.7,207.6L262,207.2L262.5,206.8L263.4,205.9L263.5,205.2L263.7,205.3L264.1,204.5L264.4,204.4L264.9,204.9L265.4,205L266,204.6L266.7,204.6L267.7,204.2L268.1,203.8L269,203.8M256.1,209.1L256.3,207.8L255.9,207.4L255.4,207.2L254.2,207.6L254.1,207.1L253.2,206.6L252.6,206L251.8,205.7L252.4,204.9L252.2,204.2L252.4,203.6L253.7,202.7L254.9,201.5M249.7,207.9L249.8,207.4L250.8,206.6L251.3,206.3L251.1,205.9L251.8,205.7M252.4,194.7L252.4,197.3L252.1,201L253,201M334,225.3L331.8,227.2L331.6,228.4L332.5,229.6L331.8,230.2L330.1,230.7L330.2,232.2L329.4,233.1L331.3,235.5M341.3,233L340.8,235.9L339.1,236.8L339.3,237.5L338.8,239.2L340,241.6L340.9,241.6L341.3,243.4L342.9,246.2M350.1,233.7L348.7,236.5L348.9,238.7L350,240.6L349.5,242L349.3,243.5L348.5,244.9M507,86.9L507.4,88L508.7,88.1L510,89.4L511.9,90.9L513.3,90.6L515.8,92.1L516.4,92.4L517.2,92.3L518.5,93.2L522.5,93.8L521.1,96L520.7,98.3L520,98.8L518.7,98.5L518.8,99.4L516.8,101.2L516.7,102.6L518.1,102.1L519,103.6L518.9,104.5L519.7,105.7L518.7,106.7L519.5,109.2L521,109.6L520.7,111M508.3,114.9L505.1,115.4L501.9,113.9L500.9,114.6L495.8,113.1L494.7,111.9M581.8,323.8L580,322.4L577.8,321.9L577,319.8L577,318.7L575.8,318.3L572.7,314.8L571.8,312.9L571.2,312.3L570.2,309.7L573.3,310.1L574.2,310.5L575.1,310.4L576.7,308.3L579.1,305.6L580.1,305.4L580.4,304.3L582,303L584.1,302.5L584.3,303.7L586.6,303.7L587.9,304.4L588.5,305.1L589.8,305.4L591.2,306.4L591.2,310.5L590.7,312.8L590.6,315.2L591,316.1L590.7,318L590.3,318.3L589.6,320.7L586.6,324.3M555.3,332.5L555.3,323L558,322.9L558.1,311.4L560.2,311.3L564.4,310.2L565.5,311.5L567.3,310.2L568.1,310.2L569.7,309.5L570.2,309.7M532.6,308.3L533.9,307.7L535.6,307.2L537.4,307.3L539.1,308.7L539.5,308.5L550.7,308.4L552.7,309.9L559.4,310.4L564.5,309L566.8,308.3L568.6,308.5L569.7,309.2L569.7,309.5M454.3,200.1L455.2,199.1L456.6,199.4L458,198.7L459.5,198.7L460.8,199.6L462.7,200.5L464.4,202.8L466.2,205.1L466.3,207.1L466.9,208.9L467.9,209.8L468.1,211.1L468,212.1L467.6,212.3L466.1,212L465.9,212.4L465.3,212.5L463.3,211.7L461.9,211.6L456.8,211.5L456.1,211.9L455.1,211.8L453.7,212.3M453.2,209.8L455.7,209.9L456.4,209.4L456.9,209.4L457.9,208.6L459.1,209.3L460.3,209.4L461.5,208.7L461,207.7L460.1,208.3L459.2,208.3L458.1,207.5L457.2,207.5L456.6,208.3L453.6,208.4M466.2,205.1L467.1,204.5L467.6,202.6L468.5,202.5L470.4,203.4L472,202.8L473.1,203L473.5,202.3L484.6,202.2L485.2,199.9L484.8,199.5L483.4,185.6L482.1,171.6L486.3,171.6L495.7,178.6L505.1,185.7L505.7,187.2L507.5,188.1L508.7,188.6L508.8,190.7L511.9,190.4L511.9,197.8L510.3,200L510.1,202L507.6,202.5L503.8,202.8L502.8,203.9L501,204.1L499.3,204.1L498.6,203.4L497,203.9L494.4,205.3L493.9,206.3L491.8,207.7L491.4,208.5L490.2,209.2L488.9,208.8L488.1,209.6L487.7,211.8L485.5,214.5L485.6,215.6L484.8,216.9L485,218.8L483.8,219.3L483.2,219.7L482.8,218.3L482,218.7L481.5,218.6L481,219.6L478.8,219.5L478.1,219L477.7,219.3L476.8,218.4L477,217.4L476.6,217.1L476.1,217.4L476.2,216.3L476.7,215.5L475.6,214.1L475.3,213.2L474.6,212.5L474.1,212.4L473.4,212.9L472.5,213.3L471.8,214L470.6,213.8L469.8,213L469.3,212.8L468.6,213.3L468.2,213.3L468,212.1M475.9,163.7L486.3,171.6M505.2,232.5L504.5,230.3L504.6,222.8L504.1,222.2L504,220.6L503,219.4L502.1,218.5L502.5,216.8L503.5,216.4L504,215L505.4,214.7L506,213.7L506.9,212.8L507.9,212.8L510,214.6L509.9,215.7L510.5,217.6L510,218.9L510.3,219.8L508.9,221.8L508.1,222.8L507.6,224.8L507.6,226.9L507.5,232.1M540.3,210.8L539.5,210.9L539.4,212L538.9,212L537,208.5L536.3,208.4L534.2,210.2L532,209.2L530.5,209L529.7,209.5L528.1,209.4L526.5,210.8L525,210.9L521.7,209.2L520.4,210L518.9,209.9L517.9,208.7L515.1,207.5L512.1,207.9L511.4,208.6L511,210.4L510.2,211.7L510,214.6M506,213.7L506,211.5L502.8,210.8L502.8,209.2L501.2,207.1L500.8,205.6L501,204.1M511.9,190.4L515.8,188.9L523.8,182.6L533.3,176.4L537.7,177.8L539.3,179.6L541.3,178.4M539.4,212L540.5,213.3L540.2,213.8L540,214.9L537.7,217.4L537,219.5L536.6,221.2L536,221.9L535.4,224.1L533.9,225.5L533.5,227.1L532.9,228.4L532.6,229.8L530.7,230.9L529.2,229.5L528.1,229.6L526.5,231.5L525.6,231.5L524.3,234.6L523.6,236.9M542.4,228.3L541,231.6L540.4,232.2L540.2,234.7L540.4,236.1L540.2,237L541.5,238.7L541.8,239.9L542.8,241.6L544.1,242.6L544.2,244.1L544.5,245L544.3,246.8L542.1,246L539.8,245.1L536.3,245L536,244.8L534.3,245.3L532.6,244.8L531.3,245L526.8,245M502.9,233.2L502.3,232L501.6,230L501.4,228.4L502,225.5L501.3,224.3L501,221.7L501,219.4L499.9,217.7L500.1,216.7L502.5,216.8M492.1,236.2L492.2,234.9L491,232.1L491.7,228.5L492.9,225.8L492.1,221.2L491.8,218.7L491.8,216.9L496.7,216.7L497.9,217L498.8,216.4L500.1,216.7M485,218.8L486.2,219.5L486.7,220.6L488,221.3L488.9,220.4L490.2,220.3L492.1,221.2M478.6,238.2L478.8,235.6L479.1,235.2L479,233.9L477.8,232.5L476.9,232.3L476.1,231.4L476.7,230L476.4,228.4L476.6,227.5L477,227.5L477.2,226.1L476.9,225.4L477.2,225L478.2,224.6L477.6,222L476.9,220.7L477.1,219.6L477.7,219.3M476.6,227.5L475.8,227.4L475.2,228.7L474.4,228.7L473.9,228L474.1,226.7L472.9,224.7L472.2,225.1L471.6,225.2L470.8,225.3L470.8,224.2L470.4,223.3L470.5,222.4L469.9,221L469.1,219.9L466.9,219.8L466.2,220.5L465.5,220.5L465,221.2L464.7,222.1L463.2,223.5M458,216.6L459.2,215.1L460,215.1L460.8,214.6L461.4,214.6L461.8,214.1L461.6,213.1L461.9,212.7L461.9,211.6M468.2,230.4L468.9,229.4L469,228.4L470.3,226.7L471.6,225.2M551.3,241L549.5,240.8L547.6,240.3L545.9,242L544.5,245M568.2,225.7L569.8,227L569.8,228.1L571.7,229.8L572.8,231.2L573.5,233.1L575.6,234.4L576,235.4M536.1,267.8L535.1,266.7L534.2,267.3L533.1,268.7M530.8,265.2L532.9,263.4L531.9,261.3L532.8,260.5L534.7,260.1L534.9,258.7L536.4,260.2L538.9,260.3L539.7,258.8L540.1,256.7L539.8,254.1L538.5,252.2L539.7,248.5L539,247.8L536.9,248.1L536.1,246.4L536.3,245M526.4,249.1L527.3,248.9L531.3,248.9L531.3,245M591,282.2L592.3,283.7L593,286.4L592.5,287.3L592,289.9L592.5,292.6L591.6,293.7L590.8,296.7L592.3,297.6L583.8,300.2L584.1,302.5M564.5,309L562.7,307L560.8,304.4L560.9,294.1L566.7,294.1L566.5,293L566.9,291.8L566.4,290.3L566.7,288.7L566.4,287.7M596,289.6L595.2,292.1L596,296.3L597,296.2L598,297.3L599.1,299.6L599.4,303.8L598.2,304.5L597.3,306.7L595.5,304.7L595.3,302.4L595.9,300.9L595.7,299.6L594.6,298.8L593.9,299.1L592.3,297.6M589.1,338.8L588.8,337.4L588.4,336M580.6,261.5L582.3,261.8L583.2,260L584.6,260.2M597.6,145.3L598.5,145.3L598.8,144.7L599.5,144.7L599.5,146L599.2,146.5L599.2,146.5L598.7,147.5L597.7,147.1L597.2,149.3L597.8,149.6L597.1,150.1L597,150.9L598.3,150.5L598.4,151.7L597,156.9L596.7,156.1L595.2,151.3M600,140.3L601.2,140.4L601.7,141.7L600.2,142.9L599.5,144.7M598.7,147.5L598.7,149.5L598.3,150.5M531.9,145.1L531.8,147.6L530.4,148.6L529.5,149.6L527.6,150.8L527.9,152.2L527.7,153.5L526.3,154.3L525.2,148.5L523.4,147.2L523.4,146.4L521.1,144.5L520.9,142L522.6,140.2L523.3,137.6L522.8,134.5L523.4,132.8M475.9,162.9L475.9,159L480.4,156.7L483.2,156.2L485.4,155.3L486.5,153.7L489.7,152.4L489.9,150L491.5,149.7L492.7,148.5L496.4,148L496.9,146.7L496.1,146L495.2,142.6L495,140.6L494,138.6M526.3,154.3L527.2,157.2L527.4,158.7L526.9,161.3L527.1,162.8L526.7,164.5L527,166.6L525.9,167.9L527.5,170.3L527.6,171.7L528.6,173.5L529.9,172.9L532.1,174.4L533.3,176.4M599.2,146.5L602.3,147.8L607.8,144.4L608.9,148.3L608.3,148.8L602.8,150.4L605.6,153.6L604.6,154.2L604.2,155.3L602.1,155.7L601.4,156.9L600.2,157.9L597.1,157.4M655.8,168.1L656.3,169.2M656.7,171.7L655.2,171.7L655,173.8L655.5,174.3L654.2,174.9L654.2,176.2L653.4,177.6L653.4,178.9L652.8,179.6L644.4,177.9L643.4,174.7L643.3,173.9M642.7,172.7L642,172.9L641.1,172.3M634.5,160L632.5,160.1L631.8,158.5L629.4,158.2L631.4,155.1L633.3,155.4M607.8,144.4L613.9,141L615,137.1L614.7,134.7L616.2,133.9L617.6,131.9L618.8,131.4L622.1,131.8L623,132.6L624.4,132.1L626.2,136L628,136.9L628.2,138.8L626.8,139.9L626.2,142.5L628.1,145.5L631.5,147.3L632.9,149.8L632.5,152.1L633.3,152.1L633.4,153.8L634.9,155.5M629.4,158.2L624.2,158L616.4,151.4L612.2,149.2L608.9,148.3M647.5,198.5L646.6,196.2L644.4,190.9L652.8,187.6L654.6,181.2L653.4,178.9M785,212.9L784.3,209L786.1,206.3L789.7,205.7L792.3,206.2L794.6,207.4L795.8,205.2L798.3,206.4L798.9,208.6L798.6,212.4L793.9,214.9L795.1,216.9L792.2,217.1L789.8,218.4M783.7,232.2L782.8,233.6L781,233.9L780.8,232.3L778.5,230.9L778,231.4M773.8,220.2L775.1,216.9L776.6,213.9L775.5,210.9L775.6,209.4L775.3,207.6L773.4,205L772.8,203.4L773.7,202.8L774.7,200L773.6,197.9L771.8,195.5L770.5,192.7L771.7,192.1L772.9,188.6L774.9,188.4L776.5,187L778.1,186.3L779.3,187.3L779.5,189.2L781.3,189.4L780.7,192.8L780.7,195.7L783.6,193.8L784.5,194.3L786.1,194.2L786.7,193.1L788.8,193.3L790.9,196L791.1,199.2L793.3,202L793.2,204.7L792.3,206.2M778.1,186.3L778.7,185.1L781.1,183L781.3,183.8L782.8,183.8L782.4,180.1L783.8,179.7L785.4,182.2L786.7,185.2L790.1,185.2L791.2,188L789.4,188.9L788.6,190L791.9,192L794.2,195.8L796,198.6L798.1,200.9L798.8,203.2L798.3,206.4M756.6,185.5L756.4,182.9L757.4,183.4L757.4,181L758.8,180.3L758.5,178.9L759.1,177.8L759.2,174.5L761.4,175.2L762.6,172.5L762.8,170.9L764.3,168.2L764.2,166.4L767.8,164.1L769.8,164.7L769.6,162.7L770.6,162.1L770.4,160.9L772,160.7L772.9,162.6L774.1,163.4L774.2,165.8L774.1,168.5L771.5,171.2L771.1,175L774.1,174.5L774.7,177.5L776.5,178.1L775.7,180.8L777.7,182L778.9,182.6L781,181.7L781.1,183M783.8,179.7L785.3,178.9L787.5,178.9L790.2,178.5L792.6,176.8L793.9,178L796.5,178.6L796,180.5L797.3,181.8L800.1,182.6M856.5,127.4L856.1,128.2L854.9,128.4L853,128.6L851.9,130L850.7,129.9L850.5,130.2M845.2,123.2L847.4,121.1L850.5,119.4L852.4,117.1L853.7,118.1L856.1,118.2L855.7,116.5L860,115.1L861.1,113.3L862.9,115.2M824.1,90.9L822.8,93.4L820.8,96.6L821.5,97.9L823.1,97.5L825.8,98L828,96.8L830.2,97.9L832.7,100.1L832.4,101.3L830.2,100.9L826.2,101.4L824.2,102.3L822.2,104.4L817.9,105.7L815.2,107.4L812.3,106.7L810.8,106.4L809.3,108.5L810.2,109.8L810.6,110.8L808.7,111.9L806.7,113.7L803.5,114.8L799.3,114.9L794.8,116L791.6,117.8L790.3,116.8L787,116.8L782.9,114.8L780.1,114.3L776.4,114.8L770.7,114L767.6,114.1L766,112.2L764.7,109.2L763,108.9L759.7,106.8L755.9,106.4L752.6,105.8L751.6,104.4L752.7,100.7L750.8,98.1L746.8,96.8L744.5,95.1L743.8,92.9M757.4,181L756,175.9L755.2,175.9L754.7,178L753.2,176.3L754.1,174.5L755.3,174.3L756.6,171.5L755,171L752.4,171L749.8,170.6L749.5,168.4L748.2,168.2L746,166.8L745,169L747,170.7L745.3,171.9L744.7,173.1L746.4,174L745.9,175.9L746.9,178.3L747.3,181M689.4,175.7L691.2,173.5L697.3,173.6L696.8,170.8L695.2,169.1L694.9,166.6L693.1,165.2L696.2,161.8L699.4,162L702.3,158.7L704,155.4L706.7,152.1L706.7,149.8L709,147.9L706.8,146.3L705.8,144.2L704.9,141.3L706.2,139.9L710.4,140.7L713.5,140.2L716.2,137.5L719.2,141.3L718.9,143.9L720,145.6L719.9,147.3L717.9,146.8L718.7,150.4L721.4,152.4L725.3,154.7L723.5,156.2L722.5,159.2L725.2,160.4L727.8,162L731.4,163.8L735.2,164.2L736.8,165.9L739,166.2L742.3,166.9L744.6,166.9L744.9,165.6L744.6,163.6L744.8,162.2L746.5,161.5L746.7,164L746.8,164.7L749.3,165.9L751,165.4L753.4,165.6L755.6,165.5L755.8,163.5L754.7,162.5L757,162.1L759.5,159.7L762.7,157.6L765,158.4L767,157.1L768.3,159.1L767.4,160.4L770.4,160.9M746.7,164L748.5,161.6L750,160.8L752,161.6L753.5,161.6L754.7,162.5M725.3,154.7L726.5,153.9L728.7,154.9L731.5,157L733.1,157.5L734,159L736.1,159.7L738.4,161.1L741.5,161.9L744.8,162.2M670.8,171.2L671.9,167.5L675.9,165.8L675.6,164.3L674.3,163.8L674.2,160.9L671.6,159.5L670.5,157.5L669.1,155.8L673.7,157.5L676.5,157L678.2,157.4L678.7,156.7L680.7,157L684.3,155.7L684.4,152.9L685.9,151.1L688,151.1L688.3,150.2L690.4,149.7L691.5,150.1L692.5,149.1L692.4,147.2L693.6,145.2L695.3,144.4L694.2,142.3L696.9,142.4L697.7,141.2L697.5,140L698.9,138.6L698.6,137L697.9,135.6L699.6,134.2L702.6,133.6L705.7,133.2L707.2,132.6L708.8,132.2L710.8,133.7L711.6,136.2L716.2,137.5M688.4,132.2L689.3,132.6L691.3,131.5L692.2,132.2L693.1,130.7L694.8,130.7L695.2,130.3L695.5,129L696.7,127.8L698.2,128.6L697.9,129.6L698.7,129.7L698.5,132.4L699.6,133.5L700.5,132.8L701.8,132.5L703.5,131L705.4,131.3L708.3,131.3L708.8,132.2M669.1,155.8L671.6,152.9L671.4,150.8L669.3,150.3L669.1,148.2L668.2,145.6L669.3,143.9L668.1,143.4L668.9,141L670,137L672.9,138.2L675,137.8L675.5,136.3L677.7,135.9L679.3,134.9L679.8,132.3L682.2,131.7L682.6,130.5L683.9,131.4L684.8,131.5M697.3,122.1L696.2,123.1L693.2,122.6L693,124.5L696,124.2L699.4,125.3L704.7,124.8L705.4,127.8L706.3,127.5L708,128.2L707.9,129.4L708.3,131.3M722.9,115.3L722.6,116.1L718.2,117.8L717.2,119.1L713.6,119.5L712.6,121.6L709.6,121.1L707.7,121.8L705.1,123.3L705.4,124L704.7,124.8M670,137L669.8,134.3L667.7,134.2L664.5,131.3L662.3,131L659.2,129.3L657.3,129L656.1,129.6L654.2,129.5L652.2,131.4L649.8,132M624.4,132.1L622.8,129.5L623.4,128.5L622.5,124.8L624.4,123.9L624.9,125.1L626.3,126.6L628.2,127L629.2,126.9L632.5,124.5L633.5,124.3L634.3,125.2L633.4,126.8L635.1,128.5L635.8,128.4M600.4,136.5L601.2,135.8L601.9,135L602.1,133.2L603,133.9L606,133L607.5,133.6L609.8,133.6L613,132.4L614.5,132.4L617.6,131.9M628.2,127L627,125.1L627.1,124.6L625.8,124.7L625,123.8L624.4,123.9L623.3,122.9L621.3,122.1L621.5,120.5L621.1,119.4L624.9,118.9L625.5,119.8L626.6,120.3L626,121.1L627.5,122.2L626.7,123.3L627.9,124.1L629.1,124.7L629.2,126.9M557.3,28.8L561.1,30.4L565.4,32.6L565.5,37.5L566.4,38.8M588.3,83.8L585.9,84L585.1,84.7L584.9,86.3L583.8,86L581.3,86.2L580.5,85.4L579.5,86L578.4,85.5L576.3,85.4L573.2,84.7L570.4,84.4L568.2,84.5L566.7,85.4L565.4,85.5L565.3,84L564.4,82.5L566.1,81.9L566.1,80.6L565.4,79.4L565.2,77.9L567.9,78L570.9,76.7L571.6,74.9L573.9,73.9L573.6,72.4L575.3,71.9L578.3,70.6M582.2,105.8L581,105.3L579.7,105.8L578.4,105.2L579.1,104.8L579.6,103.7L580.4,102.7L580.2,102.1L580.8,101.9L581,102.3L582.7,102.4L583.4,102.2L582.9,101.8L583.1,101.3L582.1,100.5L581.7,99.2L580.7,98.6L580.9,97.5L579.6,96.7L578.5,96.6L576.5,95.5L574.6,95.9L573.9,96.3L572.8,96.3L572.1,97.1L570,97.4L569.1,97.9L567.8,97.1L566,97.1L564.3,96.7L563.1,97.4L562.9,96.6L561.3,95.7L561.9,94.4L562.7,93.5L563.3,93.7L562.6,92.3L565.1,89.6L566.5,89.2L566.7,88.3L565.4,85.5M562.7,93.5L560,92.3L558,92.8L556.7,92.4L555.1,93.1L553.7,92L552.5,92.4L552.4,92.2L551.1,90.6L549,90.4L548.8,89.4L546.9,89.1L546.4,89.9L544.9,89.2L545.1,88.3L543,88.1L541.7,87L540.6,84.9L540.8,83.8L540.1,82.1L539.1,80.9L539.9,80.1L539.2,78.4M563.1,76.6L564.6,76.9L565.2,77.9M563.1,97.4L561.4,98.1L560.1,100.3L558.4,102.5L556.2,103.1L554.4,103L552.3,103.8L551.3,104.3L549,103.7L546.9,102.3L546,101.9L545.5,100.8L545,100.8M547.2,96.7L547,98L545.4,98L545.9,98.7L545,100.8L544.5,101.3L542,101.4L540.6,102.1L538.4,101.9L534.4,101L533.8,99.9L531,100.5L530.7,101.1L529,100.6L527.6,100.6L526.3,100L526.8,99.2L526.7,98.6L527.5,98.4L528.9,99.3L529.3,98.5L531.7,98.6L533.7,98L535.1,98.1L535.9,98.8L536.2,98.2L535.8,96.1L536.8,95.7L537.8,94.2L539.8,95.3L541.4,93.9L542.4,93.7L544.5,94.7L545.8,94.5L547.1,95.1L546.9,95.5L547.2,96.7L548.6,97.5L549.6,97.8L551.9,97.4L552.2,96.8L553.3,96.7L554.6,96.2L554.9,96.4L556.2,96L556.9,95.2L557.8,95L560.8,96L561.3,95.7M578.4,105.2L577.9,103.7L578.2,102.3L578.1,100.9L576.5,99L575.6,97.6L574.8,96.7L573.9,96.3M579.3,110.9L577.7,110.6L575.7,109.4L572.4,110.2L571,111L566.9,110.8L564.8,110.3L563.7,110.6L562.9,109.2L562.4,108.7L563.1,108.1L562.4,107.7L561.5,108.5L559.9,107.5L559.7,106.2L558,105.4L557.7,104.4L556.2,103.1M558.5,71.1L561.7,70.1L566.3,70.3L569.1,70L569.4,70.6L570.9,70.9L573.6,72.4M567.5,65.4L569.9,64.8L571.1,65.2L573.5,66.4L575.8,66.4M541.7,87L540.5,87.3L539.7,87L539,87.6L537,88.2L536,89L534,89.7L534.5,90.7L534.8,92.1L536.2,92.8L537.8,94.2M526.7,98.6L523.7,97.6L523.1,98.3L520.7,98.3M517.2,92.3L517.3,90.9L516.8,90.2L517.1,88L516.6,84.6L518.3,84.6L519,83.4L519.7,80.4L519.2,79.3M523.7,74.5L525.8,75L527.6,74.5M577.8,116.4L575.4,116L572.5,117L572.5,118.6L570,118.9L568,117.8L565.8,118.7L563.8,118.6L563.6,116.5L562.2,115.4L562.6,115L562.3,114.6L562.8,113.6L563.8,112.6L562.5,111.2L562.3,110L562.9,109.2M572.5,117L573.9,117.9L573,119.9L572.4,120.3M556,124.2L557.3,122.6L557.4,121.5L558.3,121.1L558.4,120.2L560.2,119.9L561.3,119.2L562.8,119.3L563.2,118.7L563.8,118.6M615.4,118L618.4,117.8L621.1,119.4M553.8,116.9L553.6,115.8L554.8,114.2L555,114.9L555.8,114.6L556.3,115.4L557,115.8L557.2,116.9L556.8,118L557.2,119.4L558.4,120.2M552.3,103.8L553,105.1L553.9,106L552.8,107.2L551.5,106.5L549.6,106.5L547.2,106L545.9,106.1L545.3,106.8L544.3,106L543.7,107.3L545.1,108.9L545.7,109.9L547,111.1L548,111.8L549.1,113.1L551.6,114.4L551.2,114.9M538.1,105.1L540,105.3L540.5,104.7L541.5,105.2L542.6,105.3L542.6,104.4L543.5,104.1L543.8,102.8L546,101.9M529,100.6L528.8,102L527.6,102.5L525.5,102.1L524.9,103.4L523.6,103.5L523.1,103L521.5,104.1L520.2,104.2L519,103.6M515.8,92.1L516.1,90.3L516.8,90.2M509.2,86.2L511.2,86.5L513.8,85.8L515.6,87.2L517.1,88M474.9,116.9L475.9,116L477,115.6L477.7,117.1L479.4,117.1L479.9,116.7L481.5,116.8L482.3,118.5L481,119.3L480.9,121.9L480.5,122.3L480.4,123.9L479.2,124.1L480.3,126.1L479.5,128.2L480.5,129.2L480.1,130L479.1,131.3L479.3,132.3M479,74L479.5,75.7L479,77.5L480.7,77.4L482.8,78.1M538.4,101.9L538,103.5L538.7,104.9M624.9,118.9L625.6,118.4L627.7,119.3L629.2,119.5L629.5,119.1L628.2,117.4L628.9,116.9M817.2,237.7L818.5,239.4L819.1,238.3L820.4,238.4L820.6,236.3L820.7,234.7M547.1,95.1L547.5,94.4L548.7,94.5L549.7,94.1L549.8,93.8L550.3,93.7L550.5,92.9L551.1,92.8L551.5,92.2L552.4,92.2M619.7,211.3L618.8,212.1L617.6,211.8L616.7,210.7L615.6,208.8L614.3,207.8L613.6,206.7L611.2,205.4L609.3,205.3L608.6,204.7L607,205.4L605.3,204L604.4,206.4L601.2,205.7M618.8,199.5L620,198.4L619.8,197.1L620.5,195.5L621.6,196.3L622.4,196L625.6,195.9L626.1,196.3L628.8,196.6L629.9,196.4L630.6,197.5L631.9,197L633.8,193.6L636.4,192.1L644.4,190.9M594.4,138.9L594.1,138.8L593.5,139.1L593.1,139L593,139.1L592.9,138.8L592.7,138.6L592.2,138.6L591.4,138.8L590.9,138.7M569.4,181.2L569.4,169.3L569.4,157.8L568.6,155.2L569.3,153.2L568.9,151.8L569.9,150.2M598,234.5L596.4,231L595.1,230.3L594.7,229L593.2,227.4L591.5,227.2L592.5,225.3L594,225.2L594.4,224.3M617.6,211.8L616.7,213.2L615.7,214.7L615.9,215.6L616,216.6L617.5,216.7L618.2,216.4L618.8,217L618.2,218.1L619.2,219.9L620.3,221.5L621.3,222.6L630.4,226.5L632.7,226.5M619.8,215.3L618.8,217M584.5,256L582.8,257L582.2,256.7M585.6,241L586.8,240.1L588.6,240.8L590.8,240.1L592.7,240.1L594.5,238.6M552.8,107.2L553.8,107.2L553.1,108.6L554.4,109.9L554,111.4L553.4,111.5L552.9,111.8L552,112.6L551.6,114.4M557.2,116.9L557.5,117L557.7,116.3L559.3,115.8L559.9,115.7L560.9,115.5L562.2,115.4M559.9,115.7L559.8,115.4L560.2,115L560.5,114.3L560.1,114.3L559.6,113.7L559.1,113.5L558.7,113L558.2,112.8L557.8,112.4L557.3,112.5L556.9,113.6L556.3,113.8L556.5,113.6L555.4,112.9L554.5,112.5L554.1,112.1L553.4,111.5M556.3,113.8L555.8,114.6';
function buildWorldMapBase(svg){
  if(svg.dataset.built)return;
  const ns='http://www.w3.org/2000/svg';
  const land=document.createElementNS(ns,'path');
  land.setAttribute('d',WORLD_LAND_D);
  land.setAttribute('class','geo-land');
  svg.appendChild(land);
  const borders=document.createElementNS(ns,'path');
  borders.setAttribute('d',WORLD_BORDERS_D);
  borders.setAttribute('class','geo-borders');
  svg.appendChild(borders);
  const markersLayer=document.createElementNS(ns,'g');
  markersLayer.setAttribute('id','geoMarkersLayer');
  svg.appendChild(markersLayer);
  svg.dataset.built='1';
}
// ─── DRAPEAUX DES PAYS ───
// SVG intégrés (jeu « country-flag-icons », licence MIT) : aucun accès réseau et un rendu
// identique sur tous les systèmes (les emojis drapeaux ne s'affichent pas sous Windows).
// Le code pays (ISO alpha-2) est fourni par le serveur dans le champ « country ».
const FLAGS={"TN":["Tunisie","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 85.333 512 341.333\"><path fill=\"#D80027\" d=\"M0 85.331h512v341.337H0z\"/><circle fill=\"#FFF\" cx=\"256\" cy=\"255.994\" r=\"96\"/><g fill=\"#D80027\"><path d=\"m267.826 219.291 16.47 22.695 26.673-8.649-16.496 22.676 16.468 22.695-26.664-8.681-16.495 22.676.017-28.04-26.664-8.682 26.674-8.648z\"/><path d=\"M277.818 312.724c-31.33 0-56.727-25.397-56.727-56.727s25.397-56.727 56.727-56.727c9.769 0 18.96 2.47 26.985 6.819-12.589-12.31-29.804-19.909-48.803-19.909-38.558 0-69.818 31.259-69.818 69.818s31.26 69.818 69.818 69.818c18.999 0 36.215-7.599 48.803-19.909-8.026 4.347-17.216 6.817-26.985 6.817z\"/></g></svg>"],"FR":["France","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#FFF\" d=\"M0 0h513v342H0z\"/><path fill=\"#00318A\" d=\"M0 0h171v342H0z\"/><path fill=\"#D80027\" d=\"M342 0h171v342H342z\"/></svg>"],"GB":["Royaume-Uni","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><g fill=\"#FFF\"><path d=\"M0 0h513v341.3H0V0z\"/><path d=\"M311.7 230 513 341.3v-31.5L369.3 230h-57.6zM200.3 111.3 0 0v31.5l143.7 79.8h56.6z\"/></g><g fill=\"#012169\"><path d=\"M393.8 230 513 295.7V230H393.8zm-82.1 0L513 341.3v-31.5L369.3 230h-57.6zm146.9 111.3-147-81.7v81.7h147zM90.3 230 0 280.2V230h90.3zm110 14.2v97.2H25.5l174.8-97.2zM118.2 111.3 0 45.6v65.7h118.2zm82.1 0L0 0v31.5l143.7 79.8h56.6zM53.4 0l147 81.7V0h-147zM421.7 111.3 513 61.1v50.2h-91.3zm-110-14.2V0h174.9L311.7 97.1z\"/></g><g fill=\"#c8102e\"><path d=\"M288 0h-64v138.7H0v64h224v138.7h64V202.7h224v-64H288V0z\"/><path d=\"M311.7 230 513 341.3v-31.5L369.3 230h-57.6zM143.7 230 0 309.9v31.5L200.3 230h-56.6zM200.3 111.3 0 0v31.5l143.7 79.8h56.6zM368.3 111.3 513 31.5V0L311.7 111.3h56.6z\"/></g></svg>"],"IE":["Irlande","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#FFF\" d=\"M0 0h513v342H0z\"/><path fill=\"#6DA544\" d=\"M0 0h171v342H0z\"/><path fill=\"#FF9811\" d=\"M342 0h171v342H342z\"/></svg>"],"DE":["Allemagne","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#D80027\" d=\"M0 0h513v342H0z\"/><path d=\"M0 0h513v114H0z\"/><path fill=\"#FFDA44\" d=\"M0 228h513v114H0z\"/></svg>"],"ES":["Espagne","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 22.5 15\"><path fill=\"#FFF\" d=\"M0 0h22.5v15H0V0z\"/><path fill=\"#D03433\" d=\"M0 0h22.5v4H0V0zm0 11h22.5v4H0v-4z\"/><path fill=\"#FBCA46\" d=\"M0 4h22.5v7H0V4z\"/><path fill=\"#FFF\" d=\"M7.8 7h1v.5h-1V7z\"/><path fill=\"#A41517\" d=\"M7.2 8.5c0 .3.3.5.6.5s.6-.2.6-.5L8.5 7H7.1l.1 1.5zM6.6 7c0-.3.2-.5.4-.5h1.5c.3 0 .5.2.5.4V7l-.1 1.5c-.1.6-.5 1-1.1 1-.6 0-1-.4-1.1-1L6.6 7z\"/><path fill=\"#A41517\" d=\"M6.8 7.5h2V8h-.5l-.5 1-.5-1h-.5v-.5zM5.3 6h1v3.5h-1V6zm4 0h1v3.5h-1V6zm-2.5-.5c0-.3.2-.5.5-.5h1c.3 0 .5.2.5.5v.2c0 .2-.1.3-.3.3H7c-.1 0-.2-.1-.2-.2v-.3z\"/></svg>"],"PT":["Portugal","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"red\" d=\"M0 0h513v342H0z\"/><path fill=\"#060\" d=\"M197 0v342H0V0z\"/><circle fill=\"#FF0\" cx=\"197.1\" cy=\"171\" r=\"64\"/><path fill=\"red\" d=\"M161.1 139v40c0 19.9 16.1 36 36 36s36-16.1 36-36v-40h-72z\"/><path fill=\"#FFF\" d=\"M197.1 191c-6.6 0-12-5.4-12-12v-16h24v16c0 6.6-5.3 12-12 12z\"/></svg>"],"IT":["Italie","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#F4F5F0\" d=\"M342 0H0v341.3h512V0z\"/><path fill=\"#008C45\" d=\"M0 0h171v342H0z\"/><path fill=\"#CD212A\" d=\"M342 0h171v342H342z\"/></svg>"],"NL":["Pays-Bas","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#FFF\" d=\"M0 114h513v114H0z\"/><path fill=\"#cd1f2a\" d=\"M0 0h513v114H0z\"/><path fill=\"#1d4185\" d=\"M0 228h513v114H0z\"/></svg>"],"BE":["Belgique","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#fdda25\" d=\"M0 0h513v342H0z\"/><path d=\"M0 0h171v342H0z\"/><path fill=\"#ef3340\" d=\"M342 0h171v342H342z\"/></svg>"],"LU":["Luxembourg","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#FFF\" d=\"M0 0h513v342H0z\"/><path fill=\"#D80027\" d=\"M0 0h513v114H0z\"/><path fill=\"#338AF3\" d=\"M0 228h513v114H0z\"/></svg>"],"AT":["Autriche","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#FFF\" d=\"M0 114h513v114H0z\"/><path fill=\"#D80027\" d=\"M0 0h513v114H0zM0 228h513v114H0z\"/></svg>"],"CH":["Suisse","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 768 512\"><path fill=\"red\" d=\"M0 0h768v512H0V0z\"/><path fill=\"#FFF\" d=\"M336 96h96v112h112v96H432v112h-96V304H224v-96h112V96z\"/></svg>"],"SE":["Su\u00e8de","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#004F8E\" d=\"M0 0h513v342H0z\"/><path fill=\"#F6C500\" d=\"M192.4.3h-64.2v138.8H0v64.1h128.2V342h64.2V203.2H513v-64.1H192.4z\"/></svg>"],"NO":["Norv\u00e8ge","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#ba0c2f\" d=\"M0 0h513v342H0z\"/><path fill=\"#FFF\" d=\"M513 211H202.6v131h-79.9V211H0v-80h122.7V0h79.9v131H513v61.4z\"/><path fill=\"#00205b\" d=\"M513 149.6v42.8H184V342h-42.7V192.4H0v-42.8h141.3V0H184v149.6z\"/></svg>"],"DK":["Danemark","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#c60c30\" d=\"M0 0h513v342H0z\"/><path fill=\"#FFF\" d=\"M190 0h-60v140H0v60h130v142h60V200h323v-60H190z\"/></svg>"],"FI":["Finlande","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#FFF\" d=\"M0 0h513v342H0z\"/><path fill=\"#2E52B2\" d=\"M513 129.3V212H203.7v130H121V212H0v-82.7h121V0h82.7v129.3z\"/></svg>"],"IS":["Islande","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#0052B4\" d=\"M0 0h513v342H0z\"/><path fill=\"#FFF\" d=\"M513 210.6H202.2v130.7H122.4V210.6H0V130.8h122.4V0H202.2v130.8H513v18.5l-1 42.7z\"/><path fill=\"#D80027\" d=\"M513 149.3V192H183.7v149.3H141V192H0v-42.7h141V0h42.7v149.3z\"/></svg>"],"PL":["Pologne","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#FFF\" d=\"M0 0h513v342H0z\"/><path fill=\"#dc143c\" d=\"M0 171h513v171H0z\"/></svg>"],"CZ":["Tch\u00e9quie","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#11457e\" d=\"M0 0h513v342H0z\"/><path fill=\"#d7141a\" d=\"M513 171v171H0l256.5-171z\"/><path fill=\"#FFF\" d=\"M513 0v171H256.5L0 0z\"/></svg>"],"SK":["Slovaquie","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 85.333 512 341.333\"><path fill=\"#FFF\" d=\"M0 85.337h512v341.326H0z\"/><path fill=\"#0052B4\" d=\"M0 196.641h512v118.717H0z\"/><path fill=\"#D80027\" d=\"M0 315.359h512v111.304H0z\"/><path fill=\"#FFF\" d=\"M129.468 181.799v85.136c0 48.429 63.267 63.267 63.267 63.267S256 315.362 256 266.935v-85.136H129.468z\"/><path fill=\"#D80027\" d=\"M146.126 184.294v81.941c0 5.472 1.215 10.64 3.623 15.485h85.97c2.408-4.844 3.623-10.012 3.623-15.485v-81.941h-93.216z\"/><path fill=\"#FFF\" d=\"M221.301 241.427h-21.425v-14.283h14.284v-14.283h-14.284v-14.284h-14.283v14.284h-14.282v14.283h14.282v14.283h-21.426v14.284h21.426v14.283h14.283v-14.283h21.425z\"/><path fill=\"#0052B4\" d=\"M169.232 301.658c9.204 5.783 18.66 9.143 23.502 10.636 4.842-1.494 14.298-4.852 23.502-10.636 9.282-5.833 15.79-12.506 19.484-19.939a24.878 24.878 0 0 0-14.418-4.583c-1.956 0-3.856.232-5.682.657-3.871-8.796-12.658-14.94-22.884-14.94-10.227 0-19.013 6.144-22.884 14.94a25.048 25.048 0 0 0-5.682-.657 24.88 24.88 0 0 0-14.418 4.583c3.691 7.433 10.198 14.106 19.48 19.939z\"/></svg>"],"HU":["Hongrie","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#FFF\" d=\"M0 0h513v342H0z\"/><path fill=\"#D80027\" d=\"M0 0h513v114H0z\"/><path fill=\"#6DA544\" d=\"M0 228h513v114H0z\"/></svg>"],"RO":["Roumanie","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#002b7f\" d=\"M0 0h171v342H0z\"/><path fill=\"#fcd116\" d=\"M171 0h171v342H171z\"/><path fill=\"#ce1126\" d=\"M342 0h171v342H342z\"/></svg>"],"BG":["Bulgarie","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#00966e\" d=\"M0 0h513v342H0z\"/><path fill=\"#FFF\" d=\"M0 0h513v114H0z\"/><path fill=\"#d62612\" d=\"M0 228h513v114H0z\"/></svg>"],"GR":["Gr\u00e8ce","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#FFF\" d=\"M0 0h513v342H0z\"/><g fill=\"#0d5eaf\"><path d=\"M0 0h513v38H0zM0 76h513v38H0zM0 152h513v38H0zM0 228h513v38H0zM0 304h513v38H0z\"/><path d=\"M0 0h190v190H0z\"/></g><g fill=\"#FFF\"><path d=\"M0 76h190v38H0z\"/><path d=\"M76 0h38v190H76z\"/></g></svg>"],"RS":["Serbie","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#c6363c\" d=\"M0 0h513v114H0z\"/><path fill=\"#0c4076\" d=\"M0 114h513v114H0z\"/><path fill=\"#FFF\" d=\"M0 228h513v114H0z\"/><path fill=\"#c6363c\" d=\"M120 96.8v85.1c0 48.4 63.3 63.3 63.3 63.3s63.3-14.8 63.3-63.3V96.8H120z\"/><g fill=\"#edb92e\"><path d=\"M146.1 111.6h74.2V82l-14.8 7.4-22.3-22.3L161 89.4 146.1 82zM231.7 193.8l-48.3-48.2-48.2 48.2 15.7 15.7 32.5-32.5 32.5 32.5z\"/></g><path fill=\"#FFF\" d=\"M231.7 148.7h-22.5c1.3-2.2 2-4.7 2-7.4 0-8.2-6.6-14.8-14.8-14.8-5.7 0-10.6 3.2-13.1 7.8-2.5-4.7-7.4-7.8-13.1-7.8-8.2 0-14.8 6.6-14.8 14.8 0 2.7.7 5.2 2 7.4h-22.1c0 8.2 7.1 14.8 15.3 14.8h-.6c0 8.2 6.6 14.8 14.8 14.8 0 7.3 5.2 13.3 12.1 14.6l-11.7 26.4c5.6 2.3 11.7 3.5 18.1 3.5s12.5-1.3 18.1-3.5L189.9 193c6.9-1.3 12.1-7.3 12.1-14.6 8.2 0 14.8-6.6 14.8-14.8h-.5c8.2 0 15.4-6.7 15.4-14.9z\"/></svg>"],"HR":["Croatie","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#FFF\" d=\"M0 0h513v342H0z\"/><path fill=\"#D80027\" d=\"M0 0h513v114H0z\"/><path fill=\"#0052B4\" d=\"M0 228h513v114H0z\"/><g fill=\"#338AF3\"><path d=\"M309.3 113.2v-44l17.5-14.4 17.7 14.4v44zM203.7 113.2v-44l-17.3-14.4-17.9 14.4v44zM238.9 113.2v-44l17.6-14.4 17.6 14.4v44z\"/></g><g fill=\"#0052B4\"><path d=\"M309.3 113.2h-35.2v-44l17.5-14.4 17.7 14.4zM238.9 113.2h-35.2v-44l17.5-14.4 17.7 14.4z\"/></g><path stroke=\"#D80027\" fill=\"#FFF\" d=\"M168.5 113.2v101.9c0 24.3 14.4 46.2 35.4 59.4 21.3 13.4 42.1 14.7 52.6 14.7s31.4-1.7 52.6-14.8c21-13 35.4-35.1 35.4-59.3V113.2h-176z\"/><g fill=\"#D80027\"><path d=\"M168.5 113.2h35.2v35.2h-35.2zM238.9 113.2h35.2v35.2h-35.2zM309.3 113.2h35.2v35.2h-35.2zM203.7 148.4h35.2v35.2h-35.2zM274.1 148.4h35.2v35.2h-35.2zM168.5 183.6h35.2v35.2h-35.2zM203.7 218.8h35.2V254h-35.2zM238.9 183.6h35.2v35.2h-35.2zM309.3 183.6h35.2v35.2h-35.2zM274.1 218.8h35.2V254h-35.2zM309.3 274.3c8.6-5.4 16.2-12.3 22-20.3h-22v20.3zM181.7 254c5.8 8 13.3 14.9 22 20.4V254h-22zM238.9 254v33.7c7.2 1.2 13.3 1.5 17.6 1.5 4.3 0 10.4-.3 17.6-1.6V254h-35.2z\"/></g></svg>"],"SI":["Slov\u00e9nie","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 240 160\"><path fill=\"#FF0D0B\" d=\"M0 0h240v160H0V0z\"/><path fill=\"#0E55BD\" d=\"M0 0h240v106.7H0V0z\"/><path fill=\"#FFF\" d=\"M0 0h240v53.3H0V0z\"/><path fill=\"#0E55BD\" d=\"M99.5 30.8c-12.4-5.5-26.4-5.5-38.8 0-2.2 28.7 4.4 44.8 19.3 48.7 14.9-3.9 21.5-20 19.5-48.7z\"/><g fill=\"#FF0\"><path d=\"m82 37.1-1.5.1-.5-1.3-.5 1.3-1.5-.1.8 1.2-.8 1.2 1.5-.2.5 1.2.5-1.2 1.5.2-.8-1.2zM76.5 29.7l-1.4.2-.6-1.4-.6 1.4-1.4-.2.8 1.2-.8 1.2 1.4-.1.6 1.3.6-1.3 1.4.1-.8-1.2zM87.5 29.7l-1.4.2-.6-1.4-.6 1.4-1.4-.2.8 1.2-.8 1.2 1.4-.1.6 1.3.6-1.3 1.4.1-.8-1.2z\"/></g><path fill=\"#FFF\" d=\"m87.9 48.7-3.1 4.1-4.7-9.3-4.7 9.3-3.1-4.1-7.9 10.4c0 .5.1.9.3 1.5 2.1 7.6 8 13.7 15.5 16.3 7.5-2.5 13.3-8.7 15.5-16.3.1-.4.3-.9.3-1.5l-8.1-10.4z\"/><path fill=\"#0E55BD\" d=\"M64.5 60.5c.8 2.8 3.7 4.4 6.4 3.6.4-.1.8-.3 1.2-.5 1.6-.9 3.6-.9 5.2 0s3.6.9 5.2 0 3.6-.9 5.2 0c2.5 1.5 5.7.5 7.1-1.9.3-.4.4-.8.5-1.2l.3.1v3.9c-.9 0-1.9.3-2.7.7-1.6.9-3.6.9-5.2 0s-3.6-.9-5.2 0-3.6.9-5.2 0-3.6-.9-5.2 0-3.6.9-5.2 0c-.8-.4-1.7-.7-2.7-.7v-3.9\"/><path fill=\"#0E55BD\" d=\"M64.5 63.6c.8 2.8 3.7 4.4 6.4 3.6.4-.1.8-.3 1.2-.5 1.6-.9 3.6-.9 5.2 0s3.6.9 5.2 0 3.6-.9 5.2 0c2.5 1.5 5.7.5 7.1-1.9.3-.4.4-.8.5-1.2l.3.1v3.9c-.9 0-1.9.3-2.7.7-1.6.9-3.6.9-5.2 0s-3.6-.9-5.2 0-3.6.9-5.2 0-3.6-.9-5.2 0-3.6.9-5.2 0c-.8-.4-1.7-.7-2.7-.7v-3.9\"/><path fill=\"#FF0D0B\" d=\"m99.5 30.8-1.6 24.8c-.7 10.5-7.8 19.7-17.9 22.9-10.1-3.2-17.2-12.3-17.9-22.9l-1.6-24.8c-.5.3-.9.4-1.5.7l1.5 24.1C61.3 66.9 69.1 76.7 80 80c10.9-3.3 18.7-13.1 19.3-24.4l1.5-24.1c-.4-.3-.9-.6-1.3-.7z\"/></svg>"],"TR":["Turquie","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#E30A17\" d=\"M0 0h513v342H0z\"/><path fill=\"#FFF\" d=\"M259.7 118.6c-13.1-9.5-29-14.6-45.3-14.5-40.8 0-73.8 30.8-73.8 68.9s33.1 68.9 73.8 68.9c17.1 0 32.9-5.4 45.3-14.5-30 38.6-85.7 45.6-124.3 15.5s-45.6-85.7-15.5-124.3 85.7-45.6 124.3-15.5c5.8 4.5 11 9.8 15.5 15.5zm39.9 65.8-18.1 21.9 1.2-28.4-26.4-10.4 27.3-7.6 1.8-28.3 15.6 23.7 27.5-7.1-17.5 22 15.3 23.9-26.7-9.7z\"/></svg>"],"RU":["Russie","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#FFF\" d=\"M0 0h513v114H0z\"/><path fill=\"#0039a6\" d=\"M0 114h513v114H0z\"/><path fill=\"#d52b1e\" d=\"M0 228h513v114H0z\"/></svg>"],"UA":["Ukraine","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 85.333 512 341.333\"><path fill=\"#FFDA44\" d=\"M0 85.337h512v341.326H0z\"/><path fill=\"#338AF3\" d=\"M0 85.337h512V256H0z\"/></svg>"],"EG":["\u00c9gypte","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#FFF\" d=\"M0 0h513v342H0z\"/><path fill=\"#D80027\" d=\"M0 0h513v114H0z\"/><path d=\"M0 228h513v114H0z\"/><path fill=\"#C09300\" d=\"M220.3 204.4s0-58.4 4.5-64.7c3.1-4.3 16.8 5.2 22.7 4.5 0 0 4.2-7.5 4.5-12 .3-4.6-1.1-7.6-4.9-6.2 0 0-1.2-2.1.5-3.3 1.6-1.2 5.6.1 5.6.1s-.5-1 1.6-.9c2.9.2 7.2 1.4 7.4 5.6.2 3.1.3 7.7.4 8.7.7 6.8 2.7 8.7 2.7 8.7s18.4-9.2 22-5.2c3.3 3.8 4.5 64.7 4.5 64.7l-18.1-16.8 12.1 29.5s-14.4 2.4-28.9 2.4c-14.5 0-31.1-4.2-31.1-4.2l13.8-28.2-19.3 17.3z\"/></svg>"],"DZ":["Alg\u00e9rie","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#41662E\" d=\"M0 0h513v342H0z\"/><path fill=\"#FFF\" d=\"M256.5 0H513v342H256.5z\"/><g fill=\"#D80027\"><path d=\"m341.5 105.3-29.4 40.4-47.5-15.4 29.4 40.4-29.4 40.4 47.5-15.4 29.4 40.4v-50l47.5-15.5-47.5-15.4z\"/><path d=\"M309.9 276.7c-58.5 0-106-47.5-106-106s47.5-106 106-106c18.3 0 35.4 4.6 50.4 12.7-23.5-23-55.7-37.2-91.2-37.2-72 0-130.4 58.4-130.4 130.4S197.1 301 269.1 301c35.5 0 67.7-14.2 91.2-37.2-14.9 8.2-32.1 12.9-50.4 12.9z\"/></g></svg>"],"MA":["Maroc","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#d12a46\" d=\"M0 .3V342h513V.3z\"/><path fill=\"#316525\" d=\"M359.8 148.9h-73.3l-22.7-69.7-22.7 69.7h-73.3l59.3 43.1-22.7 69.7 59.3-43.1 59.3 43.1-22.5-69.7 59.3-43.1zm-116.1 37.7 7.7-23.6h24.8l7.7 23.6-20.1 14.6-20.1-14.6zm27.9-37.7H256l7.8-24 7.8 24zm24.3 29-4.8-14.9h25.3l-20.5 14.9zM236.6 163l-4.8 14.9-20.5-14.9h25.3zm-5.3 61.8 7.8-24 12.6 9.2-20.4 14.8zm44.5-14.9 12.6-9.2 7.8 24-20.4-14.8z\"/></svg>"],"LY":["Libye","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path d=\"M0 0h513v342H0z\"/><path fill=\"#D80027\" d=\"M0 0h513v114H0z\"/><path fill=\"#496E2D\" d=\"M0 228h513v114H0z\"/><g fill=\"#FFF\"><path d=\"m281.4 150.7 9 12.3 14.5-4.7-9 12.4 9 12.3-14.5-4.7-9 12.3v-15.2l-14.5-4.7 14.5-4.7z\"/><path d=\"M257 201.5c-17 0-30.9-13.8-30.9-30.9s13.8-30.9 30.9-30.9c5.3 0 10.3 1.3 14.7 3.7-6.9-6.7-16.2-10.8-26.6-10.8-21 0-38 17-38 38s17 38 38 38c10.3 0 19.7-4.1 26.6-10.8-4.4 2.4-9.4 3.7-14.7 3.7z\"/></g></svg>"],"NG":["Nig\u00e9ria","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#FFF\" d=\"M0 0h513v342H0z\"/><g fill=\"#008751\"><path d=\"M0 0h171v342H0zM342 0h171v342H342z\"/></g></svg>"],"KE":["Kenya","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#FFF\" d=\"M0 0h513v342H0z\"/><path d=\"M0 0h512v90.579H0z\"/><path fill=\"#496E2D\" d=\"M0 251h513v91H0z\"/><path fill=\"#A2001D\" d=\"M0 114h513v114H0z\"/><g fill=\"#FFF\"><path d=\"m323.3 54.5-25.6-11.8L256 137l-41.7-94.3-25.6 11.8 51.8 116.2-51.8 116.2 25.6 11.8 41.7-94.4 41.7 94.4 25.6-11.8-51.8-116.2z\"/><path d=\"M273.4 65.6c-9.9-10.8-17.4-17-17.4-17s-7.5 6.2-17.4 17v210.1c9.9 10.8 17.4 17 17.4 17s7.5-6.2 17.4-17V65.6z\"/></g><g fill=\"#A2001D\"><path d=\"M209 105.9v129.5c10.5 18.5 23.3 33.7 32.9 43.8V62.1c-9.6 10.1-22.4 25.3-32.9 43.8zM303 105.9c-10.5-18.5-23.3-33.7-32.9-43.8v217.2c9.6-10.1 22.4-25.3 32.9-43.8V105.9z\"/></g><path d=\"M303 105.9v129.5c10.6-18.8 18.8-41 18.8-64.8s-8.2-45.9-18.8-64.7zM209 105.9v129.5c-10.6-18.8-18.8-41-18.8-64.8s8.2-45.9 18.8-64.7z\"/></svg>"],"ZA":["Afrique du Sud","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 85.333 512 341.333\"><path fill=\"#FFF\" d=\"M0 85.337h512v341.326H0z\"/><path d=\"M114.024 256.001 0 141.926v228.17z\"/><path fill=\"#ffb915\" d=\"M161.192 256 0 94.7v47.226l114.024 114.075L0 370.096v47.138z\"/><path fill=\"#007847\" d=\"M509.833 289.391c.058-.44.804-.878 2.167-1.318v-65.464H222.602L85.33 85.337H0V94.7L161.192 256 0 417.234v9.429h85.33l137.272-137.272h287.231z\"/><path fill=\"#000c8a\" d=\"M503.181 322.783H236.433l-103.881 103.88H512v-103.88z\"/><path fill=\"#e1392d\" d=\"M503.181 189.217H512V85.337H132.552l103.881 103.88z\"/></svg>"],"GH":["Ghana","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#FFDA44\" d=\"M0 0h513v342H0z\"/><path fill=\"#D80027\" d=\"M0 0h513v114H0z\"/><path fill=\"#496E2D\" d=\"M0 228h513v114H0z\"/><path d=\"m255.9 113.8 14.1 43.4 40.4 3.2-37 26.9 19.5 40.3-37-26.9-37 26.9 14.1-43.5-36.9-26.9h45.7z\"/></svg>"],"SN":["S\u00e9n\u00e9gal","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 85.333 512 341.333\"><path fill=\"#FFDA44\" d=\"M0 85.331h512v341.326H0z\"/><path fill=\"#D80027\" d=\"M330.207 85.331H512v341.337H330.207z\"/><g fill=\"#496E2D\"><path d=\"M0 85.331h181.793v341.337H0zM255.999 196.632l14.733 45.347h47.685l-38.576 28.029 14.734 45.348-38.576-28.026-38.577 28.026 14.737-45.348-38.576-28.029h47.681z\"/></g></svg>"],"ET":["\u00c9thiopie","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 22.5 15\"><path fill=\"#20AA46\" d=\"M0 0h22.5v5H0V0z\"/><path fill=\"#E92F3B\" d=\"M0 10h22.5v5H0v-5z\"/><path fill=\"#FADF50\" d=\"M0 5h22.5v5H0V5z\"/><circle fill=\"#205CCA\" cx=\"11.3\" cy=\"7.5\" r=\"5.2\"/><g stroke=\"#FFDB3D\" fill=\"none\"><g stroke-width=\".5\"><path d=\"m11.3 8.8-2.1 1.5.8-2.4-2-1.5h2.5l.8-2.3.8 2.3h2.6l-2.1 1.5.8 2.4-2.1-1.5zM10.3 6.4h1.9M9.9 7.8l1.3 1M12 6.2l.7 1.8M10 7.9l.6-1.8M11 9l1.6-1.1\"/></g><g stroke-width=\".25\"><path d=\"m8.7 3.9 1.1 1.6M13.8 3.8l-1 1.6M11.3 10.1V12M8.9 8.3 7.1 9M15.5 9l-1.8-.7\"/></g></g></svg>"],"CI":["C\u00f4te d'Ivoire","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#FFF\" d=\"M0 0h513v342H0z\"/><path fill=\"#009e60\" d=\"M342 0h171v342H342z\"/><path fill=\"#f77f00\" d=\"M0 0h171v342H0z\"/></svg>"],"CD":["RD Congo","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#338AF3\" d=\"M0 0h513v342H0z\"/><path fill=\"#FFDA44\" d=\"M513 66.9V0H411.7L0 274.4V342h100.3z\"/><path fill=\"#D80027\" d=\"M513 0v40.1L60.2 342H0v-40.8L451.8 0z\"/><path fill=\"#FFDA44\" d=\"m93.6 31.2 16.3 50.4H163l-42.9 31.2 16.4 50.5-42.9-31.2-43 31.2L67 112.8 24.1 81.6h53.1z\"/></svg>"],"CM":["Cameroun","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#ce1126\" d=\"M0 0h513v342H0z\"/><path fill=\"#007a5e\" d=\"M0 0h171v342H0z\"/><g fill=\"#fcd116\"><path d=\"M342 0h171v342H342zM256 102.2l17.2 53H329L283.9 188l17.2 53-45.1-32.7-45.1 32.7 17.2-53-45.1-32.8h55.8z\"/></g></svg>"],"AE":["\u00c9mirats arabes unis","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#FFF\" d=\"M0 0h513v342H0z\"/><path fill=\"#00843d\" d=\"M0 0h513v114H0z\"/><path d=\"M0 228h513v114H0z\"/><path fill=\"#c8102e\" d=\"M0 0h171v342H0z\"/></svg>"],"SA":["Arabie saoudite","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path d=\"M0 0h513v342H0z\" fill=\"#fff\"/><g fill=\"#00512e\"><path d=\"M0 0v342h513V0zm218 76q1 2-2 5-3 2-1 4 3 4 0 6-2 1-5-1-6-6 0-14 6-4 8 0m-102 2 1 6q0 4-2 4-5 1-5-6l2-4zm39 1 1 2q-1 4-5 4-5 0-6-4c0-4 7-6 10-2m47 2q3 4 2 9c-1 2-22 19-24 19q-3 0-4-4l6-6 6-5v-3c-5-10 7-18 14-10m35-3q3 1 0 7-2 8-6 6c-3-1-3 14 0 19q2 4 7-1c2-3 2-5 2-15 0-12 0-13 2-13q5-2 5 13 1 11 3 13 3 6 6 3 3-2-1-12-3-12 1-16h9q2-2 9 1l6 2 1-2q4-4 6-1l3 17c3 16 3 16 7 24l15 30q1 6-4 5l-8-14-6-12 1 20c1 21 1 23-4 23q-4 2-4-15l-1-27-1-16-9-13c-8-12-8-13-11-13q-6-1-2 6l2 7q0 4 3 2 6-2 6 3-1 4-8 5l-5 2-5 2q-6 0-10-3-2-3-5 1-5 5-12 2-5-1-7-10l-1-3-1 3q-3 7-9 8-7 0-9-11-3-10 2-10 4-1 5 7l2 6q1 0 4-7 6-17 13-12h1c-1-1 2-10 4-11zm-97 10 2 16 1 9h4q5 2 6-1c2-1 2-2-1-11l-3-12 3-1 11 5V82q4-2 6 1l3 23 1 20h2q5 0 5 5 1 6 3 0 3-9 8-2 1 3 2 0 3-7 4-6c5 0 6 7 1 13q-3 3-7 3l-6 1q-5 4-10 0l-2-2v5c0 21-16 42-33 42q-21-1-23-21 0-5 2-6 4-1 6 6 1 12 12 13 14 2 24-15 6-11 1-5-12 18-28 11-7-5-9-17-2-6 1-7 5-3 6 6 1 12 10 12 7 0 13-8 5-8-1-7-11-1-8-12 8-11 17-5l3 2-1-12-4-22q-3 1 0 7c2 6 2 6 0 10l-5 5q-3 3-7 1h-5l-1 16c-1 19-1 20-4 21-4 1-4-2-4-19l1-15-3 1q-3 1-6-2c-2-2-2-2-3-14l-1-12q-2-1-2 10l-1 13q-3 6-8 4c-3-2-4-4-7-15q-4-15 2-14 4-1 5 8l1 5 1-6q0-13 9-12l6 2c2 3 2 4 2 14l1 11 2-1 1-8-2-16q-3-12 2-11 3-1 5 7m163-3q4 10 1 13-4 3-7-7-3-11 2-10zm12-3 3 7q2 6 1 8v16l2 27c2 22 2 21 6 21h3l-1-5-2-23-3-33c-2-16-1-19 3-19 3 0 8 14 6 16v24c5 47 5 48 0 47q-3-2-1 4l2 6q2 1 5-9 7-18 10-16 3 0 7 5 4 7 7 7 4-1 1-33l-1-21-10-12-9-15q0-5 8-1 5 2 6 6l2 6q3 4 1-4 0-10 3-10c4 0 9 10 7 13l-1 8q1 8 4 12l3 3-2-15c-2-16-2-19 1-20q3 0 5 4l4 5h2q6-4 8 9 0 8-4 8-3 0-4-6 1-8-3-5c-1 2-1 3 1 19l2 17 5 7c6 10 12 21 11 22q-2 3-5 2l-5-8-5-8 1 8c0 11 0 11-3 11q-5 3-5-7l-1-16c0-8-1-10-4-15l-4-5v19c1 21 0 25-5 28q-6 5-13-3l-4-5-4 9q-6 14-11 15-9-1-11-16-1-8-5-1c-3 5-12 13-19 17q-8 4-8-2-1-2 6-6c9-5 17-14 17-18l-5-78q2-4 5-1m89 5 4 6q1 1-2 3l-2 2 4 24q4 25 4 35c0 10 0 11-3 17q-7 13-19 14-8 0-7-2-2-4 7-6 15-2 14-23a708 708 0 0 0-8-75c3-2 3-1 8 5m-211 19 1 4q0 3 2 2 5 0 5 4c0 4-9 6-13 2-4-5 0-17 5-12m108 1h2l4-1q4 2 0 7-7 6-11-1v-5q3-2 5 0m55 2q3 4-1 6-2 0-1 5t-1 6q-1 2-10-6-8-7-6-9 3-4 7 0l2 2 2-3q5-3 8-1m-131 13c3 2 6 14 6 17q-1 6 6 1l10-4q5-1 7-4 4-9 10-12 10 0 11 9 0 7-16 13l-2 1 13 1c16 0 16 0 13 9q-2 7-4 7l-10 2q-19 3-17 9 1 2 4 3 6 3 3 7l-7 1q-13-2-21-10l-3-3-2 3q-7 10-19 13-10 2-15-10l-5 2q-18 10-19 8l-2-3c0-2 1-2 15-11l9-6q1-3 5-4c3 1 10-5 12-9q3-6-1-15l-2-8q1-5 6-6c3 0 8 5 8 7q-1 4-4 3v1l1 10c1 9-1 13-10 21l-7 6 2 3q5 9 15-2 10-8 10-23 0-10-6-24v-4q3-4 6 1m17 2q2 3 4 0 11-3 5 5c-4 5-12 5-15-1v-4zm118 5 1 2q1 2-7 8c-8 6-9 7-11 5q-5-4 6-10c8-6 8-6 11-5m-229 3q3 2-10 10-12 8-13 3-5-3 9-10 13-8 14-3m229 10c3 2 4 9 3 11q-3 3-6 1c-2-2-3-11-1-13q1-2 4 1m-172 7q3 3 2 7 1 5-3 5-3 1-3-3l-2-4q-3-3-1-6 4-2 7 1m-79 9 1 8q0 9 3 6 4-1 4 3-1 7-7 7-7-1-7-14-1-9 2-10zm168 12q0 4 3 2t4 3q-2 6-8 8l-4-2q-3-3-4-2-4 0-5-4l5-6q7-6 7-4 3 1 2 5m102 3q4 4 1 9-7 7-7-4t6-5m-42 73c0 2 1 2 12 2q24-1 22 11 0 7-6 10c-4 1-31 2-34 0q-2-1-2-6v-4h-82l-74-1c-16 0-28-10-28-10h92c91 0 92-1 92-2q-1-3 4-5c5-2 4 4 4 5\"/><path d=\"m195 86 1 1 1-1-1-1q-2 0-1 1m84 5q-2 2 1 5l3 3-1-4-1-5zm-125 45q-3 3-1 4 5-1 6-4zm110-7q-4 5 0 3 6-4 2-4zm-31 30q0 6 4 8l5 4q3 2 2-3 0-3 4-7 4-3 2-4l-13-3h-5zm104 101q-1 3 13 2c11 0 13 0 14-2l-13-1z\"/></g></svg>"],"QA":["Qatar","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#8a1538\" d=\"M0 0h513v342H0z\"/><path fill=\"#FFF\" d=\"M0 0v342h150.3l37.7-19.6-37.7-18.9 37.7-19-37.7-18.9 37.7-19-37.7-19 37.7-18.9-37.7-19 37.7-19-37.7-18.9 37.7-19-37.7-18.9 37.7-19-37.7-19L188 57l-37.7-19L188 19.1 150.3 0z\"/></svg>"],"KW":["Kowe\u00eft","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#FFF\" d=\"M0 0h513v342H0z\"/><path fill=\"#6DA544\" d=\"M0 0h513v114H0z\"/><path fill=\"#D80027\" d=\"M0 228h513v114H0z\"/><path d=\"M167 227.6 0 341.3V0l167 113.8z\"/></svg>"],"BH":["Bahre\u00efn","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#D80027\" d=\"M0 0h513v342H0z\"/><path fill=\"#FFF\" d=\"m222.8 34.3-85.2 34.2 85.2 34.1-85.2 34.1 85.2 34.1-85.2 34.1 85.2 34.1-85.2 34.1 85.2 34.1-85.2 34.8H0V0h137.6z\"/></svg>"],"OM":["Oman","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#D80027\" d=\"M0 0h513v342H0z\"/><path fill=\"#FFF\" d=\"M138 0h377v114H138z\"/><path fill=\"#4A7C3A\" d=\"M138 230h377v114H138z\"/><g fill=\"none\" stroke=\"#FFF\" stroke-width=\"10\" stroke-miterlimit=\"2\"><path d=\"M40.3 35.2s37.1 48 50.8 54.5c13.7 6.5 17.1 5.7 17.1 5.7\"/><path d=\"M100 35.2S62.9 80.6 53.3 87.6c-9.6 7-18.7 7.8-18.7 7.8M51.7 65.3h37.9M70.7 35.2v30.1\"/></g></svg>"],"LB":["Liban","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#FFF\" d=\"M0 0h513v342H0z\"/><g fill=\"#D80027\"><path d=\"M0 .2h513v90.7H0zM.5 251.3H513V342H.5z\"/></g><path fill=\"#6DA544\" d=\"M290.3 193.5 256 130.7l-34.3 62.8h25.7v17.2h17.2v-17.2z\"/><path fill=\"#14AF5A\" d=\"M241.3 213.1c4.4-4.4 4.4-11.6-.1-16l4.7 4.7c-5.2-4.7-12.2-6.6-19.1-5.3l-23.5 5.9c-6.1 1.5-6.8-.4-1.8-4.2l27.1-20.3c5.1-3.8 4.2-6.9-2.3-6.9h-11c-6.4 0-7-2.3-1.5-5.1l25.3-12.7c5.6-2.8 5.1-5.1-1.5-5.1h-11c-6.4 0-7.4-3.1-2.3-6.9l27.1-20.3c5.6-3.8 12.9-3.8 18.5 0l27.1 20.3c5.1 3.8 4.2 6.9-2.3 6.9h-11c-6.4 0-7 2.3-1.5 5.1l25.3 12.7c5.6 2.8 5.1 5.1-1.5 5.1h-11c-6.4 0-7.4 3.1-2.3 6.9l27.1 20.3c5.1 3.8 4.4 5.7-1.8 4.2l-23.5-5.9c-6.8-1.3-13.9.6-19.1 5.3l4.7-4.7c-4.4 4.4-4.5 11.5-.1 16l6.8 6.8c4.4 4.4 3.1 8-3.3 8h-45.8c-6.2 0-7.7-3.6-3.3-8 .1.1 6.9-6.8 6.9-6.8z\"/></svg>"],"JO":["Jordanie","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#FFF\" d=\"M0 0h513v342H0z\"/><path d=\"M0 0h513v114H0z\"/><path fill=\"#6DA544\" d=\"M0 228h513v114H0z\"/><path fill=\"#D80027\" d=\"M256.5 170.7 0 341.3V0z\"/><path fill=\"#FFF\" d=\"m77.9 139.5 7.9 16.4 17.8-4.1-8 16.5 14.3 11.3-17.8 4 .1 18.3-14.3-11.5-14.2 11.5v-18.3l-17.8-4 14.3-11.3-7.9-16.5 17.7 4.1z\"/></svg>"],"IQ":["Irak","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#FFF\" d=\"M0 0h513v342H0z\"/><path fill=\"#CE1126\" d=\"M0 0h513v114H0z\"/><path d=\"M0 228h513v114H0z\"/><g fill=\"#547C31\"><path d=\"M219.2 160.7h-29.3c1.5-5.7 6.6-9.9 12.8-9.9v-19.9c-18.3 0-33.1 14.9-33.1 33.1v16.5h49.6c1.8 0 3.3 1.5 3.3 3.3v6.6h-66.2v19.9h86.1v-26.5c0-12.7-10.4-23.1-23.2-23.1zM268.8 190.5v-59.6H249v79.5h33.1v-19.9zM335 190.5v-59.6h-19.8v59.6h-6.6v-19.8h-19.9v39.7h59.6v-19.9z\"/></g></svg>"],"IR":["Iran","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#FFF\" d=\"M0 0h512v342H0z\"/><path fill=\"#6DA544\" d=\"M0 0h513v114H0z\"/><g fill=\"#D80027\"><path d=\"M0 227.9h513v114H0zM278.8 134.8c.1 2 8.7 26.2 4.4 39.4-6.6 20.3-15.8 21.8-19.8 24.5V134l-6.9-4.2-6.9 4.2v64.7c-4-2.7-12.4-2.4-19.8-24.5-4.3-12.7 5.7-37.3 5.8-39.2 0 0-9.5 8.1-15.8 24-5.9 14.8 1.9 49.6 29.5 54.8 2.3.4 4.7 5.6 7.2 5.6 2.1 0 4.1-5.2 6-5.5 28.4-4.6 35-41.7 29.9-55.6-5.4-14.6-13.6-23.5-13.6-23.5z\"/></g><g fill=\"#FFF\" opacity=\".5\"><path d=\"M44.6 98.9h22.3v24.4H44.6zM0 98.9h22.3v24.4H0zM89.2 98.9h22.3v24.4H89.2zM133.8 98.9h22.3v24.4h-22.3zM178.4 98.9h22.3v24.4h-22.3zM223 98.9h22.3v24.4H223zM267.7 98.9H290v24.4h-22.3zM312.3 98.9h22.3v24.4h-22.3zM356.9 98.9h22.3v24.4h-22.3zM401.5 98.9h22.3v24.4h-22.3zM446.1 98.9h22.3v24.4h-22.3zM490.7 98.9H513v24.4h-22.3zM44.6 216.9h22.3v25.5H44.6zM0 216.9h22.3v25.5H0zM89.2 216.9h22.3v25.5H89.2zM133.8 216.9h22.3v25.5h-22.3zM178.4 216.9h22.3v25.5h-22.3zM223 216.9h22.3v25.5H223zM267.7 216.9H290v25.5h-22.3zM312.3 216.9h22.3v25.5h-22.3zM356.9 216.9h22.3v25.5h-22.3zM401.5 216.9h22.3v25.5h-22.3zM446.1 216.9h22.3v25.5h-22.3zM490.7 216.9H513v25.5h-22.3z\"/></g></svg>"],"IL":["Isra\u00ebl","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#FFF\" d=\"M0 0h513v342H0z\"/><g fill=\"#2E52B2\"><path d=\"M340.6 122.4h-56.1l-28-48.6-28 48.6h-56.1l28 48.6-28 48.6h56.1l28 48.6 28-48.6h56.1l-28-48.6 28-48.6zM293.2 171 276 204.2h-38.9L219.8 171l17.2-33.2h38.9l17.3 33.2zm-36.7-71.8 11.9 23.3h-23.9l12-23.3zm-58.3 38.6h23.9l-10.8 21-13.1-21zm0 66.4 13-22.1 11.9 22.1h-24.9zm58.3 37.5-11.9-22.1h23.9l-12 22.1zm59.4-37.5h-25l11.9-22.1 13.1 22.1zm-26.1-66.4h26.1l-13 22.1-13.1-22.1zM0 21.3h512V64H0zM0 277.3h512V320H0z\"/></g></svg>"],"SY":["Syrie","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 900 600\"><path d=\"M0 0h900v600H0z\" fill=\"#0f0f0f\"/><path d=\"M0 0h900v400H0z\" fill=\"#fff\"/><path d=\"M0 0h900v200H0z\" fill=\"#047a3f\"/><path d=\"m176.26 375 48.738-150 48.738 150-127.6-92.705h157.72M626.256 375l48.738-150 48.738 150-127.6-92.705h157.72M401.252 375l48.738-150 48.738 150-127.6-92.705h157.72\" fill=\"#ce1126\"/></svg>"],"PK":["Pakistan","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 900 600\"><path fill=\"#FFF\" d=\"M0 0h900v600H0z\"/><path fill=\"#01411C\" d=\"M225 0h675v600H225z\"/><path fill=\"#FFF\" d=\"M562.5 120c99.4 0 180 80.6 180 180s-80.6 180-180 180-180-80.6-180-180 80.6-180 180-180z\"/><path fill=\"#01411C\" d=\"M498.9 135.8c68.1-60.5 172.4-54.4 232.9 13.7s54.4 172.4-13.7 232.9-172.4 54.4-232.9-13.7-54.4-172.4 13.7-232.9z\"/><path fill=\"#FFF\" d=\"m628.1 165.3 75.8 85.3-111.5-24.4L697 180.4l-57.7 98.5-11.2-113.6z\"/></svg>"],"IN":["Inde","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#ff6820\" d=\"M0 0h513v114H0V0z\"/><path fill=\"#FFF\" d=\"M0 114h513v114H0V114z\"/><path fill=\"#046a38\" d=\"M0 228h513v114H0V228z\"/><path fill=\"none\" stroke=\"#07038d\" stroke-width=\"4\" d=\"M256.5 136.8c18.9 0 34.2 15.3 34.2 34.2s-15.3 34.2-34.2 34.2-34.2-15.3-34.2-34.2 15.3-34.2 34.2-34.2z\"/><g stroke=\"#07038d\" stroke-width=\"2\"><path d=\"m265.3 138.2-17.6 65.7m17.6-65.7-17.6 65.7M273.5 141.6l-34 58.9M280.5 147l-48 48M285.9 154 227 188M289.3 162.2l-65.7 17.6M290.5 171h-68M289.3 179.8l-65.7-17.6M285.9 188 227 154M280.5 195l-48-48M273.5 200.4l-34-58.9M265.3 203.8l-17.6-65.7M256.5 205v-68M247.7 203.8l17.6-65.7M239.5 200.4l34-58.9M232.5 195l48.1-48.1M227.1 188l58.9-34M223.7 179.8l65.7-17.6M222.5 171h68M223.7 162.2l65.7 17.6M227.1 154l58.9 34M232.5 147l48.1 48.1M239.5 141.6l34 58.9M247.7 138.2l17.6 65.7\"/></g></svg>"],"BD":["Bangladesh","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 85.333 513 342\"><path fill=\"#2d6e41\" d=\"M0 85.331h513v342H0z\"/><circle fill=\"#F40B32\" cx=\"218.902\" cy=\"256.5\" r=\"115\"/></svg>"],"TH":["Tha\u00eflande","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 85.333 512 341.333\"><path fill=\"#FFF\" d=\"M0 85.334h512V426.66H0z\"/><path fill=\"#0052B4\" d=\"M0 194.056h512v123.882H0z\"/><g fill=\"#D80027\"><path d=\"M0 85.334h512v54.522H0zM0 372.143h512v54.522H0z\"/></g></svg>"],"SG":["Singapour","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 900 600\"><path fill=\"#FFF\" d=\"M0 0h900v600H0V0z\"/><path fill=\"#ED2939\" d=\"M0 0h900v300H0V0z\"/><g fill=\"#FFF\"><path d=\"M129.7 150c0-48.8 29-90.9 70.6-109.9-3.4-.3-6.8-.5-10.2-.5-61 0-110.4 49.4-110.4 110.4s49.4 110.4 110.4 110.4c3.4 0 6.9-.2 10.2-.5-41.7-19-70.6-61.1-70.6-109.9zM260.9 90.1l16.9-12.3H257l-6.5-19.9-6.5 19.9h-20.8l16.9 12.3-6.5 19.8 16.9-12.3 16.9 12.3zM298.2 204.6l16.9-12.2h-20.9l-6.5-19.9-6.4 19.9h-20.9l16.9 12.2-6.5 19.9 16.9-12.3 16.9 12.3z\"/><path d=\"m200.7 133.8 16.9-12.3h-20.9l-6.4-19.8-6.5 19.8h-20.9l16.9 12.3-6.4 19.9 16.9-12.3 16.9 12.3zM338.1 121.5h-20.9l-6.5-19.8-6.4 19.8h-20.9l16.9 12.3-6.5 19.9 16.9-12.3 16.9 12.3-6.4-19.9zM223.7 204.6l16.9-12.2h-20.9l-6.4-19.9-6.5 19.9h-20.9l16.9 12.2-6.4 19.9 16.9-12.3 16.9 12.3z\"/></g></svg>"],"MY":["Malaisie","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#FFF\" d=\"M0 0h513v342H0z\"/><g fill=\"#c00\"><path d=\"M0 0h513v24.4H0zM0 48.9h513v24.4H0zM0 97.7h513v24.4H0zM0 146.6h513V171H0zM0 195.4h513v24.4H0zM0 244.3h513v24.4H0zM0 293.1h513v24.4H0z\"/></g><path fill=\"#006\" d=\"M0 0h256.5v195.4H0z\"/><path d=\"M153.3 42.1C122.6 30.7 88.5 46.3 77.1 77s4.2 64.8 34.9 76.2c13.3 5 28 5 41.4 0-30.7 24.5-75.4 19.6-100-11.1s-19.6-75.4 11.1-100c26-20.7 62.9-20.7 88.8 0zm26.7 75-20.6 23.3 5.4-30.6-31-1.6 27.3-14.9L143 68l28.6 12 8.4-29.9 8.4 29.9L217 68l-18 25.4 27.3 14.9-31 1.6 5.4 30.6-20.7-23.4z\" fill=\"#fc0\"/></svg>"],"ID":["Indon\u00e9sie","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#FFF\" d=\"M0 0h513v342H0z\"/><path fill=\"#E00\" d=\"M0 0h513v171H0z\"/></svg>"],"PH":["Philippines","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#FFF\" d=\"M0 0h513v342H0z\"/><path fill=\"#0038A8\" d=\"M513 0v171H256.5L0 0z\"/><path fill=\"#CE1126\" d=\"M513 171v171H0l256.5-171z\"/><g fill=\"#FCD116\"><path d=\"m22.1 33.8 8.5 11.9 13.8-4.5-8.6 11.7 8.5 11.8-13.8-4.6-8.6 11.7V57.2L8.2 52.7 22 48.4zM22.1 270.2l8.5 11.8 13.8-4.4-8.6 11.7 8.5 11.8-13.8-4.6-8.6 11.7v-14.6l-13.7-4.5 13.8-4.3zM210.5 152l-8.5 11.8-13.8-4.4 8.6 11.7-8.4 11.8 13.7-4.6 8.6 11.7v-14.6l13.7-4.5-13.8-4.3z\"/><circle cx=\"95\" cy=\"171\" r=\"32\"/><path d=\"m95 171 11.2 56.5-3.4 3.4-3.7 3.5-4.2 4.2-4.2-4.2-3.7-3.5-3.4-3.4L95 171zM95 171l-11.2-56.5 3.4-3.4 3.7-3.5 4.2-4.2 4.2 4.2 3.7 3.5 3.4 3.4L95 171zM95 171l56.5-11.2 3.4 3.4 3.5 3.7 4.2 4.2-4.2 4.2-3.5 3.7-3.4 3.4L95 171zM95 171l-56.5 11.2-3.4-3.4-3.5-3.7-4.2-4.2 4.2-4.2 3.5-3.7 3.4-3.4L95 171zM95 171l32-47.9h4.8l5.1.2h5.9v5.9l.2 5.1v4.8L95 171zM95 171l-32 47.9h-4.8l-5.1-.2h-5.9v-5.9l-.2-5.1v-4.8L95 171zM95 171l47.9 32v4.8l-.2 5.1v5.9h-5.9l-5.1.2h-4.8L95 171zM95 171l-47.9-32v-4.8l.2-5.1v-5.9h5.9l5.1-.2h4.8L95 171z\"/></g></svg>"],"VN":["Vietnam","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 85.333 512 341.333\"><path fill=\"#D80027\" d=\"M196.641 85.337H0v341.326h512V85.337z\"/><path fill=\"#FFDA44\" d=\"m256 157.279 22.663 69.747H352l-59.332 43.106 22.664 69.749L256 296.774l-59.332 43.107 22.664-69.749L160 227.026h73.337z\"/></svg>"],"HK":["Hong Kong","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 21 14\"><path fill=\"#FFF\" d=\"M0 0h21v14H0V0z\"/><path fill=\"#EE1C25\" d=\"M0 0h21v14H0V0z\"/><path fill=\"#FFF\" d=\"M12 6.7c-.8-.5-1 .4-1 0 0-.8.9-1.5 2-1.5s2 .7 2 1.5c-.9 0-1 .5-1.5.5-.6 0-.7 0-1.5-.5zm1.2-.2c.1 0 .2-.1.2-.2s0-.3-.2-.3-.2.1-.2.2.1.3.2.3zm-1.8 2c.2-.9-.7-.8-.3-.9.8-.3 1.7.4 2 1.4.3 1.1 0 2.1-.8 2.4-.3-.9-.8-.8-.9-1.3-.1-.6-.2-.7 0-1.6zm.6 1.1c0 .1.2.2.3.2.1 0 .2-.2.2-.3s-.2-.2-.3-.2c-.2 0-.2.2-.2.3zM9.6 8.5c.9-.1.6-.9.8-.6.5.7.2 1.7-.7 2.4-.9.6-2 .6-2.5 0 .8-.6.6-1 .9-1.3.4-.4.5-.5 1.5-.5zm-.9.9c-.1 0-.2.2-.1.3s.3.2.4.1.1-.3 0-.4-.2-.1-.3 0zM9 6.7c.3.9 1 .3.8.6-.5.7-1.6.7-2.5 0-.9-.6-1.2-1.7-.7-2.4.8.6 1.1.2 1.5.5.4.4.5.4.9 1.3zm-1.1-.6c-.1-.1-.3 0-.4.1s-.1.3 0 .4c.1.1.3 0 .3-.1.2-.1.2-.3.1-.4zm2.6-.5c-.7.6.1 1.1-.3 1-.8-.3-1.2-1.4-.8-2.4.3-1 1.3-1.7 2-1.4-.3.9.2 1.1 0 1.6-.1.5-.2.6-.9 1.2zm.2-1.2c0-.1 0-.3-.1-.3s-.3 0-.3.1 0 .3.2.3c0 .1.1 0 .2-.1z\"/></svg>"],"CN":["Chine","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#D80027\" d=\"M0 0h513v342H0z\"/><g fill=\"#FFDA44\"><path d=\"m226.8 239.2-9.7-15.6-17.9 4.4 11.9-14.1-9.7-15.6 17.1 6.9 11.8-14.1-1.3 18.4 17.1 6.9-17.9 4.4zM290.6 82l-10.1 15.4 11.6 14.3-17.7-4.8-10.1 15.5-1-18.4-17.7-4.8 17.2-6.6-1-18.4 11.6 14.3zM236.2 25.4l-2 18.3 16.8 7.6-18 3.8-2 18.3-9.2-16-17.9 3.8 12.3-13.7-9.2-15.9 16.8 7.5zM292.8 161.8l-14.9 10.9 5.8 17.5-14.9-10.8-14.9 11 5.6-17.6-14.9-10.7 18.4-.1 5.6-17.6 5.8 17.5zM115 46.3l17.3 53.5h56.2l-45.4 32.9 17.3 53.5-45.4-33-45.5 33 17.4-53.5-45.5-32.9h56.3z\"/></g></svg>"],"TW":["Ta\u00efwan","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 85.333 512 341.333\"><path fill=\"#D80027\" d=\"M0 85.337h512v341.326H0z\"/><path fill=\"#0052B4\" d=\"M0 85.337h256V256H0z\"/><path fill=\"#FFF\" d=\"M186.435 170.669 162.558 181.9l12.714 23.125-25.927-4.961-3.286 26.192L128 206.993l-18.06 19.263-3.285-26.192-25.927 4.96 12.714-23.125-23.877-11.23 23.877-11.231-12.714-23.125 25.927 4.96 3.286-26.192L128 134.344l18.06-19.263 3.285 26.192 25.928-4.96-12.715 23.125z\"/><circle fill=\"#0052B4\" cx=\"128\" cy=\"170.674\" r=\"29.006\"/><path fill=\"#FFF\" d=\"M128 190.06c-10.692 0-19.391-8.7-19.391-19.391 0-10.692 8.7-19.391 19.391-19.391 10.692 0 19.391 8.7 19.391 19.391 0 10.691-8.699 19.391-19.391 19.391z\"/></svg>"],"KR":["Cor\u00e9e du Sud","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#FFF\" d=\"M0 0h513v342H0z\"/><g stroke=\"#000\" stroke-width=\"14.25\"><path d=\"M84.6 107.8 132 36.6m-29.7 83 47.4-71.1m-29.6 83 47.4-71.1M345.4 281.7l47.4-71.1m-29.6 82.9 47.4-71.1m-29.6 83 47.4-71.1M132 305.4l-47.4-71.1m65.2 59.2-47.4-71.1m65.2 59.3-47.4-71.1M392.8 131.5l-47.4-71.1m65.2 59.2-47.4-71.1m65.2 59.3L381 36.6\"/></g><g stroke=\"#FFF\" stroke-width=\"12.5\"><path d=\"m357.3 238.2 59.3 39.5M117.2 263.9 135 252m222.3-148.2L378 90m17.8-11.9 17.8-11.9\"/></g><circle fill=\"#CA163A\" cx=\"256.5\" cy=\"171\" r=\"85.5\"/><path fill=\"#0E4896\" d=\"M185.3 123.6c-13.1 19.6-7.8 46.2 11.9 59.3s46.2 7.8 59.3-11.9 39.6-24.9 59.3-11.9c19.6 13.1 24.9 39.6 11.9 59.3-26.2 39.3-79.3 49.9-118.6 23.7s-49.9-79.3-23.8-118.5z\"/></svg>"],"JP":["Japon","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#FFF\" d=\"M0 0h512v342H0z\"/><circle fill=\"#D80027\" cx=\"256.5\" cy=\"171\" r=\"96\"/></svg>"],"AU":["Australie","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#00008b\" d=\"M0 0h513v342H0z\"/><g fill=\"#FFF\"><path d=\"m188 212.6 11 22.9 24.7-5.7-11 22.8 19.9 15.8-24.8 5.6.1 25.4-19.9-15.9-19.8 15.9.1-25.4-24.8-5.6 19.9-15.8-11.1-22.8 24.8 5.7zM385.9 241.1l5.2 10.9 11.8-2.7-5.3 10.9 9.5 7.5-11.8 2.6v12.2l-9.4-7.6-9.5 7.6.1-12.2-11.8-2.6 9.5-7.5-5.3-10.9 11.8 2.7zM337.3 125.1l5.2 10.9 11.8-2.7-5.3 10.9 9.5 7.5-11.8 2.7v12.1l-9.4-7.6-9.5 7.6.1-12.1-11.9-2.7 9.5-7.5-5.3-10.9L332 136zM385.9 58.9l5.2 10.9 11.8-2.7-5.3 10.9 9.5 7.5-11.8 2.7v12.1l-9.4-7.6-9.5 7.6.1-12.1-11.8-2.7 9.5-7.5-5.3-10.9 11.8 2.7zM428.4 108.6l5.2 10.9 11.8-2.7-5.3 10.9 9.5 7.5-11.8 2.6V150l-9.4-7.6-9.5 7.6v-12.2l-11.8-2.6 9.5-7.5-5.3-10.9 11.8 2.7zM398 166.5l4.1 12.7h13.3l-10.8 7.8 4.2 12.7-10.8-7.9-10.8 7.9 4.1-12.7-10.7-7.8h13.3z\"/></g><path fill=\"#00008b\" d=\"M0 0h256.5v171H0z\"/><g fill=\"#FFF\"><path d=\"M256.5 0v30.6l-45.3 25.2h45.3v59.4h-59.2l59.2 32.9V171h-26.7l-73.7-40.9V171h-55.7v-48.7L12.8 171H0v-30.6l45.3-25.2H0V55.8h59.2L0 22.9V0h26.7l73.7 40.9V0h55.7v48.7L243.7 0z\"/><path d=\"M156.1 115.2 256.5 171v-15.8l-72-40zM100.4 55.8 0 0v15.8l72 40z\"/></g><g fill=\"red\"><path d=\"M144.3 0h-32.1v69.5H0v32h112.2V171h32.1v-69.5h112.2v-32H144.3z\"/><path d=\"M156.1 115.2 256.5 171v-15.8l-72-40zM72 115.2l-72 40V171l100.4-55.8zM100.4 55.8 0 0v15.8l72 40zM184.5 55.8l72-40V0L156.1 55.8z\"/></g></svg>"],"NZ":["Nouvelle-Z\u00e9lande","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#012169\" d=\"M0 0h513v342H0z\"/><g fill=\"#D80027\" stroke=\"#FFF\" stroke-width=\"2\" stroke-miterlimit=\"10\"><path d=\"m448.9 107.7 4.2 13.1h13.8l-11.1 8.1L460 142l-11.1-8.1-11.1 8.1 4.2-13.1-11.1-8.1h13.7zM384.7 253.6l5.1 15.6h16.4l-13.2 9.7 5 15.6-13.3-9.7-13.2 9.7 5-15.6-13.3-9.7h16.5zM384.8 43.4l4.7 14.6h15.3l-12.4 8.9 4.7 14.6-12.3-9-12.4 9 4.7-14.6-12.3-8.9H380zM320.6 129.4l4.7 14.5h15.3l-12.3 9 4.7 14.5-12.4-8.9-12.3 8.9 4.7-14.5-12.4-9h15.3z\"/></g><path fill=\"#012169\" d=\"M0 0h256.5v171H0z\"/><g fill=\"#FFF\"><path d=\"M256.5 0v30.6l-45.3 25.2h45.3v59.4h-59.2l59.2 32.9V171h-26.7l-73.7-40.9V171h-55.7v-48.7L12.8 171H0v-30.6l45.3-25.2H0V55.8h59.2L0 22.9V0h26.7l73.7 40.9V0h55.7v48.7L243.7 0z\"/><path d=\"M156.1 115.2 256.5 171v-15.8l-72-40zM100.4 55.8 0 0v15.8l72 40z\"/></g><g fill=\"#D80027\"><path d=\"M144.3 0h-32.1v69.5H0v32h112.2V171h32.1v-69.5h112.2v-32H144.3z\"/><path d=\"M156.1 115.2 256.5 171v-15.8l-72-40zM72 115.2l-72 40V171l100.4-55.8zM100.4 55.8 0 0v15.8l72 40zM184.5 55.8l72-40V0L156.1 55.8z\"/></g></svg>"],"US":["\u00c9tats-Unis","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#FFF\" d=\"M0 0h513v342H0z\"/><g fill=\"#D80027\"><path d=\"M0 0h513v26.3H0zM0 52.6h513v26.3H0zM0 105.2h513v26.3H0zM0 157.8h513v26.3H0zM0 210.5h513v26.3H0zM0 263.1h513v26.3H0zM0 315.7h513V342H0z\"/></g><path fill=\"#2E52B2\" d=\"M0 0h256.5v184.1H0z\"/><g fill=\"#FFF\"><path d=\"m47.8 138.9-4-12.8-4.4 12.8H26.2l10.7 7.7-4 12.8 10.9-7.9 10.6 7.9-4.1-12.8 10.9-7.7zM104.1 138.9l-4.1-12.8-4.2 12.8H82.6l10.7 7.7-4 12.8 10.7-7.9 10.8 7.9-4-12.8 10.7-7.7zM160.6 138.9l-4.3-12.8-4 12.8h-13.5l11 7.7-4.2 12.8 10.7-7.9 11 7.9-4.2-12.8 10.7-7.7zM216.8 138.9l-4-12.8-4.2 12.8h-13.3l10.8 7.7-4 12.8 10.7-7.9 10.8 7.9-4.3-12.8 11-7.7zM100 75.3l-4.2 12.8H82.6L93.3 96l-4 12.6 10.7-7.8 10.8 7.8-4-12.6 10.7-7.9h-13.4zM43.8 75.3l-4.4 12.8H26.2L36.9 96l-4 12.6 10.9-7.8 10.6 7.8L50.3 96l10.9-7.9H47.8zM156.3 75.3l-4 12.8h-13.5l11 7.9-4.2 12.6 10.7-7.8 11 7.8-4.2-12.6 10.7-7.9h-13.2zM212.8 75.3l-4.2 12.8h-13.3l10.8 7.9-4 12.6 10.7-7.8 10.8 7.8-4.3-12.6 11-7.9h-13.5zM43.8 24.7l-4.4 12.6H26.2l10.7 7.9-4 12.7L43.8 50l10.6 7.9-4.1-12.7 10.9-7.9H47.8zM100 24.7l-4.2 12.6H82.6l10.7 7.9-4 12.7L100 50l10.8 7.9-4-12.7 10.7-7.9h-13.4zM156.3 24.7l-4 12.6h-13.5l11 7.9-4.2 12.7 10.7-7.9 11 7.9-4.2-12.7 10.7-7.9h-13.2zM212.8 24.7l-4.2 12.6h-13.3l10.8 7.9-4 12.7 10.7-7.9 10.8 7.9-4.3-12.7 11-7.9h-13.5z\"/></g></svg>"],"CA":["Canada","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#FFF\" d=\"M0 0h513v342H0z\"/><g fill=\"red\"><path d=\"M0 0h142v342H0zM371 0h142v342H371zM306.5 206l50.4-25.2-25.2-12.6V143l-50.4 25.2 25.2-50.4h-25.2L256.1 80l-25.2 37.8h-25.2l25.2 50.4-50.4-25.2v25.2l-25.2 12.6 50.4 25.2-12.6 25.2h50.4V269h25.2v-37.8h50.4z\"/></g></svg>"],"MX":["Mexique","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#dc2339\" d=\"M342 0H0v341.3h513V0z\"/><path fill=\"#11865d\" d=\"M0 0h171v342H0z\"/><path fill=\"#FFF\" d=\"M171 0h171v342H171z\"/><path fill=\"#8C9157\" d=\"M195.8 171.2c0 21.6 11.5 41.7 30.3 52.5 5.8 3.4 13.2 1.4 16.6-4.4 3.4-5.8 1.4-13.2-4.4-16.6-11.3-6.5-18.2-18.5-18.2-31.5 0-6.7-5.4-12.1-12.1-12.1-6.7 0-12.2 5.4-12.2 12.1zm93.4 51.1c17.5-11.1 28-30.4 28-51.1 0-6.7-5.4-12.1-12.1-12.1s-12.1 5.4-12.1 12.1c0 12.4-6.3 24-16.8 30.7-5.7 3.5-7.5 10.9-4.1 16.7s10.9 7.5 16.7 4.1c0-.2.2-.3.4-.4z\"/><ellipse fill=\"#C59262\" cx=\"256.5\" cy=\"159.1\" rx=\"24.3\" ry=\"36.4\"/></svg>"],"BR":["Br\u00e9sil","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#009b3a\" d=\"M0 0h513v342H0z\"/><path fill=\"#fedf00\" d=\"m256.5 19.3 204.9 151.4L256.5 322 50.6 170.7z\"/><circle fill=\"#FFF\" cx=\"256.5\" cy=\"171\" r=\"80.4\"/><path fill=\"#002776\" d=\"M215.9 165.7c-13.9 0-27.4 2.1-40.1 6 .6 43.9 36.3 79.3 80.3 79.3 27.2 0 51.3-13.6 65.8-34.3-24.9-31-63.2-51-106-51zM334.9 186c.9-5 1.5-10.1 1.5-15.4 0-44.4-36-80.4-80.4-80.4-33.1 0-61.5 20.1-73.9 48.6 10.9-2.2 22.1-3.4 33.6-3.4 46.8.1 89 19.5 119.2 50.6z\"/></svg>"],"AR":["Argentine","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#FFF\" d=\"M0 0h512v342H0z\"/><path fill=\"#338AF3\" d=\"M0 0h512v114H0zM0 228h512v114H0z\"/><circle fill=\"#FFDA44\" stroke=\"#d6ab00\" stroke-width=\"5\" cx=\"256.5\" cy=\"171\" r=\"40\"/></svg>"],"CL":["Chili","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#D80027\" d=\"M0 0h513v342H0z\"/><path fill=\"#FFF\" d=\"M196 0h317v171H196z\"/><path fill=\"#0037A1\" d=\"M0 0h196v171H0z\"/><path fill=\"#FFF\" d=\"M98 24.5 113.1 71H162l-39.6 28.7 15.2 46.5L98 117.5l-39.6 28.7 15.2-46.5L34 71h48.9z\"/></svg>"],"CO":["Colombie","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#FFDA44\" d=\"M0 0h513v342H0z\"/><path fill=\"#D80027\" d=\"M0 256.5h513V342H0z\"/><path fill=\"#0052B4\" d=\"M0 171h513v85.5H0z\"/></svg>"],"PE":["P\u00e9rou","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#FFF\" d=\"M0 0h513v342H0z\"/><g fill=\"#D80027\"><path d=\"M0 0h171v342H0zM342 0h171v342H342z\"/></g><path fill=\"#00A854\" d=\"M224.3 119c4.3 7.6-17.1 71.9-6.5 87.8s21.9 22.4 38.2 22.4c16.3 0 30.7-6.8 39.6-24.7s-14.2-76.3-6.8-85.4c7.4-9.1 29.3 8.5 35.5 35.2 6.2 26.7-8.4 60.2-20.7 70-12.3 9.7-20.3 9.7-47.6 9.7s-32.4-1.4-45-10.7c-12.5-9.4-28.9-35.8-22.7-67.3 6.2-31.7 31.7-44.6 36-37z\"/><path fill=\"#D91023\" stroke=\"#E7CA00\" stroke-width=\"2\" d=\"M272.6 218.9H240c-7.5 0-13.6-6.1-13.6-13.6v-45c0-7.5 6.1-13.6 13.6-13.6h32.6c7.5 0 13.6 6.1 13.6 13.6v45c.1 7.5-6 13.6-13.6 13.6z\"/><ellipse fill=\"#E7CA00\" cx=\"256.4\" cy=\"200.9\" rx=\"25\" ry=\"13.2\"/><path fill=\"#FFF\" d=\"M256.4 147.6v35.1h28.9c0-38.3 3.9-35.1-28.9-35.1z\"/><path fill=\"#007EA8\" d=\"M256.4 147.6v35.1h-28.9c0-37.9-3.7-35.1 28.9-35.1z\"/><path fill=\"#947245\" d=\"M270.1 158.1h1.6v21.5h-1.6z\"/><circle fill=\"#D18219\" stroke=\"#FFF\" cx=\"241.8\" cy=\"165.2\" r=\"8.1\"/><ellipse fill=\"none\" stroke=\"#00A854\" stroke-width=\"6\" cx=\"256.4\" cy=\"122.2\" rx=\"20.5\" ry=\"16.5\"/><ellipse fill=\"none\" stroke=\"#E7CA00\" stroke-width=\"3\" cx=\"256.4\" cy=\"122.2\" rx=\"20.5\" ry=\"16.5\"/><circle fill=\"#E7CA00\" cx=\"270.9\" cy=\"163.1\" r=\"8.9\"/></svg>"],"VE":["Venezuela","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 22.5 15\"><path fill=\"#FFCE00\" d=\"M0 0h22.5v5H0z\"/><path fill=\"#203899\" d=\"M0 5h22.5v5H0z\"/><path fill=\"#D82B2B\" d=\"M0 10h22.5v5H0z\"/><path fill=\"#FFF\" d=\"m12.13 5.38.14.46.45.14-.39.27.01.48-.38-.29-.45.15.15-.44-.28-.38h.47zM10.36 5.38l.27.39h.47l-.28.38.15.45-.45-.16-.38.28.01-.47-.39-.28.45-.13zM13.8 5.94l-.01.48.39.27-.46.14-.14.45-.27-.38h-.48l.29-.38-.15-.45.45.16zM8.69 5.96l.4.27.44-.18-.13.46.3.36-.47.02-.26.4-.16-.44-.46-.12.37-.3zM15.24 7.13l-.2.44.25.4-.47-.05-.31.36-.1-.46-.43-.18.41-.24.03-.47.35.31zM15.43 8.25l.21.43.47.07-.34.33.08.46-.42-.22-.42.22.08-.46-.34-.33.47-.07zM7 8.2l.21.43.47.06-.34.33.08.47L7 9.27l-.42.22.08-.47-.34-.33.47-.06zM7.31 7.05l.45.14.38-.29v.48l.39.26-.45.15-.13.46-.28-.38-.47.01.27-.38z\"/></svg>"],"CU":["Cuba","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#FF9811\" d=\"M0-40.8v422.9-211.4z\"/><path fill=\"#FFF\" d=\"M0 0h513v342H0z\"/><g fill=\"#0052B4\"><path d=\"M0 0h513v68.3H0zM0 136.5h513v68.3H0zM0 273.1h513v68.3H0z\"/></g><path fill=\"#D80027\" d=\"M256 170.7 0 342V0z\"/><path fill=\"#FFF\" d=\"m86.5 111.4 12.7 39.2h41.3l-33.4 24.2 12.8 39.3-33.4-24.2-33.4 24.2 12.8-39.3-33.4-24.2h41.2z\"/></svg>"],"PA":["Panama","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#FFF\" d=\"M0 0h513v342H0z\"/><path fill=\"#da121a\" d=\"M256.5 0H513v171H256.5z\"/><path fill=\"#072357\" d=\"M0 171h256.5v171H0zM128 37.9l11.8 36.4h38.3l-31 22.4 11.9 36.4-31-22.5-31 22.5 11.9-36.4-31-22.4h38.3z\"/><path fill=\"#da121a\" d=\"m384 208.9 11.8 36.4h38.3l-31 22.4 11.9 36.4-31-22.5-31 22.5 11.9-36.4-31-22.4h38.3z\"/></svg>"],"UY":["Uruguay","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#FFF\" d=\"M0 0h513v342H0z\"/><g fill=\"#0038a8\"><path d=\"M0 38h513v38H0zM0 114h513v38H0zM0 190h513v38H0zM0 266h513v38H0z\"/></g><path fill=\"#FFF\" d=\"M0 0h256.5v190H0z\"/><path fill=\"#FED443\" d=\"m128.3 138.7-15.1 22.6-3.8-26.9L86 148.2l8.2-25.9-27.2 2.2 18.6-19.8L60.3 95l25.4-9.7L67 65.5l27.1 2.3-8.2-25.9 23.4 13.8 3.9-26.9 15.1 22.6 15.1-22.6 3.8 26.9 23.4-13.8-8.2 25.9 27.1-2.2-18.6 19.8 25.4 9.7-25.4 9.7 18.6 19.8-27.1-2.2 8.2 25.9-23.4-13.8-3.9 26.9-15-22.7zm0-.5c23.9.9 44-17.6 44.9-41.5.9-23.9-17.6-44-41.5-44.9h-3.4c-23.9.8-42.6 20.8-41.8 44.6.6 22.8 19 41.1 41.8 41.8zm0-10.6c-18.8 0-34-15.2-34-34s15.2-34 34-34 34 15.2 34 34-15.3 34-34 34z\"/></svg>"],"EC":["\u00c9quateur","<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 513 342\"><path fill=\"#FFDA44\" d=\"M0 0h513v342H0z\"/><path fill=\"#0052B4\" d=\"M0 170.7h513V256H0z\"/><path fill=\"#D80027\" d=\"M0 256h513v85.3H0z\"/><circle fill=\"#FFDA44\" cx=\"256\" cy=\"171\" r=\"60\"/><path fill=\"#4C1F00\" d=\"M369.2 79.9s-27.2-13.8-33.9-16c-6.7-2.2-49.1-13.2-49.1-7 0 10.3-13.5 12-15.5 12s-2-5.3-14.8-5.3-13.7 4.8-15.8 4.8c-2.2 0-14.5-1-14.5-11.5 0-5.2-42.9 4-49.1 7-6.1 3.1-33.9 16-33.9 16s45.8 2.7 53.9 6.2 43.8 16.5 43.8 16.5l-2.8 13.3h37.8l-3.8-13.3s35.1-12.7 43.8-16.5 53.9-6.2 53.9-6.2z\"/><path fill=\"#57BA17\" d=\"M217.7 171.7c0 21.1 17.2 38.3 38.3 38.3 21.1 0 38.3-17.2 38.3-38.3v-11.5h-76.5v11.5z\"/><path fill=\"#338AF3\" d=\"M256 110.5c-21.1 0-38.3 17.2-38.3 38.3v11.5h76.5v-11.5c.1-21.1-17.1-38.3-38.2-38.3z\"/></svg>"]};
const FLAG_URI={};
function flagHtml(cc){
  const f=cc&&FLAGS[cc];
  if(!f)return '<span class="flag flag-none" aria-hidden="true"></span>';
  if(!FLAG_URI[cc])FLAG_URI[cc]='data:image/svg+xml;utf8,'+encodeURIComponent(f[1]);
  return `<img class="flag" src="${FLAG_URI[cc]}" alt="${cc}" title="${esc(f[0])}">`;
}
function geoFraudColor(rate){
  // taux de fraude 0 % -> bleu ... 100 % -> rouge (dégradé via violet, cf. légende)
  const t=Math.max(0,Math.min(1,(rate||0)/100));
  const c1=[0,212,255], c2=[191,95,255], c3=[255,45,120];
  let a,b,f;
  if(t<0.5){a=c1;b=c2;f=t/0.5;}else{a=c2;b=c3;f=(t-0.5)/0.5;}
  const r=Math.round(a[0]+(b[0]-a[0])*f);
  const g=Math.round(a[1]+(b[1]-a[1])*f);
  const bl=Math.round(a[2]+(b[2]-a[2])*f);
  return `rgb(${r},${g},${bl})`;
}
// ─────────────────────────────────────────────────────────────────────────
// ─── CARTE TEMPS RÉEL ─────────────────────────────────────────────────────
// • Flux SSE (/stats/geo/stream) : le SERVEUR pousse les données dès que la base
//   change. Si SSE est indisponible, repli automatique sur l'interrogation de
//   /stats/geo toutes les GEO_POLL_MS (3 s).
// • Les marqueurs sont MIS À JOUR sur place (pas reconstruits) : taille et
//   couleur s'animent en douceur, un halo apparaît quand une ville reçoit de
//   nouvelles transactions.
// • SURFACE du cercle strictement proportionnelle au nombre de transactions
//   (rayon = R_MAX x racine(total / total_max)) ; seul un plancher de 1,5 px
//   garde les toutes petites villes visibles.
// • COULEUR = nombre de fraudes de la ville, du bleu (0) au rouge (maximum).
//   Les rayons sont calculés en PIXELS ÉCRAN avec la même formule que la légende,
//   donc les cercles de la carte correspondent exactement à ceux de la légende.
// ─────────────────────────────────────────────────────────────────────────
const GEO_POLL_MS=3000;
const GEO_R_MAX=9, GEO_R_FLOOR=1.5;              // rayon écran max (px) pour la ville la plus active (vue Monde) / plancher de visibilité
const GEO_LEGEND_FRACTIONS=[0.05,0.2,0.5,1];     // fractions du volume max affichées dans la légende
const GEO_ASPECT=GEO_VB_W/GEO_VB_H, GEO_MAX_ZOOM=25;
const geoState={markers:new Map(),cities:new Map(),lastData:[],maxTotal:1,maxFraud:0,hoverKey:null,busy:false,first:true,streamOk:false,es:null,
  vb:{x:0,y:0,w:GEO_VB_W,h:GEO_VB_H},zoom:1,region:'world',onlyFraud:false,noFlash:false,anim:null,drag:null,zoomTimer:null};

// Régions du zoom rapide : [lonMin, lonMax, latMin, latMax] ('world' = vue complète)
const GEO_REGIONS={
  world:null,
  europe:[-12,45,34,62],
  namerica:[-170,-50,8,72],
  samerica:[-85,-32,-56,14],
  africa:[-20,55,-36,38],
  asia:[25,150,-12,60],
  oceania:[105,180,-50,-8],
};
function geoClampView(v){
  v.w=Math.min(GEO_VB_W,Math.max(GEO_VB_W/GEO_MAX_ZOOM,v.w));
  v.h=v.w/GEO_ASPECT;
  v.x=Math.min(GEO_VB_W-v.w,Math.max(0,v.x));
  v.y=Math.min(GEO_VB_H-v.h,Math.max(0,v.y));
  return v;
}
function geoRegionView(name){
  const r=GEO_REGIONS[name];
  if(!r)return {x:0,y:0,w:GEO_VB_W,h:GEO_VB_H};
  const [x1,y1]=geoProject(r[0],r[3]), [x2,y2]=geoProject(r[1],r[2]);
  let w=(x2-x1)*1.06, h=(y2-y1)*1.06;
  const cx=(x1+x2)/2, cy=(y1+y2)/2;
  if(w/h>GEO_ASPECT)h=w/GEO_ASPECT; else w=h*GEO_ASPECT;
  return geoClampView({x:cx-w/2,y:cy-h/2,w:w,h:h});
}
// Zone réellement occupée par la carte dans le SVG (letterboxing « meet »)
function geoContent(svg){
  const r=svg.getBoundingClientRect();
  const w=Math.min(r.width,r.height*GEO_ASPECT)||1;
  const h=w/GEO_ASPECT;
  return {rect:r,w:w,h:h,offX:(r.width-w)/2,offY:(r.height-h)/2};
}
function geoScale(svg){
  return geoContent(svg).w/geoState.vb.w;      // pixels écran par unité de viewBox
}
function geoRadiusPx(total,maxTotal){
  // surface ∝ total  =>  rayon ∝ racine(total) ; léger agrandissement quand on zoome (max x1,7)
  const mult=Math.min(1.7,1+0.15*(geoState.zoom-1));
  return Math.max(GEO_R_FLOOR, GEO_R_MAX*mult*Math.sqrt((total||0)/Math.max(1,maxTotal)));
}
function geoSyncControls(){
  document.querySelectorAll('#geoRegions .chip').forEach(b=>b.classList.toggle('on',b.dataset.region===geoState.region));
  const wrap=document.querySelector('.geo-map-wrap');
  if(wrap)wrap.classList.toggle('zoomed',geoState.zoom>1.01);
}
function geoApplyView(){
  const svg=document.getElementById('geoMap'), vb=geoState.vb;
  svg.setAttribute('viewBox',`${vb.x} ${vb.y} ${vb.w} ${vb.h}`);
  geoState.zoom=GEO_VB_W/vb.w;
  geoSyncControls();
  if(geoState.lastData.length)renderGeoMap(geoState.lastData);   // recalcule les rayons en pixels écran
  else updateGeoLegend(geoState.maxTotal,geoState.maxFraud,false);
}
function geoSetZooming(on){
  const wrap=document.querySelector('.geo-map-wrap');
  if(wrap)wrap.classList.toggle('zooming',on);
}
function geoAnimateTo(target,region){
  if(geoState.anim)cancelAnimationFrame(geoState.anim);
  const from={...geoState.vb}, to=geoClampView({...target}), t0=performance.now(), D=450;
  geoState.region=region||null;
  geoSetZooming(true);
  const ease=t=>t<.5?4*t*t*t:1-Math.pow(-2*t+2,3)/2;
  const step=(now)=>{
    const p=Math.min(1,(now-t0)/D), e=ease(p);
    geoState.vb={x:from.x+(to.x-from.x)*e, y:from.y+(to.y-from.y)*e, w:from.w+(to.w-from.w)*e, h:from.h+(to.h-from.h)*e};
    geoApplyView();
    if(p<1){geoState.anim=requestAnimationFrame(step);}
    else{geoState.anim=null;geoSetZooming(false);}
  };
  geoState.anim=requestAnimationFrame(step);
}
// zoom immédiat (molette / glisser) : ancre = fraction (u,v) de la carte sous le curseur
function geoZoomAt(factor,u,v){
  if(geoState.anim){cancelAnimationFrame(geoState.anim);geoState.anim=null;}
  const vb=geoState.vb, nw=Math.min(GEO_VB_W,Math.max(GEO_VB_W/GEO_MAX_ZOOM,vb.w*factor));
  const ax=vb.x+u*vb.w, ay=vb.y+v*vb.h, nh=nw/GEO_ASPECT;
  geoState.vb=geoClampView({x:ax-u*nw,y:ay-v*nh,w:nw,h:nh});
  geoState.region=null;
  geoNoTransition();
  geoApplyView();
}
function geoNoTransition(){
  geoSetZooming(true);
  clearTimeout(geoState.zoomTimer);
  geoState.zoomTimer=setTimeout(()=>geoSetZooming(false),200);
}
function initGeoControls(){
  const svg=document.getElementById('geoMap'), wrap=svg.parentElement;
  document.querySelectorAll('#geoRegions .chip').forEach(b=>{
    b.addEventListener('click',()=>geoAnimateTo(geoRegionView(b.dataset.region),b.dataset.region));
  });
  document.getElementById('geoZoomIn').addEventListener('click',()=>{
    if(geoState.anim){cancelAnimationFrame(geoState.anim);geoState.anim=null;}
    const vb=geoState.vb, nw=Math.max(GEO_VB_W/GEO_MAX_ZOOM,vb.w/1.6), nh=nw/GEO_ASPECT;
    geoAnimateTo({x:vb.x+vb.w/2-nw/2,y:vb.y+vb.h/2-nh/2,w:nw,h:nh},null);
  });
  document.getElementById('geoZoomOut').addEventListener('click',()=>{
    if(geoState.anim){cancelAnimationFrame(geoState.anim);geoState.anim=null;}
    const vb=geoState.vb, nw=Math.min(GEO_VB_W,vb.w*1.6), nh=nw/GEO_ASPECT;
    geoAnimateTo({x:vb.x+vb.w/2-nw/2,y:vb.y+vb.h/2-nh/2,w:nw,h:nh},nw>=GEO_VB_W-0.01?'world':null);
  });
  const fb=document.getElementById('geoFraudOnly');
  fb.addEventListener('click',()=>{
    geoState.onlyFraud=!geoState.onlyFraud;
    fb.classList.toggle('on',geoState.onlyFraud);
    geoState.noFlash=true;                       // pas d'éclair de « nouvelle ville » lors du filtrage
    renderGeoMap(geoState.lastData);
    geoState.noFlash=false;
  });
  // molette : zoom centré sur le curseur
  svg.addEventListener('wheel',(ev)=>{
    ev.preventDefault();
    const c=geoContent(svg);
    const u=Math.min(1,Math.max(0,(ev.clientX-c.rect.left-c.offX)/c.w));
    const v=Math.min(1,Math.max(0,(ev.clientY-c.rect.top-c.offY)/c.h));
    geoZoomAt(Math.exp(ev.deltaY*0.0015),u,v);
  },{passive:false});
  // glisser pour déplacer la carte
  svg.addEventListener('pointerdown',(ev)=>{
    if(ev.button!==0||geoState.zoom<=1.01)return;
    if(geoState.anim){cancelAnimationFrame(geoState.anim);geoState.anim=null;geoSetZooming(false);}
    geoState.drag={x:ev.clientX,y:ev.clientY,vx:geoState.vb.x,vy:geoState.vb.y,moved:false};
  });
  window.addEventListener('pointermove',(ev)=>{
    const d=geoState.drag; if(!d)return;
    const dx=ev.clientX-d.x, dy=ev.clientY-d.y;
    if(!d.moved&&Math.hypot(dx,dy)<3)return;
    if(!d.moved){d.moved=true;wrap.classList.add('dragging');geoState.hoverKey=null;document.getElementById('geoTooltip').style.display='none';}
    const s=geoScale(svg);
    geoState.vb=geoClampView({x:d.vx-dx/s,y:d.vy-dy/s,w:geoState.vb.w,h:geoState.vb.h});
    geoState.region=null;
    geoNoTransition();
    geoApplyView();
  });
  const endDrag=()=>{ if(geoState.drag){geoState.drag=null;wrap.classList.remove('dragging');} };
  window.addEventListener('pointerup',endDrag);
  window.addEventListener('pointercancel',endDrag);
  geoSyncControls();
}
function geoShortNum(n){
  n=Math.max(1,Math.round(n));
  if(n>=10000)return Math.round(n/1000)+'k';
  if(n>=1000)return (n/1000).toFixed(1).replace('.0','')+'k';
  return String(n);
}
function geoTooltipHtml(c){
  const hot='color:var(--c4);font-weight:700';
  return `<b>${c.country?flagHtml(c.country):''}${esc(c.city)}${c.state?', '+esc(c.state):''}</b><br>`
    +`Total : ${(c.total||0).toLocaleString('fr-FR')} Trans.<br>`
    +`Fraude : <span style="${hot}">${(c.fraud||0).toLocaleString('fr-FR')}</span><br>`
    +`Taux : <span style="${hot}">${c.rate}%</span>`;
}
function showGeoStatus(title,detail){
  const el=document.getElementById('geoStatus');
  if(!el)return;
  if(!title){el.style.display='none';return;}
  el.textContent='';
  const b=document.createElement('b');b.textContent=title;el.appendChild(b);
  if(detail)el.appendChild(document.createTextNode(detail));
  el.style.display='block';
}
function geoDiagMessage(d){
  if(!d)return ['Aucune ville à afficher pour le moment.',''];
  const n=(v)=>(v||0).toLocaleString('fr-FR');
  if(!d.rows)return ['Aucune transaction dans la base.','La carte se remplira dès que des transactions seront enregistrées.'];
  if(!d.with_city)return ['Aucun lieu de transaction renseigné.',n(d.rows)+' transactions en base, mais aucune ne contient de coordonnées de lieu de transaction.'];
  let m=n(d.rows)+' transactions · '+n(d.with_city)+' avec lieu de transaction · '+n(d.with_coords)+' avec coordonnées GPS.';
  if(d.unlocated&&d.unlocated.length)m+=' Lieux non localisables : '+d.unlocated.join(', ')+'.';
  return ['Aucune ville géolocalisable.',m];
}
function updateGeoLegend(maxTotal,maxFraud,hasData){
  const box=document.getElementById('geoLegendDots');
  if(!box)return;
  box.innerHTML=GEO_LEGEND_FRACTIONS.map(f=>{
    const d=Math.round(geoRadiusPx(f*maxTotal,maxTotal)*2);
    return `<div class="geo-legend-item"><i style="width:${d}px;height:${d}px"></i><em>${hasData?geoShortNum(f*maxTotal):'—'}</em></div>`;
  }).join('');
}
function geoFlash(layer,x,y,radius,color){
  const ns='http://www.w3.org/2000/svg';
  const f=document.createElementNS(ns,'circle');
  f.setAttribute('cx',x);f.setAttribute('cy',y);f.setAttribute('r',radius);
  f.setAttribute('fill','none');f.setAttribute('stroke',color);f.setAttribute('stroke-width','2');
  f.setAttribute('class','geo-flash');
  f.addEventListener('animationend',()=>f.remove());
  layer.appendChild(f);
}
function renderGeoMap(cities,diag){
  const svg=document.getElementById('geoMap');
  const tooltip=document.getElementById('geoTooltip');
  const wrap=svg.parentElement;
  buildWorldMapBase(svg);
  const layer=document.getElementById('geoMarkersLayer');
  const ns='http://www.w3.org/2000/svg';
  const valid=(cities||[]).filter(c=>c.lat!=null&&c.long!=null);
  geoState.lastData=valid;
  if(diag!==undefined)geoState.diag=diag;
  // échelles calculées sur TOUTES les villes : les tailles/couleurs restent stables quand on filtre
  const maxTotal=Math.max(1,...valid.map(c=>c.total||0));
  const maxFraud=Math.max(0,...valid.map(c=>c.fraud||0));
  geoState.maxTotal=maxTotal;geoState.maxFraud=maxFraud;
  updateGeoLegend(maxTotal,maxFraud,valid.length>0);
  const shown=geoState.onlyFraud?valid.filter(c=>(c.fraud||0)>0):valid;
  const scale=geoScale(svg);
  const seen=new Set();

  // gros cercles dessinés en premier (dessous) : les petits restent survolables
  shown.slice().sort((a,b)=>(b.total||0)-(a.total||0)).forEach(c=>{
    const key=c.city+'|'+(c.state||'');
    seen.add(key);
    const [x,y]=geoProject(c.long,c.lat);
    const radius=geoRadiusPx(c.total,maxTotal)/scale;
    const color=geoFraudColor(c.rate);
    const prev=geoState.cities.get(key);
    geoState.cities.set(key,c);

    let m=geoState.markers.get(key);
    if(!m){
      const pulse=document.createElementNS(ns,'circle');
      pulse.setAttribute('fill','none');pulse.setAttribute('stroke-width','1.5');
      pulse.setAttribute('class','geo-marker-pulse');
      const circle=document.createElementNS(ns,'circle');
      circle.setAttribute('fill-opacity','0.55');circle.setAttribute('stroke-width','1.4');
      circle.setAttribute('class','geo-marker');
      circle.addEventListener('mousemove',(ev)=>{
        const cc=geoState.cities.get(key); if(!cc)return;
        geoState.hoverKey=key;
        const r=wrap.getBoundingClientRect();
        tooltip.style.left=(ev.clientX-r.left)+'px';
        tooltip.style.top=(ev.clientY-r.top)+'px';
        tooltip.style.display='block';
        tooltip.innerHTML=geoTooltipHtml(cc);
      });
      circle.addEventListener('mouseleave',()=>{geoState.hoverKey=null;tooltip.style.display='none';});
      layer.appendChild(pulse);layer.appendChild(circle);
      m={pulse,circle};geoState.markers.set(key,m);
      if(!geoState.first&&!geoState.noFlash&&!geoState.anim&&!geoState.drag)geoFlash(layer,x,y,radius,color);   // nouvelle ville apparue en direct
    }else if(prev&&(c.total||0)>(prev.total||0)){
      geoFlash(layer,x,y,radius,color);                       // nouvelles transactions reçues
    }

    // mise à jour sur place (les transitions CSS animent taille et couleur)
    m.circle.setAttribute('cx',x);m.circle.setAttribute('cy',y);
    m.circle.setAttribute('r',radius);m.circle.style.setProperty('r',radius+'px');
    m.circle.setAttribute('fill',color);m.circle.setAttribute('stroke',color);
    m.pulse.setAttribute('cx',x);m.pulse.setAttribute('cy',y);m.pulse.setAttribute('r',radius);
    m.pulse.setAttribute('stroke',color);
    m.pulse.style.display=((c.fraud||0)>0&&(c.rate||0)>=70)?'':'none';   // halo permanent sur les villes au taux de fraude élevé (>= 70 %)
    layer.appendChild(m.pulse);layer.appendChild(m.circle);   // ré-ordonne (gros dessous, petits dessus)
  });

  // villes disparues
  geoState.markers.forEach((m,key)=>{
    if(!seen.has(key)){m.pulse.remove();m.circle.remove();geoState.markers.delete(key);geoState.cities.delete(key);}
  });

  // info-bulle survolée : rafraîchie en direct
  if(geoState.hoverKey&&geoState.cities.has(geoState.hoverKey)&&tooltip.style.display==='block'){
    tooltip.innerHTML=geoTooltipHtml(geoState.cities.get(geoState.hoverKey));
  }
  if(shown.length){showGeoStatus(null);}
  else if(valid.length&&geoState.onlyFraud){showGeoStatus('Aucune fraude détectée.','Désactivez « Uniquement les fraudes » pour revoir toutes les villes.');}
  else{const [t,m]=geoDiagMessage(geoState.diag);showGeoStatus(t,m);}
  geoState.first=false;
}
async function refreshGeo(){
  if(geoState.busy||document.hidden)return;
  geoState.busy=true;
  try{
    const g=await api('/stats/geo');
    renderGeoMap((g&&g.cities)||[],g&&g.diagnostic);
  }catch(e){
    console.error(e);
    if(!geoState.lastData.length)showGeoStatus('Impossible de charger la carte.',String(e&&e.message||e));
  }finally{
    geoState.busy=false;
  }
}
function openGeoStream(){
  if(!window.EventSource)return;
  const es=new EventSource('/stats/geo/stream');
  geoState.es=es;
  es.addEventListener('open',()=>{geoState.streamOk=true;});
  es.addEventListener('geo',(ev)=>{
    geoState.streamOk=true;
    try{const g=JSON.parse(ev.data);renderGeoMap((g&&g.cities)||[],g&&g.diagnostic);}
    catch(e){console.error('SSE geo :',e);}
  });
  es.addEventListener('ping',()=>{geoState.streamOk=true;});
  es.addEventListener('geo_error',(ev)=>{
    let m=ev.data;try{m=JSON.parse(ev.data);}catch(e){}
    console.error('SSE geo_error :',m);
    if(!geoState.lastData.length)showGeoStatus('Erreur côté serveur (carte).',String(m));
  });
  es.addEventListener('unauthorized',()=>{es.close();window.location.href='/login';});
  es.onerror=()=>{
    geoState.streamOk=false;                            // le repli par interrogation prend le relais
    if(es.readyState===2){setTimeout(openGeoStream,10000);}   // flux fermé (ex. 401/503) : nouvelle tentative plus tard
  };
}
function startGeoRealtime(){
  initGeoControls();                                     // régions, filtre fraudes, zoom molette / glisser
  refreshGeo();                                        // premier affichage immédiat
  openGeoStream();                                       // flux poussé par le serveur
  setInterval(()=>{if(!geoState.streamOk)refreshGeo();},GEO_POLL_MS);   // repli si le flux est coupé
  document.addEventListener('visibilitychange',()=>{if(!document.hidden&&!geoState.streamOk)refreshGeo();});
  // recalcul des rayons (en pixels écran) si la carte est redimensionnée
  const svg=document.getElementById('geoMap');
  if(window.ResizeObserver){
    let t=null;
    new ResizeObserver(()=>{clearTimeout(t);t=setTimeout(()=>{if(geoState.lastData.length)renderGeoMap(geoState.lastData);},120);}).observe(svg.parentElement);
  }
}

function renderFraudCategories(categories){
  const el=document.getElementById('catList');
  if(!categories.length){
    el.innerHTML='<div class="empty">Aucune transaction signalée pour le moment</div>';
    return;
  }
  const palette=['var(--c4)','var(--c3)','var(--c1)','var(--c5)','#8bacc8'];
  el.innerHTML=categories.map((c,i)=>{
    const color=palette[i%palette.length];
    return `<div class="cat-row">
      <span class="cat-name" title="${esc(catDisplay(c.category))}">${esc(catDisplay(c.category))}</span>
      <div class="cat-bar-track"><div class="cat-bar-fill" style="width:${c.pct}%;background:${color};color:${color}"></div></div>
      <span class="cat-pct">${c.pct}%</span>
    </div>`;
  }).join('');
}

function renderTrendChart(daily){
  const days=(daily.days||[]).slice(-7);
  const n=days.length;
  const block=(daily.block||[]).slice(-7);
  const rev=(daily.manual_review||[]).slice(-7);
  const allow=(daily.allow||[]).slice(-7);
  const volume=days.map((_,i)=>(block[i]||0)+(rev[i]||0)+(allow[i]||0));
  const highRisk=days.map((_,i)=>(block[i]||0)+(rev[i]||0));
  const labels=days.map(d=>{try{return new Date(d).toLocaleDateString('fr-FR',{day:'2-digit',month:'2-digit'})}catch{return d}});
  if(cTrend)cTrend.destroy();
  cTrend=new Chart(document.getElementById('cTrend').getContext('2d'),{
    type:'line',
    data:{
      labels:labels.length?labels:['—'],
      datasets:[
        {
          type:'line',
          label:'Transactions autorisées',
          data:allow.length?allow:[0],
          borderColor:'#180fcb',backgroundColor:'rgba(24,15,203,.12)',
          borderWidth:2,pointRadius:3,pointHoverRadius:5,pointBackgroundColor:'#180fcb',
          pointBorderColor:'#03060f',pointBorderWidth:1,tension:.3,fill:true,
          order:3,
        },
        {
          type:'line',
          label:'Volume de transactions',
          data:volume.length?volume:[0],
          borderColor:'#00d4ff',backgroundColor:'rgba(0,212,255,.12)',
          borderWidth:2,pointRadius:3,pointHoverRadius:5,pointBackgroundColor:'#00d4ff',
          pointBorderColor:'#03060f',pointBorderWidth:1,tension:.3,fill:true,
          order:1,
        },
        {
          type:'line',
          label:'High risque (Bloquées + Revue)',
          data:highRisk.length?highRisk:[0],
          borderColor:'#ff4d6d',backgroundColor:'rgba(255,77,109,.12)',
          borderWidth:2,pointRadius:3,pointHoverRadius:5,pointBackgroundColor:'#ff4d6d',
          pointBorderColor:'#03060f',pointBorderWidth:1,tension:.3,fill:true,
          order:2,
        }
      ]
    },
    options:{
      responsive:true,maintainAspectRatio:false,
      animation:(fsId==='cTrend')?false:{},  // recréé pendant le plein écran : pas d'animation d'entrée
      interaction:{intersect:false,mode:'index'},
      plugins:{
        legend:{position:'bottom',labels:{color:'#8bacc8',font:{size:10},boxWidth:8,padding:8,usePointStyle:true,pointStyle:'circle'}},
        tooltip:{backgroundColor:'rgba(9,15,31,.95)',borderColor:'rgba(0,212,255,.2)',borderWidth:1,titleColor:'#00d4ff',bodyColor:'#8bacc8',padding:12}
      },
      scales:{
        y:{beginAtZero:true,ticks:{precision:0,color:'#4a6a88',font:{size:10}},grid:{color:'rgba(0,212,255,.05)'}},
        x:{grid:{display:false},ticks:{color:'#4a6a88',font:{size:10}}}
      }
    }
  });
}

// ─── TABLE ───
function applyFilters(){
  const q=document.getElementById('srch').value.toLowerCase();
  filtered=all.filter(a=>{
    if(srcF!=='all'&&a.channel!==srcF)return false;
    if(decF&&a.decision!==decF)return false;
    if(q){const s=(a.transaction_id+' '+a.merchant+' '+a.cc_num_masked+' '+a.merch_city+' '+a.city+' '+a.decision).toLowerCase();if(!s.includes(q))return false}
    return true;
  });
  page=1;renderTable();
}

function renderTable(){
  const st=(page-1)*PS,en=st+PS,pg=filtered.slice(st,en);
  const tb=document.getElementById('tbody');
  if(!pg.length){
    tb.innerHTML='<tr><td colspan="8"><div class="empty">Aucune transaction correspondante</div></td></tr>';
    document.getElementById('pInfo').textContent='0 résultats';
    document.getElementById('pBtns').innerHTML='';
    return;
  }
  tb.innerHTML=pg.map(a=>{
    const sc=Math.round((a.fraud_probability||0)*100);
    const sc_col=scColor(sc);
    const rowCls=a.decision==='BLOCK'?'row-esc':a.decision==='MANUAL_REVIEW'?'row-rev':a.decision==='ALLOW'?'row-cls':'';
    const amt=a.amt!=null?Number(a.amt).toFixed(2)+' €':'—';
    const amtCls=(a.amt||0)>500?'danger':(a.amt||0)>200?'warn':'';
    const tid=esc(a.transaction_id);
    return `<tr class="${rowCls}" onclick="openModal('${tid}')">
      <td><span class="idwrap"><span class="cell-id" title="${tid}">${esc(shortId(a.transaction_id))}</span><button class="copy-btn" title="Copier l'ID" onclick="copyId(event,'${tid}')">⧉</button></span></td>
      <td><span class="cell-amt ${amtCls}">${amt}</span></td>
      <td><span class="cell-rule" title="${esc(a.merchant)} — ${esc(a.category)}">${esc(a.merchant||'—')} <span style="color:var(--txt3)">· ${catDisplay(a.category)}</span></span></td>
      <td><span class="cell-ip">${esc(a.cc_num_masked||'—')}</span></td>
      <td><span class="cell-city" title="Ville de naissance du titulaire : ${esc(a.city||'—')}">${a.merch_city?flagHtml(a.merch_country):''}${esc(a.merch_city||'—')}</span></td>
      <td><div class="sbar"><div class="strack"><div class="sfill" style="width:${sc}%;background:${sc_col};color:${sc_col}"></div></div><span class="slabel" style="color:${sc_col}">${sc}</span></div></td>
      <td>${dBadge(a.decision)}</td>
      <td><span class="cell-date">${fmt(a.processed_at)}</span></td>
     </tr>`;
  }).join('');

  const tot=Math.ceil(filtered.length/PS);
  document.getElementById('pInfo').textContent=`${filtered.length.toLocaleString('fr-FR')} transactions · Page ${page}/${tot}`;
  let ph='';
  ph+=`<button class="pbtn" ${page<=1?'disabled':''} onclick="goP(${page-1})">&#9664;</button>`;
  let sp=Math.max(1,page-2),ep=Math.min(tot,sp+4);
  for(let i=sp;i<=ep;i++)ph+=`<button class="pbtn ${i===page?'on':''}" onclick="goP(${i})">${i}</button>`;
  ph+=`<button class="pbtn" ${page>=tot?'disabled':''} onclick="goP(${page+1})">&#9654;</button>`;
  document.getElementById('pBtns').innerHTML=ph;
}
function goP(p){page=p;renderTable();document.querySelector('.table-wrap').scrollIntoView({behavior:'smooth'})}

function chBadge(s){
  if(s==='pos')return '<span class="badge badge-fs">Point de vente</span>';
  if(s==='online')return '<span class="badge badge-qr">En ligne</span>';
  return '<span class="badge">'+esc(s)+'</span>';
}
// Statut Analyste réel (feedback_history), distinct de la Décision automatique du modèle.
function analystBadge(status){
  if(status==='TP')return '<span class="vbadge vb-tp">Fraude confirmée</span>';
  if(status==='FP')return '<span class="vbadge vb-fp">Légitime</span>';
  return '<span class="vbadge vb-pend">En attente</span>';
}
function dBadge(d){
  if(d==='BLOCK')return '<span class="dbadge db-esc">Bloquée</span>';
  if(d==='MANUAL_REVIEW')return '<span class="dbadge db-rev">Revue</span>';
  if(d==='ALLOW')return '<span class="dbadge db-cls">Autorisée</span>';
  return '<span class="dbadge db-unk">'+(d||'N/A')+'</span>';
}


// ─── MODAL ───
async function openModal(tid){
  document.getElementById('ov').classList.add('open');
  document.getElementById('mBody').innerHTML='<div style="text-align:center;padding:30px"><span class="spin"></span></div>';
  try{
    const a=await api('/debug/scored-transactions/'+encodeURIComponent(tid));
    const score=Math.round((a.fraud_probability||0)*100),anomaly=a.anomaly_score||0,dec=(a.decision||'unknown');
    const sc_col=scColor(score);
    const exp=a.llm_explanation||'Aucune analyse disponible.';

    let bm=null;
    try{ bm=await api('/debug/behavioral-metrics/'+encodeURIComponent(tid)); }catch(e){ bm=null; }

    let related=[];
    try{ related=await api('/debug/related-transactions/'+encodeURIComponent(a.cc_num||'')+'?exclude='+encodeURIComponent(tid)); }catch(e){ related=[]; }

    let decTxt='';
    if(dec==='BLOCK')decTxt='BLOQUÉE';
    else if(dec==='MANUAL_REVIEW')decTxt='REVUE MANUELLE';
    else if(dec==='ALLOW')decTxt='AUTORISÉE';
    else decTxt='Scoring indisponible';

    const actions = a.recommended_actions || [];
    let actHtml='';
    if(actions.length){
      actHtml=`<div class="actions-box">
        <div class="actions-title">⚡ Actions recommandées (IA)</div>
        <ul class="actions-list">${actions.map(x=>'<li>'+esc(x)+'</li>').join('')}</ul>
      </div>`;
    }

    document.getElementById('mBody').innerHTML=`
      <div class="hero">
        <div class="score-ring" style="--ring-color:${sc_col}">${score}</div>
        <div class="hero-info">
          <div class="hero-decision" style="color:${sc_col}">${esc(decTxt)}</div>
          ${actHtml}
        </div>
        <button class="m-close" style="align-self:flex-start;flex-shrink:0" onclick="closeAlertModal()" title="Fermer">&#x2715;</button>
      </div>

      <div class="explain">
        <div class="explain-title">⚡ Analyse Fraude (IA)</div>
        <div class="explain-text">${esc(exp)}</div>
      </div>

      <div class="igrid">
        <div class="ipanel pc1">
          <div class="ptitle">Informations Transaction</div>
          ${ir('Marchand',a.merchant)}${ir('Catégorie',catDisplay(a.category))}
          ${ir('Montant',a.amt!=null?a.amt.toFixed(2)+' €':null,(a.amt||0)>500?'warn':'')}
          ${ir('Carte',a.cc_num_masked,'hi')}${ir('Titulaire',a.cardholder_name)}
          ${ir('Lieu de transaction',a.merch_city)}${ir('Ville de naissance',a.city)}${ir('État/Région',a.state)}
          ${ir('Date transaction',a.trans_date_trans_time)}
        </div>
        <div class="ipanel pc2">
          <div class="ptitle">Métriques Comportementales</div>
          ${bm&&Array.isArray(bm.rows)?bm.rows.filter(r=>!/feature|score de risque|facteurs? principa|verdict/i.test((r.key||'')+' '+(r.label||''))).map(r=>ir(r.label,(r.category!==undefined?catDisplay(r.category)+' '+r.suffix:r.value),r.level)).join(''):ir('Métriques comportementales','Indisponibles (erreur de calcul)','warn')}
        </div>
      </div>

      <button class="jtoggle" onclick="this.nextElementSibling.classList.toggle('open')">{ } Afficher le JSON complet</button>
      <div class="jviewer">${esc(JSON.stringify(buildCleanJson(a,bm,exp),null,2))}</div>

      <!-- Historique récent de cette carte ── -->
      <div style="margin-top:20px;">
        <h4 style="color:#00fff2;border-bottom:1px solid rgba(0,255,242,0.2);padding-bottom:5px;">
          📜 Historique récent de cette carte
        </h4>
        ${(()=>{
          if(!related.length)return '<p style="color:#666;text-align:center;padding:10px;">Aucune autre transaction associée à cette carte.</p>';
          const thStyle='padding:8px 10px;border:1px solid rgba(0,255,242,0.2);white-space:nowrap;font-weight:700;font-size:.78em;letter-spacing:.04em;';
          let rows='';
          related.forEach(t=>{
            const merch = esc(t.merchant||'N/A');
            const cat   = esc(t.category||'-');
            const amt   = t.amt!=null ? t.amt.toFixed(2)+' €' : '-';
            const dec2  = esc(t.decision||'-');
            const decColor = t.decision==='BLOCK'?'#ff4d6d':t.decision==='MANUAL_REVIEW'?'#ffaa00':'#00ff9d';
            const dt    = esc(t.trans_date_trans_time||'N/A');
            rows += `<tr style="border-bottom:1px solid rgba(255,255,255,0.04);transition:background .15s;" onmouseover="this.style.background='rgba(0,255,242,0.04)'" onmouseout="this.style.background=''">
              <td style="padding:8px 10px;color:#00d4ff;font-family:'Calisto MT',Georgia,'Times New Roman',serif;font-size:.75em;white-space:nowrap;">${esc(t.transaction_id)}</td>
              <td style="padding:8px 10px;color:#fff;font-weight:600;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${merch}">${merch}</td>
              <td style="padding:8px 10px;color:#8bacc8;font-size:.78em;max-width:140px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${cat}">${cat}</td>
              <td style="padding:8px 10px;color:#ffaa00;font-family:'Calisto MT',Georgia,'Times New Roman',serif;font-size:.75em;white-space:nowrap;">${amt}</td>
              <td style="padding:8px 10px;font-size:.82em;white-space:nowrap;color:${decColor};font-weight:700;">${dec2}</td>
              <td style="padding:8px 10px;color:#4a6a88;font-family:'Calisto MT',Georgia,'Times New Roman',serif;font-size:.73em;white-space:nowrap;">${dt}</td>
              </tr>`;
          });
          return `<div style="overflow-x:auto;max-height:420px;overflow-y:auto;border:1px solid rgba(0,255,242,0.12);border-radius:8px;">
            <table style="width:100%;border-collapse:collapse;font-size:0.85em;text-align:left;min-width:800px;">
              <thead style="position:sticky;top:0;z-index:2;">
                <tr style="background:rgba(0,20,40,0.98);color:#00fff2;">
                  <th style="${thStyle}">ID Transaction</th>
                  <th style="${thStyle}">Marchand</th>
                  <th style="${thStyle}">Catégorie</th>
                  <th style="${thStyle}">Montant</th>
                  <th style="${thStyle}">Décision</th>
                  <th style="${thStyle}">Date</th>
                  </tr>
              </thead>
              <tbody>${rows}</tbody>
            </table>
          </div>`;
        })()}
      </div>
    `;
  }catch(e){
    document.getElementById('mBody').innerHTML=`<div class="empty" style="color:var(--c4)">Erreur : ${esc(e.message)}</div>`;
  }
}

// ─── JSON unique et propre : transaction + analyse IA, sans null / vide / champs inutiles ───
const JSON_DROP_KEYS=new Set(['state','city_pop','job','foreign_transaction','content_hash','analyst_status','llm_explanation','recommended_actions','top_factors','cc_num_masked','model_version',
  'is_weekend','is_night','txn_count_1h','new_merchant','fraud_probability','anomaly_score','decision','age']);
function pruneJson(v){
  if(Array.isArray(v)){
    const arr=v.map(pruneJson).filter(x=>x!==undefined);
    return arr.length?arr:undefined;
  }
  if(v&&typeof v==='object'){
    const out={};
    Object.keys(v).forEach(k=>{const p=pruneJson(v[k]);if(p!==undefined)out[k]=p;});
    return Object.keys(out).length?out:undefined;
  }
  if(v===null||v===undefined||v==='')return undefined;
  if(typeof v==='number'&&!Number.isFinite(v))return undefined;
  return v;
}
function buildCleanJson(a,bm,exp){
  const tx={};
  Object.keys(a||{}).forEach(k=>{if(!JSON_DROP_KEYS.has(k))tx[k]=a[k];});
  const factors=(a.top_factors||[]).map(f=>({
    critere:f.label||f.feature, valeur:f.display, effet:f.direction==='up'?'augmente le risque':'réduit le risque'
  }));
  const full={
    transaction:tx,
    analyse_ia:{explication:exp, facteurs:factors, actions_recommandees:a.recommended_actions}
  };
  return pruneJson(full)||{};
}

function ir(k,v,cls){
  if(v===undefined||v===null||v==='')return '';
  const c=cls?' '+cls:'';
  return `<div class="irow"><span class="ikey">${esc(k)}</span><span class="ival${c}">${esc(String(v))}</span></div>`;
}

// ─── WIRING ───
document.querySelectorAll('.chip[data-f]').forEach(b=>{
  b.addEventListener('click',function(){
    document.querySelectorAll('.chip[data-f]').forEach(x=>x.classList.remove('on'));
    this.classList.add('on');
    srcF=this.dataset.f;decF=null;
    document.querySelectorAll('.chip[data-d]').forEach(x=>x.classList.remove('on'));
    applyFilters();
  });
});
document.querySelectorAll('.chip[data-d]').forEach(b=>{
  b.addEventListener('click',function(){
    const was=this.classList.contains('on');
    document.querySelectorAll('.chip[data-d]').forEach(x=>x.classList.remove('on'));
    if(!was){this.classList.add('on');decF=this.dataset.d}else{decF=null}
    applyFilters();
  });
});
document.getElementById('srch').addEventListener('input',applyFilters);
function closeAlertModal(){
  document.getElementById('ov').classList.remove('open');
}

document.getElementById('ov').addEventListener('click',e=>{if(e.target===document.getElementById('ov'))closeAlertModal()});

document.addEventListener('keydown',e=>{
  if(e.key==='Escape'){ if(fsBox){closeChartFullscreen();} else {closeAlertModal();} }
});


// ─── GRAPHIQUES EN PLEIN ÉCRAN (clic sur « Distribution des Risk Scores » / « Tendance temporelle ») ───
// Un clic ouvre le graphique sur tout l'écran ; l'échelle s'adapte (polices et points plus grands,
// graduations plus fines, marge en haut, valeurs affichées, plages de scores sélectionnables).
// Fermeture : bouton « Fermer », touche Échap.
let fsBox=null, fsHome=null, fsId=null;

function niceStepMax(maxVal){
  // plafond « propre » de l'axe Y avec une marge de ~18 %
  const m=Math.max(1,maxVal)*1.18;
  const mag=Math.pow(10,Math.floor(Math.log10(m)));
  const step=m<=10?1:(m<=20?2:(m<=50?5:(m/mag<2?mag/5:mag/2)));
  return {max:Math.ceil(m/step)*step, step};
}

function setRiskChartMode(full){
  const ch=cRisk; if(!ch)return;
  const ds=ch.data.datasets[0];
  const crit=(ch.options.plugins.blinkPoint||{}).index;
  const counts=ds.data;
  ch._fsScale=full?1.7:1;
  ds.pointRadius=counts.map((_,i)=>i===crit?(full?7:5):(full?(counts[i]>0?5:3):2.5));
  ds.pointHoverRadius=full?9:6;
  ds.borderWidth=full?3:2;
  const x=ch.options.scales.x, y=ch.options.scales.y;
  if(full){
    const lo=0, hi=Math.max(100,counts.length-1);   // échelle complète 0–100
    const visible=counts.slice(lo,hi+1);
    const {max,step}=niceStepMax(Math.max(1,...visible));
    x.min=lo; x.max=hi;
    x.grid={display:true,color:'rgba(0,212,255,.05)'};
    x.ticks={color:'#8bacc8',font:{size:13},autoSkip:true,maxTicksLimit:(hi-lo)<=30?(hi-lo+1):21,maxRotation:0};
    x.title={display:true,text:'Risk Score',color:'#8bacc8',font:{size:14,weight:'600'}};
    y.min=0; y.max=max; y.suggestedMax=undefined;
    y.ticks={precision:0,stepSize:step,color:'#8bacc8',font:{size:13}};
    y.title={display:true,text:'Nombre de transactions',color:'#8bacc8',font:{size:14,weight:'600'}};
    ch.options.plugins.tooltip.titleFont={size:14}; ch.options.plugins.tooltip.bodyFont={size:13};
  }else{
    x.min=undefined; x.max=undefined;
    x.grid={display:false};
    x.ticks={color:'#4a6a88',font:{size:9},autoSkip:true,maxTicksLimit:11,maxRotation:0};
    x.title={display:true,text:'Risk Score',color:'#4a6a88',font:{size:10}};
    y.min=undefined; y.max=undefined; y.suggestedMax=Math.max(4,...counts);
    y.ticks={precision:0,stepSize:undefined,color:'#4a6a88',font:{size:10}};
    y.title={display:true,text:'Nombre de transactions',color:'#4a6a88',font:{size:10}};
    ch.options.plugins.tooltip.titleFont=undefined; ch.options.plugins.tooltip.bodyFont=undefined;
  }
  ch.update();
}

function setTrendChartMode(full){
  const ch=cTrend; if(!ch)return;
  let maxV=1;
  ch.data.datasets.forEach(d=>{
    d.pointRadius=full?6:3; d.pointHoverRadius=full?9:5; d.borderWidth=full?3:2;
    (d.data||[]).forEach(v=>{if(v>maxV)maxV=v;});
  });
  const x=ch.options.scales.x, y=ch.options.scales.y, lg=ch.options.plugins.legend.labels;
  if(full){
    const {max,step}=niceStepMax(maxV);
    y.min=0; y.max=max; y.ticks={precision:0,stepSize:step,color:'#8bacc8',font:{size:13}};
    y.grid={color:'rgba(0,212,255,.07)'};
    y.title={display:true,text:'Nombre de transactions',color:'#8bacc8',font:{size:14,weight:'600'}};
    x.grid={display:true,color:'rgba(0,212,255,.05)'};
    x.offset=true;
    x.ticks={color:'#8bacc8',font:{size:13}};
    x.title={display:true,text:'Jour',color:'#8bacc8',font:{size:14,weight:'600'}};
    lg.font={size:13}; lg.padding=18; lg.boxWidth=9; lg.boxHeight=9;
    ch.options.plugins.tooltip.titleFont={size:14}; ch.options.plugins.tooltip.bodyFont={size:13};
  }else{
    y.min=undefined; y.max=undefined; y.ticks={precision:0,color:'#4a6a88',font:{size:10}};
    y.grid={color:'rgba(0,212,255,.05)'}; y.title={display:false};
    x.offset=false; x.grid={display:false}; x.ticks={color:'#4a6a88',font:{size:10}}; x.title={display:false};
    lg.font={size:10}; lg.padding=8; lg.boxWidth=8; lg.boxHeight=undefined;
    ch.options.plugins.tooltip.titleFont=undefined; ch.options.plugins.tooltip.bodyFont=undefined;
  }
  ch.update();
}

function applyFsMode(id,full){
  const ch=id==='cRisk'?cRisk:cTrend; if(!ch)return;
  ch.options.animation=false;                 // changement d'échelle immédiat (pas d'état intermédiaire)
  if(id==='cRisk')setRiskChartMode(full);
  else setTrendChartMode(full);
  ch.options.animation=(fsId===id)?false:{};  // plein écran : pas d'animation ; sinon animation normale
}

function openChartFullscreen(box){
  if(fsBox)return;
  fsBox=box; fsId=box.dataset.fs;
  fsHome=document.createElement('div');                    // repère pour remettre la boîte à sa place
  fsHome.style.display='none';
  box.parentNode.insertBefore(fsHome,box);
  document.body.appendChild(box);                          // hors de la grille : vrai plein écran
  box.classList.add('chart-full');
  document.body.classList.add('chart-full-open');
  const close=document.createElement('button');
  close.type='button'; close.className='chart-full-close';
  close.innerHTML='✕';
  close.setAttribute('aria-label','Fermer');
  close.addEventListener('click',e=>{e.stopPropagation();closeChartFullscreen();});
  box.appendChild(close);
  applyFsMode(fsId,true);
  requestAnimationFrame(()=>{const ch=fsId==='cRisk'?cRisk:cTrend; if(ch)ch.resize();});
}

function closeChartFullscreen(){
  if(!fsBox)return;
  const box=fsBox, id=fsId;
  box.querySelectorAll('.chart-full-close').forEach(n=>n.remove());
  box.classList.remove('chart-full');
  document.body.classList.remove('chart-full-open');
  if(fsHome&&fsHome.parentNode){fsHome.parentNode.insertBefore(box,fsHome);fsHome.remove();}
  fsBox=null; fsHome=null; fsId=null;
  applyFsMode(id,false);
  const chc=id==='cRisk'?cRisk:cTrend; if(chc)chc.options.animation={};
  requestAnimationFrame(()=>{const ch=id==='cRisk'?cRisk:cTrend; if(ch)ch.resize();});
}

// Les graphiques sont recréés à chaque rafraîchissement : si l'un est ouvert, on réapplique le mode plein écran.
function reapplyChartFullscreen(){
  if(!fsBox)return;
  applyFsMode(fsId,true);
  const ch=fsId==='cRisk'?cRisk:cTrend; if(ch)ch.resize();
}

document.querySelectorAll('.chart-box[data-fs]').forEach(box=>{
  box.addEventListener('click',()=>{ if(!box.classList.contains('chart-full'))openChartFullscreen(box); });
  box.addEventListener('keydown',e=>{
    if((e.key==='Enter'||e.key===' ')&&!box.classList.contains('chart-full')){e.preventDefault();openChartFullscreen(box);}
  });
});

// ─── DÉCONNEXION ───
document.getElementById('btnLogout').addEventListener('click', async () => {
  try{ await fetch('/api/auth/logout', {method:'POST'}); }catch(e){}
  window.location.href = '/login';
});

// ─── CLOCK ───
function tick(){document.getElementById('clk').textContent=new Date().toLocaleString('fr-FR',{day:'2-digit',month:'2-digit',year:'numeric',hour:'2-digit',minute:'2-digit',second:'2-digit'}).replace(',','')}
tick();setInterval(tick,1000);

// ─── INIT ───
async function refresh(){
  try{await Promise.all([loadData(),loadCharts()]);}catch(e){console.error(e)}
}
refresh();
setInterval(refresh,30000);
startGeoRealtime();   // carte géographique : flux temps réel SSE (repli : interrogation toutes les 3 s)
</script>
</body>
</html>'''


# ============================================================================
# Page de login — authentification 100% CÔTÉ SERVEUR.
# ============================================================================
AUTH_HTML = r'''<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FraudGuard ML // Access — Authentification</title>
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Cdefs%3E%3ClinearGradient id='g' x1='0' y1='0' x2='1' y2='1'%3E%3Cstop offset='0' stop-color='%23ffb52e'/%3E%3Cstop offset='1' stop-color='%23e67e00'/%3E%3C/linearGradient%3E%3C/defs%3E%3Cpath d='M16 2 L28 6.5 V15 C28 22.5 22.8 27.8 16 30 C9.2 27.8 4 22.5 4 15 V6.5 Z' fill='url%28%23g%29'/%3E%3Ccircle cx='14.5' cy='14' r='5' fill='none' stroke='%230c1220' stroke-width='2.6'/%3E%3Cpath d='M18.3 17.8 L23 22.5' stroke='%230c1220' stroke-width='3' stroke-linecap='round'/%3E%3C/svg%3E">
<style>
:root{
  --c1:#ffb52e;   /* ambre  */
  --c2:#e67e00;   /* orange */
  --c3:#00ff9d;   /* vert (succès) */
  --c4:#ff4d6d;   /* rouge  (erreur) */
  --c1d:rgba(255,181,46,.12);
  --c2d:rgba(230,126,0,.12);
  --c3d:rgba(0,255,157,.10);
  --c4d:rgba(255,77,109,.12);
  --bg0:#070b14;
  --bg1:#0c1220;
  --bg2:#080d18;
  --border:rgba(214,150,30,.38);
  --txt1:#f5f7fb;
  --txt2:#a3adc2;
  --txt3:#6b768c;
  --radius:12px;
  --radius2:22px;
  --glow1:0 0 18px rgba(255,170,30,.22);
  --glow3:0 0 18px rgba(0,255,157,.22);
  --glow4:0 0 18px rgba(255,77,109,.25);
}
*{box-sizing:border-box;margin:0;padding:0;}
html,body{height:100%;}
body{
  min-height:100vh;
  background:var(--bg0);
  color:var(--txt1);
  font-family:'Calisto MT',Georgia,'Times New Roman',serif;
  display:flex;
  align-items:center;
  justify-content:center;
  overflow-x:hidden;
  position:relative;
}
body::before{
  content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
  background:
    radial-gradient(circle at 14% 22%, rgba(190,120,35,.30), transparent 38%),
    radial-gradient(circle at 88% 78%, rgba(30,85,210,.30), transparent 42%);
}
button,input{font-family:inherit;}


.wrap{
  position:relative;z-index:1;
  width:100%;
  max-width:960px;
  display:grid;
  grid-template-columns:1.05fr 1fr;
  background:linear-gradient(180deg, rgba(10,17,32,.92) 0%, rgba(6,10,20,.92) 100%);
  border:1px solid rgba(214,150,30,.55);
  border-radius:var(--radius2);
  box-shadow:0 0 60px rgba(0,0,0,.55), 0 0 22px rgba(255,170,30,.10);
  overflow:hidden;
  margin:24px;
}

.side{
  padding:44px 40px;
  background:linear-gradient(160deg, rgba(255,170,30,.06) 0%, transparent 55%);
  border-right:1px solid var(--border);
  display:flex;
  flex-direction:column;
  justify-content:space-between;
  position:relative;
}
.brand{display:flex;align-items:center;gap:14px;}
.logo-slot{
  width:56px;height:56px;flex-shrink:0;border-radius:14px;
  background:linear-gradient(135deg,var(--c2),var(--c1));
  display:flex;align-items:center;justify-content:center;
  overflow:hidden;position:relative;font-weight:900;color:#fff;font-size:1.2rem;
  box-shadow:0 0 18px rgba(255,170,30,.25);
}
.brand-txt h1{
  font-size:1.4rem;letter-spacing:.08em;
  background:linear-gradient(90deg,var(--c1),#ffd98a);
  -webkit-background-clip:text;background-clip:text;color:transparent;
  font-weight:800;
}
.brand-txt p{font-size:.68rem;color:#ffffff;letter-spacing:.08em;text-transform:uppercase;margin-top:3px;}

.pitch{margin-top:38px;}
.pitch h2{
  font-size:1.5rem;line-height:1.35;font-weight:700;color:var(--txt1);
  margin-bottom:14px;
}
.pitch h2 span{color:var(--c1);}

.status-line{
  display:flex;align-items:center;gap:8px;
  font-family:'Calisto MT',Georgia,'Times New Roman',serif;font-size:.74rem;color:var(--c1);
  letter-spacing:.05em;margin-top:36px;
}
.live-dot{
  width:7px;height:7px;border-radius:50%;background:var(--c1);
  box-shadow:0 0 8px var(--c1),0 0 16px var(--c1);
  animation:pulse 1.8s ease-in-out infinite;
}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}



.panel{
  padding:44px 40px;
  display:flex;
  flex-direction:column;
  background:var(--bg1);
}
.tabs{
  display:flex;gap:4px;background:var(--bg2);border:1px solid var(--border);
  border-radius:40px;padding:4px;margin-bottom:30px;
}
.tab{
  flex:1;text-align:center;padding:9px 0;border-radius:36px;cursor:pointer;
  font-size:.78rem;font-weight:700;letter-spacing:.03em;color:var(--txt2);
  transition:all .2s ease;user-select:none;
}
.tab.on{
  background:linear-gradient(90deg,var(--c2),var(--c1));
  color:#fff;box-shadow:var(--glow1);
}

.form{display:none;flex-direction:column;gap:16px;animation:fadeIn .25s ease;}
.form.on{display:flex;}
@keyframes fadeIn{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:translateY(0)}}

.field{display:flex;flex-direction:column;gap:6px;}
.field label{
  font-size:.68rem;letter-spacing:.07em;text-transform:uppercase;color:#ffffff;font-weight:700;
}
.field .input-wrap{
  position:relative;display:flex;align-items:center;
}
.field input{
  width:100%;background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius);
  padding:11px 40px 11px 14px;color:var(--txt1);font-size:.85rem;outline:none;
  font-family:inherit;transition:border-color .2s ease, box-shadow .2s ease;
}
.field input::placeholder{color:var(--txt3);}
.field input:focus{border-color:var(--c1);box-shadow:var(--glow1);}
.field input.err{border-color:var(--c4);box-shadow:var(--glow4);}
.field .toggle-pass{
  position:absolute;right:13px;cursor:pointer;pointer-events:all;
  font-size:.68rem;color:var(--c1);letter-spacing:.03em;font-weight:700;background:none;border:none;
}
.hint{font-size:.68rem;color:var(--txt3);margin-top:-4px;}
.hint.bad{color:var(--c4);}
.hint.good{color:var(--c3);}

.strength{height:3px;border-radius:2px;background:rgba(255,255,255,.06);overflow:hidden;margin-top:2px;}
.strength i{display:block;height:100%;width:0%;background:var(--c4);transition:all .25s ease;}

.link{color:var(--c1);cursor:pointer;text-decoration:none;font-weight:600;}
.link:hover{text-decoration:underline;}

.btn{
  margin-top:6px;padding:12px 0;border:none;border-radius:var(--radius);
  background:linear-gradient(90deg,var(--c2),var(--c1));
  color:#fff;font-weight:800;text-shadow:0 1px 2px rgba(0,0,0,.25);font-size:.85rem;letter-spacing:.03em;cursor:pointer;
  box-shadow:var(--glow1);transition:transform .15s ease, box-shadow .15s ease;
  display:flex;align-items:center;justify-content:center;gap:8px;
}
.btn:hover{transform:translateY(-1px);box-shadow:0 0 26px rgba(255,170,30,.4);}
.btn:active{transform:translateY(0);}
.btn:disabled{opacity:.5;cursor:not-allowed;transform:none;}
.btn .spinner{
  width:14px;height:14px;border-radius:50%;border:2px solid rgba(255,255,255,.4);
  border-top-color:#fff;animation:spin .6s linear infinite;display:none;
}
.btn.loading .spinner{display:inline-block;}
.btn.loading .btn-label{opacity:.7;}
@keyframes spin{to{transform:rotate(360deg)}}

.banner{
  display:none;align-items:flex-start;gap:10px;padding:11px 13px;border-radius:var(--radius);
  font-size:.76rem;line-height:1.5;margin-bottom:4px;
}
.banner.show{display:flex;}
.banner.error{background:var(--c4d);color:var(--c4);border:1px solid rgba(255,77,109,.3);}
.banner.success{background:var(--c3d);color:var(--c3);border:1px solid rgba(0,255,157,.3);}

.switch-line{text-align:center;font-size:.78rem;color:var(--txt2);margin-top:10px;}

.mobile-brand{
  display:none;
  align-items:center;gap:12px;
  margin-bottom:26px;padding-bottom:22px;
  border-bottom:1px solid var(--border);
}
.mobile-brand .logo-slot{width:44px;height:44px;border-radius:11px;font-size:.9rem;}
.mobile-brand h1{
  font-size:1.15rem;letter-spacing:.08em;
  background:linear-gradient(90deg,var(--c1),#ffd98a);
  -webkit-background-clip:text;background-clip:text;color:transparent;
  font-weight:800;
}
.mobile-brand p{font-size:.62rem;color:#ffffff;letter-spacing:.08em;text-transform:uppercase;margin-top:2px;}

@media (max-width:820px){
  .wrap{grid-template-columns:1fr;}
  .side{display:none;}
  .panel{padding:36px 26px;}
  .mobile-brand{display:flex;}
}
</style>
</head>
<body>

<div class="wrap">

  <!-- ─────────── LEFT : branding / status ─────────── -->
  <div class="side">
    <div>
      <div class="brand">
        <div class="logo-slot">FG</div>
        <div class="brand-txt">
          <h1>FraudGuard ML</h1>
          <p>Real-Time Fraud Detection</p>
        </div>
      </div>

      <div class="pitch">
        <h2>Le portail d'accès sécurisé de <span>FraudGuard ML.</span></h2>
      </div>
    </div>

    <div>
      <div class="status-line"><span class="live-dot"></span> MOTEUR DE SCORING ACTIF</div>
    </div>
  </div>

  <!-- ─────────── RIGHT : formulaires ─────────── -->
  <div class="panel">

    <div class="mobile-brand">
      <div class="logo-slot">FG</div>
      <div>
        <h1>FraudGuard ML</h1>
        <p>Real-Time Fraud Detection</p>
      </div>
    </div>

    <div class="tabs">
      <div class="tab on" data-tab="signin">Connexion</div>
      <div class="tab" data-tab="signup">Inscription</div>
    </div>

    <div id="banner" class="banner"></div>

    <!-- ─── SIGN IN ─── -->
    <form class="form on" id="formSignin" autocomplete="off">
      <div class="field">
        <label for="siIdentifiant">Identifiant</label>
        <div class="input-wrap">
          <input type="text" id="siIdentifiant" placeholder="j.dupont" required>
        </div>
      </div>
      <div class="field">
        <label for="siPass">Mot de passe</label>
        <div class="input-wrap">
          <input type="password" id="siPass" placeholder="••••••••" required>
          <button type="button" class="toggle-pass" data-target="siPass">AFFICHER</button>
        </div>
      </div>
      <button type="submit" class="btn" id="btnSignin">
        <span class="spinner"></span><span class="btn-label">Se connecter</span>
      </button>
      <div class="switch-line">Pas encore de compte ? <span class="link" data-tab="signup">Créer un accès</span></div>
    </form>

    <!-- ─── SIGN UP ─── -->
    <form class="form" id="formSignup" autocomplete="off">
      <div class="field">
        <label for="suName">Nom d'utilisateur</label>
        <div class="input-wrap">
          <input type="text" id="suName" placeholder="j.dupont" required minlength="3">
        </div>
      </div>
      <div class="field">
        <label for="suPass">Mot de passe</label>
        <div class="input-wrap">
          <input type="password" id="suPass" placeholder="8 caractères minimum" required minlength="8">
          <button type="button" class="toggle-pass" data-target="suPass">AFFICHER</button>
        </div>
        <div class="strength"><i id="strengthBar"></i></div>
        <div class="hint" id="strengthHint"></div>
      </div>
      <div class="field">
        <label for="suPass2">Confirmer le mot de passe</label>
        <div class="input-wrap">
          <input type="password" id="suPass2" placeholder="••••••••" required>
          <button type="button" class="toggle-pass" data-target="suPass2">AFFICHER</button>
        </div>
      </div>
      <button type="submit" class="btn" id="btnSignup">
        <span class="spinner"></span><span class="btn-label">Créer mon compte</span>
      </button>
      <div class="switch-line">Déjà inscrit ? <span class="link" data-tab="signin">Se connecter</span></div>
    </form>

  </div>
</div>

<script>
const DASHBOARD_URL = "/dashboard";

function showBanner(msg, type){
  const b = document.getElementById("banner");
  b.textContent = msg;
  b.className = "banner show " + type;
}
function hideBanner(){
  document.getElementById("banner").className = "banner";
}
function setLoading(btn, loading){
  btn.disabled = loading;
  btn.classList.toggle("loading", loading);
}

(async () => {
  try{
    const r = await fetch('/api/auth/me', { credentials: 'same-origin' });
    if(r.ok){ window.location.href = DASHBOARD_URL; }
  }catch(e){ /* pas de session : on reste sur le formulaire */ }
})();

function switchTab(target){
  document.querySelectorAll(".tab").forEach(t => t.classList.toggle("on", t.dataset.tab === target));
  document.getElementById("formSignin").classList.toggle("on", target === "signin");
  document.getElementById("formSignup").classList.toggle("on", target === "signup");
  hideBanner();
}
document.querySelectorAll("[data-tab]").forEach(el=>{
  el.addEventListener("click", () => switchTab(el.dataset.tab));
});

document.querySelectorAll(".toggle-pass").forEach(btn=>{
  btn.addEventListener("click", ()=>{
    const input = document.getElementById(btn.dataset.target);
    const show = input.type === "password";
    input.type = show ? "text" : "password";
    btn.textContent = show ? "MASQUER" : "AFFICHER";
  });
});

const suPass = document.getElementById("suPass");
const strengthBar = document.getElementById("strengthBar");
const strengthHint = document.getElementById("strengthHint");
suPass.addEventListener("input", ()=>{
  const v = suPass.value;
  let score = 0;
  if(v.length >= 8) score++;
  if(/[A-Z]/.test(v) && /[a-z]/.test(v)) score++;
  if(/\d/.test(v)) score++;
  if(/[^A-Za-z0-9]/.test(v)) score++;
  const pct = [0, 25, 55, 80, 100][score];
  const colors = ["#ff4d6d","#ff4d6d","#ffaa00","#ffd166","#00ff9d"];
  strengthBar.style.width = pct + "%";
  strengthBar.style.background = colors[score];
  const labels = ["Trop court","Faible","Moyen","Bon","Excellent"];
  strengthHint.textContent = v ? "Robustesse : " + labels[score] : "";
  strengthHint.className = "hint " + (score >= 3 ? "good" : (v ? "bad" : ""));
});

document.getElementById("formSignup").addEventListener("submit", async (e)=>{
  e.preventDefault();
  hideBanner();

  const name  = document.getElementById("suName").value.trim();
  const pass  = document.getElementById("suPass").value;
  const pass2 = document.getElementById("suPass2").value;
  const btn   = document.getElementById("btnSignup");

  if(name.length < 3){
    showBanner("Le nom d'utilisateur doit contenir au moins 3 caractères.", "error");
    return;
  }
  if(pass.length < 8){
    showBanner("Le mot de passe doit contenir au moins 8 caractères.", "error");
    return;
  }
  if(pass !== pass2){
    showBanner("Les mots de passe ne correspondent pas.", "error");
    return;
  }

  setLoading(btn, true);
  try{
    const resp = await fetch('/api/auth/signup', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      credentials: 'same-origin',
      body: JSON.stringify({ username: name, password: pass })
    });
    const result = await resp.json().catch(()=>({}));
    setLoading(btn, false);

    if(!resp.ok){
      showBanner(result.detail || "Erreur lors de la création du compte.", "error");
      return;
    }

    showBanner("Compte créé avec succès. Vous pouvez maintenant vous connecter.", "success");
    document.getElementById("formSignup").reset();
    strengthBar.style.width = "0%";
    strengthHint.textContent = "";
    strengthHint.className = "hint";

    setTimeout(()=>{
      switchTab("signin");
      document.getElementById("siIdentifiant").value = name;
      showBanner("Compte prêt — connectez-vous pour accéder au dashboard.", "success");
    }, 900);
  }catch(err){
    setLoading(btn, false);
    showBanner("Erreur de communication avec le serveur.", "error");
  }
});

document.getElementById("formSignin").addEventListener("submit", async (e)=>{
  e.preventDefault();
  hideBanner();

  const identity = document.getElementById("siIdentifiant").value.trim();
  const pass     = document.getElementById("siPass").value;
  const btn      = document.getElementById("btnSignin");

  if(!identity || !pass){
    showBanner("Merci de renseigner votre identifiant et votre mot de passe.", "error");
    return;
  }

  setLoading(btn, true);
  try{
    const resp = await fetch('/api/auth/login', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      credentials: 'same-origin',
      body: JSON.stringify({ username: identity, password: pass })
    });
    const result = await resp.json().catch(()=>({}));

    if(!resp.ok){
      setLoading(btn, false);
      showBanner(result.detail || "Authentification échouée.", "error");
      return;
    }

    showBanner("Authentification réussie — redirection vers le dashboard…", "success");
    setTimeout(() => { window.location.href = DASHBOARD_URL; }, 500);
  }catch(err){
    setLoading(btn, false);
    showBanner("Erreur de communication avec le serveur.", "error");
  }
});
</script>
</body>
</html>
'''

# ============================================================================
# AUTHENTIFICATION SERVEUR — sessions HttpOnly + base SQLite dédiée
# (logique générique, inchangée par rapport au dashboard SOC d'origine —
#  ne dépend d'aucun concept métier SIEM/fraude)
#
# TOUTES les écritures ci-dessous (création de compte, session, tentative de
# connexion) sont horodatées en UTC : datetime.now(timezone.utc) pour les
# colonnes texte ISO, time.time() (UTC epoch) pour les colonnes numériques
# (created_at/expires_at des sessions, fenêtre de rate-limit). Volontairement
# inchangé par rapport à la version précédente.
# ============================================================================
import hashlib
import hmac
import secrets
from typing import Tuple

from fastapi import Depends, Response, Cookie
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

SESSION_COOKIE_NAME = os.getenv("DASHBOARD_SESSION_COOKIE", "fraudguard_dashboard_session")
SESSION_TTL_SECONDS = int(os.getenv("DASHBOARD_SESSION_TTL_SECONDS", str(8 * 3600)))  # 8h par défaut

COOKIE_SECURE = os.getenv("DASHBOARD_COOKIE_SECURE", "false").strip().lower() in ("1", "true", "yes")
if not COOKIE_SECURE:
    logger.warning(
        "DASHBOARD_COOKIE_SECURE=false — le cookie de session sera envoyé même en HTTP. "
        "En production, exposez ce service en HTTPS et positionnez DASHBOARD_COOKIE_SECURE=true."
    )

# Base SQLite dédiée à l'authentification du dashboard — distincte de fraud.db.
# Ce service est le SEUL écrivain de cette base-ci.
AUTH_DB_PATH = Path(os.getenv("DASHBOARD_AUTH_DB", str(DATA_DIR / "dashboard_auth.db"))).resolve()

PBKDF2_ITERATIONS = 200_000

MAX_LOGIN_ATTEMPTS = 6
LOGIN_ATTEMPT_WINDOW_SECONDS = 600  # 10 minutes
_failed_login_attempts: Dict[str, List[float]] = {}

def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"

def _rate_limit_key(ip: str, username: str) -> str:
    return f"{ip}::{username}"

def _check_login_rate_limit(ip: str, username: str) -> None:
    key = _rate_limit_key(ip, username)
    now = time.time()
    attempts = [t for t in _failed_login_attempts.get(key, []) if now - t < LOGIN_ATTEMPT_WINDOW_SECONDS]
    _failed_login_attempts[key] = attempts
    if len(attempts) >= MAX_LOGIN_ATTEMPTS:
        raise HTTPException(
            status_code=429,
            detail="Trop de tentatives de connexion échouées. Réessayez dans quelques minutes.",
        )

def _register_failed_login(ip: str, username: str) -> None:
    key = _rate_limit_key(ip, username)
    _failed_login_attempts.setdefault(key, []).append(time.time())

def _reset_login_rate_limit(ip: str, username: str) -> None:
    _failed_login_attempts.pop(_rate_limit_key(ip, username), None)

def _auth_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(AUTH_DB_PATH), timeout=5)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn

def init_auth_db() -> None:
    AUTH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = _auth_connect()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dashboard_users (
                username TEXT PRIMARY KEY,
                salt_hex TEXT NOT NULL,
                password_hash_hex TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dashboard_sessions (
                token TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_username ON dashboard_sessions(username)")
        conn.commit()
    finally:
        conn.close()
    logger.info(f"Base d'authentification du dashboard prête — {AUTH_DB_PATH}")

def _hash_password(password: str, salt: Optional[bytes] = None) -> Tuple[str, str]:
    if salt is None:
        salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return salt.hex(), digest.hex()

def create_user(username: str, password: str) -> bool:
    """Retourne False si le nom d'utilisateur existe déjà."""
    conn = _auth_connect()
    try:
        existing = conn.execute(
            "SELECT 1 FROM dashboard_users WHERE username=?", (username,)
        ).fetchone()
        if existing:
            return False
        salt_hex, hash_hex = _hash_password(password)
        conn.execute(
            "INSERT INTO dashboard_users (username, salt_hex, password_hash_hex, created_at) VALUES (?,?,?,?)",
            (username, salt_hex, hash_hex, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        return True
    finally:
        conn.close()

def verify_user_credentials(username: str, password: str) -> bool:
    conn = _auth_connect()
    try:
        row = conn.execute(
            "SELECT salt_hex, password_hash_hex FROM dashboard_users WHERE username=?", (username,)
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return False
    salt_hex, expected_hex = row
    try:
        _, computed_hex = _hash_password(password, bytes.fromhex(salt_hex))
    except ValueError:
        return False
    return hmac.compare_digest(computed_hex, expected_hex)

def create_session(username: str) -> str:
    token = secrets.token_urlsafe(32)
    now = time.time()
    conn = _auth_connect()
    try:
        conn.execute(
            "INSERT INTO dashboard_sessions (token, username, created_at, expires_at) VALUES (?,?,?,?)",
            (token, username, now, now + SESSION_TTL_SECONDS),
        )
        conn.commit()
    finally:
        conn.close()
    return token

def get_session_username(token: str) -> Optional[str]:
    conn = _auth_connect()
    try:
        row = conn.execute(
            "SELECT username, expires_at FROM dashboard_sessions WHERE token=?", (token,)
        ).fetchone()
        if not row:
            return None
        username, expires_at = row
        if time.time() > expires_at:
            conn.execute("DELETE FROM dashboard_sessions WHERE token=?", (token,))
            conn.commit()
            return None
        return username
    finally:
        conn.close()

def delete_session(token: str) -> None:
    conn = _auth_connect()
    try:
        conn.execute("DELETE FROM dashboard_sessions WHERE token=?", (token,))
        conn.commit()
    finally:
        conn.close()

def cleanup_expired_sessions() -> None:
    conn = _auth_connect()
    try:
        conn.execute("DELETE FROM dashboard_sessions WHERE expires_at < ?", (time.time(),))
        conn.commit()
    finally:
        conn.close()

def get_current_username(request: Request) -> Optional[str]:
    """Lit le cookie de session et retourne le username si la session est valide, sinon None."""
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return None
    return get_session_username(token)

async def require_user_api(request: Request) -> str:
    """Dépendance pour les routes API/JSON : lève 401 si non authentifié."""
    username = await asyncio.to_thread(get_current_username, request)
    if not username:
        raise HTTPException(status_code=401, detail="Authentification requise. Merci de vous reconnecter.")
    return username

class AuthCredentials(BaseModel):
    username: str
    password: str

# ============================================================================
# Application FastAPI — service dashboard (lecture seule sur fraud.db)
# ============================================================================
app = FastAPI(title="FraudGuard ML Dashboard Service (read-only)")

_http_client: Optional[httpx.AsyncClient] = None

@app.on_event("startup")
async def _startup():
    global _http_client
    _http_client = httpx.AsyncClient(timeout=30.0)
    await asyncio.to_thread(init_auth_db)
    await asyncio.to_thread(cleanup_expired_sessions)
    logger.info(f"FraudGuard dashboard démarré sur le port {DASHBOARD_PORT} — DB={DB_PATH} (mode=ro)")

@app.on_event("shutdown")
async def _shutdown():
    if _http_client:
        await _http_client.aclose()

# ── Pages HTML (routes PUBLIQUES : login) ───────────────────────────────────
@app.get("/login", response_class=HTMLResponse)
async def login_page():
    return HTMLResponse(content=AUTH_HTML)

@app.get("/logout")
async def logout_redirect(request: Request, response: Response):
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if token:
        await asyncio.to_thread(delete_session, token)
    redirect = RedirectResponse(url="/login", status_code=303)
    redirect.delete_cookie(SESSION_COOKIE_NAME, path="/")
    return redirect

# ── Page du dashboard (PROTÉGÉE : redirige vers /login si non authentifié) ──
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    username = await asyncio.to_thread(get_current_username, request)
    if not username:
        return RedirectResponse(url="/login", status_code=303)
    return HTMLResponse(content=DASHBOARD_HTML)

@app.get("/")
async def root():
    return {
        "message": "FraudGuard ML Dashboard Service (lecture seule) — sert le dashboard fraude et lit fraud.db en mode=ro",
        "db_path": str(DB_PATH),
        "db_exists": DB_PATH.exists(),
        "agent_api_url": AGENT_API_URL,
        "dashboard": "/dashboard",
        "login": "/login",
        "docs": "/docs",
    }

@app.get("/health")
async def health():
    try:
        count = get_failed_transaction_count()
        return {"status": "healthy", "db_reachable": True, "failed_transactions": count}
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "db_reachable": False, "error": str(e)})

# ── Endpoints d'authentification (PUBLICS) ──────────────────────────────────
@app.post("/api/auth/signup")
async def api_signup(payload: AuthCredentials):
    username = payload.username.strip().lower()
    password = payload.password

    if len(username) < 3:
        raise HTTPException(status_code=400, detail="Le nom d'utilisateur doit contenir au moins 3 caractères.")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Le mot de passe doit contenir au moins 8 caractères.")

    created = await asyncio.to_thread(create_user, username, password)
    if not created:
        raise HTTPException(status_code=409, detail="Ce nom d'utilisateur existe déjà.")
    logger.info(f"Nouveau compte dashboard créé : {username}")
    return {"status": "success"}

@app.post("/api/auth/login")
async def api_login(payload: AuthCredentials, request: Request, response: Response):
    ip = _client_ip(request)
    username = payload.username.strip().lower()

    await asyncio.to_thread(_check_login_rate_limit, ip, username)

    valid = await asyncio.to_thread(verify_user_credentials, username, payload.password)
    if not valid:
        await asyncio.to_thread(_register_failed_login, ip, username)
        raise HTTPException(status_code=401, detail="Identifiant ou mot de passe incorrect.")

    await asyncio.to_thread(_reset_login_rate_limit, ip, username)
    token = await asyncio.to_thread(create_session, username)

    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
        max_age=SESSION_TTL_SECONDS,
        path="/",
    )
    logger.info(f"Connexion réussie : {username} (ip={ip})")
    return {"status": "success", "username": username}

@app.post("/api/auth/logout")
async def api_logout(request: Request, response: Response):
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if token:
        await asyncio.to_thread(delete_session, token)
    response.delete_cookie(SESSION_COOKIE_NAME, path="/")
    return {"status": "success"}

@app.get("/api/auth/me")
async def api_me(request: Request):
    username = await asyncio.to_thread(get_current_username, request)
    if not username:
        raise HTTPException(status_code=401, detail="Non authentifié")
    return {"username": username}

# ── Endpoints /debug/* (lecture seule — PROTÉGÉS) ───────────────────────────
@app.get("/debug/scored-transactions")
async def debug_scored(limit: int = 20, channel: str = None, hours: int = None, user: str = Depends(require_user_api)):
    return await asyncio.to_thread(get_scored_transactions, limit, channel, hours)

@app.get("/debug/scored-transactions/{transaction_id}")
async def debug_scored_one(transaction_id: str, user: str = Depends(require_user_api)):
    result = await asyncio.to_thread(get_scored_transaction, transaction_id)
    if not result:
        raise HTTPException(404, "Not found")
    return result

@app.get("/debug/related-transactions/{cc_num}")
async def debug_related(cc_num: str, exclude: str = None, limit: int = 20, user: str = Depends(require_user_api)):
    return await asyncio.to_thread(get_related_transactions, cc_num, exclude, limit)

@app.get("/debug/search-transactions")
async def search_transactions(transaction_id: str = None, channel: str = None, cardholder_name: str = None, limit: int = 100, user: str = Depends(require_user_api)):
    return await asyncio.to_thread(search_scored_transactions, transaction_id, channel, cardholder_name, limit)

@app.get("/debug/card-verdict/{cc_num}")
async def card_verdict(cc_num: str, user: str = Depends(require_user_api)):
    return await asyncio.to_thread(get_card_verdict, cc_num)

@app.get("/debug/behavioral-metrics/{transaction_id}")
async def behavioral_metrics(transaction_id: str, user: str = Depends(require_user_api)):
    result = await asyncio.to_thread(compute_behavioral_metrics, transaction_id)
    if result is None:
        raise HTTPException(404, "Not found")
    return _bm_strip_hidden(result)

@app.get("/debug/failed-transactions")
async def debug_failed(limit: int = 50, user: str = Depends(require_user_api)):
    return await asyncio.to_thread(get_recent_failed_transactions, limit)

@app.get("/debug/failed-transactions/search")
async def search_failed(transaction_id: str = None, channel: str = None, reason: str = None, limit: int = 50, user: str = Depends(require_user_api)):
    return await asyncio.to_thread(search_failed_transactions, transaction_id, channel, reason, limit)

@app.get("/debug/failed-transactions/{failed_id}")
async def get_failed_transaction(failed_id: int, user: str = Depends(require_user_api)):
    def _g():
        conn = _ro_connect()
        try:
            cur = conn.execute("SELECT id, transaction_id, channel, raw_payload_json, reason, attempts, replay_count, created_at, dead_at, fingerprint FROM failed_transactions WHERE id=?", (failed_id,))
            row = cur.fetchone()
        finally:
            conn.close()
        if not row:
            return None
        return {"id": row[0], "transaction_id": row[1], "channel": row[2], "raw_payload": json.loads(row[3]), "reason": row[4],
                "attempts": row[5], "replay_count": row[6], "created_at": row[7], "dead_at": row[8], "fingerprint": row[9]}
    result = await asyncio.to_thread(_g)
    if not result:
        raise HTTPException(404, "Not found")
    return result

# ── Endpoints /stats/* (lecture seule — PROTÉGÉS) ───────────────────────────
@app.get("/stats/daily-decisions")
async def daily_decision_stats(user: str = Depends(require_user_api)):
    return await asyncio.to_thread(get_daily_decision_stats)

@app.get("/stats/channels")
async def channel_stats(user: str = Depends(require_user_api)):
    return await asyncio.to_thread(get_channel_distribution)

@app.get("/stats/daily")
async def daily_stats(user: str = Depends(require_user_api)):
    return await asyncio.to_thread(get_daily_stats)

@app.get("/stats/categories")
async def category_stats(user: str = Depends(require_user_api)):
    return await asyncio.to_thread(get_category_distribution)

@app.get("/stats/decisions")
async def decision_stats(user: str = Depends(require_user_api)):
    return await asyncio.to_thread(get_decision_distribution)

@app.get("/stats/decisions-by-channel")
async def decision_stats_by_channel(user: str = Depends(require_user_api)):
    return await asyncio.to_thread(get_decision_distribution_by_channel)

@app.get("/stats/fraud-scores")
async def fraud_score_stats(user: str = Depends(require_user_api)):
    return await asyncio.to_thread(get_fraud_score_distribution)

@app.get("/stats/fraud-categories")
async def fraud_categories_stats(limit: int = 5, user: str = Depends(require_user_api)):
    return await asyncio.to_thread(get_top_fraud_categories, limit)

@app.get("/stats/kpis")
async def kpi_stats(window_days: int = 1, user: str = Depends(require_user_api)):
    return await asyncio.to_thread(get_kpi_metrics, window_days)

# ── Carte géographique temps réel ────────────────────────────────────────────
# Le service dashboard est en lecture seule : les transactions sont écrites par
# l'agent de scoring dans fraud.db. On détecte donc un changement à moindre coût
# via la signature (mtime + taille) de fraud.db et de son journal WAL, puis on
# ne rejoue le GROUP BY que si la base a réellement changé. Le snapshot est
# partagé entre tous les clients connectés (JSON classique ET flux SSE).
_GEO_CACHE: Dict[str, Any] = {"ts": 0.0, "sig": None, "data": None}
_GEO_CACHE_LOCK = threading.Lock()
_GEO_MIN_INTERVAL = 0.5      # jamais plus d'un recalcul toutes les 0,5 s
_GEO_MAX_AGE = 30.0          # recalcul forcé au moins toutes les 30 s
_GEO_STREAM_TICK = 1.0       # fréquence de vérification du flux SSE
_GEO_STREAM_FORCE = 10.0     # recalcul de contrôle même sans changement détecté
_GEO_STREAM_PING = 5.0       # ping SSE (garde la connexion vivante + heure « live »)
_GEO_AUTH_RECHECK = 60.0     # revalidation de la session sur un flux ouvert

def _db_signature() -> tuple:
    sig = []
    for suffix in ("", "-wal"):
        try:
            st = os.stat(f"{DB_PATH}{suffix}")
            sig.append((st.st_mtime_ns, st.st_size))
        except OSError:
            sig.append(None)
    return tuple(sig)

def _geo_snapshot() -> Dict[str, Any]:
    with _GEO_CACHE_LOCK:
        now = time.time()
        sig = _db_signature()
        age = now - _GEO_CACHE["ts"]
        if _GEO_CACHE["data"] is not None and (
            age < _GEO_MIN_INTERVAL or (_GEO_CACHE["sig"] == sig and age < _GEO_MAX_AGE)
        ):
            return _GEO_CACHE["data"]
        data = get_geo_distribution()
        _GEO_CACHE.update(ts=time.time(), sig=sig, data=data)
        return data

@app.get("/stats/geo")
async def geo_stats(user: str = Depends(require_user_api)):
    return await asyncio.to_thread(_geo_snapshot)

@app.get("/stats/geo/stream")
async def geo_stream(request: Request, user: str = Depends(require_user_api)):
    """Flux Server-Sent Events de la carte : le serveur POUSSE un événement
    « geo » dès que la base change (transaction ajoutée ou décision mise à jour),
    plus un « ping » toutes les 5 s. Le navigateur (EventSource) se reconnecte
    tout seul en cas de coupure."""
    async def event_gen():
        last_sig = None
        last_payload = None
        last_compute = 0.0
        last_ping = 0.0
        last_auth = time.time()
        last_error = None
        yield "retry: 3000\n\n"
        while True:
            if await request.is_disconnected():
                break
            now = time.time()
            try:
                if now - last_auth >= _GEO_AUTH_RECHECK:
                    last_auth = now
                    if not await asyncio.to_thread(get_current_username, request):
                        yield "event: unauthorized\ndata: {}\n\n"
                        break
                sig = _db_signature()
                if sig != last_sig or (now - last_compute) >= _GEO_STREAM_FORCE:
                    data = await asyncio.to_thread(_geo_snapshot)
                    last_sig, last_compute = sig, now
                    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
                    if payload != last_payload:
                        last_payload = payload
                        yield f"event: geo\ndata: {payload}\n\n"
                if now - last_ping >= _GEO_STREAM_PING:
                    last_ping = now
                    yield f"event: ping\ndata: {int(now * 1000)}\n\n"
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"Flux SSE /stats/geo/stream : {e}")
                msg = json.dumps(str(e)[:200], ensure_ascii=False)
                if msg != last_error:
                    last_error = msg
                    yield f"event: geo_error\ndata: {msg}\n\n"
                last_sig = None          # on retentera au prochain tour
            await asyncio.sleep(_GEO_STREAM_TICK)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )

@app.get("/dashboard/transactions")
async def dashboard_transactions(channel: str = None, limit: int = 200, user: str = Depends(require_user_api)):
    return await asyncio.to_thread(get_transactions_for_dashboard, channel, limit)

# ============================================================================
# Proxy d'écriture — SEUL point du service qui déclenche une écriture (sur la
# base fraud.db), et il ne le fait jamais lui-même : il transmet tel quel le
# feedback HITL (clic Fraude confirmée / Transaction légitime) à l'agent de
# scoring, seul propriétaire de cette base et seul à ré-entraîner les modèles.
# Le dashboard ne devient jamais un second écrivain de fraud.db.
# ============================================================================
@app.post("/api/v1/feedback")
async def proxy_feedback(request: Request, user: str = Depends(require_user_api)):
    body = await request.body()
    # On ajoute l'analyste connecté au corps (colonne feedback_history.analyst).
    try:
        payload = json.loads(body or b"{}")
    except ValueError:
        raise HTTPException(status_code=400, detail="Corps JSON invalide")
    if isinstance(payload, dict):
        payload["analyst"] = str(user)
        body = json.dumps(payload).encode("utf-8")
    try:
        resp = await _http_client.post(
            f"{AGENT_API_URL}/api/v1/feedback",
            content=body,
            headers={"Content-Type": "application/json"},
        )
    except httpx.RequestError as e:
        logger.error(f"Proxy vers l'agent de scoring ({AGENT_API_URL}) indisponible : {e}")
        raise HTTPException(
            status_code=502,
            detail=f"Agent de scoring injoignable à {AGENT_API_URL} — le feedback n'a pas été enregistré.",
        )
    return JSONResponse(status_code=resp.status_code, content=resp.json())

# ============================================================================
# Main
# ============================================================================
async def main():
    config = uvicorn.Config(app, host="0.0.0.0", port=DASHBOARD_PORT, log_level="info", access_log=False)
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Arrêt demandé")