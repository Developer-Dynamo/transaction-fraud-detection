#!/usr/bin/env python3
"""
Rejoue History.csv à travers l'API de scoring (API_Score1.py) AVANT de
démarrer le flux en direct (ingestion1.py), afin que les premières vraies
transactions de chaque carte bénéficient déjà d'un historique pour le
calcul des features de vélocité (nb_tx_1h/24h, amt_zscore_card,
merchant_freq, is_new_merchant, ...).

Chaque ligne est envoyée avec "is_history": true : API_Score1.py (v2, avec
la colonne is_history) l'utilise pour calculer les features ET la stocke
dans transactions_scored, mais le dashboard (app.py) doit filtrer sur
is_history = 0 et ne l'affichera donc jamais.

Prérequis : API_Score1.py doit être démarré (uvicorn, port 8001 par défaut)
AVANT de lancer ce script, car /score a besoin du modèle chargé et écrit
dans la même base SQLite que le serveur.

MODÈLE DE DONNÉES (identique à ingestion.py) :
  - chaque carte (cc_num) a UNE ville de naissance/domicile fixe
    (city/lat/long identiques sur toutes ses lignes) ;
  - le LIEU DE TRANSACTION (merch_lat/merch_long) est aléatoire pour toutes
    les villes du monde (world_cities.py, y compris les villes de naissance),
    sans lien avec le client.
Le script VÉRIFIE ces deux règles avant de rejouer et s'arrête si le domicile
d'une carte varie. Si le fichier date de l'ancien modèle (lieux de transaction
concentrés sur quelques marchands), utiliser --regenerate (réécrit le CSV) ou
--randomize-locations (corrige seulement à la volée pendant le rejeu).

Usage :
    python seed_history.py                     # rejoue History.csv
    python seed_history.py --regenerate         # réécrit History.csv (lieux de
                                                  # transaction aléatoires
                                                  # monde entier) puis quitte
    python seed_history.py --purge              # purge d'abord les anciennes
                                                  # lignes de History.csv déjà
                                                  # présentes dans fraud.db
                                                  # (pollution due à un rejeu
                                                  # fait AVANT ce correctif)
    python seed_history.py --csv autre.csv
"""

import argparse
import csv
import os
import sqlite3
import sys
from pathlib import Path

import random

import requests

from world_cities import (
    WORLD_CITIES, nearest_world_city, random_transaction_location, with_extra_cities,
)

CSV_PATH_DEFAULT = "History.csv"
SCORE_API_URL = os.getenv("FRAUD_SCORE_API_URL", "http://localhost:8001/score")
SCORE_API_TIMEOUT = 10

# Même résolution de chemin que DB_PATH dans API_Score1.py, pour que
# --purge cible exactement la même base que celle utilisée par le serveur.
DATA_DIR = Path(os.getenv("FRAUD_DATA_DIR", "data")).resolve()
DB_PATH = DATA_DIR / "fraud.db"

# Champs bruts attendus par TransactionInput (API_Score1.py). "unix_time"
# n'en fait pas partie (journal local uniquement, comme dans ingestion1.py).
TRANSACTION_INPUT_FIELDS = [
    "transaction_id", "cc_num", "merchant", "category", "amt", "age",
    "lat", "long", "merch_lat", "merch_long",
    "trans_date_trans_time", "channel", "city",
]


def load_history_rows(csv_path: str) -> list[dict]:
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # Ordre chronologique STRICT : les fenêtres de vélocité (1h/6h/24h/7d)
    # calculées côté serveur dépendent de l'ordre d'arrivée des transactions
    # par carte, pas seulement de leur horodatage individuel.
    rows.sort(key=lambda r: (r.get("unix_time") or r.get("trans_date_trans_time") or ""))
    return rows


def to_payload(row: dict) -> dict:
    payload = {k: row[k] for k in TRANSACTION_INPUT_FIELDS if k in row and row[k] != ""}
    payload["amt"] = float(payload["amt"])
    payload["age"] = float(payload["age"])
    for k in ("lat", "long", "merch_lat", "merch_long"):
        payload[k] = float(payload[k])
    payload["is_history"] = True
    return payload


def check_data_model(rows: list[dict]) -> None:
    """Vérifie que History.csv respecte le modèle : domicile FIXE par carte
    (sinon on s'arrête : dist_home_merch_km et l'heure locale seraient faux) et
    lieux de transaction variés (sinon simple avertissement)."""
    homes: dict[str, set] = {}
    for r in rows:
        homes.setdefault(r["cc_num"], set()).add((r["city"], r["lat"], r["long"]))
    bad = {cc: h for cc, h in homes.items() if len(h) > 1}
    if bad:
        cc, h = next(iter(bad.items()))
        sys.exit(f"[ERREUR] {len(bad)} carte(s) avec plusieurs domiciles (ex. cc_num={cc}: "
                 f"{sorted(h)[:3]}). Une carte doit avoir UNE ville de naissance fixe.")
    pool = with_extra_cities([(c, float(la), float(lo)) for c, la, lo in
                              {next(iter(h)) for h in homes.values()}])
    tx_cities = set()
    for r in rows:
        near = nearest_world_city(float(r["merch_lat"]), float(r["merch_long"]), 60.0, pool)
        tx_cities.add(near[0] if near else (round(float(r["merch_lat"])), round(float(r["merch_long"]))))
    print(f"[contrôle] {len(homes)} cartes, domicile fixe OK ; "
          f"{len(tx_cities)} lieux de transaction distincts sur {len(rows)} lignes "
          f"(pool possible : {len(pool)} villes).")
    if len(tx_cities) < min(30, len(pool) // 2):
        print("[AVERTISSEMENT] Lieux de transaction peu variés (ancien modèle ?). "
              "Utiliser --regenerate ou --randomize-locations.")


def randomize_locations(rows: list[dict]) -> None:
    """Remplace merch_lat/merch_long de chaque ligne par une ville du monde tirée
    au hasard (villes de naissance des clients incluses), indépendamment du client."""
    homes = {(r["city"], float(r["lat"]), float(r["long"])) for r in rows}
    pool = with_extra_cities(list(homes))
    for r in rows:
        _, r["merch_lat"], r["merch_long"] = random_transaction_location(pool)


def regenerate_csv(csv_path: str) -> None:
    """Réécrit History.csv avec des lieux de transaction aléatoires (sauvegarde
    .bak). Tout le reste (cartes, domicile, dates, montants) est conservé."""
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    backup = csv_path + ".bak"
    os.replace(csv_path, backup)
    randomize_locations(rows)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[regenerate] {csv_path} réécrit ({len(rows)} lignes) ; ancien fichier -> {backup}. "
          f"Pensez à relancer avec --purge pour retirer les anciennes lignes de fraud.db.")


def purge_history_rows(csv_path: str) -> None:
    """Supprime de fraud.db les lignes dont le transaction_id vient de
    History.csv — utile si l'historique a été rejoué AVANT que la colonne
    is_history n'existe (cas de pollution déjà constatée dans le
    dashboard)."""
    if not DB_PATH.exists():
        print(f"[purge] {DB_PATH} n'existe pas encore, rien à purger.")
        return

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        ids = [r["transaction_id"] for r in csv.DictReader(f)]

    conn = sqlite3.connect(DB_PATH, timeout=10)
    try:
        with conn:
            cur = conn.executemany(
                "DELETE FROM transactions_scored WHERE transaction_id = ?",
                [(i,) for i in ids],
            )
        print(f"[purge] {cur.rowcount if cur.rowcount is not None else len(ids)} lignes "
              f"potentiellement supprimées de transactions_scored (transaction_id ∈ History.csv).")
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=CSV_PATH_DEFAULT, help="Chemin vers History.csv")
    parser.add_argument("--purge", action="store_true",
                         help="Supprime d'abord de fraud.db les lignes déjà rejouées "
                              "AVANT le correctif is_history (nettoyage ponctuel).")
    parser.add_argument("--regenerate", action="store_true",
                         help="Réécrit History.csv avec des lieux de transaction aléatoires "
                              "(monde entier) puis quitte, sans rejouer.")
    parser.add_argument("--randomize-locations", action="store_true",
                         help="Tire des lieux de transaction aléatoires À LA VOLÉE pendant le "
                              "rejeu, sans modifier le fichier.")
    parser.add_argument("--seed", type=int, default=None, help="Graine aléatoire (reproductibilité).")
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    if not os.path.isfile(args.csv):
        sys.exit(f"Fichier introuvable : {args.csv}")

    if args.regenerate:
        regenerate_csv(args.csv)
        return

    if args.purge:
        purge_history_rows(args.csv)

    rows = load_history_rows(args.csv)
    if args.randomize_locations:
        randomize_locations(rows)
    check_data_model(rows)
    print(f"{len(rows)} transactions à rejouer depuis {args.csv} vers {SCORE_API_URL} "
          f"(is_history=true, invisibles dans le dashboard)\n")

    ok, failed = 0, 0
    for i, row in enumerate(rows, start=1):
        payload = to_payload(row)
        try:
            resp = requests.post(SCORE_API_URL, json=payload, timeout=SCORE_API_TIMEOUT)
            resp.raise_for_status()
            ok += 1
            if i % 50 == 0 or i == len(rows):
                print(f"  [{i}/{len(rows)}] ok={ok} échecs={failed}")
        except requests.exceptions.RequestException as e:
            failed += 1
            print(f"  [{i}/{len(rows)}] ÉCHEC transaction_id={payload.get('transaction_id')} : {e}")

    print(f"\nTerminé : {ok} rejouées avec succès, {failed} échecs.")
    if failed:
        print("Les échecs n'ont PAS été journalisés dans transactions_scored "
              "(seulement dans failed_transactions si l'API a pu répondre).")


if __name__ == "__main__":
    main()