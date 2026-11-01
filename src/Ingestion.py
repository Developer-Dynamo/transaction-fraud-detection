#!/usr/bin/env python3
"""
Générateur de transactions en continu (une transaction toutes les 5 secondes)
— aligné sur score1.py / API_Score.py.

Il produit un mélange CONTRÔLÉ de trois classes de transactions :

  NORMAL    (~55 %)  comportement habituel : proche du domicile, montant courant.
  ANOMALY   (~20 %)  inhabituel mais LÉGITIME (voyage régional, panier plus
                     gros que d'habitude, achat tardif...). Ce sont les
                     transactions « pas totalement normales ».
  FRAUD     (~25 %)  scénarios de fraude explicites (voir ci-dessous).

Les proportions se règlent par variables d'environnement :
    FRAUD_PROBABILITY=0.25     part des transactions FRAUDULEUSES
    ANOMALY_PROBABILITY=0.20   part des transactions ANOMALES (légitimes)
    (le reste = NORMAL)

IMPORTANT : ces proportions s'appliquent aux TRANSACTIONS réellement émises
(pas aux « événements »). Comme certains scénarios de fraude contiennent
plusieurs transactions (rafale), une petite régulation adaptative corrige le
tirage pour que la part réelle reste proche des cibles.

----------------------------------------------------------------------
SCÉNARIOS DE FRAUDE (tous construits pour être détectables)
----------------------------------------------------------------------
IMPOSSIBLE_TRAVEL  achat POS près du domicile, puis 5 à 45 min plus tard un
                   achat POS à PLUS de 5000 km (vitesse impossible).
EXTREME_AMOUNT     un achat en ligne de 800 à 3200 € (misc_net / shopping_net).
CARD_TESTING       2 à 4 micro-achats en ligne (0,5 à 5 €) en quelques
                   secondes, puis un gros achat de 1500 à 3000 € (misc_net).
VELOCITY_ATTACK    4 à 8 transactions sur la MÊME carte, 2 à 20 s d'écart.
NIGHT_HIGH_RISK    achat de 400 à 1500 € entre 02:00 et 04:00 UTC, catégorie
                   à risque (misc_net / misc_pos).

----------------------------------------------------------------------
VÉRITÉ TERRAIN
----------------------------------------------------------------------
Le payload envoyé à l'API est INCHANGÉ (aucun champ en plus : l'API ne voit
jamais l'étiquette). L'étiquette réelle de chaque transaction est écrite à
part dans Ingestion_labels.csv (transaction_id, label, pattern), ce qui
permet de comparer ensuite avec la décision de l'API. Un résumé (répartition
réelle + décisions de l'API par classe) est affiché toutes les 20 transactions.

----------------------------------------------------------------------
MODÈLE DE DONNÉES (inchangé)
----------------------------------------------------------------------
- Les clients (cc_num + age + domicile FIXE) sont lus dans History.csv
  (FRAUD_HISTORY_CSV, ./History.csv, data/History.csv, dossier du script) ;
  sinon repli sur 15 clients aléatoires.
- lat/long/city envoyés = domicile FIXE du client (contrat de score1.py).
- merch_lat/merch_long = lieu de la transaction. Pour une transaction NORMALE
  il est proche du domicile (rayon FRAUD_LOCAL_RADIUS_KM, 120 km par défaut) ;
  les scénarios anormaux/frauduleux utilisent des lieux plus ou moins lointains.
- RÈGLE 1 : le lieu de la transaction est TOUJOURS différent de la ville de
  naissance / domicile du client (même nom de ville exclu, et au moins
  FRAUD_MIN_CITY_SEPARATION_KM = 25 km par défaut). Une transaction « locale »
  se fait donc dans une ville VOISINE, jamais dans la ville du domicile.
- RÈGLE 2 : un même client n'apparaît JAMAIS deux fois de suite dans le flux.
  Les rafales (CARD_TESTING, VELOCITY_ATTACK, ...) restent intactes, mais une
  transaction d'un AUTRE client est intercalée entre deux transactions de la
  même carte (horodatage placé entre les deux, ordre chronologique conservé).
- RÈGLE 3 : les lieux de transaction sont ÉQUILIBRÉS géographiquement. Le tirage
  se fait par région (cellule de FRAUD_REGION_CELL_DEG = 15° par défaut) puis par
  ville, en favorisant les régions/villes les moins utilisées : aucun pays ou
  continent n'est sur-représenté parce que la liste de villes y est plus dense.
  Les clients sont aussi tirés par zone (40°) pour que History.csv, s'il est
  dominé par un pays, ne concentre pas tout le flux dans ce pays.
- Les features dérivées (vélocité, vitesse, z-score...) sont calculées côté
  serveur par l'API à partir de l'historique : ce script n'en envoie aucune.

Variables d'environnement existantes conservées : FRAUD_SIM_START_TIME,
FRAUD_SIM_SPEED, FRAUD_REALISTIC_CADENCE, FRAUD_TX_PER_CARD_PER_DAY,
FRAUD_SCORE_API_URL, FRAUD_HISTORY_CSV, FRAUD_LOCAL_RADIUS_KM.
Nouvelles : FRAUD_PROBABILITY, ANOMALY_PROBABILITY, FRAUD_FAR_MIN_KM,
FRAUD_SEED (graine aléatoire pour reproduire une exécution),
FRAUD_REGION_CELL_DEG (taille des régions d'équilibrage, 15° par défaut),
FRAUD_LOCAL_MIN_CHOICES (5), FRAUD_BALANCE_CLIENT_ZONES (1), FRAUD_CLIENT_ZONE_DEG (40).
"""

import csv
import math
import os
import random
import re
import time
import unicodedata
import uuid
from collections import Counter, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from world_cities import WORLD_CITIES, nearest_world_city, random_transaction_location, with_extra_cities

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------

CSV_PATH = "Ingestion.csv"
LABELS_PATH = "Ingestion_labels.csv"
INTERVAL_SECONDS = 5  # une transaction envoyée toutes les 5 secondes (temps réel)
SUMMARY_EVERY = 20    # affichage du résumé toutes les N transactions

_seed_env = os.getenv("FRAUD_SEED")
if _seed_env:
    random.seed(_seed_env)


def _env_prob(name, default):
    try:
        v = float(os.getenv(name, str(default)))
        if not 0.0 <= v <= 1.0:
            raise ValueError("doit être entre 0 et 1")
        return v
    except ValueError as e:
        print(f"[AVERTISSEMENT] {name} invalide ({e}) : {default} utilisé.")
        return default


FRAUD_PROBABILITY = _env_prob("FRAUD_PROBABILITY", 0.25)
ANOMALY_PROBABILITY = _env_prob("ANOMALY_PROBABILITY", 0.20)
if FRAUD_PROBABILITY + ANOMALY_PROBABILITY > 1.0:
    _s = FRAUD_PROBABILITY + ANOMALY_PROBABILITY
    print("[AVERTISSEMENT] FRAUD_PROBABILITY + ANOMALY_PROBABILITY > 1 : valeurs normalisées.")
    FRAUD_PROBABILITY, ANOMALY_PROBABILITY = FRAUD_PROBABILITY / _s, ANOMALY_PROBABILITY / _s
NORMAL_PROBABILITY = max(0.0, 1.0 - FRAUD_PROBABILITY - ANOMALY_PROBABILITY)

LOCAL_RADIUS_KM = float(os.getenv("FRAUD_LOCAL_RADIUS_KM", "120"))
FAR_MIN_KM = float(os.getenv("FRAUD_FAR_MIN_KM", "5000"))   # « très loin » (impossible travel)
REGIONAL_MIN_KM, REGIONAL_MAX_KM = 300.0, 2500.0            # voyage légitime plausible
# Distance minimale entre la ville de domicile et la ville de transaction (règle 1).
MIN_CITY_SEPARATION_KM = float(os.getenv("FRAUD_MIN_CITY_SEPARATION_KM", "25"))
# La liste mondiale est clairsemée : souvent aucune ville à < LOCAL_RADIUS_KM du domicile.
# Pour ne pas retomber toujours sur LA même ville voisine, on complète avec les villes
# les plus proches jusqu'à disposer d'au moins FRAUD_LOCAL_MIN_CHOICES choix.
LOCAL_MIN_CHOICES = max(1, int(float(os.getenv("FRAUD_LOCAL_MIN_CHOICES", "5"))))
# Équilibrage des CLIENTS par zone géographique (cellules de FRAUD_CLIENT_ZONE_DEG degrés) :
# évite qu'un pays très représenté dans History.csv concentre la plupart des transactions.
BALANCE_CLIENT_ZONES = os.getenv("FRAUD_BALANCE_CLIENT_ZONES", "1").strip() not in ("0", "false", "False", "")
CLIENT_ZONE_DEG = float(os.getenv("FRAUD_CLIENT_ZONE_DEG", "40"))

# ----------------------------------------------------------------------
# HORLOGE SIMULÉE
# ----------------------------------------------------------------------
_SIM_START_ENV = os.getenv("FRAUD_SIM_START_TIME")
SIM_START: datetime | None = None
if _SIM_START_ENV:
    try:
        SIM_START = datetime.strptime(_SIM_START_ENV.strip(), "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        print(f"[AVERTISSEMENT] FRAUD_SIM_START_TIME={_SIM_START_ENV!r} invalide "
              f"(format attendu : 'AAAA-MM-JJ HH:MM:SS') : heure réelle utilisée à la place.")

try:
    SIM_SPEED = float(os.getenv("FRAUD_SIM_SPEED", "1.0"))
    if SIM_SPEED <= 0:
        raise ValueError("doit être strictement positif")
except ValueError as e:
    print(f"[AVERTISSEMENT] FRAUD_SIM_SPEED invalide ({e}) : vitesse 1.0 utilisée.")
    SIM_SPEED = 1.0

_REAL_START = datetime.now(timezone.utc)

REALISTIC_CADENCE = os.getenv("FRAUD_REALISTIC_CADENCE", "1").strip() not in ("0", "false", "False", "")
try:
    TX_PER_CARD_PER_DAY = float(os.getenv("FRAUD_TX_PER_CARD_PER_DAY", "2.0"))
    if TX_PER_CARD_PER_DAY <= 0:
        raise ValueError("doit être strictement positif")
except ValueError as e:
    print(f"[AVERTISSEMENT] FRAUD_TX_PER_CARD_PER_DAY invalide ({e}) : 2.0 utilisé.")
    TX_PER_CARD_PER_DAY = 2.0

_sim_clock: datetime | None = None


def simulated_now(force_gap_seconds: float | None = None, night: bool = False) -> datetime:
    """Horodatage (UTC) de la PROCHAINE transaction, toujours strictement
    croissant (l'API suppose un ordre chronologique causal).

    force_gap_seconds : impose l'écart avec la transaction précédente (rafales,
                        voyage impossible). Sinon écart aléatoire (loi exponentielle).
    night             : avance l'horloge jusqu'à la prochaine fenêtre
                        02:00-04:00 UTC (scénario NIGHT_HIGH_RISK).
    """
    global _sim_clock
    if REALISTIC_CADENCE:
        if _sim_clock is None:
            _sim_clock = SIM_START or datetime.now(timezone.utc)
        else:
            if force_gap_seconds is not None:
                gap = force_gap_seconds
            else:
                mean_gap_s = 86400.0 / (TX_PER_CARD_PER_DAY * max(1, len(CLIENTS)))
                gap = random.expovariate(1.0 / mean_gap_s)
            _sim_clock += timedelta(seconds=max(1.0, gap))
    else:
        # Mode « horloge réelle » : on garde l'heure réelle, mais on force la
        # croissance stricte pour que les rafales gardent l'écart voulu.
        if SIM_START is None:
            t = datetime.now(timezone.utc)
        else:
            real_elapsed_s = (datetime.now(timezone.utc) - _REAL_START).total_seconds()
            t = SIM_START + timedelta(seconds=real_elapsed_s * SIM_SPEED)
        if _sim_clock is not None:
            t = max(t, _sim_clock + timedelta(seconds=max(1.0, force_gap_seconds or 1.0)))
        _sim_clock = t

    if night and not (2 <= _sim_clock.hour < 4):
        target = _sim_clock.replace(hour=random.randint(2, 3), minute=random.randint(0, 59),
                                    second=random.randint(0, 59), microsecond=0)
        if target <= _sim_clock:
            target += timedelta(days=1)
        _sim_clock = target
    return _sim_clock


SCORE_API_URL = os.getenv("FRAUD_SCORE_API_URL", "http://localhost:8001/score")
SCORE_API_TIMEOUT = 5  # secondes

FIELDNAMES = [
    "transaction_id", "cc_num", "trans_date_trans_time", "unix_time",
    "amt", "merchant", "category", "channel", "city", "age",
    "lat", "long", "merch_lat", "merch_long",
]
LABEL_FIELDNAMES = ["transaction_id", "label", "pattern"]

# ----------------------------------------------------------------------
# CLIENTS / CARTES
# ----------------------------------------------------------------------
NB_CLIENTS = 15  # nombre de clients générés en repli (si History.csv est introuvable)
_AGE_POOL = [18, 22, 25, 28, 31, 34, 37, 40, 44, 48, 52, 56, 60, 65, 70]


def _find_history_csv():
    candidates = []
    env_path = os.getenv("FRAUD_HISTORY_CSV")
    if env_path:
        candidates.append(Path(env_path))
    here = Path(__file__).resolve().parent
    candidates += [Path.cwd() / "History.csv", Path("data") / "History.csv",
                   Path("data") / "sample" / "History.csv", here / "History.csv"]
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_clients_from_history():
    """Lit les clients (cc_num + age + domicile fixe) dans History.csv.
    Retourne None si le fichier est absent/illisible."""
    path = _find_history_csv()
    if path is None:
        return None
    clients = {}
    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cc = (row.get("cc_num") or "").strip()
                if not cc or cc in clients:
                    continue
                try:
                    age = int(round(float(row.get("age") or 0)))
                except ValueError:
                    age = 0
                try:
                    home_lat = float(row.get("lat"))
                    home_long = float(row.get("long"))
                except (TypeError, ValueError):
                    print(f"[AVERTISSEMENT] Domicile (lat/long) illisible pour cc_num={cc} "
                          f"dans {path} : repli sur des clients aléatoires.")
                    return None
                clients[cc] = {
                    "cc_num": cc,
                    "age": age if age > 0 else random.choice(_AGE_POOL),
                    "home_city": (row.get("city") or "").strip() or "Domicile inconnu",
                    "home_lat": home_lat,
                    "home_long": home_long,
                }
    except (OSError, csv.Error) as e:
        print(f"[AVERTISSEMENT] Lecture de {path} impossible : {e}")
        return None
    if not clients:
        return None
    print(f"{len(clients)} cartes chargées depuis {path} (domicile fixe par carte).")
    return list(clients.values())


def _random_clients():
    """Repli : cc_num uniques + un domicile fixe par client."""
    suffixes = random.sample(range(10000), NB_CLIENTS)
    homes = random.sample(WORLD_CITIES, min(NB_CLIENTS, len(WORLD_CITIES)))
    return [
        {
            "cc_num": f"400012345678{suffix:04d}",
            "age": _AGE_POOL[i % len(_AGE_POOL)],
            "home_city": homes[i % len(homes)][0],
            "home_lat": homes[i % len(homes)][1],
            "home_long": homes[i % len(homes)][2],
        }
        for i, suffix in enumerate(suffixes)
    ]


CLIENTS = load_clients_from_history()
if CLIENTS is None:
    print("[AVERTISSEMENT] History.csv introuvable ou incomplet : cartes ALÉATOIRES utilisées "
          "-> aucun historique comportemental pour ces cartes.")
    CLIENTS = _random_clients()
NB_CLIENTS = len(CLIENTS)

# Catégories à utiliser = celles vues à l'entraînement (category_map).
MERCHANTS = [
    {"name": "fraud_Carrefour",      "category": "grocery_pos"},
    {"name": "fraud_Amazon",         "category": "shopping_net"},
    {"name": "fraud_TotalEnergies",  "category": "gas_transport"},
    {"name": "fraud_UberEats",       "category": "food_dining"},
    {"name": "fraud_Fnac",           "category": "shopping_pos"},
    {"name": "fraud_Zara",           "category": "misc_pos"},
    {"name": "fraud_CryptoExchange", "category": "misc_net"},
]
POS_MERCHANTS = [m for m in MERCHANTS if not m["category"].endswith("_net")]
NET_MERCHANTS = [m for m in MERCHANTS if m["category"].endswith("_net")]
RISKY_NET_MERCHANTS = [m for m in MERCHANTS if m["category"] in ("misc_net", "shopping_net")]
NIGHT_RISK_MERCHANTS = [m for m in MERCHANTS if m["category"] in ("misc_net", "misc_pos")]
CRYPTO_MERCHANT = next(m for m in MERCHANTS if m["category"] == "misc_net")

TRANSACTION_CITIES = with_extra_cities([(c["home_city"], c["home_lat"], c["home_long"]) for c in CLIENTS])

# ----------------------------------------------------------------------
# GÉOGRAPHIE
# ----------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def _norm_city(name):
    """Nom de ville normalisé (sans accents, minuscules, sans ponctuation)."""
    t = unicodedata.normalize("NFKD", str(name or ""))
    t = "".join(ch for ch in t if not unicodedata.combining(ch)).casefold()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def _is_other_city(lat, lon, city_name, home_city):
    """Vrai si le point (lat, lon) correspond à une ville DIFFÉRENTE du domicile :
    ni la ville tirée, ni la ville la plus proche du point (celle que le dashboard
    affichera comme « lieu de transaction ») ne portent le nom du domicile."""
    home_n = _norm_city(home_city)
    if home_n and _norm_city(city_name) == home_n:
        return False
    near = nearest_world_city(lat, lon, cities=TRANSACTION_CITIES)
    if near and home_n and _norm_city(near[0]) == home_n:
        return False
    return True


def _cities_other_than_home(home_city):
    home_n = _norm_city(home_city)
    others = [c for c in TRANSACTION_CITIES if _norm_city(c[0]) != home_n]
    return others or list(TRANSACTION_CITIES)


# --- Équilibrage géographique (RÈGLE 3) ---------------------------------
# Sans correction, tirer « une ville au hasard dans la liste » favorise les
# régions qui contiennent beaucoup de villes (et le mode local retombe toujours
# sur les mêmes voisines). On tire donc en deux temps, avec rééquilibrage :
#   1) une RÉGION (cellule de REGION_CELL_DEG degrés) parmi celles éligibles,
#   2) une VILLE de cette région,
# chaque niveau favorisant ce qui a été le MOINS utilisé jusqu'ici. Les
# compteurs sont globaux : les régions/villes sur-utilisées sont délaissées.
REGION_CELL_DEG = float(os.getenv("FRAUD_REGION_CELL_DEG", "15"))
_region_use = Counter()
_city_use = Counter()


def _region_key(lat, lon):
    return (int(math.floor(lat / REGION_CELL_DEG)), int(math.floor(lon / REGION_CELL_DEG)))


def _city_key(c):
    return (c[0], round(c[1], 2), round(c[2], 2))


def _record_use(c):
    _region_use[_region_key(c[1], c[2])] += 1
    _city_use[_city_key(c)] += 1


def _balanced_pick(candidates):
    """Choisit une ville : région la moins utilisée d'abord, puis ville la moins
    utilisée de cette région (poids = 1 / (1 + retard sur le minimum)^2)."""
    by_region = {}
    for c in candidates:
        by_region.setdefault(_region_key(c[1], c[2]), []).append(c)
    regions = list(by_region)
    min_r = min(_region_use[r] for r in regions)
    r = random.choices(regions, weights=[1.0 / (1.0 + _region_use[x] - min_r) for x in regions])[0]
    cities = by_region[r]
    min_c = min(_city_use[_city_key(c)] for c in cities)
    return random.choices(cities, weights=[1.0 / (1.0 + _city_use[_city_key(c)] - min_c) ** 2 for c in cities])[0]


def _draw_location(candidates, home_city, accept=None):
    """Tirage équilibré + bruit léger, revérifié (ville différente du domicile,
    condition `accept` éventuelle). Retourne (lat, lon) ou None."""
    for _ in range(30):
        c = _balanced_pick(candidates)
        _name, lat, lon = random_transaction_location([c])
        if _is_other_city(lat, lon, c[0], home_city) and (accept is None or accept(lat, lon)):
            _record_use(c)
            return lat, lon
    return None


def geo_stats():
    """(nb lieux, nb villes distinctes, ville la + fréquente, sa part,
    nb régions, part de la région la + fréquente) ou None."""
    total = sum(_city_use.values())
    if not total:
        return None
    top_city, n_city = _city_use.most_common(1)[0]
    _top_reg, n_reg = _region_use.most_common(1)[0]
    return total, len(_city_use), top_city[0], n_city / total, len(_region_use), n_reg / total


def _local_point(home_lat, home_long, home_city=""):
    """Lieu « local » d'une transaction NORMALE : tiré (de façon équilibrée) parmi
    les villes de la liste situées entre MIN_CITY_SEPARATION_KM et LOCAL_RADIUS_KM
    du domicile, EN EXCLUANT la ville de domicile (même nom). Si aucune ville
    voisine ne convient, on prend la ville la plus proche qui soit différente du
    domicile. Si la liste est clairsemée (moins de FRAUD_LOCAL_MIN_CHOICES villes
    éligibles), on la complète avec les villes les plus proches. Un léger bruit est
    appliqué, puis le résultat est revérifié."""
    ranked = sorted(_cities_other_than_home(home_city),
                    key=lambda c: haversine_km(home_lat, home_long, c[1], c[2]))
    nearby = [c for c in ranked
              if MIN_CITY_SEPARATION_KM <= haversine_km(home_lat, home_long, c[1], c[2]) <= LOCAL_RADIUS_KM]
    if len(nearby) < LOCAL_MIN_CHOICES:
        # complète avec les villes les plus proches (au-delà du rayon) pour varier les lieux
        extra = [c for c in ranked if c not in nearby
                 and haversine_km(home_lat, home_long, c[1], c[2]) >= MIN_CITY_SEPARATION_KM]
        nearby = nearby + extra[:LOCAL_MIN_CHOICES - len(nearby)]
        if not nearby:
            nearby = [ranked[0]]
    pt = _draw_location(nearby, home_city)
    if pt:
        return pt
    c = nearby[0]  # repli : coordonnées exactes de la ville voisine choisie
    _record_use(c)
    return round(c[1], 6), round(c[2], 6)


def pick_location(home_lat, home_long, mode, home_city=""):
    """Lieu de la transaction (TOUJOURS une ville différente du domicile).
    mode = "local"    : ville voisine du domicile (comportement normal)
           "regional" : 300 à 2500 km (voyage légitime plausible)
           "far"      : au moins FAR_MIN_KM (5000 km)
    Les modes regional/far filtrent d'abord TOUTES les villes éligibles par
    distance, puis tirent de façon équilibrée par région et par ville (aucun
    pays/continent n'est favorisé par le nombre de villes qu'il contient)."""
    if mode == "local":
        return _local_point(home_lat, home_long, home_city)

    lo, hi = (FAR_MIN_KM, float("inf")) if mode == "far" else (REGIONAL_MIN_KM, REGIONAL_MAX_KM)
    others = _cities_other_than_home(home_city)
    candidates = [c for c in others if lo <= haversine_km(home_lat, home_long, c[1], c[2]) <= hi]
    if candidates:
        pt = _draw_location(candidates, home_city,
                            accept=lambda la, lg: lo <= haversine_km(home_lat, home_long, la, lg) <= hi)
        if pt:
            return pt
    if mode == "far":
        # repli : parmi les 15 villes les plus éloignées (et non toujours la même)
        farthest = sorted(others, key=lambda c: -haversine_km(home_lat, home_long, c[1], c[2]))[:15]
        c = _balanced_pick(farthest)
        _record_use(c)
        return round(c[1], 6), round(c[2], 6)
    return _local_point(home_lat, home_long, home_city)


# ----------------------------------------------------------------------
# MONTANTS
# ----------------------------------------------------------------------

BASE_RANGES = {
    "grocery_pos": (5, 120),
    "shopping_net": (10, 250),
    "gas_transport": (15, 90),
    "food_dining": (5, 70),
    "shopping_pos": (10, 150),
    "misc_pos": (5, 100),
    "misc_net": (5, 150),
}


def normal_amount(category):
    """Montant courant selon la catégorie (beaucoup de petits montants).
    Plus d'outlier aléatoire ici : les montants inhabituels sont désormais
    produits explicitement (et étiquetés) par les scénarios ANOMALY/FRAUD."""
    low, high = BASE_RANGES.get(category, (5, 120))
    return round(random.triangular(low, high, low + (high - low) * 0.25), 2)


# ----------------------------------------------------------------------
# CONSTRUCTION D'UNE TRANSACTION
# ----------------------------------------------------------------------

def _build_row(client, merchant, amt, ts, loc_mode):
    category = merchant["category"]
    merch_lat, merch_long = pick_location(client["home_lat"], client["home_long"], loc_mode, client["home_city"])
    return {
        "transaction_id": str(uuid.uuid4()),
        "cc_num": client["cc_num"],
        "trans_date_trans_time": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "unix_time": int(ts.timestamp()),
        "amt": round(amt, 2),
        "merchant": merchant["name"],
        "category": category,
        "channel": "online" if category.endswith("_net") else "pos",
        "city": client["home_city"],        # domicile FIXE
        "age": client["age"],
        "lat": client["home_lat"],          # domicile FIXE
        "long": client["home_long"],
        "merch_lat": merch_lat,             # lieu de la transaction
        "merch_long": merch_long,
    }


# ----------------------------------------------------------------------
# SCÉNARIOS — chacun renvoie une liste de (row, label, pattern)
# ----------------------------------------------------------------------

def _ts(ts, **kw):
    """Horodatage imposé (transaction intercalée) ou prochain horodatage simulé."""
    return ts if ts is not None else simulated_now(**kw)


def scenario_normal(client, ts=None):
    m = random.choice(MERCHANTS)
    row = _build_row(client, m, normal_amount(m["category"]), _ts(ts), "local")
    return [(row, "NORMAL", "NORMAL")]


# --- ANOMALIES : inhabituel mais légitime ------------------------------

ANOMALY_PATTERNS = ["AMOUNT_ABOVE_USUAL", "REGIONAL_TRAVEL", "LATE_NIGHT_PURCHASE", "UNUSUAL_ONLINE_BASKET"]


# Anomalies sans contrainte horaire : utilisables pour une transaction intercalée.
FILLER_ANOMALY_PATTERNS = ["AMOUNT_ABOVE_USUAL", "REGIONAL_TRAVEL", "UNUSUAL_ONLINE_BASKET"]


def scenario_anomaly(client, ts=None, patterns=None):
    pattern = random.choice(patterns or ANOMALY_PATTERNS)

    if pattern == "AMOUNT_ABOVE_USUAL":
        # 1,5x à 2,5x le plafond habituel de la catégorie (reste sous les fraudes).
        m = random.choice(MERCHANTS)
        high = BASE_RANGES[m["category"]][1]
        row = _build_row(client, m, random.uniform(high * 1.5, high * 2.5), _ts(ts), "local")

    elif pattern == "REGIONAL_TRAVEL":
        # Achat en point de vente à 300-2500 km du domicile (déplacement).
        m = random.choice(POS_MERCHANTS)
        row = _build_row(client, m, normal_amount(m["category"]), _ts(ts), "regional")

    elif pattern == "LATE_NIGHT_PURCHASE":
        # Achat nocturne (02:00-04:00 UTC) mais de montant courant.
        m = random.choice(MERCHANTS)
        row = _build_row(client, m, normal_amount(m["category"]), simulated_now(night=True), "local")

    else:  # UNUSUAL_ONLINE_BASKET
        m = random.choice(NET_MERCHANTS)
        row = _build_row(client, m, random.uniform(250, 600), _ts(ts), "local")

    return [(row, "ANOMALY", pattern)]


# --- FRAUDES -----------------------------------------------------------

FRAUD_PATTERNS = ["IMPOSSIBLE_TRAVEL", "EXTREME_AMOUNT", "CARD_TESTING", "VELOCITY_ATTACK", "NIGHT_HIGH_RISK"]


# Poids par scénario : les rafales (CARD_TESTING, VELOCITY_ATTACK) contiennent
# plusieurs transactions, on les tire donc moins souvent pour que le mélange
# TRANSACTION par TRANSACTION reste équilibré entre les 5 patterns.
FRAUD_PATTERN_WEIGHTS = [3, 3, 1.5, 1, 3]


def scenario_fraud(client):
    pattern = random.choices(FRAUD_PATTERNS, weights=FRAUD_PATTERN_WEIGHTS)[0]
    out = []

    if pattern == "IMPOSSIBLE_TRAVEL":
        # 1) achat POS normal près du domicile (précédent légitime)
        m1 = random.choice(POS_MERCHANTS)
        out.append((_build_row(client, m1, normal_amount(m1["category"]), simulated_now(), "local"),
                    "NORMAL", "IMPOSSIBLE_TRAVEL_precursor"))
        # 2) 5 à 45 min plus tard : achat POS à > 5000 km
        m2 = random.choice(POS_MERCHANTS)
        ts = simulated_now(force_gap_seconds=random.uniform(300, 2700))
        out.append((_build_row(client, m2, normal_amount(m2["category"]) * random.uniform(1.0, 2.0), ts, "far"),
                    "FRAUD", pattern))

    elif pattern == "EXTREME_AMOUNT":
        m = random.choice(RISKY_NET_MERCHANTS)
        mode = random.choice(["local", "far"])
        out.append((_build_row(client, m, random.uniform(800, 3200), simulated_now(), mode), "FRAUD", pattern))

    elif pattern == "CARD_TESTING":
        mode = random.choice(["local", "far"])
        # micro-achats en ligne en quelques secondes...
        for i in range(random.randint(2, 4)):
            ts = simulated_now() if i == 0 else simulated_now(force_gap_seconds=random.uniform(5, 60))
            out.append((_build_row(client, random.choice(NET_MERCHANTS), random.uniform(0.5, 5.0), ts, mode),
                        "FRAUD", pattern))
        # ... puis gros achat immédiat
        ts = simulated_now(force_gap_seconds=random.uniform(10, 90))
        out.append((_build_row(client, CRYPTO_MERCHANT, random.uniform(1500, 3000), ts, mode), "FRAUD", pattern))

    elif pattern == "VELOCITY_ATTACK":
        # rafale sur la MÊME carte : 4 à 8 transactions, 2 à 20 s d'écart
        for i in range(random.randint(4, 8)):
            ts = simulated_now() if i == 0 else simulated_now(force_gap_seconds=random.uniform(2, 20))
            m = random.choice(MERCHANTS)
            out.append((_build_row(client, m, random.uniform(20, 400), ts, "local"), "FRAUD", pattern))

    else:  # NIGHT_HIGH_RISK
        m = random.choice(NIGHT_RISK_MERCHANTS)
        mode = random.choice(["local", "far"])
        ts = simulated_now(night=True)
        out.append((_build_row(client, m, random.uniform(400, 1500), ts, mode), "FRAUD", pattern))

    return out


# ----------------------------------------------------------------------
# CHOIX DE LA CLASSE (avec régulation pour tenir les proportions)
# ----------------------------------------------------------------------

_sent = Counter()             # transactions émises par classe
_decisions = Counter()        # (classe, décision API)
_pending = deque()            # transactions déjà construites (rafales) en attente d'envoi

_TARGETS = {"NORMAL": NORMAL_PROBABILITY, "ANOMALY": ANOMALY_PROBABILITY, "FRAUD": FRAUD_PROBABILITY}


def choose_class():
    """Tire la classe du prochain événement. Les rafales contiennent plusieurs
    transactions : on corrige donc le tirage selon l'écart entre la part réelle
    déjà émise et la cible, pour que les proportions TRANSACTION par
    TRANSACTION restent proches de FRAUD_PROBABILITY / ANOMALY_PROBABILITY."""
    total = sum(_sent.values())
    if total < 20:
        weights = dict(_TARGETS)
    else:
        weights = {k: max(0.0, t + 8.0 * (t - _sent[k] / total)) for k, t in _TARGETS.items()}
    if sum(weights.values()) <= 0:
        weights = dict(_TARGETS)
    return random.choices(list(weights), weights=list(weights.values()))[0]


_last_cc = None   # carte de la dernière transaction émise
_last_ts = None   # son horodatage (texte)


_client_zone_use = Counter()


def _zone_key(client):
    return (int(math.floor(client["home_lat"] / CLIENT_ZONE_DEG)),
            int(math.floor(client["home_long"] / CLIENT_ZONE_DEG)))


def _pick_client(exclude_cc=None):
    """Client DIFFÉRENT du dernier client émis (règle 2). Quand
    BALANCE_CLIENT_ZONES est actif, on tire d'abord la zone géographique la moins
    utilisée, puis un client de cette zone (règle 3 : pas de pays sur-représenté)."""
    pool = [c for c in CLIENTS if c["cc_num"] != exclude_cc] if len(CLIENTS) > 1 else CLIENTS
    if not BALANCE_CLIENT_ZONES:
        return random.choice(pool)
    zones = {}
    for c in pool:
        zones.setdefault(_zone_key(c), []).append(c)
    zl = list(zones)
    mn = min(_client_zone_use[z] for z in zl)
    z = random.choices(zl, weights=[1.0 / (1.0 + _client_zone_use[x] - mn) for x in zl])[0]
    _client_zone_use[z] += 1
    return random.choice(zones[z])


def _parse_row_ts(txt):
    return datetime.strptime(txt, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def _filler_transaction(exclude_cc, ts):
    """Transaction simple (normale ou anomalie légitime) d'un AUTRE client,
    intercalée au milieu d'une rafale pour que la même carte ne soit jamais
    émise deux fois de suite."""
    client = _pick_client(exclude_cc)
    tn, ta = _TARGETS["NORMAL"], _TARGETS["ANOMALY"]
    if ta > 0 and random.random() < ta / max(tn + ta, 1e-9):
        return scenario_anomaly(client, ts=ts, patterns=FILLER_ANOMALY_PATTERNS)[0]
    return scenario_normal(client, ts=ts)[0]


def next_transaction():
    """Retourne (row, label, pattern). Vide d'abord la file des rafales, en
    intercalant une transaction d'un autre client dès que la suivante serait
    de la même carte que la précédente."""
    global _last_cc, _last_ts
    if not _pending:
        client = _pick_client(_last_cc)
        cls = choose_class()
        builder = {"NORMAL": scenario_normal, "ANOMALY": scenario_anomaly, "FRAUD": scenario_fraud}[cls]
        _pending.extend(builder(client))

    if len(CLIENTS) > 1 and _last_cc is not None and _pending[0][0]["cc_num"] == _last_cc:
        # Horodatage placé entre la transaction précédente et la suivante de la rafale :
        # l'ordre chronologique du flux reste strictement croissant.
        ts = None
        try:
            lo = _parse_row_ts(_last_ts)
            hi = _parse_row_ts(_pending[0][0]["trans_date_trans_time"])
            gap = (hi - lo).total_seconds()
            if gap >= 2:
                ts = lo + timedelta(seconds=int(gap // 2))
        except (TypeError, ValueError):
            ts = None
        row, label, pattern = _filler_transaction(_last_cc, ts)
    else:
        row, label, pattern = _pending.popleft()

    _sent[label] += 1
    _last_cc = row["cc_num"]
    _last_ts = row["trans_date_trans_time"]
    return row, label, pattern


# ----------------------------------------------------------------------
# ENVOI À L'API DE SCORING
# ----------------------------------------------------------------------

def send_to_scoring_api(row):
    """Transfère la transaction à l'API /score. N'interrompt jamais la génération."""
    payload = {k: v for k, v in row.items() if k != "unix_time"}
    try:
        response = requests.post(SCORE_API_URL, json=payload, timeout=SCORE_API_TIMEOUT)
        response.raise_for_status()
        result = response.json()
        print(f"  -> scoré : decision={result.get('decision')} "
              f"risk_score={result.get('risk_score')} "
              f"(proba modèle={result.get('model_probability')})")
        return result
    except requests.exceptions.ConnectionError:
        print(f"  -> [ERREUR] Impossible de joindre l'API de scoring ({SCORE_API_URL}). L'API est-elle démarrée ?")
    except requests.exceptions.Timeout:
        print(f"  -> [ERREUR] Timeout lors de l'appel à l'API de scoring ({SCORE_API_URL}).")
    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            pass
        print(f"  -> [ERREUR] L'API de scoring a renvoyé {e.response.status_code} : {detail}")
    except requests.exceptions.RequestException as e:
        print(f"  -> [ERREUR] Échec de l'appel à l'API de scoring : {e}")
    return None


# ----------------------------------------------------------------------
# FICHIERS CSV
# ----------------------------------------------------------------------

def prepare_csv():
    """True si l'en-tête doit être écrit. Un ancien Ingestion.csv avec un autre
    schéma est mis de côté au lieu d'être complété avec des lignes désalignées."""
    if not os.path.isfile(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
        return True
    with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
        existing_header = next(csv.reader(f), [])
    if existing_header == FIELDNAMES:
        return False
    backup = f"Ingestion_old_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    os.rename(CSV_PATH, backup)
    print(f"Ancien schéma détecté dans {CSV_PATH} -> sauvegardé sous {backup}")
    return True


def _labels_need_header():
    return not os.path.isfile(LABELS_PATH) or os.path.getsize(LABELS_PATH) == 0


# ----------------------------------------------------------------------
# AFFICHAGE
# ----------------------------------------------------------------------

_ICON = {"NORMAL": "✅", "ANOMALY": "🟡", "FRAUD": "⚠️ "}


def print_summary():
    total = sum(_sent.values())
    if not total:
        return
    print(f"\n----- Résumé après {total} transactions -----")
    for label in ("NORMAL", "ANOMALY", "FRAUD"):
        n = _sent[label]
        decisions = {d: c for (lbl, d), c in _decisions.items() if lbl == label}
        extra = f"  décisions API : {decisions}" if decisions else ""
        print(f"  {label:<8} {n:>5} ({n / total:5.1%}){extra}")
    g = geo_stats()
    if g:
        n, cities, top_city, top_share, regions, reg_share = g
        print(f"  Lieux : {cities} villes distinctes, {regions} régions ; ville la plus fréquente : "
              f"{top_city} ({top_share:.1%}) ; région la plus fréquente : {reg_share:.1%}")
    print("---------------------------------------------\n")


def main():
    write_header = prepare_csv()
    write_label_header = _labels_need_header()

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f, \
         open(LABELS_PATH, "a", newline="", encoding="utf-8") as lf:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        label_writer = csv.DictWriter(lf, fieldnames=LABEL_FIELDNAMES)
        if write_header:
            writer.writeheader()
            f.flush()
        if write_label_header:
            label_writer.writeheader()
            lf.flush()

        print(f"🚀 Génération de transactions démarrée -> {CSV_PATH} (étiquettes : {LABELS_PATH})")
        print(f"🎯 Cibles : NORMAL {NORMAL_PROBABILITY:.0%} | ANOMALY {ANOMALY_PROBABILITY:.0%} "
              f"| FRAUD {FRAUD_PROBABILITY:.0%}")
        print(f"👥 {NB_CLIENTS} clients (1 carte unique chacun, jamais deux fois de suite)")
        print(f"📍 Lieu de transaction toujours différent de la ville de naissance "
              f"(écart minimal {MIN_CITY_SEPARATION_KM:g} km)")
        if REALISTIC_CADENCE:
            print(f"⏱️  Cadence réaliste : ~{TX_PER_CARD_PER_DAY:g} transaction(s)/jour/carte en temps simulé "
                  f"(envoi réel toutes les {INTERVAL_SECONDS} s)")
        print(f"📡 Envoi vers {SCORE_API_URL}")
        print("Ctrl+C pour arrêter.\n")

        count = 0
        try:
            while True:
                row, label, pattern = next_transaction()
                writer.writerow(row)
                label_writer.writerow({"transaction_id": row["transaction_id"], "label": label, "pattern": pattern})
                f.flush()
                lf.flush()

                near = nearest_world_city(row["merch_lat"], row["merch_long"], cities=TRANSACTION_CITIES)
                merch_city = near[0] if near else "?"
                print(f"{_ICON[label]} [{label}:{pattern}] {row['trans_date_trans_time']} | "
                      f"carte …{row['cc_num'][-4:]} | {row['amt']}€ | {row['category']} | {row['channel']} | "
                      f"lieu : {merch_city} (domicile : {row['city']})")
                result = send_to_scoring_api(row)
                if result is not None:
                    _decisions[(label, result.get("decision"))] += 1

                count += 1
                if count % SUMMARY_EVERY == 0:
                    print_summary()
                time.sleep(INTERVAL_SECONDS)
        except KeyboardInterrupt:
            print("\nGénération arrêtée par l'utilisateur.")
            print_summary()


if __name__ == "__main__":
    main()