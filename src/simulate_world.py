"""Simulateur de transactions bancaires MONDIALES (cartes voyageuses + fraudes).

Sorties (dossier --out) :
  world_train_features.csv   même schéma que fraudTrain_features.csv (+ is_fraud)
  priors.json                stats catégorie + fréquences marchand (serving)
  Ingestion.csv              flux à rejouer vers l'API (colonnes de votre Ingestion.csv)
  Ingestion_labels.csv       transaction_id,is_fraud (vérité terrain, pour évaluer)
  History.csv                historique antérieur des cartes du flux

Usage :
  python simulate_world.py --out data_world --cards 1500 --days 365
  python simulate_world.py --out data_world --base fraudTrain_features.csv   # fusion avec la base US
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from features_builder import build_features, haversine_km, OUTPUT_COLUMNS

# (ville, lat, lon, poids)
CITIES = [
    ("Tunis", 36.81, 10.18, 3), ("Sfax", 34.74, 10.76, 1), ("Paris", 48.86, 2.35, 4), ("Lyon", 45.76, 4.84, 1.5),
    ("Marseille", 43.30, 5.37, 1.2), ("Luxembourg", 49.61, 6.13, 1), ("Bruxelles", 50.85, 4.35, 1.5),
    ("Berlin", 52.52, 13.40, 2.5), ("Munich", 48.14, 11.58, 1.5), ("Madrid", 40.42, -3.70, 2),
    ("Barcelone", 41.39, 2.17, 1.5), ("Rome", 41.90, 12.50, 2), ("Milan", 45.46, 9.19, 1.5),
    ("Londres", 51.51, -0.13, 4), ("Amsterdam", 52.37, 4.90, 1.5), ("Zurich", 47.37, 8.54, 1),
    ("Casablanca", 33.57, -7.59, 1.5), ("Alger", 36.75, 3.06, 1.2), ("Le Caire", 30.04, 31.24, 1.5),
    ("Dakar", 14.72, -17.47, 0.8), ("Lagos", 6.52, 3.38, 1), ("Nairobi", -1.29, 36.82, 0.8),
    ("Johannesburg", -26.20, 28.05, 1), ("Dubaï", 25.20, 55.27, 2), ("Istanbul", 41.01, 28.98, 2),
    ("Riyad", 24.71, 46.68, 1), ("Mumbai", 19.08, 72.88, 2), ("Delhi", 28.61, 77.21, 2),
    ("Singapour", 1.35, 103.82, 2), ("Bangkok", 13.76, 100.50, 1.2), ("Shanghai", 31.23, 121.47, 2.5),
    ("Pékin", 39.90, 116.40, 2), ("Tokyo", 35.68, 139.69, 3), ("Séoul", 37.57, 126.98, 1.5),
    ("Sydney", -33.87, 151.21, 1.5), ("Auckland", -36.85, 174.76, 0.5), ("Moscou", 55.76, 37.62, 1.5),
    ("New York", 40.71, -74.01, 4), ("Los Angeles", 34.05, -118.24, 2.5), ("Chicago", 41.88, -87.63, 2),
    ("Houston", 29.76, -95.37, 1.5), ("Miami", 25.76, -80.19, 1.2), ("Toronto", 43.65, -79.38, 1.5),
    ("Mexico", 19.43, -99.13, 1.5), ("São Paulo", -23.55, -46.63, 2), ("Buenos Aires", -34.60, -58.38, 1),
    ("Bogota", 4.71, -74.07, 0.8), ("Lima", -12.05, -77.04, 0.6), ("Santiago", -33.45, -70.67, 0.6),
]
CITY_LAT = np.array([c[1] for c in CITIES]); CITY_LON = np.array([c[2] for c in CITIES])
CITY_W = np.array([c[3] for c in CITIES], float); CITY_W /= CITY_W.sum()

# catégorie : (poids, mu, sigma) du log-montant — ordres de grandeur de Sparkov
CATS = {
    "grocery_pos": (0.10, 3.9, 0.75), "gas_transport": (0.10, 3.8, 0.55), "home": (0.09, 3.7, 0.9),
    "shopping_pos": (0.09, 3.9, 1.0), "kids_pets": (0.09, 3.5, 0.8), "shopping_net": (0.09, 4.1, 1.0),
    "entertainment": (0.07, 3.7, 0.9), "food_dining": (0.07, 3.0, 0.7), "personal_care": (0.07, 3.3, 0.8),
    "health_fitness": (0.06, 3.3, 0.8), "misc_pos": (0.06, 3.5, 1.0), "misc_net": (0.04, 3.6, 1.0),
    "grocery_net": (0.04, 4.0, 0.7), "travel": (0.03, 4.4, 1.2),
}
CAT_NAMES = list(CATS)
ONLINE_CATS = [c for c in CAT_NAMES if c.endswith("_net")]
BASE_W = np.array([CATS[c][0] for c in CAT_NAMES]); BASE_W /= BASE_W.sum()
PREFIX = ["Blue", "Grand", "Nova", "Prime", "Urban", "Royal", "Eco", "Metro", "Alpha", "Sun", "Delta", "Orion",
          "Atlas", "Zen", "Terra", "Lumi", "Vega", "Mistral", "Cedar", "Harbor"]
SUFFIX = ["Market", "Group", "Store", "SA", "Ltd", "& Sons", "Co", "Express", "Plaza", "Hub", "Corner", "LLC"]


def _name(rng):
    return f"fraud_{rng.choice(PREFIX)} {rng.choice(SUFFIX)} {rng.integers(1, 999)}"


def build_merchants(rng, per_city_cat=6, online_per_cat=14):
    pos = {}
    for ci in range(len(CITIES)):
        for cat in CAT_NAMES:
            if cat in ONLINE_CATS:
                continue
            lat = CITY_LAT[ci] + rng.normal(0, 0.06, per_city_cat)
            lon = CITY_LON[ci] + rng.normal(0, 0.08, per_city_cat)
            pos[(ci, cat)] = [(_name(rng), float(a), float(b)) for a, b in zip(lat, lon)]
    online = {}
    for cat in ONLINE_CATS:
        hq = rng.choice(len(CITIES), online_per_cat)
        online[cat] = [(_name(rng), float(CITY_LAT[h]), float(CITY_LON[h])) for h in hq]
    return pos, online


def _pick(rng, lst, w=None):
    if w is None:
        w = 1.0 / np.arange(1, len(lst) + 1)          # Zipf : quelques marchands favoris
        w = w / w.sum()
    return lst[rng.choice(len(lst), p=w)]


def _local_hour_sample(rng, n):
    h = rng.choice(24, size=n, p=_HOUR_P)
    return h + rng.random(n)


_h = np.array([1, .5, .3, .2, .2, .4, 1, 2.5, 4, 5, 5.5, 6.5, 7.5, 6.5, 5.5, 5.5, 6, 7, 7.5, 7, 5.5, 4, 3, 2], float)
_HOUR_P = _h / _h.sum()


def simulate(n_cards, days, start_ts, seed, fraud_rate=0.006, trips_per_year=3.0):
    rng = np.random.default_rng(seed)
    pos_m, onl_m = build_merchants(rng)
    rows = []
    cards = []
    for k in range(n_cards):
        home = rng.choice(len(CITIES), p=CITY_W)
        hlat = float(CITY_LAT[home] + rng.normal(0, 0.05)); hlon = float(CITY_LON[home] + rng.normal(0, 0.06))
        age = int(np.clip(rng.normal(42, 15), 18, 85))
        dob = (pd.Timestamp(start_ts, unit="s") - pd.Timedelta(days=int(age * 365.25 + rng.integers(0, 365)))).strftime("%Y-%m-%d")
        cards.append((k, home, hlat, hlon, dob))
        rate = float(np.clip(rng.lognormal(np.log(0.7), 0.6), 0.1, 4.0))       # tx / jour
        cw = rng.dirichlet(BASE_W * 12 + 0.3)                                    # préférences de la carte
        scale = float(rng.lognormal(0, 0.35))
        n_tx = rng.poisson(rate * days)
        if n_tx == 0:
            continue
        # voyages : (début_jour, fin_jour, ville)
        trips = []
        for _ in range(rng.poisson(trips_per_year * days / 365)):
            s = rng.uniform(0, max(days - 3, 1)); dur = rng.uniform(2, 12)
            far = rng.choice(len(CITIES), p=CITY_W)
            if far != home:
                trips.append((s, min(s + dur, days), far))
        day = np.sort(rng.uniform(0, days, n_tx))
        city = np.full(n_tx, home)
        for s, e, c in trips:
            inside = (day >= s + 1.0) & (day <= e)               # 1 jour de trajet à l'aller
            city[inside] = c
            day[(day > s) & (day < s + 1.0)] = -1               # pas d'achat pendant le vol
            day[(day > e) & (day < e + 1.0)] = -1               # ni au retour
        keep = day >= 0
        day, city = day[keep], city[keep]
        cat_i = rng.choice(len(CAT_NAMES), size=len(day), p=cw)
        for d, c, ci in zip(day, city, cat_i):
            cat = CAT_NAMES[ci]
            if cat in ONLINE_CATS:
                m = _pick(rng, onl_m[cat])
            else:
                m = _pick(rng, pos_m[(c, cat)])
            lh = _local_hour_sample(rng, 1)[0]
            off = round(CITY_LON[c] / 15.0)
            ts = start_ts + int(np.floor(d)) * 86400 + int((lh - off) * 3600)
            mu, sg = CATS[cat][1], CATS[cat][2]
            a = float(np.exp(rng.normal(mu, sg)) * scale)
            a = round(max(a, 1.0), 2) if rng.random() > 0.05 else float(round(a / 10) * 10 or 10)
            rows.append((k, ts, a, cat, m[0], hlat, hlon, m[1], m[2], dob, 0))
    legit = pd.DataFrame(rows, columns=["card", "unix_time", "amt", "category", "merchant", "lat", "long",
                                        "merch_lat", "merch_long", "dob", "is_fraud"])
    legit = legit.sort_values(["card", "unix_time"]).reset_index(drop=True)
    fraud = _inject_fraud(rng, legit, cards, pos_m, onl_m, fraud_rate, start_ts, days)
    allr = pd.concat([legit, fraud], ignore_index=True)
    allr["cc_num"] = (4000_0000_0000_0000 + allr["card"] * 7919).astype("int64").astype(str)
    allr["tz_offset_h"] = np.round(allr["long"] / 15.0)
    allr = allr.drop(columns="card")
    return allr.sort_values("unix_time").reset_index(drop=True)


def _inject_fraud(rng, legit, cards, pos_m, onl_m, fraud_rate, start_ts, days):
    target = int(len(legit) * fraud_rate)
    out = []
    by_card = {k: g for k, g in legit.groupby("card")}
    ids = list(by_card)
    cinfo = {c[0]: c for c in cards}
    made = 0
    while made < target:
        k = ids[rng.integers(len(ids))]
        g = by_card[k]
        anchor = g.iloc[rng.integers(len(g))]
        _, home, hlat, hlon, dob = cinfo[k]
        typ = rng.choice(["clone_far", "cnp_burst", "card_test", "home_night"], p=[.30, .30, .15, .25])
        amax = float(g["amt"].max()); amed = float(g["amt"].median())

        def row(ts, amt, cat, m):
            return (k, int(ts), round(float(amt), 2), cat, m[0], hlat, hlon, m[1], m[2], dob, 1)

        rows = []
        if typ == "clone_far":
            dist = haversine_km(CITY_LAT, CITY_LON, anchor["merch_lat"], anchor["merch_long"])
            far = rng.choice(np.flatnonzero(dist > 2500))
            t = anchor["unix_time"] + rng.uniform(600, 3 * 3600)
            for _ in range(rng.integers(1, 5)):
                cat = rng.choice(["shopping_pos", "misc_pos", "gas_transport", "grocery_pos", "travel", "entertainment"])
                rows.append(row(t, amed * rng.uniform(2.5, 9), cat, _pick(rng, pos_m[(far, cat)], w=np.ones(6) / 6)))
                t += rng.uniform(120, 2400)
        elif typ == "cnp_burst":
            t0 = anchor["unix_time"]
            off = round(hlon / 15.0)
            t = t0 - (t0 + off * 3600) % 86400 + (rng.choice([23, 0, 1, 2, 3, 4]) - off) * 3600 + 86400 + rng.integers(0, 3000)
            for _ in range(rng.integers(2, 8)):
                cat = rng.choice(["shopping_net", "misc_net", "grocery_net"])
                rows.append(row(t, max(amed * rng.uniform(3, 12), amax * rng.uniform(0.8, 3)), cat, _pick(rng, onl_m[cat], w=np.ones(14) / 14)))
                t += rng.uniform(120, 5400)
        elif typ == "card_test":
            t = anchor["unix_time"] + rng.uniform(3600, 86400)
            for _ in range(rng.integers(3, 6)):
                cat = rng.choice(["shopping_net", "misc_net"])
                rows.append(row(t, rng.uniform(1, 6), cat, _pick(rng, onl_m[cat], w=np.ones(14) / 14)))
                t += rng.uniform(30, 600)
            cat = rng.choice(["shopping_net", "misc_net"])
            rows.append(row(t + 300, rng.uniform(300, 2500), cat, _pick(rng, onl_m[cat], w=np.ones(14) / 14)))
        else:   # home_night : local, nuit, montant au-dessus du max habituel
            off = round(hlon / 15.0)
            t0 = anchor["unix_time"]
            t = t0 - (t0 + off * 3600) % 86400 + (rng.choice([23, 0, 1, 2, 3]) - off) * 3600 + 86400 + rng.integers(0, 3000)
            for _ in range(rng.integers(1, 4)):
                cat = rng.choice(["shopping_pos", "misc_pos", "gas_transport", "grocery_pos"])
                rows.append(row(t, amax * rng.uniform(1.1, 3.5), cat, _pick(rng, pos_m[(home, cat)], w=np.ones(6) / 6)))
                t += rng.uniform(300, 3600)
        out.extend(rows)
        made += len(rows)
    return pd.DataFrame(out, columns=legit.columns)


def to_ingestion(df, with_ids=True):
    """Colonnes de Ingestion.csv ; `channel` = online si catégorie *_net."""
    d = df.copy()
    d["transaction_id"] = [f"sim{i:08d}" for i in range(len(d))]
    d["trans_date_trans_time"] = pd.to_datetime(d["unix_time"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")
    d["channel"] = np.where(d["category"].str.endswith("_net"), "online", "pos")
    d["age"] = np.round((pd.to_datetime(d["unix_time"], unit="s") - pd.to_datetime(d["dob"])).dt.days / 365.25).astype(int)
    d["city"] = ""
    cols = ["transaction_id", "cc_num", "trans_date_trans_time", "unix_time", "amt", "merchant", "category",
            "channel", "city", "age", "lat", "long", "merch_lat", "merch_long"]
    return d[cols], d[["transaction_id", "is_fraud"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_world")
    ap.add_argument("--cards", type=int, default=1500)
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--base", default=None, help="fraudTrain_features.csv à fusionner (recalculé avec les mêmes conventions)")
    ap.add_argument("--stream-cards", type=int, default=300)
    ap.add_argument("--stream-days", type=int, default=14)
    ap.add_argument("--history-days", type=int, default=60)
    ap.add_argument("--end-date", default="2026-10-11")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    start = int(pd.Timestamp("2025-06-01").timestamp())
    sim = simulate(a.cards, a.days, start, a.seed)
    sim["trans_num"] = [f"w{i:09d}" for i in range(len(sim))]
    parts = [sim]
    if a.base:
        base = pd.read_csv(a.base, usecols=["trans_num", "cc_num", "unix_time", "amt", "category", "merchant", "lat", "long",
                                              "merch_lat", "merch_long", "dob", "is_fraud"])
        base["tz_offset_h"] = 0          # unix_time de fraudTrain = heure locale
        base["cc_num"] = base["cc_num"].astype(str)
        parts.insert(0, base)
        # alignement temporel : la base US est décalée pour précéder la simulation
        shift = int(sim["unix_time"].min() - base["unix_time"].max() - 86400)
        base["unix_time"] = base["unix_time"] + shift
    raw = pd.concat(parts, ignore_index=True)
    feats, priors = build_features(raw)
    feats.to_csv(out / "world_train_features.csv", index=False)
    json.dump({"category_stats": priors["category_stats"], "merchant_counts": priors["merchant_counts"]},
              open(out / "priors.json", "w"))
    print(f"[train] {len(feats):,} tx, {feats['is_fraud'].sum():,} fraudes ({feats['is_fraud'].mean()*100:.3f} %)")

    # ---- flux d'ingestion : autres cartes, historique + derniers jours
    end = int(pd.Timestamp(a.end_date).timestamp())
    s0 = end - (a.history_days + a.stream_days) * 86400
    st = simulate(a.stream_cards, a.history_days + a.stream_days, s0, a.seed + 1, fraud_rate=0.01)
    cut = end - a.stream_days * 86400
    hist, stream = st[st["unix_time"] < cut], st[(st["unix_time"] >= cut) & (st["unix_time"] <= end)]
    ing, lab = to_ingestion(stream)
    hi, hl = to_ingestion(hist)
    hi["transaction_id"] = "h" + hi["transaction_id"]
    ing.to_csv(out / "Ingestion.csv", index=False)
    lab.to_csv(out / "Ingestion_labels.csv", index=False)
    hi.drop(columns=["unix_time"]).to_csv(out / "History.csv", index=False)
    print(f"[stream] history={len(hi):,} stream={len(ing):,} fraudes stream={int(lab['is_fraud'].sum())}")


if __name__ == "__main__":
    main()