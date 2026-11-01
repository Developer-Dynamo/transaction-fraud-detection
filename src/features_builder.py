"""Construction des 46 features FraudGuard à partir de transactions BRUTES.

Même définition que fraudTrain_features.csv (conventions vérifiées ligne par
ligne sur la tranche fournie), avec deux choix explicites pour le monde entier :
  * achat EN LIGNE (catégorie *_net) : dist_home_merch = 0, dist_from_prev = 0,
    speed = 0 (comme l'API) ; la « position précédente » est la dernière
    transaction PHYSIQUE de la carte ;
  * heure locale = heure UTC + tz_offset_h (colonne facultative ; défaut
    round(long/15)). Mettre tz_offset_h = 0 pour fraudTrain (unix_time y est
    déjà une heure locale).

Colonnes brutes requises : cc_num, unix_time, amt, category, merchant, lat,
long, merch_lat, merch_long, dob (AAAA-MM-JJ), [is_fraud], [trans_num].
"""
from __future__ import annotations
import numpy as np
import pandas as pd

ONLINE_SUFFIX = "_net"
FEATURE_COLUMNS = [
    "hour", "day_of_week", "month", "is_weekend", "is_night", "hour_sin", "hour_cos", "age",
    "log_amt", "is_round_amt", "amt_vs_category_z", "amt_to_category_mean", "dist_home_merch_km",
    "tx_rank", "is_first_tx", "time_since_last_tx", "log_time_since_last_tx", "time_since_first_tx",
    "dist_from_prev_merch_km", "speed_kmh", "card_mean_amt_prev", "card_std_amt_prev",
    "card_max_amt_prev", "amt_zscore_card", "amt_to_card_mean", "amt_to_card_max", "is_new_max_amt",
    "card_mean_dist_prev_km", "dist_to_card_mean", "nb_prev_same_merchant", "is_new_merchant",
    "is_new_category", "category_share_card", "hour_share_card", "merchant_freq",
    "nb_tx_1h", "sum_amt_1h", "nb_tx_6h", "sum_amt_6h", "nb_tx_24h", "sum_amt_24h",
    "nb_tx_7d", "sum_amt_7d", "amt_share_of_24h", "tx_accel_1h_vs_24h",
]
# Ordre exact des colonnes de fraudTrain_features.csv
OUTPUT_COLUMNS = [
    "trans_num", "cc_num", "unix_time", "amt", "category", "merchant", "lat", "long",
    "merch_lat", "merch_long", "dob", "hour", "day_of_week", "month", "is_weekend", "is_night",
    "hour_sin", "hour_cos", "age", "log_amt", "is_round_amt", "amt_vs_category_z",
    "amt_to_category_mean", "dist_home_merch_km", "tx_rank", "is_first_tx", "time_since_last_tx",
    "log_time_since_last_tx", "time_since_first_tx", "dist_from_prev_merch_km", "speed_kmh",
    "card_mean_amt_prev", "card_std_amt_prev", "card_max_amt_prev", "amt_zscore_card",
    "amt_to_card_mean", "amt_to_card_max", "is_new_max_amt", "card_mean_dist_prev_km",
    "dist_to_card_mean", "nb_prev_same_merchant", "is_new_merchant", "is_new_category",
    "category_share_card", "hour_share_card", "merchant_freq", "nb_tx_1h", "sum_amt_1h",
    "nb_tx_6h", "sum_amt_6h", "nb_tx_24h", "sum_amt_24h", "nb_tx_7d", "sum_amt_7d",
    "amt_share_of_24h", "tx_accel_1h_vs_24h", "is_fraud",
]


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _window_stats(ts: np.ndarray, amt: np.ndarray, delta_s: float):
    """Pour un tableau trié `ts` : nb et somme des tx dans [ts_i - delta, ts_i) (causal)."""
    left = np.searchsorted(ts, ts - delta_s, side="left")
    idx = np.arange(len(ts))
    csum = np.concatenate([[0.0], np.cumsum(amt)])
    return (idx - left).astype(float), csum[idx] - csum[left]


def build_features(raw: pd.DataFrame, category_stats: dict | None = None,
                   merchant_counts: dict | None = None, online_suffix: str | None = ONLINE_SUFFIX):
    """Retourne (df_features_au_format_CSV, priors). `category_stats` /
    `merchant_counts` : à fournir pour réutiliser ceux de l'entraînement
    (serving) ; sinon calculés sur `raw`."""
    df = raw.copy()
    if "tz_offset_h" not in df:
        df["tz_offset_h"] = np.round(df["long"] / 15.0)
    if "is_fraud" not in df:
        df["is_fraud"] = 0
    if "trans_num" not in df:
        df["trans_num"] = [f"tx{i:09d}" for i in range(len(df))]
    df = df.sort_values(["cc_num", "unix_time"], kind="mergesort").reset_index(drop=True)

    ts = df["unix_time"].to_numpy(dtype="int64")
    amt = df["amt"].to_numpy(dtype=float)
    local = pd.to_datetime(ts + (df["tz_offset_h"].to_numpy() * 3600).astype("int64"), unit="s")
    hour = local.hour.to_numpy()
    df["hour"] = hour
    df["day_of_week"] = local.dayofweek.to_numpy()
    df["month"] = local.month.to_numpy()
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_night"] = ((hour >= 22) | (hour < 6)).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    dob = pd.to_datetime(df["dob"])
    df["age"] = np.round((pd.to_datetime(ts, unit="s") - dob).dt.days / 365.25, 2)
    df["log_amt"] = np.log1p(np.maximum(amt, 0))
    df["is_round_amt"] = (np.abs(amt - np.round(amt / 10) * 10) < 1e-6).astype(int)

    # --- priors globaux (catégorie, marchand)
    if category_stats is None:
        g = df.groupby("category")["amt"].agg(["mean", "std"])
        category_stats = {c: (float(r["mean"]), float(r["std"])) for c, r in g.iterrows()}
    if merchant_counts is None:
        merchant_counts = df["merchant"].value_counts().to_dict()
    cm = df["category"].map(lambda c: category_stats.get(c, (np.nan, np.nan))[0]).to_numpy(float)
    cs = df["category"].map(lambda c: category_stats.get(c, (np.nan, np.nan))[1]).to_numpy(float)
    df["amt_vs_category_z"] = np.where(cs > 0, (amt - cm) / cs, 0.0)
    df["amt_to_category_mean"] = np.where(cm > 0, amt / cm, np.nan)
    df["merchant_freq"] = df["merchant"].map(merchant_counts).fillna(0).astype(float)

    # --- géographie
    online = (df["category"].str.endswith(online_suffix).to_numpy() if online_suffix
              else np.zeros(len(df), dtype=bool))
    d_home = haversine_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"]).to_numpy()
    d_home = np.where(online, 0.0, d_home)
    df["dist_home_merch_km"] = d_home

    n = len(df)
    card = df["cc_num"].to_numpy()
    starts = np.r_[0, np.flatnonzero(card[1:] != card[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    cols = {k: np.zeros(n) for k in [
        "tx_rank", "is_first_tx", "time_since_last_tx", "log_time_since_last_tx", "time_since_first_tx",
        "dist_from_prev_merch_km", "speed_kmh", "card_mean_amt_prev", "card_std_amt_prev",
        "card_max_amt_prev", "amt_zscore_card", "amt_to_card_mean", "amt_to_card_max",
        "is_new_max_amt", "card_mean_dist_prev_km", "dist_to_card_mean", "nb_tx_1h", "sum_amt_1h",
        "nb_tx_6h", "sum_amt_6h", "nb_tx_24h", "sum_amt_24h", "nb_tx_7d", "sum_amt_7d"]}
    mlat = df["merch_lat"].to_numpy(float)
    mlon = df["merch_long"].to_numpy(float)
    for s, e in zip(starts, ends):
        t, a, d = ts[s:e], amt[s:e], d_home[s:e]
        k = e - s
        i = np.arange(k)
        cols["tx_rank"][s:e] = i
        cols["is_first_tx"][s:e] = (i == 0)
        gap = np.r_[-1.0, np.diff(t).astype(float)]
        cols["time_since_last_tx"][s:e] = gap
        cols["log_time_since_last_tx"][s:e] = np.where(i == 0, -1.0, np.log1p(np.maximum(gap, 0)))
        cols["time_since_first_tx"][s:e] = (t - t[0]).astype(float)
        # moyennes / écarts-types / max sur l'historique strict (1re tx : sa propre valeur)
        cs1 = np.concatenate([[0.0], np.cumsum(a)])
        cs2 = np.concatenate([[0.0], np.cumsum(a * a)])
        cd = np.concatenate([[0.0], np.cumsum(d)])
        cnt = np.maximum(i, 1).astype(float)
        mean_prev = np.where(i == 0, a, cs1[i] / cnt)
        var_prev = np.where(i >= 2, (cs2[i] - cnt * (cs1[i] / cnt) ** 2) / np.maximum(i - 1, 1), 0.0)
        std_prev = np.sqrt(np.maximum(var_prev, 0.0))
        max_prev = np.where(i == 0, a, np.maximum.accumulate(np.r_[a[0], a[:-1]]))
        cols["card_mean_amt_prev"][s:e] = mean_prev
        cols["card_std_amt_prev"][s:e] = std_prev
        cols["card_max_amt_prev"][s:e] = max_prev
        ratio_mean = np.where(mean_prev > 0, a / np.where(mean_prev > 0, mean_prev, 1), 1.0)
        z = np.where(std_prev > 0, (a - mean_prev) / np.where(std_prev > 0, std_prev, 1), 0.0)
        cols["amt_zscore_card"][s:e] = np.where(i == 0, 0.0, z)
        cols["amt_to_card_mean"][s:e] = np.where(i == 0, 1.0, ratio_mean)
        cols["amt_to_card_max"][s:e] = np.where(i == 0, 1.0, a / np.where(max_prev > 0, max_prev, 1))
        cols["is_new_max_amt"][s:e] = np.where(i == 0, 0.0, (a > max_prev).astype(float))
        dmean = np.where(i == 0, d, cd[i] / cnt)
        cols["card_mean_dist_prev_km"][s:e] = dmean
        cols["dist_to_card_mean"][s:e] = np.where((i == 0) | (dmean <= 0), 1.0,
                                                 d / np.where(dmean > 0, dmean, 1))
        # vélocité causale
        for lab, delta in [("1h", 3600), ("6h", 6 * 3600), ("24h", 86400), ("7d", 7 * 86400)]:
            nb, sm = _window_stats(t.astype(float), a, float(delta))
            cols[f"nb_tx_{lab}"][s:e] = nb
            cols[f"sum_amt_{lab}"][s:e] = sm
        # distance / vitesse depuis la dernière tx PHYSIQUE
        phys = ~online[s:e]
        last_idx = -1
        dprev = np.full(k, -1.0)
        spd = np.full(k, -1.0)
        la, lo = mlat[s:e], mlon[s:e]
        for j in range(k):
            if j == 0:
                pass
            elif not phys[j]:
                dprev[j] = 0.0
                spd[j] = 0.0
            elif last_idx >= 0:
                dd = float(haversine_km(la[last_idx], lo[last_idx], la[j], lo[j]))
                hrs = (t[j] - t[last_idx]) / 3600.0
                dprev[j] = dd
                spd[j] = dd / hrs if hrs > 0 else 0.0
            if phys[j]:
                last_idx = j
        cols["dist_from_prev_merch_km"][s:e] = dprev
        cols["speed_kmh"][s:e] = spd
    for k_, v in cols.items():
        df[k_] = v
    # habitudes (cumcount = nb d'occurrences strictement antérieures)
    g = df.groupby(["cc_num", "merchant"]).cumcount()
    df["nb_prev_same_merchant"] = g.astype(float)
    df["is_new_merchant"] = (g == 0).astype(float)
    gc = df.groupby(["cc_num", "category"]).cumcount()
    df["is_new_category"] = (gc == 0).astype(float)
    rank = df["tx_rank"].to_numpy()
    df["category_share_card"] = np.where(rank > 0, gc / np.maximum(rank, 1), 0.0)
    gh = df.groupby(["cc_num", "hour"]).cumcount()
    df["hour_share_card"] = np.where(rank > 0, gh / np.maximum(rank, 1), 0.0)
    denom = df["sum_amt_24h"] + df["amt"]
    df["amt_share_of_24h"] = np.where(denom > 0, df["amt"] / denom, 0.0)
    df["tx_accel_1h_vs_24h"] = df["nb_tx_1h"] / ((df["nb_tx_24h"] + 1.0) / 24.0)
    df["is_fraud"] = df["is_fraud"].astype(int)
    out = df.sort_values("unix_time", kind="mergesort").reset_index(drop=True)
    return out[OUTPUT_COLUMNS], {"category_stats": category_stats, "merchant_counts": merchant_counts}


if __name__ == "__main__":
    # Validation : recalcule les features de la tranche fournie et compare.
    import sys
    ref = pd.read_csv(sys.argv[1])
    raw = ref[["trans_num", "cc_num", "unix_time", "amt", "category", "merchant", "lat", "long",
               "merch_lat", "merch_long", "dob", "is_fraud"]].copy()
    raw["tz_offset_h"] = 0          # unix_time de fraudTrain = heure locale
    # les features d'une catégorie en ligne suivent la nouvelle convention (dist=0) :
    # on compare donc seulement les lignes physiques.
    out, _ = build_features(raw, online_suffix=None)   # fraudTrain : tout est « physique »
    out = out.set_index("trans_num").loc[ref["trans_num"]].reset_index()
    skip = {"amt_vs_category_z", "amt_to_category_mean", "merchant_freq"}
    phys = np.ones(len(ref), dtype=bool)
    bad = 0
    for c in FEATURE_COLUMNS:
        if c in skip:
            continue
        a, b = ref[c].to_numpy(float)[phys], out[c].to_numpy(float)[phys]
        diff = np.nanmax(np.abs(a - b)) if len(a) else 0
        ok = diff < 1e-5 * max(1.0, np.nanmax(np.abs(a)))
        if not ok:
            bad += 1
            print(f"ECART {c}: max|diff|={diff:.6g}")
    print(f"{len(ref)} lignes ({phys.sum()} physiques) ; colonnes en écart : {bad}")