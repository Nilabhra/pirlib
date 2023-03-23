import json

import numpy as np
import pandas as pd


def build_features(raw, census, co_est, window_lens, target="microbusiness_density", lags=6):
    feats = []
    for lag in range(1, lags):
        raw[f"mbd_lag_{lag}"] = raw.groupby("cfips")[target].shift(lag)
        feats.append(f"mbd_lag_{lag}")

    lag = 1
    for window in window_lens:
        raw[f"mbd_rollmea{window}_{lag}"] = raw.groupby("cfips")[f"mbd_lag_{lag}"].transform(
            lambda s: s.rolling(window, min_periods=1).sum()
        )
        feats.append(f"mbd_rollmea{window}_{lag}")

    census_columns = list(census.columns)
    census_columns.remove("cfips")

    raw = raw.merge(census, on="cfips", how="left")
    feats += census_columns

    co_est["cfips"] = co_est.STATE * 1000 + co_est.COUNTY
    co_columns = [
        "SUMLEV",
        "DIVISION",
        "ESTIMATESBASE2020",
        "POPESTIMATE2020",
        "POPESTIMATE2021",
        "NPOPCHG2020",
        "NPOPCHG2021",
        "BIRTHS2020",
        "BIRTHS2021",
        "DEATHS2020",
        "DEATHS2021",
        "NATURALCHG2020",
        "NATURALCHG2021",
        "INTERNATIONALMIG2020",
        "INTERNATIONALMIG2021",
        "DOMESTICMIG2020",
        "DOMESTICMIG2021",
        "NETMIG2020",
        "NETMIG2021",
        "RESIDUAL2020",
        "RESIDUAL2021",
        "GQESTIMATESBASE2020",
        "GQESTIMATES2020",
        "GQESTIMATES2021",
        "RBIRTH2021",
        "RDEATH2021",
        "RNATURALCHG2021",
        "RINTERNATIONALMIG2021",
        "RDOMESTICMIG2021",
        "RNETMIG2021",
    ]
    raw = raw.merge(co_est, on="cfips", how="left")
    feats += co_columns
    return raw, feats


def rot(df):
    rot_feats = []
    for angle in [15, 30, 45]:
        xfeat = f"rot_{angle}_x"
        df[xfeat] = (np.cos(np.radians(angle)) * df["lat"]) + (
            np.sin(np.radians(angle)) * df["lng"]
        )

        yfeat = f"rot_{angle}_y"
        df[yfeat] = (np.cos(np.radians(angle)) * df["lat"]) - (
            np.sin(np.radians(angle)) * df["lng"]
        )

        rot_feats.append(xfeat)
        rot_feats.append(yfeat)

    return df, rot_feats


def engineer_features(raw_file, census_file, indicator_file, hparam_file):
    raw = pd.read_csv(raw_file)
    census = pd.read_csv(census_file)
    co_est = pd.read_csv(indicator_file, encoding="latin-1")

    with open(hparam_file, "r") as f:
        hparams = json.load(f)

    window_lens = hparams["window_lens"]
    lag = hparams["lag"]

    raw, feats = build_features(raw, census, co_est, window_lens=window_lens, lags=lag)

    features = ["state_i"]
    features += feats
    features += ["lng", "lat", "scale"]

    coordinates = raw[["lng", "lat"]].values

    # Encoding tricks
    emb_size = 20
    precision = 1e6

    latlon = np.expand_dims(coordinates, axis=-1)

    m = np.exp(np.log(precision) / emb_size)
    angle_freq = m ** np.arange(emb_size)
    angle_freq = angle_freq.reshape(1, 1, emb_size)
    latlon = latlon * angle_freq
    latlon[..., 0::2] = np.cos(latlon[..., 0::2])

    raw, rot_feats = rot(raw)

    features += rot_feats

    return raw, features
