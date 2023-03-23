import numpy as np
import pandas as pd


def preprocessing(train_file, reveal_test_file, test_file, sub_file, cfips_file):
    train = pd.read_csv(train_file)
    reveal_test = pd.read_csv(reveal_test_file)
    train = (
        pd.concat([train, reveal_test])
        .sort_values(by=["cfips", "first_day_of_month"])
        .reset_index()
    )
    test = pd.read_csv(test_file)
    drop_index = (test.first_day_of_month == "2022-11-01") | (
        test.first_day_of_month == "2022-12-01"
    )
    test = test.loc[~drop_index, :]

    sub = pd.read_csv(sub_file)
    coords = pd.read_csv(cfips_file)
    print(train.shape, test.shape, sub.shape)

    train["istest"] = 0
    test["istest"] = 1
    raw = pd.concat((train, test)).sort_values(["cfips", "row_id"]).reset_index(drop=True)
    raw = raw.merge(coords.drop("name", axis=1), on="cfips")

    raw["state_i1"] = raw["state"].astype("category")
    raw["county_i1"] = raw["county"].astype("category")
    raw["first_day_of_month"] = pd.to_datetime(raw["first_day_of_month"])
    raw["county"] = raw.groupby("cfips")["county"].ffill()
    raw["state"] = raw.groupby("cfips")["state"].ffill()
    raw["dcount"] = raw.groupby(["cfips"])["row_id"].cumcount()
    raw["county_i"] = (raw["county"] + raw["state"]).factorize()[0]
    raw["state_i"] = raw["state"].factorize()[0]
    raw["scale"] = (raw["first_day_of_month"] - raw["first_day_of_month"].min()).dt.days
    raw["scale"] = raw["scale"].factorize()[0]

    # Remove anomalies.
    for o in raw.cfips.unique():
        indices = raw["cfips"] == o
        tmp = raw.loc[indices].copy().reset_index(drop=True)
        var = tmp.microbusiness_density.values.copy()
        for i in range(37, 2, -1):
            thr = 0.10 * np.mean(var[:i])
            difa = var[i] - var[i - 1]
            if (difa >= thr) or (difa <= -thr):
                if difa > 0:
                    var[:i] += difa - 0.003
                else:
                    var[:i] += difa + 0.003
        var[0] = var[1] * 0.99
        raw.loc[indices, "microbusiness_density"] = var

    lag = 1
    raw[f"mbd_lag_{lag}"] = raw.groupby("cfips")["microbusiness_density"].shift(lag).bfill()
    raw["dif"] = (raw["microbusiness_density"] / raw[f"mbd_lag_{lag}"]).fillna(1).clip(0, None) - 1
    raw.loc[(raw[f"mbd_lag_{lag}"] == 0), "dif"] = 0
    raw.loc[(raw[f"microbusiness_density"] > 0) & (raw[f"mbd_lag_{lag}"] == 0), "dif"] = 1
    raw["dif"] = raw["dif"].abs()

    raw["target"] = raw.groupby("cfips")["microbusiness_density"].shift(-1)

    # Convert targets.
    raw["target"] = raw["target"] / raw["microbusiness_density"] - 1

    raw.loc[raw["cfips"] == 28055, "target"] = 0.0
    raw.loc[raw["cfips"] == 48269, "target"] = 0.0

    raw["lastactive"] = raw.groupby("cfips")["active"].transform("last")

    return raw
