import json

import catboost as cat
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

ACT_THR = 150


def smape(y_true, y_pred):
    smap = np.zeros(len(y_true))

    num = np.abs(y_true - y_pred)
    dem = (np.abs(y_true) + np.abs(y_pred)) / 2

    pos_ind = (y_true != 0) | (y_pred != 0)
    smap[pos_ind] = num[pos_ind] / dem[pos_ind]

    return 100 * np.mean(smap)


def get_base_models(hparams):
    # LGBM model

    lgb_model = lgb.LGBMRegressor(**hparams["lgbm"])

    xgb_model = xgb.XGBRegressor(**hparams["xgb"])

    cat_model = cat.CatBoostRegressor(**hparams["cat"])

    models = {}
    models["xgb"] = xgb_model
    models["lgbm"] = lgb_model
    models["cat"] = cat_model

    return models


def get_meta_model(hparams):
    cat_model = cat.CatBoostRegressor(**hparams)
    return cat_model


def get_base_model_preds(raw_file, hparam_file, features_file):
    TS = 40
    raw = pd.read_csv(raw_file)

    with open(hparam_file, "r") as f:
        hparams = json.load(f)

    features = joblib.load(features_file)

    train_indices = (
        (raw.istest == 0) & (raw.dcount < TS) & (raw.dcount >= 1) & (raw.lastactive > ACT_THR)
    )
    valid_indices = raw.dcount == TS

    # Train the base models
    models = get_base_models(hparams)
    model0 = models["lgbm"]
    model1 = models["xgb"]
    model2 = models["cat"]

    model0.fit(
        raw.loc[train_indices, features], raw.loc[train_indices, "target"].clip(-0.002, 0.006)
    )

    model1.fit(
        raw.loc[train_indices, features], raw.loc[train_indices, "target"].clip(-0.002, 0.006)
    )

    model2.fit(
        raw.loc[train_indices, features], raw.loc[train_indices, "target"].clip(-0.002, 0.006)
    )

    tr_pred0 = model0.predict(raw.loc[train_indices, features])
    tr_pred1 = model1.predict(raw.loc[train_indices, features])
    tr_pred2 = model2.predict(raw.loc[train_indices, features])
    train_preds = np.column_stack((tr_pred0, tr_pred1, tr_pred2))

    val_preds0 = model0.predict(raw.loc[valid_indices, features])
    val_preds1 = model1.predict(raw.loc[valid_indices, features])
    val_preds2 = model2.predict(raw.loc[valid_indices, features])
    valid_preds = np.column_stack((val_preds0, val_preds1, val_preds2))

    return train_preds, valid_preds


def get_meta_model_preds(raw_file, hparams_file, train_pred_file, valid_pred_file):
    TS = 40

    raw = pd.read_csv(raw_file)

    with open(hparams_file, "r") as f:
        hparams = json.load(f)

    train_preds = np.load(train_pred_file)
    valid_preds = np.load(valid_pred_file)

    train_indices = (
        (raw.istest == 0) & (raw.dcount < TS) & (raw.dcount >= 1) & (raw.lastactive > ACT_THR)
    )
    valid_indices = raw.dcount == TS

    meta_model = get_meta_model(hparams)
    meta_model.fit(train_preds, raw.loc[train_indices, "target"].clip(-0.002, 0.006))

    ypred = meta_model.predict(valid_preds)

    valid_smape = smape(raw.loc[valid_indices, "target"], ypred)
    print("Validaton SMAPE:", valid_smape)
    return ypred
