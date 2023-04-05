from math import sqrt
from typing import Any, Dict

import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

NFOLDS = 3
SEED = 0
NROWS = None


class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params["random_state"] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict_proba(x)[:, 1]


class CatboostWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params["random_seed"] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict_proba(x)[:, 1]


class LightGBMWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params["feature_fraction_seed"] = seed
        params["bagging_seed"] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict_proba(x)[:, 1]


class XgbWrapper(object):
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param["seed"] = seed
        self.nrounds = params.pop("nrounds", 250)

    def train(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))


def get_oof(clf, x_train, y_train, x_test, n_folds=3):
    oof_train = np.zeros((x_train.shape[0],))
    oof_test = np.zeros((x_test.shape[0],))
    oof_test_skf = np.empty((NFOLDS, x_test.shape[0]))

    kf = KFold(n_splits=n_folds, shuffle=True)
    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train.loc[train_index]
        y_tr = y_train.loc[train_index]
        x_te = x_train.loc[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


def train_basemodels(
    train: pd.DataFrame, test: pd.DataFrame, params: Dict[str, Any]
) -> Dict[str, Any]:
    # Initiate the meta models.
    xg = XgbWrapper(seed=SEED, params=params["xgb_params"])
    et = SklearnWrapper(clf=ExtraTreesClassifier, seed=SEED, params=params["et_params"])
    rf = SklearnWrapper(clf=RandomForestClassifier, seed=SEED, params=params["rf_params"])
    cb = CatboostWrapper(clf=CatBoostClassifier, seed=SEED, params=params["catboost_params"])

    # Separate features from targets.
    y_train = train["TARGET"]
    y_test = test["TARGET"]
    x_train = train.drop("TARGET", axis=1)
    x_test = test.drop("TARGET", axis=1)

    # Train the meta modles.
    xg_oof_train, xg_oof_test = get_oof(xg, x_train, y_train, x_test)
    et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test)
    rf_oof_train, rf_oof_test = get_oof(rf, x_train, y_train, x_test)
    cb_oof_train, cb_oof_test = get_oof(cb, x_train, y_train, x_test)

    x_train = np.concatenate((xg_oof_train, et_oof_train, rf_oof_train, cb_oof_train), axis=1)
    x_test = np.concatenate((xg_oof_test, et_oof_test, rf_oof_test, cb_oof_test), axis=1)

    return {"train": x_train, "train_targets": y_train, "test": x_test, "test_targets": y_test}


def train_metamodel(
    train: np.ndarray,
    train_targets: np.ndarray,
    test: np.ndarray,
    test_targets: np.ndarray,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    logistic_regression = LogisticRegression(**params["lr_params"])
    logistic_regression.fit(train, train_targets)

    test_preds = logistic_regression.predict_proba(test)[:, 1]
    au_roc = roc_auc_score(test_targets, test_preds)

    return {"AU_ROC": au_roc}
