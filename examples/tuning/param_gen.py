import copy
import json
import random

base_model_params = {
    "lgbm": {
        "n_iter": 300,
        "boosting_type": "dart",
        "verbosity": -1,
        "objective": "l1",
        "random_state": 42,
        "colsample_bytree": 0.8841279649367693,
        "colsample_bynode": 0.10142964450634374,
        "max_depth": 8,
        "learning_rate": 0.003647749926797374,
        "lambda_l2": 0.5,
        "num_leaves": 61,
        "seed": 42,
        "min_data_in_leaf": 213,
    },
    "xgb": {
        "objective": "reg:pseudohubererror",
        "tree_method": "hist",
        "n_estimators": 795,
        "learning_rate": 0.0075,
        "max_leaves": 17,
        "subsample": 0.20,
        "colsample_bytree": 0.50,
        "max_bin": 4096,
        "n_jobs": 2,
    },
    "cat": {
        "iterations": 2000,
        "loss_function": "MAPE",
        "verbose": 0,
        "grow_policy": "SymmetricTree",
        "learning_rate": 0.035,
        "colsample_bylevel": 0.8,
        "max_depth": 5,
        "l2_leaf_reg": 0.2,
        "subsample": 0.70,
        "max_bin": 4096,
    },
}


def write_hparams(hparams, filename):
    base_dir = "./data"
    with open(f"{base_dir}/{filename}", "w") as f:
        json.dump(hparams, f)


def generate_preprocess_hparams():
    return {"no_params": "no_values"}


def generate_fe_hparams():
    lags = [6, 9]
    window_lens = [[2, 4, 6, 8, 10], [6, 8, 20]]

    hparams = {}
    hparams["lag"] = random.choice(lags)
    hparams["window_lens"] = random.choice(window_lens)

    return hparams


def generate_base_model_hparams():
    lgbm_max_depth = [7, 8]
    xgb_subsample = [0.1, 0.2]
    cat_max_depth = [3, 5]

    hparams = copy.copy(base_model_params)
    hparams["lgbm"]["max_depth"] = random.choice(lgbm_max_depth)
    hparams["xgb"]["subsample"] = random.choice(xgb_subsample)
    hparams["cat"]["max_depth"] = random.choice(cat_max_depth)

    return hparams


def generate_meta_model_hparams():
    cat_max_depth = [3, 5]
    hparams = copy.copy(base_model_params["cat"])
    hparams["max_depth"] = random.choice(cat_max_depth)
    return hparams


def generate_hparams():
    write_hparams(generate_preprocess_hparams(), "preprocess_hp.json")
    write_hparams(generate_fe_hparams(), "fe_hp.json")
    write_hparams(generate_base_model_hparams(), "base_model_hp.json")
    write_hparams(generate_meta_model_hparams(), "meta_model_hp.json")


if __name__ == "__main__":
    generate_hparams()
