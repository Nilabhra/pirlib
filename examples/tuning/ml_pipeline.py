import pickle

import numpy as np

from pirlib.iotypes import DirectoryPath, FilePath
from pirlib.pipeline import pipeline
from pirlib.task import task

from .feature_engineering import engineer_features
from .modelling import get_base_model_preds, get_meta_model_preds
from .preprocessing import preprocessing


@task(cache=True, cache_key_file="hparams")
def preprocess(data_dir: DirectoryPath, *, hparams: FilePath) -> DirectoryPath:
    # File paths for inputs.
    train_file = data_dir / "train.csv"
    reveal_test_file = data_dir / "revealed_test.csv"
    test_file = data_dir / "test.csv"
    sub_file = data_dir / "sample_submission.csv"
    cfips_file = data_dir / "cfips_location.csv"

    processed_data = preprocessing(train_file, reveal_test_file, test_file, sub_file, cfips_file)

    output_dir = task.context().output

    processed_data.to_csv(output_dir / "raw.csv", index=False)

    return output_dir


@task(cache=True, cache_key_file="hparams")
def feature_engineer(
    data_dir: DirectoryPath, preprocess_dir: DirectoryPath, *, hparams: FilePath
) -> DirectoryPath:
    census_file = data_dir / "census_starter.csv"
    raw_file = preprocess_dir / "raw.csv"
    co_indicator = data_dir / "co-est2021-alldata.csv"

    engineered_data, features = engineer_features(raw_file, census_file, co_indicator, hparams)

    output_dir = task.context().output

    engineered_op = output_dir / "raw_engineered.csv"
    engineered_data.to_csv(engineered_op, index=False)

    with open(output_dir / "features.pkl", "wb") as f:
        pickle.dump(features, f)

    return output_dir


@task(cache=True, cache_key_file="hparams")
def base_model_training(engineered_data: DirectoryPath, *, hparams: FilePath) -> DirectoryPath:
    raw_file = engineered_data / "raw_engineered.csv"
    features_file = engineered_data / "features.pkl"

    train_preds, valid_preds = get_base_model_preds(raw_file, hparams, features_file)

    output_dir = task.context().output

    # Save predictions.
    np.save(output_dir / "train_preds.npy", train_preds)
    np.save(output_dir / "valid_preds.npy", valid_preds)

    return output_dir


@task(cache=True, cache_key_file="hparams")
def meta_model_training(
    engineered_data: DirectoryPath, base_model_output: DirectoryPath, *, hparams: FilePath
) -> DirectoryPath:
    raw_file = engineered_data / "raw_engineered.csv"
    train_pred_file = base_model_output / "train_preds.npy"
    valid_pred_file = base_model_output / "valid_preds.npy"

    meta_valid_preds = get_meta_model_preds(raw_file, hparams, train_pred_file, valid_pred_file)

    output_dir = task.context().output

    # Save final predictions.
    np.save(output_dir / "valid_preds_meta.npy", meta_valid_preds)

    return output_dir


@pipeline
def ml_job(
    raw_data: DirectoryPath,
    preproc_hp: FilePath,
    fe_hp: FilePath,
    base_model_hp: FilePath,
    meta_model_hp: FilePath,
) -> DirectoryPath:
    # Preprocess.
    preproc_data = preprocess(raw_data, preproc_hp)

    # Feature engineer.
    fe_data = feature_engineer(raw_data, preproc_data, fe_hp)

    # Train base models.
    base_model_preds = base_model_training(fe_data, base_model_hp)

    # Train meta models.
    meta_model_preds = meta_model_training(fe_data, base_model_preds, meta_model_hp)

    return meta_model_preds
