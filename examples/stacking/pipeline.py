import json
import pickle

import pandas as pd

from pirlib.iotypes import DirectoryPath, FilePath
from pirlib.pipeline import pipeline
from pirlib.task import task

from .data_processing import prepare_data
from .modelling import train_basemodels, train_metamodel


@task(cache=True, cache_key_file="hparams", timer=True)
def preprocess(
    train_path: FilePath, prev_app_path: FilePath, test_path: FilePath, *, hparams: FilePath
) -> DirectoryPath:
    # Read the train and test data.
    train = pd.read_csv(train_path)
    prev = pd.read_csv(prev_app_path)
    test = pd.read_csv(test_path)

    # Prepare data for training and evaluation.
    train = prepare_data(train, prev)
    test = prepare_data(test, prev)

    # Write preprocessed data to the output directory.
    output_dir = task.context().output
    train.to_csv(output_dir / "train_preprocessed.csv", index=False)
    test.to_csv(output_dir / "test_preprocessed.csv", index=False)

    return output_dir


@task(cache=True, cache_key_file="hparams", timer=True)
def build_basemodels(data_dir: DirectoryPath, *, hparams: FilePath) -> DirectoryPath:
    # Read preprocessed data.
    train = pd.read_csv(data_dir / "train_preprocessed.csv")
    test = pd.read_csv(data_dir / "test_preprocessed.csv")

    # Read hyperparameters.
    with hparams.open() as f:
        hp = json.load(f)

    # Train the base models.
    oof_data = train_basemodels(train, test, hp)

    # Write the OOF data to disk.
    output_dir = task.context().output
    with (output_dir / "oof_data.pkl").open("wb") as f:
        pickle.dump(oof_data, f)

    return output_dir


@task(cache=True, cache_key_file="hparams", timer=True)
def build_metamodel(data_dir: DirectoryPath, *, hparams: FilePath) -> DirectoryPath:
    # Read the OOF predictions.
    with (data_dir / "oof_data.pkl").open("rb") as f:
        oof_data = pickle.load(f)

    # Extract data fields.
    x_train = oof_data["train"]
    y_train = oof_data["train_targets"]

    x_test = oof_data["test"]
    y_test = oof_data["test_targets"]

    # Read hyperparameters.
    with hparams.open("r") as f:
        hp = json.load(f)

    # Train the metamodel.
    au_roc = train_metamodel(x_train, y_train, x_test, y_test, hp)

    # Write test AU ROC to disk.
    output_dir = task.context().output
    with (output_dir / "score.json").open("w") as f:
        json.dump(au_roc, f)

    return output_dir


@pipeline
def ml_job(
    train_path: FilePath,
    prev_app_path: FilePath,
    test_path: FilePath,
    preproc_hp: FilePath,
    base_model_hp: FilePath,
    meta_model_hp: FilePath,
) -> DirectoryPath:
    # Preprocess.
    preprocessed_data = preprocess(train_path, prev_app_path, test_path, hparams=preproc_hp)

    # Train base models.
    base_model_op = build_basemodels(preprocessed_data, hparams=base_model_hp)

    # Train the meta model and generate test AU-ROC score.
    meta_model_op = build_metamodel(base_model_op, hparams=meta_model_hp)

    return meta_model_op
