from datasets import load_dataset
import pandas as pd

def fetch_data(config,  mode="train"):
    # Function that will be called to fetch data for training.
    # It should return the data in the format X,A,Y, where:
    # X: A list of the confounders/state
    # A: The action/treatment
    # Y: The outcome/label
    # All three should be in the form of pandas dataframes that can then be further processed
    # Take the parameters for fetching the data from the config
    
    dataset = load_dataset(
        config["dataset_name"],
        split=config.get("split", "train"),
        data_dir=config.get("data_dir"),
        revision=config.get("revision")
    )

    df = dataset.to_pandas()

    train_seasons = set(config["training_seasons"])
    test_seasons = set(config["testing_seasons"])

    if mode == "train":
        df = df[df["season"].isin(train_seasons)].copy()
    elif mode == "test":
        df = df[df["season"].isin(test_seasons)].copy()
    else:
        raise ValueError("mode must be 'train' or 'test'")

    X_cols = config["X_columns"]
    A_cols = config["A_columns"]
    Y_cols = config["Y_columns"]

    required = X_cols + A_cols + Y_cols
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = df[X_cols].copy()
    A = df[A_cols].copy()
    Y = df[Y_cols].copy()

    return X, A, Y