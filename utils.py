"""Data loading and evaluation helpers shared by all scripts.

Datasets:
car.csv   (classification): UCI Car Evaluation. 6 ordinal features; the
          target is turned into a binary label: acceptable (True) or not.
house.csv (regression):     house prices with 12 features; target = price.
"""
from typing import Tuple

import numpy as np
import pandas as pd


CAR_COLUMNS = ["buying price", "maintenance cost", "doors",
               "number of persons", "lug_boot", "safety", "decision"]
CAR_CLASSES = {"unacc", "acc", "good", "vgood"}


def _load_car(path: str) -> Tuple[pd.DataFrame, np.ndarray]:
    # Works both with UCI's raw car.data (no header) and with a CSV that has a
    # header row: we detect the header and use our own column names either way.
    df = pd.read_csv(path, header=None, dtype=str)
    if df.iloc[0, -1].strip() not in CAR_CLASSES:
        df = df.iloc[1:]
    df.columns = CAR_COLUMNS
    df = df.apply(lambda col: col.str.strip())

    y = df["decision"].values != "unacc"
    X_df = df.drop("decision", axis=1)

    # The features are ordered categories, so we encode them as ordered numbers
    # (a threshold split "x < t" then means "lower than some level").
    level4 = {"vhigh": 4, "high": 3, "med": 2, "low": 1}
    X_df["buying price"] = X_df["buying price"].map(level4)
    X_df["maintenance cost"] = X_df["maintenance cost"].map(level4)
    X_df["doors"] = X_df["doors"].map({"2": 2, "3": 3, "4": 4, "5more": 5})
    X_df["number of persons"] = X_df["number of persons"].map({
        "2": 2,
        "4": 4,
        "more": 6,
    })
    X_df["lug_boot"] = X_df["lug_boot"].map({"small": 1, "med": 2, "big": 3})
    X_df["safety"] = X_df["safety"].map({"low": 1, "med": 2, "high": 3})
    return X_df, y


def _load_house(path: str) -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(path)
    y = df["price"].values.astype(float)
    X_df = df.drop("price", axis=1)

    # binary yes/no columns -> 1/0
    for col in ["mainroad", "guestroom", "basement", "hotwaterheating",
                "airconditioning", "prefarea"]:
        X_df[col] = X_df[col].map({"yes": 1, "no": 0})
    X_df["furnishingstatus"] = X_df["furnishingstatus"].map({
        "furnished": 2,
        "semi-furnished": 1,
        "unfurnished": 0,
    })
    return X_df, y


def load_data(
        path: str,
        test_split: float = 0.2,
        random_state: int = 2024,
        is_regression: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return X_train, X_test, y_train, y_test."""
    X_df, y = _load_house(path) if is_regression else _load_car(path)
    if X_df.isnull().values.any():
        raise ValueError(f"Unexpected values in {path}: check the column contents")

    # split data
    X = X_df.values.astype(float)
    shuffled_indices = np.random.RandomState(random_state).permutation(len(X))
    test_set_size = int(len(X) * test_split)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test


def standardize_targets(y_train: np.ndarray, y_test: np.ndarray):
    """Scale prices to zero mean / unit std (using train statistics only), so
    that errors are comparable across datasets and RMSE = 1 means 'as bad as
    always predicting the mean'."""
    y_mu, y_std = y_train.mean(), y_train.std()
    return (y_train - y_mu) / y_std, (y_test - y_mu) / y_std


def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def report_classification(name: str, model, X_train, X_test, y_train, y_test) -> None:
    """Fit a classifier and print its train/test accuracy."""
    model.fit(X_train, y_train)
    train_acc = (model.predict(X_train) == y_train).mean()
    test_acc = (model.predict(X_test) == y_test).mean()
    print(f"{name:24s} train accuracy: {train_acc:.4f} | test accuracy: {test_acc:.4f}")


def report_regression(name: str, model, X_train, X_test, y_train, y_test) -> None:
    """Fit a regressor and print its train/test RMSE (on standardized prices)."""
    model.fit(X_train, y_train)
    print(f"{name:24s} train RMSE: {rmse(model.predict(X_train), y_train):.4f} | "
          f"test RMSE: {rmse(model.predict(X_test), y_test):.4f}")
