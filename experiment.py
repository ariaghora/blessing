import json
import time

import numpy as np
import pandas as pd
from boruta import BorutaPy
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from baselinewrapper import SPECSelector, RFSSelector
from blessing import TorchBlessing as Blessing
from skblessing import Blessing as SKBlessing


def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        print(f"function '{f.__name__}' took: {te-ts:2.4} sec")
        return result

    return timed


def dataset_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test


def load_iris_noise(n_noise: int):
    X, y = load_iris(return_X_y=True)
    X = np.hstack([X, np.random.randn(X.shape[0], n_noise // 2)])
    X = np.hstack([X, np.ones(shape=(X.shape[0], n_noise // 2))])
    np.random.shuffle(X.T)
    return dataset_split(X, y)


def load_mnist():
    df = pd.read_csv("data/train.csv")
    X_train = df.drop("label", axis=1).values
    y_train = df["label"].values
    df = pd.read_csv("data/test.csv")
    X_test = df.drop("label", axis=1).values
    y_test = df["label"].values
    return X_train, X_test, y_train, y_test


@timeit
def run_blessing(X_train: np.ndarray, y_train: np.ndarray, k: int):
    selector = Blessing(k)
    selector.fit(X_train)
    return selector


@timeit
def run_spec(X_train: np.ndarray, y_train: np.ndarray, k: int):
    selector = SPECSelector(k)
    selector.fit(X_train)
    return selector


@timeit
def run_skblessing(X_train: np.ndarray, y_train: np.ndarray, k: int):
    selector = SKBlessing(k)
    selector.fit(X_train)
    return selector


@timeit
def run_boruta(X_train: np.ndarray, y_train: np.ndarray, k: int):
    forest = RandomForestClassifier()
    selector = BorutaPy(forest, n_estimators="auto")
    selector.fit(X_train, y_train)
    return selector


@timeit
def run_rfe(X_train: np.ndarray, y_train: np.ndarray, k: int):
    forest = RandomForestClassifier()
    selector = RFE(forest, n_features_to_select=k)
    selector.fit(X_train, y_train)
    return selector


@timeit
def run_rfs(X_train: np.ndarray, y_train: np.ndarray, k: int):
    selector = RFSSelector(k)
    selector.fit(X_train, y_train)
    return selector


def run_classifier(
    selector_func,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    k: int,
):
    ts = time.time()
    selector = selector_func(X_train, y_train, k)
    te = time.time()

    X_train_new = selector.transform(X_train)
    X_test_new = selector.transform(X_test)

    clf = DecisionTreeClassifier().fit(X_train_new, y_train)
    acc = (clf.predict(X_test_new) == y_test).mean()

    print(f"Acc {type(selector).__name__} = ", acc)
    return {
        "selector": type(selector).__name__,
        "accuracy": acc,
        "time": f"{te-ts:2.4}",
    }


def run_experiment(
    name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    k: int,
    selector_runners: list,
):
    print(f"Running {name}...")
    res = []
    for f in selector_runners:
        try:
            res.append(run_classifier(f, X_train, X_test, y_train, y_test, k))
        except MemoryError:
            print(f"Cannot run {f.__class__.__name__} due to memory error")
    return res


def run_varying_noise(levels: list, selector_runners: list):
    table = pd.DataFrame()
    res = []
    for level in levels:
        res.append(
            {
                "noise_level": level,
                "selectors_acc": run_experiment(
                    f"Iris (noise columns = {level})",
                    *load_iris_noise(level),
                    k=4,
                    selector_runners=selector_runners,
                ),
            }
        )
    return res


if __name__ == "__main__":
    np.random.seed(42)  # for consistent result across machine ¯\_(ツ)_/¯

    selector_runners = [run_skblessing, run_spec, run_boruta]
    #selector_runners = [run_rfs]

    levels = [10, 50, 100, 500, 1000]
    res = run_varying_noise(levels=levels, selector_runners=selector_runners)
    with open("result/run_varying_noise.json", "w") as f:
        json.dump(res, f, indent=2)

    run_experiment("MNIST", *load_mnist(), k=100, selector_runners=selector_runners)
