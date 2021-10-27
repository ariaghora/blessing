import sys
import time

sys.path.append("..")

from typing import List

import numpy as np
import pandas as pd
import scipy.io
from baselinewrapper import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from blessing import Blessing, BlessingPlus


def dataset_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test


def load_mat_dataset(path):
    mat = scipy.io.loadmat(path)
    X = mat["X"]
    y = mat["Y"].ravel()
    return dataset_split(X, y)


def run_classifier(
    selector: BaseBaselineSelector,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    k: int,
):
    ts = time.time()
    te = time.time()

    X_train_new = selector.transform(X_train)
    X_test_new = selector.transform(X_test)

    clf = DecisionTreeClassifier().fit(X_train_new, y_train)
    acc = (clf.predict(X_test_new) == y_test).mean()

    return acc, (te - ts)


def run_one_mat_dataset(
    title: str,
    dataset_path: str,
    selectors: List,  # list of selector class type
    k_list: List[int],
) -> pd.DataFrame:
    """
    Generate a table as the result of running several selectors with
    different number of selected features
    """
    index = [selector.__name__.replace("Selector", "") for selector in selectors]
    columns = pd.MultiIndex.from_product([[title], [f"k={k}" for k in k_list]])
    result_df = pd.DataFrame(index=index, columns=columns)

    X_train, X_test, y_train, y_test = load_mat_dataset(dataset_path)
    for i, Selector in enumerate(selectors):
        print(f"Running {Selector.__name__}...")
        for j, k in enumerate(k_list):
            selector = Selector(k=k).fit(X_train, y_train)
            acc, time = run_classifier(selector, X_train, X_test, y_train, y_test, k)
            result_df.iloc[i, j] = acc

    return result_df


if __name__ == "__main__":
    np.random.seed(42)  # for consistent result across machine ¯\_(ツ)_/¯

    # df_madelon = run_one_mat_dataset(
    #     "Madelon",
    #     "data/madelon.mat",
    #     [Blessing, BlessingPlus, SPECSelector, LapScoreSelector, MCFSSelector],
    #     [50, 100, 150, 200],
    # )
    # print(df_madelon)

    # df_lyphoma = run_one_mat_dataset(
    #     "Lymphoma",
    #     "data/lymphoma.mat",
    #     [Blessing, SPECSelector, LapScoreSelector, MCFSSelector],
    #     [50, 100, 150, 200],
    # )
    # print(df_lyphoma)

    df_basehock = run_one_mat_dataset(
        "Basehock",
        "data/BASEHOCK.mat",
        [Blessing, SPECSelector, LapScoreSelector, MCFSSelector],
        [50, 100, 150, 200],
    )
    print(df_basehock)
