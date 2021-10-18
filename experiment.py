import time

import numpy as np

from blessing import Blessing


def timeit(f):
    def timed(*args, **kw):

        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        print(f"func: {f.__name__} took: {te-ts:2.4} sec")
        return result

    return timed


@timeit
def run_blessing(X_train: np.ndarray):
    selector = Blessing(k=3)
    selector.fit(X_train)
    return selector


@timeit
def run_boruta(X_train: np.ndarray, y_train: np.ndarray):
    forest = RandomForestClassifier()
    selector = BorutaPy(forest, n_estimators="auto")
    selector.fit(X_train, y_train)
    return selector


@timeit
def run_select_k_best(X_train: np.ndarray, y_train: np.ndarray):
    selector = SelectKBest(k=3)
    selector.fit(X_train, y_train)
    return selector


@timeit
def run_RFE(X_train: np.ndarray, y_train: np.ndarray):
    forest = RandomForestClassifier()
    selector = RFE(forest, n_features_to_select=3)
    selector.fit(X_train, y_train)
    return selector


if __name__ == "__main__":
    import pandas as pd
    from boruta import BorutaPy
    from pandas.api.types import is_numeric_dtype
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import RFE, SelectKBest
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    X, y = load_iris(return_X_y=True)
    X = np.hstack([X, np.random.randn(X.shape[0], 1000) + 2])
    np.random.shuffle(X.T)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

    blessing = run_blessing(X_train)
    X_train_bless = blessing.transform(X_train)
    X_test_bless = blessing.transform(X_test)

    clf = RandomForestClassifier().fit(X_train_bless, y_train)
    acc = (clf.predict(X_test_bless) == y_test).mean()
    print("Acc blessing = ", acc)

    boruta_selector = run_boruta(X_train, y_train)
    X_train_boruta = boruta_selector.transform(X_train)
    X_test_boruta = boruta_selector.transform(X_test)

    clf = RandomForestClassifier().fit(X_train_boruta, y_train)
    acc = (clf.predict(X_test_boruta) == y_test).mean()
    print("Acc boruta = ", acc)

    boruta_selector = run_select_k_best(X_train, y_train)
    X_train_boruta = boruta_selector.transform(X_train)
    X_test_boruta = boruta_selector.transform(X_test)

    clf = RandomForestClassifier().fit(X_train_boruta, y_train)
    acc = (clf.predict(X_test_boruta) == y_test).mean()
    print("Acc SelectKBest = ", acc)

    boruta_selector = run_RFE(X_train, y_train)
    X_train_boruta = boruta_selector.transform(X_train)
    X_test_boruta = boruta_selector.transform(X_test)

    clf = RandomForestClassifier().fit(X_train_boruta, y_train)
    acc = (clf.predict(X_test_boruta) == y_test).mean()
    print("Acc RFE = ", acc)
