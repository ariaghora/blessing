import numpy as np
from boruta import BorutaPy
from skfeature.function.similarity_based import SPEC
from skfeature.function.sparse_learning_based import RFS
from skfeature.utility.sparse_learning import feature_ranking
from sklearn.ensemble import RandomForestClassifier


class BaseBaselineSelector:
    def __init__(self, k):
        self.k = k


class BorutaSelector(BaseBaselineSelector):
    def __init__(self, k):
        super().__init__(k)

    def fit(self, X, y):
        forest = RandomForestClassifier()
        self.selector = BorutaPy(forest, n_estimators="auto")
        self.selector.fit(X, y)
        return self

    def transform(self, X):
        return X[:, np.argsort(self.selector.ranking_)[: self.k]]


class SPECSelector(BaseBaselineSelector):
    def __init__(self, k, style=-1):
        super().__init__(k)
        self.style = style
        self.scores = None
        self.idx = None

    def fit(self, X):
        self.scores = SPEC.spec(X, style=self.style)
        self.idx = SPEC.feature_ranking(self.scores, style=self.style)
        return self

    def transform(self, X):
        return X[:, self.idx[: self.k]]


class RFSSelector(BaseBaselineSelector):
    def __init__(self, k):
        super().__init__(k)
        self.weights = None
        self.idx = None

    def fit(self, X, y):
        y = y.reshape(-1, 1)
        self.weights = RFS.rfs(X, y, gamma=0.1)
        self.idx = feature_ranking(self.weights)
        return self

    def transform(self, X):
        return X[:, self.idx[: self.k]]
