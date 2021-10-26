import numpy as np
from boruta import BorutaPy
from skfeature.function.similarity_based import SPEC, lap_score
from skfeature.function.sparse_learning_based import NDFS, RFS, UDFS, MCFS
from skfeature.utility.construct_W import construct_W
from skfeature.utility.sparse_learning import feature_ranking
from sklearn.ensemble import RandomForestClassifier

from baselines.dlufs.construct_L import construct_L
from baselines.dlufs.DLUFS import dlufs
from JELSR import feature_selection as jelsr_feature_selection


class BaseBaselineSelector:
    def __init__(self, k):
        self.k = k
        self.idx = None

    def transform(self, X):
        return X[:, self.idx[: self.k]]


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

    def fit(self, X):
        self.scores = SPEC.spec(X, style=self.style)
        self.idx = SPEC.feature_ranking(self.scores, style=self.style)
        return self


class RFSSelector(BaseBaselineSelector):
    def __init__(self, k):
        super().__init__(k)
        self.weights = None

    def fit(self, X, y):
        y = y.reshape(-1, 1)
        self.weights = RFS.rfs(X, y, gamma=0.1)
        self.idx = feature_ranking(self.weights)
        return self


class MCFSSelector(BaseBaselineSelector):
    def __init__(self, k):
        super().__init__(k)
        self.weights = None

    def fit(self, X):
        kwargs = {
            "metric": "euclidean",
            "neighborMode": "knn",
            "weightMode": "heatKernel",
            "k": 5,
            "t": 1,
        }
        W = construct_W(X, **kwargs)
        self.weights = MCFS.mcfs(X, n_selected_features=self.k, W=W, n_clusters=20)
        self.idx = MCFS.feature_ranking(self.weights)
        return self


class LapScoreSelector(BaseBaselineSelector):
    def __init__(self, k):
        super().__init__(k)
        self.weights = None

    def fit(self, X):
        kwargs_W = {
            "metric": "euclidean",
            "neighbor_mode": "knn",
            "weight_mode": "heat_kernel",
            "k": 5,
            "t": 1,
        }
        self.weights = construct_W(X, **kwargs_W)
        score = lap_score.lap_score(X, W=self.weights)
        self.idx = lap_score.feature_ranking(score)
        return self


class JELSRSelector(BaseBaselineSelector):
    def __init__(self, k):
        super().__init__(k)

    def fit(self, X):
        self.idx = jelsr_feature_selection(X.T, self.k, self.k)
        return self

    def transform(self, X):
        return X[:, self.idx]


class NDFSSelector(BaseBaselineSelector):
    def __init__(self, k):
        super().__init__(k)
        self.weights = None

    def fit(self, X, y):
        y = y.reshape(-1, 1)
        kwargs = {
            "metric": "euclidean",
            "neighborMode": "knn",
            "weightMode": "heatKernel",
            "k": 5,
            "t": 1,
        }
        W = construct_W(X, **kwargs)
        self.weights = NDFS.ndfs(X, W=W, n_clusters=5)
        self.idx = feature_ranking(self.weights)
        return self


class DLUFSSelector(BaseBaselineSelector):
    def __init__(self, k):
        super().__init__(k)

    def fit(self, X):
        Parm = [1e-4, 1e-2, 1, 1e2, 1e4]

        p = len(X[0])
        n = len(X[:, 0])

        step = 50
        num_fea = self.k

        X = (X - X.mean(axis=0)) / X.std(axis=0)

        L = construct_L(X)

        idx = np.zeros((p, 25, 6), dtype=np.int)

        for r in range(step, num_fea + 1, step):
            count = 0
            for Parm1 in Parm:
                for Parm2 in Parm:
                    Weight = dlufs(X, L=L, r=r, alpha=Parm1, lambd=Parm2)
                    idx[0:p, count, (r / step) - 1] = self.feature_ranking(Weight)
                    count += 1
        self.idx = idx
        return self

    def feature_ranking(self, W):
        T = (W * W).sum(1)
        idx = np.argsort(T, 0)
        return idx[::-1]
