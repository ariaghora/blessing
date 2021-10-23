import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler


class Blessing:
    def __init__(self, k: int, max_iter: int = 200, scale_input: bool = True):
        self.k = k
        self.max_iter = max_iter
        self.scale_input = scale_input

        self.hidden_layer_sizes = (40, self.k, 40)
        self.ae: BaseEstimator = None
        self.column_scores: np.ndarray = None
        self.chosen_column_idx: np.ndarray = None

    def fit(self, X: np.ndarray):
        """'Nudge' the values of constant columns"""
        X = X + (
            np.random.randn(*X.shape) * 10 ** (np.ceil(np.log10(np.abs(X) + 1e-8)) - 2)
        )
        if self.scale_input:
            X = MinMaxScaler().fit_transform(X)

        self.ae = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes, max_iter=self.max_iter
        )

        self.ae.fit(X, X)
        X_hat = self.ae.predict(X)
        col_mse = ((X_hat - X) ** 2).mean(0)
        col_penalty = np.log(X.var(0) + 1e-8)
        self.col_scores = 1 / (col_mse - col_penalty)
        self.chosen_column_idx = np.argsort(self.col_scores)[-self.k :][::-1]
        return self

    def transform(self, X: np.ndarray):
        return X[:, self.chosen_column_idx]
