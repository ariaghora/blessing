from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


class Autoencoder(torch.nn.Module):
    def __init__(self, n_feat: int, k: int):
        super().__init__()
        self.enc1 = torch.nn.Linear(n_feat, 20)
        self.enc2 = torch.nn.Linear(20, k)
        self.dec1 = torch.nn.Linear(k, 20)
        self.dec2 = torch.nn.Linear(20, n_feat)
        self.lrelu = torch.nn.LeakyReLU()
        self.enc_seq = [self.enc1, self.lrelu, self.enc2, self.lrelu]
        self.dec_seq = [self.dec1, self.lrelu, self.dec2]

    def forward(self, X: torch.Tensor):
        z = X
        for f in self.enc_seq:
            z = f(z)
        X_hat = z
        for f in self.dec_seq:
            X_hat = f(X_hat)
        return X_hat, z


class TorchBlessing:
    def __init__(
        self,
        k: int,
        max_iter: int = 200,
        scale_input: bool = True,
        verbose: bool = True,
    ):
        self.k = k
        self.max_iter = max_iter
        self.scale_input = scale_input
        self.verbose = verbose
        self.ae: Autoencoder = None
        self.column_scores: np.ndarray = None
        self.chosen_column_idx: np.ndarray = None

    def _ensure_float_tensor(self, X: Any) -> torch.FloatTensor:
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        elif isinstance(X, pd.DataFrame):
            X = torch.from_numpy(X.values)
        else:
            raise Exception("Incompatible input")
        return X.float()

    def fit(self, X: np.ndarray):
        if self.scale_input:
            X = MinMaxScaler().fit_transform(X)

        self.ae = Autoencoder(n_feat=X.shape[1], k=self.k)
        X = self._ensure_float_tensor(X)

        mse = torch.nn.MSELoss()
        opt = torch.optim.Adam(self.ae.parameters())
        for i in tqdm(range(self.max_iter)):
            opt.zero_grad()
            X_hat, z = self.ae(X)
            loss = mse(X_hat, X)
            loss.backward()
            opt.step()
        col_mse = ((X_hat.detach().numpy() - X.detach().numpy()) ** 2).mean(0)
        col_penalty = np.log(X.detach().numpy().var(0) + 1e-8)
        self.col_scores = 1 / (col_mse - col_penalty)
        self.chosen_column_idx = np.argsort(self.col_scores)[-self.k :][::-1]
        return self

    def transform(self, X: np.ndarray):
        return X[:, self.chosen_column_idx]
