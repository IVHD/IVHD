import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
import torch
from torch.optim import (
    Adadelta,
    Adagrad,
    Adam,
    AdamW,
    SparseAdam,
    Adamax,
    ASGD,
    LBFGS,
    NAdam,
    RAdam,
    RMSprop,
    Rprop,
    SGD
)

_optimizer_mapping = {cls.__name__.lower(): cls for cls in
                      [Adadelta, Adagrad, Adam, AdamW, SparseAdam, Adamax, ASGD, LBFGS, NAdam, RAdam, RMSprop, Rprop,
                       SGD]}


class IVHDGrad(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            n_components: int = 2,
            nn: int = 2,
            rn: int = 1,
            optimizer: str = 'adam',
            optimizer_params=None,
            steps: int = 200,
            epsilon: float = 1e-15,
            re_draw_remote_neighbors: bool = False,
            verbose: bool = False,
    ) -> None:
        if optimizer_params is None:
            optimizer_params = {'lr': 0.01}
        self.n_components = n_components
        self.nn = nn
        self.rn = rn
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.simulation_steps = steps
        self.epsilon = epsilon
        self.re_draw_remote_neighbors = re_draw_remote_neighbors
        self.verbose = verbose

    def transform(self, X):
        nns = self._get_nearest_neighbors_indexes(X)
        rns = self._get_remote_neighbors_indexes(X)

        x = torch.rand(X.shape[0], self.n_components, requires_grad=True)
        print(f" X shape: {x.shape}")
        optimizer = _optimizer_mapping[self.optimizer]([x], **self.optimizer_params)

        for _ in range(self.simulation_steps):
            neigborhoods = x[nns]
            if self.re_draw_remote_neighbors:
                rns = self._get_remote_neighbors_indexes(X)
            remote_neigborhoods = x[rns]

            xu = x.unsqueeze(1)

            dist_emb_neighbor = ((xu - neigborhoods) ** 2 + self.epsilon).sum(dim=2).sqrt()
            dist_emb_remote = ((xu - remote_neigborhoods) ** 2 + self.epsilon).sum(dim=2).sqrt()

            del_n = 0
            del_r = 1

            cost_emb_neighbor = ((dist_emb_neighbor - del_n) ** 2).sum()
            cost_emb_remote = ((dist_emb_remote - del_r) ** 2).sum()

            cost = cost_emb_neighbor + cost_emb_remote

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

        return x.detach().numpy()

    def _get_nearest_neighbors_indexes(self, X: np.ndarray) -> np.ndarray:
        # for every point in X find indexes of its 'nn' nearest neighbors
        knn_model = NearestNeighbors(n_neighbors=self.nn + 1)
        knn_model.fit(X)
        _, indices = knn_model.kneighbors(X)
        return indices[:, 1:]

    def _get_remote_neighbors_indexes(self, X: np.ndarray) -> np.ndarray:
        # for every point in X sample indices of its 'rn' remote neighbors
        return np.random.randint(low=0, high=X.shape[0], size=(X.shape[0], self.rn))

    def __sklearn_is_fitted__(self) -> bool:
        return True  # all calculations are performed in 'transform' method

    def fit(self, X, y=None, **fit_params):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)
