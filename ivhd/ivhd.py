import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors


class IVHD(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_components: int = 2,
        nn: int = 2,
        rn: int = 1,
        c: float = 0.1,
        lambda_: float = 0.3,
        simulation_steps: int = 200,
        verbose: bool = False,
        distance: str = 'euclidean'
    ) -> None:
        self.n_components = n_components
        self.nn = nn
        self.rn = rn
        self.c = c
        self.lambda_ = lambda_
        self.simulation_steps = simulation_steps
        self.verbose = verbose
        self.distance = distance

    def transform(self, X):
        nns = self._get_nearest_neighbors_indexes(X)
        rns = self._get_remote_neighbors_indexes(X)

        a = (1 - self.lambda_) / (1 + self.lambda_)
        b = self.c / (1 + self.lambda_)

        x = np.random.rand(X.shape[0], self.n_components)
        x_delta = np.random.rand(X.shape[0], self.n_components)

        for _ in range(self.simulation_steps):
            f = self._calculate_forces(x, nns, rns)
            x_delta = a * x_delta + b * f
            x = x + x_delta

        return x

    def _calculate_forces(self, x: np.ndarray, nns: np.ndarray, rns: np.ndarray):
        # (X.shape[0], nn, n_components)
        x_i = np.repeat(x[:, np.newaxis, :], self.nn, axis=1)
        x_j = x[nns]

        # (X.shape[0], n_components)
        fnn = np.sum(x_i - x_j, axis=1)

        # (X.shape[0], rn, n_components)
        x_i = np.repeat(x[:, np.newaxis, :], self.rn, axis=1)
        x_k = x[rns]
        x_ik = x_i - x_k

        # (X.shape[0], rn)
        d_ik = self._calculate_distance(self.distance, x_i, x_k)
        # (X.shape[0], rn, n_components)
        d_ik = np.repeat(d_ik[:, :, np.newaxis], self.n_components, axis=-1) + 1e8

        # (X.shape[0], n_components)
        frn = np.sum((1 - d_ik) * x_ik / d_ik, axis=1)

        f = -fnn - self.c * frn
        return f

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

    def _calculate_distance(self, distance_string, x_i, x_k):
        if distance_string == 'cosine':
            top = np.sum(x_i * x_k, axis=-1)
            bottom_i = np.linalg.norm(x_i, axis=-1)
            bottom_k = np.linalg.norm(x_k, axis=-1)
            return 1 - (top / (bottom_i * bottom_k))

        elif distance_string == 'binary':
            return np.sum(x_i != x_k, axis=-1)

        else:
            return np.sum((x_i - x_k) ** 2, axis=-1)
