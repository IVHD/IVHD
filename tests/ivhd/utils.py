from typing import Optional

import numpy as np
from sklearn.datasets import fetch_openml


def load_mnist(subset_size: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

    if subset_size is not None:
        X, y = X[:subset_size], y[:subset_size]

    X = X.reshape(-1, 28 * 28)
    y = y.astype(np.int32)
    return X, y
