import numpy as np

from ivhd import IVHD


def test_ivhd():
    X = np.random.randn(100, 5)
    ivhd = IVHD()
    ivhd.fit_transform(X)
