import numpy as np

from ivhd import IVHDGrad


def test_ivhdgrad():
    X = np.random.randn(100, 5)
    ivhdgrad = IVHDGrad()
    ivhdgrad.fit_transform(X)
