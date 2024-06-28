import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from ivhd import IVHDGrad
from tests.ivhd.utils import load_mnist


def test_ivhd_interface():
    X = np.random.randn(100, 30)

    ivhd = IVHDGrad()
    X_ivhd = ivhd.fit_transform(X)

    assert isinstance(X_ivhd, np.ndarray), "Expected numpy array"
    assert X_ivhd.shape == (100, 2)


def test_ivhd_performance_on_mnist():
    # Load a subset of MNIST dataset to limit execution times
    SUBSET_SIZE = 10000  # / 60000
    X, y = load_mnist(SUBSET_SIZE)

    ivhd = IVHDGrad(
        steps=2000,
        nn=5,
        rn=2,
        re_draw_remote_neighbors=True,
        optimizer_params={"lr": 0.005},
    )
    X_ivhd = ivhd.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_ivhd, y)

    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    classification_accuracy = accuracy_score(y_test, y_pred)

    # 72% was experimentally determined to be an average accuracy
    # on this dataset with these parameters.
    # By comparison:
    #   PCA ~ 40%
    #   T-SNE ~ 95%
    # The minimal accuracy threshold to pass the test is set to 65%.
    assert (
        classification_accuracy >= 0.65
    ), "Classification accuracy should be equal or grater than 65%"
