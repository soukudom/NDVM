from sklearn.base import ClassifierMixin, BaseEstimator
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC


class PCASSE(ClassifierMixin, BaseEstimator):
    def __init__(self, n_components=4, subspace_size=4):
        self.n_components = n_components
        self.subspace_size = subspace_size

    def fit(self, X, y, classes=None):
        # Calculate PCA components
        components = np.abs(
            PCA(n_components=self.n_components, svd_solver="full").fit(X).components_
        )

        # Gather ensemble
        self.subspaces = np.array(
            [np.argsort(-row)[: self.subspace_size] for row in components]
        )

        # Build ensemble
        self.ensemble = [SVC().fit(X[:, subspace], y) for subspace in self.subspaces]

        return self

    def predict(self, X):
        return (
            np.mean(
                np.array(
                    [
                        self.ensemble[i].decision_function(X[:, subspace])
                        for i, subspace in enumerate(self.subspaces)
                    ]
                ),
                axis=0,
            )
            > 0
        ).astype(int)


class PCASSEE(ClassifierMixin, BaseEstimator):
    def __init__(self, distribuant_treshold=0.1, subspace_size=4):
        self.distribuant_treshold = distribuant_treshold
        self.subspace_size = subspace_size

    def fit(self, X, y, classes=None):
        # Calculate PCA components

        pca = PCA(svd_solver="full").fit(X)
        components = np.abs(pca.components_)

        # Z EVR
        evrd = np.add.accumulate(pca.explained_variance_ratio_)
        self.n_components = np.where(evrd > self.distribuant_treshold)[0][0]

        if self.n_components == 0:
            self.n_components = 1

        # print(evrd)
        # print("%i COMPONENTS" % self.n_components)
        components = components[: self.n_components, :]

        # Calculate subspace size
        self.subspace_size = 4

        # Gather ensemble
        self.subspaces = np.array(
            [np.argsort(-row)[: self.subspace_size] for row in components]
        )

        # Build ensemble
        self.ensemble = [SVC().fit(X[:, subspace], y) for subspace in self.subspaces]

        return self

    def predict(self, X):
        return (
            np.mean(
                np.array(
                    [
                        self.ensemble[i].decision_function(X[:, subspace])
                        for i, subspace in enumerate(self.subspaces)
                    ]
                ),
                axis=0,
            )
            > 0
        ).astype(int)


class RS(ClassifierMixin, BaseEstimator):
    def __init__(self, n_estimators=20, subspace_size=4):
        self.n_estimators = n_estimators
        self.subspace_size = subspace_size

    def fit(self, X, y, classes=None):
        # Calculate PCA components

        pca = PCA(svd_solver="full").fit(X)
        components = np.abs(pca.components_)

        # Gather ensemble
        self.subspaces = np.random.randint(
            X.shape[1], size=(self.n_estimators, self.subspace_size)
        )

        # Build ensemble
        self.ensemble = [SVC().fit(X[:, subspace], y) for subspace in self.subspaces]

        return self

    def predict(self, X):
        return (
            np.mean(
                np.array(
                    [
                        self.ensemble[i].decision_function(X[:, subspace])
                        for i, subspace in enumerate(self.subspaces)
                    ]
                ),
                axis=0,
            )
            > 0
        ).astype(int)
