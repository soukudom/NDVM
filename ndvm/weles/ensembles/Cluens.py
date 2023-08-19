from sklearn.base import ClassifierMixin, BaseEstimator, clone
from sklearn.cluster import KMeans
import numpy as np


class Cluens(ClassifierMixin, BaseEstimator):
    def __init__(self, base_estimator, n_estimators=3):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.clstrs = [
            KMeans(n_clusters=self.n_estimators).fit(X[y == c]) for c in self.classes
        ]

        self.estimators_ = []
        for clstr in self.clstrs:
            y_pred = clstr.predict(X)
            for cls in range(self.n_estimators):
                mask = y_pred == cls
                if len(np.unique(y[mask])) == 2:
                    clf = clone(self.base_estimator).fit(X[mask], y[mask])
                    self.estimators_.append(clf)

        return self
