from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import StandardScaler
#from sklearn.neighbors import DistanceMetric
from sklearn.metrics import DistanceMetric
import numpy as np
import math


class ExposerClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, radius=None, p=2, memory=250):
        self.radius = radius
        self.p = p
        self.memory = memory

    def fit(self, X, y):
        self.scaler = StandardScaler().fit(X)
        self.X_ = self.scaler.transform(X)
        self.y_ = np.copy(y)
        self.classes_, self.prior = np.unique(y, return_counts=True)
        self.metric = DistanceMetric.get_metric(metric="euclidean")
        return self

    def partial_fit(self, X, y, classes=None):
        self.scaler = StandardScaler().fit(X)
        self.X_ = (
            np.concatenate((self.X_, self.scaler.transform(X)), axis=0)
            if hasattr(self, "X_")
            else self.scaler.transform(X)
        )
        self.y_ = (
            np.concatenate((self.y_, y), axis=0) if hasattr(self, "y_") else np.copy(y)
        )

        self.classes_ = classes
        _, self.prior = np.unique(y, return_counts=True)
        if self.classes_ is None:
            self.classes_, self.prior = np.unique(y, return_counts=True)

        self.metric = DistanceMetric.get_metric(metric="euclidean")

        if self.X_.shape[0] > self.memory:
            self.X_, self.y_ = self.X_[-self.memory :, :], self.y_[-self.memory :]
        return self

    def predict_proba(self, X):
        X_ = self.scaler.transform(X)
        if self.radius is None:
            self.radius = np.sum(np.std(X_, axis=0))

        distances = self.metric.pairwise(X_, self.X_)
        return np.array(
            [
                np.sum((self.radius - distances[:, self.y_ == label]).clip(0), axis=1)
                / self.prior[label]
                for label in self.classes_
            ]
        )

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=0)

    def minkowski_distance(self, x, y):
        diff = np.fabs(x - y)
        diff_p = diff ** self.p
        return max(math.pow(np.sum(diff_p), 1 / self.p), np.nextafter(0, 1))
        # return 1.0
