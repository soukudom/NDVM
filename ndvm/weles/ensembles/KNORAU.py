"""
KNORA-U
"""
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import math
#from sklearn.neighbors import DistanceMetric
from sklearn.metrics import DistanceMetric


class KNORAU(BaseEstimator, ClassifierMixin):
    """
    Implementation of the KNORA-Union des method.
    """

    def __init__(self, ensemble=[], k=7, metric="euclidean"):
        self.ensemble = ensemble
        self.k = k
        self.metric = metric

    def fit(self, X, y):
        self.X_dsel = X
        self.y_dsel = y

        self.knn = KNeighborsClassifier(n_neighbors=self.k, metric="euclidean")

        self.knn.fit(self.X_dsel, self.y_dsel)

    def estimate_competence(self, X):
        self.competences = np.zeros((X.shape[0], len(self.ensemble)))
        _, self.neighbors = self.knn.kneighbors(X=X, n_neighbors=self.k)

        local_X = np.reshape(self.X_dsel[self.neighbors], (-1, X.shape[-1]))
        local_y = np.reshape(self.y_dsel[self.neighbors], (-1))

        self.competences = np.sum(
            np.array(
                [
                    np.reshape(clf.predict(local_X) == local_y, (X.shape[0], self.k))
                    for clf in self.ensemble
                ]
            ),
            axis=2,
        ).T
        # print(self.competences)

    def ensemble_matrix(self, X):
        """EM."""
        return np.array([member_clf.predict(X) for member_clf in self.ensemble]).T

    def predict(self, X):
        if self.X_dsel.shape[0] >= self.k:
            self.estimate_competence(X)
            em = self.ensemble_matrix(X)
            predict = []

            for i, row in enumerate(em):
                decision = np.bincount(row, weights=self.competences[i])
                predict.append(np.argmax(decision))
        else:
            em = self.ensemble_matrix(X)
            predict = []

            for i, row in enumerate(em):
                decision = np.bincount(row)
                predict.append(np.argmax(decision))

        return np.array(predict)

    def score(self, X, y):
        return balanced_accuracy_score(y, self.predict(X))
