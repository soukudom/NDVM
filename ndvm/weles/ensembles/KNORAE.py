"""
KNORA-E
"""
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import accuracy_score
import math
#from sklearn.neighbors import DistanceMetric
from sklearn.metrics import DistanceMetric
from torch import cdist, from_numpy


class KNORAE(BaseEstimator, ClassifierMixin):
    """
    Implementation of the KNORA-Eliminate des method.
    """

    def __init__(self, ensemble=[], k=70, p=2):
        self.ensemble = ensemble
        self.k = k
        self.p = p

    def fit(self, X, y):
        self.X_dsel = X
        self.y_dsel = y

    def estimate_competence(self, X):
        # kontener na wagi
        self.competences = np.zeros((X.shape[0], len(self.ensemble))).astype(int)
        # dystanse od testowych do DSEL
        all_distances = cdist(from_numpy(X), from_numpy(self.X_dsel), p=self.p).numpy()
        reduce_local = True

        while reduce_local:
            # lokalne sasiedztwo dla kazdej testowej
            self.neighbors_indx_ = np.argsort(all_distances)[:, :self.k]
            # szukamy wyroczni
            for i, clf in enumerate(self.ensemble):
                # predykcja całego dsel
                pred = clf.predict(self.X_dsel)
                # predykcje lokalnych regionow
                local_pred = self.y_dsel[self.neighbors_indx_]
                # print(local_pred.shape)
                # prawdziwe etykiety lokalnych regionow
                local_true = pred[self.neighbors_indx_]
                # czy jest wyrocznia czy nie?
                self.competences[:,i] = (local_true == local_pred).all(axis=1)

            # sprawdzamy, czy gdzieś nie ma lokalnych wyroczni
            no_oracles = np.argwhere(np.sum(self.competences, axis=1)==0).reshape(1,-1)
            # jezeli nie ma, to zmniejszamy k i sprawdzamy na nowo
            if no_oracles.shape[1] != 0 and self.k >= 2:
                self.k -= 1
            # jezeli sa wszedzie, to idziemy dalej
            else:
                reduce_local = False

        # Gdyby gdzies nie bylo zadnej wyroczni, to kombinujemy wszystkie
        self.competences[no_oracles] = [1,1,1]


    def ensemble_matrix(self, X):
        """EM."""
        return np.array([member_clf.predict(X) for member_clf in self.ensemble]).T

    def predict(self, X):
        self.estimate_competence(X)
        em = self.ensemble_matrix(X)
        predict = []

        for i, row in enumerate(em):
            decision = np.bincount(row, weights=self.competences[i])
            predict.append(np.argmax(decision))

        return np.array(predict)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
