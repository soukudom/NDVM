import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from torch import cdist, dist, from_numpy
from scipy.stats import mode

class fkNN(BaseEstimator, ClassifierMixin):
    """
    Nearest Neighbors Classifier based on pytorch distances and able to employ
    fractional distances during neighborhood search.
    """
    def __init__(self, k=5, p=2):
        self.k = k
        self.p = p

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.X_, self.y_ = X, y

    def predict_proba(self, X):
        all_distances = cdist(from_numpy(X), from_numpy(self.X_), p=self.p).numpy()
        self.neighbors_indx_ = np.argsort(all_distances)[:, :self.k]
        self.neighbors_pred_ = self.y_[self.neighbors_indx_]
        proba = np.array([np.sum(self.neighbors_pred_==l, axis=1)/self.k
        for l in self.classes_]).T

        return proba

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        proba = self.predict_proba(X)
        pred = np.argmax(proba, axis=1)

        return pred
