import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class LinearClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        p = np.array([np.mean(X[y == c], axis=0) for c in np.unique(y)])
        self.b = np.mean(p, axis=0)
        self.w = p[1] - self.b
        return self

    def decision_function(self, X):
        return (X - self.b).dot(self.w)

    def predict(self, X):
        return self.decision_function(X) > 0
