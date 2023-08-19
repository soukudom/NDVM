import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator, clone
from sklearn.naive_bayes import GaussianNB

class MetaPreproc(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=GaussianNB(), preprocessing=None):
        self.base_estimator = base_estimator
        self.preprocessing = preprocessing

    def fit(self, X, y, classes=None):
        # if not hasattr(self, "clf"):
        self.clf = clone(self.base_estimator)
        self.preproc = clone(self.preprocessing)
        if self.preprocessing != None:
            X, y = self.preproc.fit_resample(X, y)
        return self.clf.fit(X, y)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def predict(self, X):
        return self.clf.predict(X)
