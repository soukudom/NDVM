import numpy as np
from scipy import stats
from sklearn.base import clone, ClassifierMixin, BaseEstimator
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

RULES = ["mean", "gmean", "hmean", "rbh"]
KUNCHEVA = 0.000001


class Geometron(ClassifierMixin, BaseEstimator):
    def __init__(
        self, base_estimator, n_estimators=3, random_state=None, rule="gmean", sigma=3
    ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.rule = rule
        self.sigma = sigma

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.classes_ = np.unique(y)
        self.n_samples, self.n_features = X.shape
        self.means = []
        self.ensemble = []
        for i in range(self.n_estimators):
            sel = np.random.randint(self.n_samples, size=int(self.n_samples))
            self.ensemble.append(clone(self.base_estimator).fit(X[sel], y[sel]))

        # Test scale for the needs of geometric means
        decfuncs = np.array(
            [clf.decision_function(X) for clf in self.ensemble]
        ).T.reshape(-1)
        self.std = np.sqrt(np.sum(np.power(decfuncs, 2), axis=0) / len(decfuncs))

        return self

    def decfunc(self, X):
        decfuncs = np.array([clf.decision_function(X) for clf in self.ensemble])
        s_decfuncs = decfuncs / self.std
        s_decfuncs += self.sigma
        s_decfuncs = s_decfuncs / (self.sigma * 2)
        s_decfuncs = np.clip(s_decfuncs, KUNCHEVA, 1 - KUNCHEVA)

        if self.rule == "gmean":
            decfunc = (stats.gmean(s_decfuncs, axis=0) - 0.5) * self.sigma * 2
        elif self.rule == "hmean":
            decfunc = (stats.hmean(s_decfuncs, axis=0) - 0.5) * self.sigma * 2
        elif self.rule == "mean":
            decfunc = np.mean(decfuncs, axis=0)
        elif self.rule == "rbh":
            # decfunc = np.mean(decfuncs, axis=0)
            gmin = np.min(decfuncs, axis=0)
            gmax = np.max(decfuncs, axis=0)
            gmed = np.median(decfuncs, axis=0)

            h1 = gmax - gmed
            h2 = gmed - gmin

            hm = stats.hmean([h1, h2], axis=0)
            ghm = np.zeros((gmin.shape))
            mask = gmed > (gmax + gmin)
            mask2 = gmed <= (gmax + gmin)
            ghm[mask] = gmax[mask] - hm[mask]
            ghm[mask2] = gmin[mask2] + hm[mask2]

            decfunc = ghm

        return decfunc

    def predict(self, X):
        decfunc = self.decfunc(X)

        y_pred = decfunc > 0

        return y_pred
