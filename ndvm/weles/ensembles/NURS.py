from sklearn.base import ClassifierMixin, BaseEstimator, clone
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import numpy as np



class NURS(BaseEstimator, ClassifierMixin):
    """
    Non-uniform random subspace
    For drifted data streams
    """
    def __init__(self, best=False, rs=False, n_f=2, n=20, ensemble=False):
        self.coefs = []
        self.best = best
        self.rs = rs
        self.n_f = n_f
        self.n = n  # number of subspaces
        self.ensemble = ensemble

        self.goods = []
        self.bads = []
        np.random.seed(123)

    def partial_fit(self, X, y, classes=None):
        self._X = np.copy(X)
        self._y = np.copy(y)
        n_features = X.shape[1]

        self.scaler = StandardScaler()
        rescaled_X = self.scaler.fit_transform(self._X)
        self.clf = LogisticRegression(solver="lbfgs", multi_class="auto")
        self.clf.fit(rescaled_X, y)

        coef = np.copy(np.abs(self.clf.coef_[0]))
        coef -= np.min(coef)
        coef /= np.sum(coef)
        self.coefs.append(coef)

        self.good = np.argsort(-coef)[: self.n_f]
        self.bad = np.argsort(coef)[: self.n_f]

        self.goods.append(self.good)
        self.bads.append(self.bad)

        # Train
        # Ensemble
        if self.ensemble:
            if self.best:
                self.good_subspaces = np.random.choice(
                    list(range(n_features)), size=(self.n, self.n_f), p=coef
                )
                self.good_ensemble = [
                    DecisionTreeClassifier().fit(X[:, subspace], y)
                    for subspace in self.good_subspaces
                ]
            if not self.best:
                bcoef = np.copy(coef)
                bcoef = 1 - bcoef
                bcoef -= np.min(bcoef)
                bcoef /= np.sum(bcoef)

                self.bad_subspaces = np.random.choice(
                    list(range(n_features)), size=(self.n, self.n_f), p=bcoef
                )

                self.bad_ensemble = [
                    DecisionTreeClassifier().fit(X[:, subspace], y)
                    for subspace in self.bad_subspaces
                ]
            if self.rs:
                self.random_subspaces = np.random.choice(
                    list(range(n_features)), size=(self.n, self.n_f)
                )

                self.random_ensemble = [
                    DecisionTreeClassifier().fit(X[:, subspace], y)
                    for subspace in self.random_subspaces
                ]
        else:
            if self.rs:
                self.rand = np.random.randint(X.shape[1], size=self.n_f)
                self.clf_rand = DecisionTreeClassifier()
                self.clf_rand.fit(X[:, self.rand], y)
            if self.best:
                self.clf_good = DecisionTreeClassifier()
                self.clf_good.fit(X[:, self.good], y)
            if not self.best:
                self.clf_bad = DecisionTreeClassifier()
                self.clf_bad.fit(X[:, self.bad], y)

    def predict(self, X):
        if self.ensemble:
            if self.rs:
                esm = np.array(
                    [
                        clf.predict_proba(X[:, self.random_subspaces[i]])
                        for i, clf in enumerate(self.random_ensemble)
                    ]
                )
                fesm = np.sum(esm, axis=0)
                y_pred = np.argmax(fesm, axis=1)

                return y_pred
            if self.best:
                esm = np.array(
                    [
                        clf.predict_proba(X[:, self.good_subspaces[i]])
                        for i, clf in enumerate(self.good_ensemble)
                    ]
                )
                fesm = np.sum(esm, axis=0)
                y_pred = np.argmax(fesm, axis=1)
                return y_pred
            if not self.best:
                esm = np.array(
                    [
                        clf.predict_proba(X[:, self.bad_subspaces[i]])
                        for i, clf in enumerate(self.bad_ensemble)
                    ]
                )
                fesm = np.sum(esm, axis=0)
                y_pred = np.argmax(fesm, axis=1)

                return y_pred
        else:
            if self.rs:
                set = X[:, self.rand]
                return self.clf_rand.predict(set)

            if self.best:
                set = X[:, self.good]
                return self.clf_good.predict(set)
            if not self.best:
                set = X[:, self.bad]
                return self.clf_bad.predict(set)


class SDTC(BaseEstimator, ClassifierMixin):
    def __init(self):
        pass

    def partial_fit(self, X, y, classes=None):
        self.clf = DecisionTreeClassifier()
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)
