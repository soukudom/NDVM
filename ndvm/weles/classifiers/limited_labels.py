import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator, clone
#import strlearn as sl


class BLS(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, budget=0.5, random_state=None):
        self.budget = budget
        self.base_estimator = base_estimator
        self.random_state = random_state

    def partial_fit(self, X, y, classes=None):
        np.random.seed(self.random_state)
        # First train
        if not hasattr(self, "clf"):
            # Pierwszy chunk na pelnym
            self.clf = clone(self.base_estimator)

        # Get random subset
        limit = int(self.budget * len(y))
        idx = np.array(list(range(len(y))))
        selected = np.random.choice(idx, size=limit, replace=False)

        # print(X[selected].shape)
        # Partial fit
        self.clf.partial_fit(X[selected], y[selected], classes)

    def predict(self, X):
        return self.clf.predict(X)


class ALS(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, treshold=0.2, budget=1.0):
        self.treshold = treshold
        self.base_estimator = base_estimator
        self.budget = budget

    def partial_fit(self, X, y, classes=None):
        # First train
        limit = int(self.budget * len(y))
        if not hasattr(self, "clf"):
            # Pierwszy chunk na pelnym
            self.clf = clone(self.base_estimator).partial_fit(X, y, classes=classes)
            self.usage = []

        else:
            supports = np.abs(self.clf.predict_proba(X)[:, 0] - 0.5)
            closest = np.argsort(supports)[:limit]
            selected = closest[supports[closest]<self.treshold]
            # selected = supports < self.treshold
            # print(closest)
            # print(y[selected])
            if np.sum(selected) > 0:
                self.clf.partial_fit(X[selected], y[selected], classes)

                # score = sl.metrics.balanced_accuracy_score(
                    # y[selected], self.clf.predict(X[selected])
                # )

                # self.treshold = 0.5 - score / 2

            self.usage.append(selected.shape[0]/X.shape[0])
            # print(np.mean(np.array(self.usage)))
            self.used = (np.mean(np.array(self.usage)))
            # sys.stdout.write("%f%%   \r" % (np.mean(np.array(self.usage))) )
            # sys.stdout.flush()

    def predict(self, X):
        return self.clf.predict(X)


class BALS(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, treshold=0.2, budget=0.2, random_state=None):
        self.treshold = treshold
        self.budget = budget
        self.base_estimator = base_estimator
        self.random_state = random_state

    def partial_fit(self, X, y, classes=None):
        np.random.seed(self.random_state)
        # First train
        if not hasattr(self, "clf"):
            # Pierwszy chunk na pelnym
            self.clf = clone(self.base_estimator).partial_fit(X, y, classes=classes)
            self.usage = []

        else:
            supports = np.abs(self.clf.predict_proba(X)[:, 0] - 0.5)
            selected = supports < self.treshold

            if np.sum(selected) > 0:
                self.clf.partial_fit(X[selected], y[selected], classes)

                score = 0 #sl.metrics.balanced_accuracy_score(
                    #y[selected], self.clf.predict(X[selected])
                #)

                # self.treshold = 0.5 - score / 2

            self.usage.append(np.sum(selected) / selected.shape)

            # Get random subset
            limit = int(self.budget * len(y))
            idx = np.array(list(range(len(y))))
            selected = np.random.choice(idx, size=limit, replace=False)

            # Partial fit
            self.clf.partial_fit(X[selected], y[selected], classes)

    def predict(self, X):
        return self.clf.predict(X)
