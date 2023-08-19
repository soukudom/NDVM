"""
Genetic Ensemble Selection
"""

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
import numpy as np
from scipy.special import binom

VAR = 10000000
np.set_printoptions(suppress=True, precision=3)


class GES(BaseEstimator, ClassifierMixin):
    """
    Genetic Ensemble Selection
    """

    def __init__(
        self,
        pool_size=20,
        ensemble_size=5,
        num_iter=50,
        base_clf=LogisticRegression(solver="lbfgs"),
        elite_limit=1,
        p_crossing=0.025,
        p_mutation=0.01,
        alpha=0,
        beta=0,
        metric=f1_score,
        random_state=13,
    ):
        # Ensemble parameters
        self.pool_size = pool_size  # Number of ensembles in pool
        self.ensemble_size = ensemble_size  # Number of classifiers in ensemble
        self.base_clf = base_clf

        # Genetic parameters
        self.num_iter = num_iter
        self.elite_limit = elite_limit
        self.p_crossing = p_crossing
        self.p_mutation = p_mutation
        self.random_state = random_state

        # Regularization parameters
        self.alpha = alpha  # strength of parameters usage
        self.beta = beta  # strength of hamming distance
        self.metric = metric

    # @profile
    def fit(self, X, y):
        """
        Process training set.
        """
        # Setting random state
        np.random.seed(self.random_state)

        # Store training set properties
        self.X_, self.y_ = X, y
        self.n_features = X.shape[1]
        self.n_objects = X.shape[0]
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)

        # Build base model on all features and store default coef
        self.base_clf.fit(self.X_, self.y_)
        if hasattr(self.base_clf, "coef_"):
            self.coef_ = np.copy(self.base_clf.coef_[0])
        if hasattr(self.base_clf, "sigma_"):
            self.sigma_ = np.copy(self.base_clf.sigma_)

        # Prepare storage for processing
        self.all_scores = []
        self.all_qualities = []

        # Prepare random feature selection model
        self.model_ = (
            np.random.randint(
                2, size=(self.pool_size, self.ensemble_size, self.n_features)
            )
            == 1
        )

        # Prepare probas storage
        self.probas_storage = {}

        # Iterate epochs
        for i in range(self.num_iter):
            # Perform mutation
            self.mutate()

            # Perform crossing
            self.cross()

            # Get model probas, calculate supports and return scores
            probas = self.get_model_probas()
            supports = np.mean(probas, axis=1)
            y_preds = np.argmax(supports, axis=2)
            scores = np.array([self.metric(self.y_, y_pred) for y_pred in y_preds])

            # Calculate regularizations and quality
            a = self.reg_a()
            b = self.reg_b()
            q = scores - (self.alpha * a) + (self.beta * b)

            # Update history
            self.all_scores.append(scores)
            self.all_qualities.append(q)

            # Sort model by obtained measure
            sorter = (-q).argsort()
            self.model_ = self.model_[sorter]
            scores = scores[sorter]
            q = q[sorter]

            # Selection
            if np.min(q) < 0:  # Deal with negative qualities
                p = q - np.min(q)
            else:
                p = q
            p[p == 0] = 0.0000001
            p = p[self.elite_limit :] / np.sum(p[self.elite_limit :])
            c = np.random.choice(
                range(self.elite_limit, self.pool_size),
                size=self.pool_size - self.elite_limit,
                p=p,
            )
            self.model_[self.elite_limit :] = self.model_[c]
            q[self.elite_limit :] = q[c]

        # Convert scores
        self.all_scores = np.array(self.all_scores)
        self.all_qualities = np.array(self.all_qualities)

    def reg_a(self):
        return np.sum(np.max(self.model_, axis=1), axis=1) / self.n_features

    def reg_b(self):
        d = np.zeros((self.pool_size)).astype(int)
        for e, ensemble in enumerate(self.model_):
            for i in range(self.ensemble_size):
                for j in range(i + 1, self.ensemble_size):
                    d[e] += len(np.bitwise_xor(ensemble[i], ensemble[j]).nonzero()[0])
        return d / binom(self.ensemble_size, 2) / self.n_features

    def get_model_probas(self):
        # Prepare structure
        probas = np.zeros(
            (self.pool_size, self.ensemble_size, self.n_objects, self.n_classes)
        )

        # Iterate ensembles
        for e, ensemble in enumerate(self.model_):
            for s, subspace in enumerate(ensemble):
                # Check if probing already done for subspace
                if tuple(subspace) not in self.probas_storage:
                    if hasattr(self, "coef_"):
                        self.probas_storage.update(
                            {
                                tuple(subspace): self._predict_proba_with_coef(
                                    self.X_, self.coef_ * subspace
                                )
                            }
                        )
                    if hasattr(self, "sigma_"):
                        self.probas_storage.update(
                            {
                                tuple(subspace): self._predict_proba_with_sigma(
                                    self.X_, self.sigma_ * subspace
                                )
                            }
                        )
                # Store
                probas[e, s] = self.probas_storage[tuple(subspace)]
        return probas

    def predict(self, X):
        if hasattr(self, "coef_"):
            predict_probas = np.array(
                [
                    self._predict_proba_with_coef(X, self.coef_ * subspace)
                    for subspace in self.model_[0]
                ]
            )
        if hasattr(self, "sigma_"):
            predict_probas = np.array(
                [
                    self._predict_proba_with_sigma(X, self.sigma_ * subspace)
                    for subspace in self.model_[0]
                ]
            )
        predict_probas = np.mean(predict_probas, axis=0)
        prediction = np.argmax(predict_probas, axis=1)
        return prediction

    def _predict_proba_with_coef(self, X, coef_):
        self.base_clf.coef_[0] = coef_
        predict_proba = self.base_clf.predict_proba(X)
        return predict_proba

    def _predict_proba_with_sigma(self, X, sigma_):
        sigma_[sigma_ == 0] = VAR
        self.base_clf.sigma_ = sigma_
        predict_proba = self.base_clf.predict_proba(X)
        return predict_proba

    def mutate(self):
        # Generate random mutation mask
        mask = (
            np.random.rand(self.pool_size, self.ensemble_size, self.n_features)
            < self.p_mutation
        )
        # Conserve elite
        mask[: self.elite_limit] = False

        # Mutate
        self.model_[mask] = np.invert(self.model_[mask])

    def cross(self):
        # Copy elite
        _elite = np.copy(self.model_[: self.elite_limit])

        # Match partners with given probability
        to_replace = np.where(
            np.random.rand(self.pool_size - self.elite_limit) < self.p_crossing
        )[0]
        partners = np.random.randint(self.pool_size, size=to_replace.size)

        # Generate crossing pattern
        pattern = (
            np.random.randint(
                2, size=(to_replace.size, self.ensemble_size, self.n_features)
            )
            == 1
        )

        # Perform all crossings
        for m, (i, j) in enumerate(zip(to_replace, partners)):
            self.model_[i, pattern[m]] = self.model_[j, pattern[m]]

        # Restore elite
        self.model_[: self.elite_limit] = _elite
