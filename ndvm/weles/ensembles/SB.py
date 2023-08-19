"""
Stratified Bagging.
"""
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import (check_X_y, check_array,
                                      check_is_fitted,
                                      check_random_state)
import numpy as np
from sklearn.metrics import accuracy_score
from scipy.stats import mode


class SB(BaseEnsemble, ClassifierMixin):
    def __init__(self, base_estimator=None, n_estimators=10, voting='soft',
                 random_state=None):
        """Initialization."""
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.voting = voting
        self.random_state = random_state
        self.rng = check_random_state(self.random_state)

    def fit(self, X, y):
        """Fitting."""
        X, y = check_X_y(X, y)
        self.X_, self.y_ = X, y

        self.classes_ = np.unique(y)
        self.estimators_ = []

        for i in range(self.n_estimators):
            selected_samples = [
                self.rng.randint(
                    0, self.X_[self.y_ == label].shape[0],
                    self.X_[self.y_ == label].shape[0])
                for label in self.classes_
            ]

            X_train = np.concatenate(
                (self.X_[self.y_ == 0][selected_samples[0]],
                 self.X_[self.y_ == 1][selected_samples[1]]), axis=0)
            y_train = np.concatenate(
                (self.y_[self.y_ == 0][selected_samples[0]],
                 self.y_[self.y_ == 1][selected_samples[1]]), axis=0)

            self.estimators_.append(
                clone(self.base_estimator).fit(X_train, y_train)
            )
        return self

    def ensemble_support_matrix(self, X):
        """ESM."""
        return np.array(
            [member_clf.predict_proba(X) for member_clf in self.estimators_]
        )

    def predict_proba(self, X):
        """Predict proba"""
        # Check is fit had been called
        check_is_fitted(self, "classes_")
        # Input validation
        X = check_array(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("Number of features does not match")

        esm = self.ensemble_support_matrix(X)
        average_support = np.mean(esm, axis=0)

        return average_support

    def predict(self, X):
        # Input validation
        X = check_array(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("Number of features does not match")

        # Support Accumulation
        if self.voting == 'soft':
            esm = self.ensemble_support_matrix(X)
            average_support = np.mean(esm, axis=0)
            prediction = np.argmax(average_support, axis=1)
        # Majority Voting
        elif self.voting == 'hard':
            predictions = np.array(
                [member_clf.predict(X) for member_clf in self.estimators_]
             )
            prediction = np.squeeze(mode(predictions, axis=0)[0])
        else:
            raise ValueError("Invalid voting type")
        return prediction

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
