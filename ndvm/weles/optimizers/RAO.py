"""
Regression-aided optimizer.
"""

from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import explained_variance_score
import numpy as np
from scipy.optimize import minimize, rosen, rosen_der


class RAO(BaseEstimator):
    def __init__(
        self,
        base_estimator,
        n_splits=5,
        n_guesses=100000,
        metric=explained_variance_score,
        random_state=None,
        bounds=[(1201, 3500), (601, 1200), (0, 600)],
    ):
        self.base_estimator = base_estimator
        self.n_splits = n_splits
        self.metric = metric
        self.random_state = random_state
        self.n_guesses = n_guesses
        self.bounds = bounds

    def fit(self, X, y):
        # Store features and labels
        self.X = np.copy(X)
        self.y = np.copy(y)

        # Prepare storage for ensemble parameters
        self.ensemble = []
        self.feature_scalers = []
        self.label_scalers = []
        self.scores = np.zeros(self.n_splits)

        # Build regression model
        kfold = KFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
        for fold, (train, test) in enumerate(kfold.split(X)):
            # Train scalers
            feature_scaler = StandardScaler().fit(X[train])
            label_scaler = StandardScaler().fit(y[train].reshape(-1, 1))

            # Build member model
            reg = clone(self.base_estimator)
            reg.fit(
                feature_scaler.transform(X[train]),
                label_scaler.transform(y[train].reshape(-1, 1)),
            )

            # Test member model
            y_pred = reg.predict(feature_scaler.transform(X[test]))
            scaled_y_test = label_scaler.transform(y[test].reshape(-1, 1))
            score = self.metric(scaled_y_test, y_pred)

            # Store member model
            self.ensemble.append(reg)
            self.feature_scalers.append(feature_scaler)
            self.label_scalers.append(label_scaler)
            self.scores[fold] = score

        # Normalize scores to weights
        self.weights = self.scores / np.sum(self.scores)

        # Estimate optimization bounds
        # min_bounds = np.min(X, axis=0)
        # max_bounds = np.max(X, axis=0)
        # self.bounds = [(min_bounds[i], max_bounds[i]) for i in range(X.shape[1])]

        return self

    def predict(self, X):
        # Acquire all predictions
        y_pred_ensemble = np.array(
            [
                reg.predict(self.feature_scalers[i].transform(X))
                for i, reg in enumerate(self.ensemble)
            ]
        )

        if len(y_pred_ensemble.shape) == 3:
            y_pred_ensemble = y_pred_ensemble[:, :, 0]
        print("YPREDENS", y_pred_ensemble.shape)

        # print(y_pred_ensemble.shape)

        # Rescale them
        rescaled_y_pred_ensemble = np.array(
            [
                self.label_scalers[i].inverse_transform(y_pred)
                for i, y_pred in enumerate(y_pred_ensemble)
            ]
        )
        """ FORMER
        rescaled_y_pred_ensemble = np.mean(
            np.array(
                [
                    self.label_scalers[i].inverse_transform(y_pred)
                    for i, y_pred in enumerate(y_pred_ensemble)
                ]
            ),
            axis=2,
        )
        """
        # print(rescaled_y_pred_ensemble.shape)

        # print("REAL", self.y[:3], self.y[-3:])
        # print("RESC", rescaled_y_pred_ensemble, rescaled_y_pred_ensemble.shape)

        # Weight and flattern them
        y_pred = np.sum(rescaled_y_pred_ensemble * self.weights[:, np.newaxis], axis=0)

        return y_pred

    def _optfun(self, x):
        # y =
        # print("OPTFUN", x, y)
        return self.predict([x])

    def optimize(self):
        """
        Monte Carlo optimizer.
        """
        np.random.seed(self.random_state)
        # Generate guesses
        X = np.array(
            [
                np.random.randint(bound[0], bound[1], self.n_guesses)
                for bound in self.bounds
            ]
        ).T

        # Calculate predictions
        y_pred = self.predict(X)

        # Gather minimized prediction
        min_idx = np.argmin(y_pred)

        return (X[min_idx], y_pred[min_idx])

        print(X, X.shape)
        print(y_pred, y_pred.shape)

        exit()
        # res = minimize(self._optfun, [self.stored_minimum()[0]], bounds=self.bounds)
        # return (res.x.astype(int), res.fun[0])

    def inner_confidence(self):
        return np.mean(self.scores)

    def stored_minimum(self):
        """
        Returns x and y for stored minimum, supplemented with predicted y.
        """
        min_idx = np.argmin(self.y)
        y_pred = self.predict([self.X[min_idx]])[0]
        return (self.X[min_idx].astype(int), y_pred, self.y[min_idx])
