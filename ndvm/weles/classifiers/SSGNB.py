"""
Subspaced Gaussian Naive Bayes
"""
from sklearn.base import ClassifierMixin, BaseEstimator
import numpy as np


class SSGNB(ClassifierMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y, classes=None):
        # Gather prior info
        self.classes, self.prior = np.unique(y, return_counts=True)
        self.prior = self.prior / X.shape[1]
        self.n_features = X.shape[1]
        self.n_classes = len(self.classes)

        # Calculate current norm
        self.means = np.array(
            [np.mean(X[y == label], axis=0) for label in self.classes]
        )
        self.stds = np.array([np.std(X[y == label], axis=0) for label in self.classes])

        return self

    def predict(self, X, subspaces=None):
        if subspaces is None:
            # Calculate distribution density
            ps = np.product(
                [self._pdf(X, self.stds[c], self.means[c]).T for c in self.classes],
                axis=1,
            )

            # Establish and return prediction
            y_pred = np.argmax(self.prior[:, np.newaxis] * ps, axis=0)
            return y_pred
        else:
            # One estimation to meet all requirements
            # To jest fantastyczny tensor z niezależnymi wartościami
            # funkcji gęstości rozkładu dla predykowanego zbioru.
            # Ma wymiary:
            # - klasa
            # - cecha
            # - testowany wzorzec
            # [gęstości rozkładu]
            psf = np.array(
                [self._pdf(X, self.stds[c], self.means[c]).T for c in self.classes]
            )

            # Możemy sobie teraz przeiterować podprzestrzenie i dla każdej z
            # nich wyznaczyć podprzestrzenny produkt. Otrzymamy tensor pse o
            # wymiarach:
            # - identyfikator podprzestrzeni
            # - klasa
            # - testowany wzorzec
            # [gęstości rozkładu]
            pse = np.array([np.product(psf[:, ss, :], axis=1) for ss in subspaces])

            # Z takiego cudu możliwa jest to prostego wyliczenia macierz
            # predykcji komitetu o wymiarach:
            # - identyfikator podprzestrzeni
            # - predykcja
            # [predykcje]
            y_preds = np.argmax(self.prior[np.newaxis, :, np.newaxis] * pse, axis=1)

            return y_preds

    def _pdf(self, x, std=1, mean=0):
        """
        Probability density function.
        """
        return np.exp(-(0.5) * np.power((x - mean) / std, 2)) / (
            std * np.sqrt(2 * np.pi)
        )
