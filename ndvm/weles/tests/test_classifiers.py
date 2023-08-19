from sklearn.datasets import make_classification
import weles as ws


def dataset():
    return make_classification(random_state=1410)


def test_linear_classifier():
    clf = ws.classifiers.LinearClassifier()
    X, y = dataset()
    clf.fit(X, y)
    y_pred = clf.predict(X)


def test_exposer_classifier():
    clf = ws.classifiers.ExposerClassifier()
    X, y = dataset()
    clf.fit(X, y)
    y_pred = clf.predict(X)

    for i in range(5):
        clf.partial_fit(X, y)


def test_ssgnb():
    clf = ws.classifiers.SSGNB()
    X, y = dataset()
    clf.fit(X, y)
    y_pred = clf.predict(X)
    subspaces = [[0, 1, 2], [3], [7, 9]]
    y_preds = clf.predict(X, subspaces)
