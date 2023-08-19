from sklearn.datasets import make_classification
import weles as ws


def dataset():
    return make_classification(random_state=1410)


def test_geometron():
    base_estimator = ws.classifiers.LinearClassifier()
    clf = ws.ensembles.Geometron(base_estimator)
    X, y = dataset()
    clf.fit(X, y)
    y_pred = clf.predict(X)
