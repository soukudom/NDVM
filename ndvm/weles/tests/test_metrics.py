from sklearn.datasets import make_classification
import sklearn as skl
import weles as ws


def multiclass_dataset():
    return make_classification(random_state=1410, n_classes=5,
                               n_informative=11)


def test_balanced_accuracy_score():
    clf = ws.classifiers.LinearClassifier()
    X, y = multiclass_dataset()
    clf.fit(X, y)
    y_pred = clf.predict(X)
    score_1 = ws.metrics.balanced_accuracy_score(y, y_pred)
    score_2 = skl.metrics.balanced_accuracy_score(y, y_pred)
    assert score_1 == score_2
