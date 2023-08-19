import numpy as np


def balanced_accuracy_score(y_true, y_pred):
    classes = np.unique(y_true)
    n_classes = classes.shape[0]
    _ = np.arange(n_classes*n_classes).reshape(n_classes, n_classes)
    _ = (_ % n_classes, _ // n_classes)
    cm = np.sum((y_true[:, np.newaxis] == classes[np.newaxis, :]).T[_[0]] *
                (y_pred[:, np.newaxis] == classes[np.newaxis, :]).T[_[1]],
                axis=2)
    return np.mean(cm.diagonal() / np.sum(cm, axis=0))
