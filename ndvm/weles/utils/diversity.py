import numpy as np

# Samo E
def entropy_measure_e(ensemble, X, y):
    L = len(ensemble)
    return np.mean(
        (
            L // 2
            - np.abs(
                np.sum(
                    y[np.newaxis, :] == np.array([clf.predict(X) for clf in ensemble]),
                    axis=0,
                )
                - L // 2
            )
        )
        / (L / 2)
    )

def _make_relationship_table(d_pred):
    rt = np.zeros(shape=(2, 2))

    for idx, val in zip(*np.unique(d_pred.T, axis=0, return_counts=True)):
        rt[tuple(idx)] = val

    return rt


def make_relationship_tables(predictions, full_matrix=False):
    pool_len = len(predictions)

    if full_matrix:
        return np.array([
            [
                _make_relationship_table(predictions[(i, k), :])
                for k in range(pool_len)
            ]
            for i in range(pool_len)
        ]).T

    return np.array([
        _make_relationship_table(predictions[(i, k), :])
        for i in range(pool_len) for k in range(i + 1, pool_len)
    ]).T


# Q-statistic
def Q_statistic(relationship_tables):
    (n00, n01), (n10, n11) = relationship_tables
    divisor = ((n11 * n00) + (n01 * n10))
    divisor[divisor==0] = 0.000000000001
    return ((n11 * n00) - (n01 * n10)) / divisor


# Correlation coefficient
def correlation_coefficient(relationship_tables):
    (n00, n01), (n10, n11) = relationship_tables
    return ((n11 * n00) - (n01 * n10)) / np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))


# Disagreement measure
def disagreement_measure(relationship_tables):
    (n00, n01), (n10, n11) = relationship_tables
    # return (n01 + n10) / (n11 + n10 + n01 + n00)
    return (n01 + n10) / relationship_tables.sum(axis=(0, 1))


# Double-fault measure
def double_fault_measure(relationship_tables):
    (n00, n01), (n10, n11) = relationship_tables
    # return n00 / (n11 + n10 + n01 + n00)
    return n00 / relationship_tables.sum(axis=(0, 1))

# Calculates all 5 metrics
def calc_diversity_measures(X, y, classifier_pool, p=0):
    L = len(classifier_pool)
    predictions = np.array([np.equal(_.predict(X), y).astype(np.int) for _ in classifier_pool])
    tables = make_relationship_tables(predictions)

    q = Q_statistic(tables).mean()
    dis = disagreement_measure(tables).mean()
    kw = ((L-1) / (2*L)) * dis
    k = 1 - (1/(2*p*(1-p)))*dis
    e = np.mean(
        (
            L // 2
            - np.abs(
                np.sum(
                    y[np.newaxis, :] == np.array([member_clf.predict(X) for member_clf in classifier_pool]),
                    axis=0,
                )
                - L // 2
            )
        )
        / (L / 2)
    )
    return e, k, kw, dis, q

# Calculates only a given metric
def calc_diversity_measures2(X, y, classifier_pool, subspaces, p=0, measure=None):
    L = len(classifier_pool)
    if measure == "e":
        e = np.mean(
            (
                L // 2
                - np.abs(
                    np.sum(
                        y[np.newaxis, :] == np.array([member_clf.predict(X[:, subspaces[clf_ind]]) for clf_ind, member_clf in enumerate(classifier_pool)]),
                        axis=0,
                    )
                    - L // 2
                )
            )
            / (L / 2)
        )
        return e
    elif measure == "q":
        predictions = np.array([np.equal(_.predict(X[:, subspaces[i]]), y).astype(np.int) for i, _ in enumerate(classifier_pool)])
        tables = make_relationship_tables(predictions)
        q = Q_statistic(tables).mean()
        return q
    elif measure == "dis":
        predictions = np.array([np.equal(_.predict(X[:, subspaces[i]]), y).astype(np.int) for i, _ in enumerate(classifier_pool)])
        tables = make_relationship_tables(predictions)
        dis = disagreement_measure(tables).mean()
        return dis
    elif measure == "k":
        predictions = np.array([np.equal(_.predict(X[:, subspaces[i]]), y).astype(np.int) for i, _ in enumerate(classifier_pool)])
        tables = make_relationship_tables(predictions)
        dis = disagreement_measure(tables).mean()
        k = 1 - (1/(2*p*(1-p)))*dis
        return k
    elif measure == "kw":
        predictions = np.array([np.equal(_.predict(X[:, subspaces[i]]), y).astype(np.int) for i, _ in enumerate(classifier_pool)])
        tables = make_relationship_tables(predictions)
        dis = disagreement_measure(tables).mean()
        kw = ((L-1) / (2*L)) * dis
        return kw
    else:
        print("Not a valid diversity measure")

# Subspaces -- If you need diversity for random subspace
def calc_diversity_measures(X, y, classifier_pool, subspaces, p=0):
    L = len(classifier_pool)
    predictions = np.array([np.equal(_.predict(X[:, subspaces[i]]), y).astype(np.int) for i, _ in enumerate(classifier_pool)])
    tables = make_relationship_tables(predictions)

    q = Q_statistic(tables).mean()
    dis = disagreement_measure(tables).mean()
    kw = ((L-1) / (2*L)) * dis
    k = 1 - (1/(2*p*(1-p)))*dis
    e = np.mean(
        (
            L // 2
            - np.abs(
                np.sum(
                    y[np.newaxis, :] == np.array([member_clf.predict(X[:, subspaces[clf_ind]]) for clf_ind, member_clf in enumerate(classifier_pool)]),
                    axis=0,
                )
                - L // 2
            )
        )
        / (L / 2)
    )
    # The same but different
    # k2 = 1 - (len(classifier_pool)/((len(classifier_pool)-1)*p*(1-p)))*kw
    return e, k, kw, dis, q
