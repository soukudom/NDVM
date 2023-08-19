import numpy as np
from scipy import stats


def t_test_corrected(a, b, J=5, k=5):
    """
    Corrected t-test for repeated cross-validation.
    input, two 2d arrays. Repetitions x folds
    As default for 5x5CV
    """
    if J*k != a.shape[0]:
        raise Exception('%i scores received, but J=%i, k=%i (J*k=%i)' % (
            a.shape[0], J, k, J*k
        ))

    d = a - b
    bar_d = np.mean(d)
    bar_sigma_2 = np.var(d.reshape(-1), ddof=1)
    bar_sigma_2_mod = (1 / (J * k) + 1 / (k - 1)) * bar_sigma_2
    t_stat = bar_d / np.sqrt(bar_sigma_2_mod)
    pval = stats.t.sf(np.abs(t_stat), (k * J) - 1) * 2
    return t_stat, pval


def t_test_13(a, b, corr=0.6):
    """
    Corrected t-test for repeated cross-validation.
    input, two 2d arrays. Repetitions x folds
    """
    k = len(a)  # J - repetitions, k - folds
    d = a - b
    bar_d = np.mean(d)
    bar_sigma_2 = np.var(d.reshape(-1), ddof=1)
    bar_sigma_2_mod = (1 / (k * (1 - corr))) * bar_sigma_2
    t_stat = bar_d / np.sqrt(bar_sigma_2_mod)
    pval = stats.t.sf(np.abs(t_stat), k - 1) * 2
    return t_stat, pval


def t_test_rel(a, b):
    """
    Paired, relative t-test.
    """
    J = len(a)
    d = a - b
    bar_d = np.mean(d)
    bar_sigma_2 = np.var(d.reshape(-1), ddof=1)
    bar_sigma_2_mod = (1 / J) * bar_sigma_2
    t_stat = bar_d / np.sqrt(bar_sigma_2_mod)
    pval = stats.t.sf(np.abs(t_stat), J - 1) * 2
    return t_stat, pval


IMPLEMENTED_TESTS = {
    "t_test_corrected": (t_test_corrected, {'J': 5, 'k': 5}),
    "t_test_13": (t_test_13, {'corr': 0.6}),
    "t_test_rel": (t_test_rel, {})
}
