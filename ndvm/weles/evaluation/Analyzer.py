"""
Absolutnie bez sensu. Powiela verbose z Evaluator.score().
"""

import numpy as np
from scipy.stats import rankdata
from tabulate import tabulate
from scipy.stats import wilcoxon


class Analyzer():
    def __init__(self, scores, clfs, metrics, datasets):
        self.scores = scores
        self.clfs = clfs
        self.metrics = metrics
        self.datasets = datasets

    def analyze(self, alpha=.05, stat_rank=wilcoxon):
        for m, metric in enumerate(self.metrics):
            print("################ ", metric, " ################")
            scores = self.scores[:,:,m]

            names_column = np.array(list(self.datasets.keys())).reshape(len(self.datasets), -1)
            scores_table = np.concatenate((names_column, scores), axis=1)
            print(tabulate(scores_table, headers=self.clfs.keys(), floatfmt=".3f"), scores_table.shape)

            # ranks
            ranks = []
            for row in scores:
                ranks.append(rankdata(row).tolist())
            ranks = np.array(ranks)
            mean_ranks = np.mean(ranks, axis=0)

            # strasznie głupi test globalny do texa
            # p_value = np.zeros((len(self.clfs), len(self.clfs)))
            #
            # for i in range(len(self.clfs)):
            #     for j in range(len(self.clfs)):
            #         _, p_value[i, j] = stat_rank(ranks.T[i], ranks.T[j], zero_method="zsplit")
            # dependency = p_value>alpha
            #
            # text = "\\ "
            # for i in range(mean_ranks.shape[0]):
            #     text += "& "
            #     a = np.where(dependency[i] == 0)[0]
            #
            #     for value in a:
            #         if mean_ranks[i] < mean_ranks[value]:
            #             a = a[a != value]
            #
            #     if a.size == mean_ranks.shape[0]-1:
            #         text += "$_{all}$"
            #     elif a.size == 0:
            #         text += "$_{-}$"
            #     else:
            #         a += 1
            #         text +=  "$_{" + ", ".join(["%i" % i for i in a]) + "}$"
            #
            # text += "\\\\\n"

            print("################ Mean ranks ################")
            print(tabulate(mean_ranks[np.newaxis,:], headers=self.clfs.keys(), floatfmt=".3f" ))
