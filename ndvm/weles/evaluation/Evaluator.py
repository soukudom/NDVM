"""
Class description

Date:
Authors:
"""

# imports
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from tabulate import tabulate
from tqdm import tqdm
from scipy.stats import rankdata
from os import path, listdir
from hashlib import md5
import inspect
from multiprocessing import Pool

VERBOSE_COLUMNS = 80


class Evaluator():
    def __init__(self, datasets, protocol=(1, 5, None), protocol2=(False, 5, None), store=None):
        self.datasets = datasets
        self.protocol = protocol
        self.protocol2 = protocol2
        self.store = store

        # Check storage
        if store is not None:
            if path.isdir(store):
                self.stored = listdir(self.store)
            else:
                raise Exception("There is no path '%s' to store results" %
                                store)
        else:
            pass
            #print("Store is none")

    def process(self, clfs, verbose=False):
        """
        This function is used to process declared evaluation protocol
        through all given datasets and classifiers.
        It results with stored predictions and corresponding labels.

        Input atguments description:
        clfs: dictonary that contains estimators names and objects
              ["name"] : obj
        """
        self.clfs = clfs

        # Establish protocol
        self.m, self.k, self.random_state = self.protocol
        self.s, self.k, self.random_state = self.protocol2
        if self.random_state is None:
            self.store = None

        skf = RepeatedStratifiedKFold(n_splits=self.k, n_repeats=self.m,
                                      random_state=self.random_state)
        
        # default StratifiedKFold(n_splits=5, shuffle=False, random_state=None)
        # KFold that returns stratified folds.
        # Suffle - Whether to shuffle each class’s samples before splitting into batches. 
        # Note that the samples within each split will not be shuffled.
        # we call protocol=(shuffle, n_splits, random_state)
        nkf = StratifiedKFold(n_splits=self.k, shuffle=self.s, random_state=self.random_state)
        
        self.predictions = np.zeros([len(self.datasets), len(self.clfs),
                                     self.m * self.k], dtype=object)
        self.true_values = np.zeros([len(self.datasets), self.m * self.k],
                                    dtype=object)

        # Iterate over datasets
        for dataset_id, dataset_name in enumerate(tqdm(self.datasets,
                                                       desc="DTS",
                                                       ascii=True,
                                                       disable=not verbose)):
            X, y = self.datasets[dataset_name]
            for fold_id, (train, test) in enumerate(nkf.split(X, y)):
                str_gt = self._storage_key_gt(X, y, fold_id)
                self.true_values[dataset_id, fold_id] = y[test]

                for clf_id, clf_name in enumerate(self.clfs):
                    str_clf = self._storage_key_pred(str_gt,
                                                     self.clfs[clf_name])

                    if self.store is not None and str_clf+".npy" in self.stored:
                        y_pred = np.load("%s/%s.npy" % (self.store, str_clf))
                    else:
                        clf = clone(self.clfs[clf_name])
                        clf.fit(X[train], y[train])
                        y_pred = clf.predict(X[test])
                        if self.store is not None:
                            np.save("%s/%s" % (self.store, str_clf), y_pred)
                    self.predictions[dataset_id, clf_id, fold_id] = y_pred

        return self

    def _storage_key_pred(self, gt, clf):
        if self.random_state is None:
            return None

        m = md5()
        m.update(gt.encode("ascii"))
        with open(inspect.getfile(clf.__class__), 'r') as file:
            m.update(file.read().encode("utf8"))
        m.update(str(clf).encode("ascii"))

        return(m.hexdigest())

    def _storage_key_gt(self, X, y, fold_id):
        if self.random_state is None:
            return None

        m = md5()
        m.update(X.copy(order='C'))
        m.update(y)
        m.update(str(fold_id).encode("ascii"))
        m.update(str(self.m).encode("ascii"))
        m.update(str(self.k).encode("ascii"))
        m.update(str(self.random_state).encode("ascii"))
        return(m.hexdigest())

    def score(self, metrics, verbose=False, return_flatten=True):
        """
        description

        Input arguments description:
        metrics: dictonary that contains metrics names and functions
                 ["name"] : function
        """
        self.metrics = metrics

        # Prepare storage for scores
        # DB x CLF x FOLD x METRIC
        self.scores = np.array([[[[
            metrics[m_name](
                self.true_values[db_idx, f_idx],
                self.predictions[db_idx, clf_idx, f_idx])
            for m_name in self.metrics]
            for f_idx in range(self.m * self.k)]
            for clf_idx, clf in enumerate(self.clfs)]
            for db_idx, db_name in enumerate(self.datasets)])

        # Store mean scores and stds
        # DB x CLF x METRIC
        self.mean_scores = np.mean(self.scores, axis=2)
        self.stds = np.std(self.scores, axis=2)

        lmn = len(max(list(self.metrics.keys()), key=len))
        lmc = (VERBOSE_COLUMNS-lmn)//2
        self.mean_ranks = []
        self.ranks = []
        for m, metric in enumerate(self.metrics):
            scores_ = self.mean_scores[:, :, m]

            # ranks
            ranks = []
            for row in scores_:
                ranks.append(rankdata(row).tolist())
            ranks = np.array(ranks)
            self.ranks.append(ranks)
            mean_ranks = np.mean(ranks, axis=0)
            self.mean_ranks.append(mean_ranks)
            names_column = np.array(list(self.datasets.keys())).reshape(
                len(self.datasets), -1)
            scores_table = np.concatenate((names_column, scores_), axis=1)
            if verbose:
                print(lmc*"#", metric.center(lmn), lmc*"#")
                print(tabulate(scores_table, headers=self.clfs.keys(),
                               floatfmt=".3f"))

                print(lmc*"-", "Mean ranks".center(lmn), lmc*"-")
                print(tabulate(mean_ranks[np.newaxis, :],
                               headers=self.clfs.keys(), floatfmt=".3f"))
        self.mean_ranks = np.array(self.mean_ranks)
        self.ranks = np.array(self.ranks)
        # Give output
        return {
            True: (self.mean_scores, self.stds),
            False: self.scores,
        }[return_flatten]
