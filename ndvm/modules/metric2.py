"""
    Dataset Label a Calculation
"""
import warnings
import weles as ws
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.cm as cm
from multiprocessing import Pool, Process, Manager, current_process
from core import AbstractMetric

# import dataframe_image as dfi
import yaml
from yaml.loader import SafeLoader
from pandas import read_csv
import os
import argparse
import sys
import pickle
from sklearn.metrics import auc


class Association(AbstractMetric):
    def __init__(self, dataset, label, multiclass, verbose):
        self.raw_dataset = dataset
        self.label = label
        self.datasets = None
        self.clfs = ["RF", "DT", "XGB"]
        self.clfs_ver1 = []
        self.clfs_ver2 = {}
        self.ev = None
        self.X1 = None
        self.y1 = None
        self.dataset_params = {}
        self.a = None
        self.eval_metrics = ["F1"]
        self.metrics = {}
        self.perc = [50, 40, 30, 20, 10, 1]
        self.perm = None
        self.corr = None
        self.nperm = 100
        self.output = None
        self.cores = 1
        self.verbose = verbose
        self.auc_score = 0
        self.MULTICLASS = multiclass
        self.max_clf_metric_score = 0
        self.raw_slope = 0
        self.raw_auc = 0

    def get_details(self):
        pass

    def get_name(self):
        return "Association"

    def load_dataset(self):
        """
            Load dataset
        """
        try:
            self.y1 = self.raw_dataset[self.label]
            self.X1 = self.raw_dataset.drop(columns=[self.label])
            self.y1 = self.y1.astype('category')
            self.y1 = self.y1.cat.codes
            self.X1 = MinMaxScaler().fit_transform(self.X1)
        except ValueError as err:
            print()
            print("Error: convert string value to number.")
            print("Full Error Message:", err)
            sys.exit(2)

        self.datasets = {"all": (self.X1, self.y1)}

    # TODO handle unknown or duplicit models defined in configuration files
    def run_classifiers(self):
        """
            Initialize listed pool of classifiers
        """
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
        from xgboost import XGBClassifier
        from sklearn.neural_network import MLPClassifier

        # Pool of available models for selected pool of classifiers
        if self.MULTICLASS:
            clfs_pool = {
                #"KNN": KNeighborsClassifier(),
                "DT": DecisionTreeClassifier(criterion="gini"),
                "RF": RandomForestClassifier(class_weight="balanced", criterion="gini"),
                #"MLP": MLPClassifier(
                #    hidden_layer_sizes=(80, 100), activation="relu", batch_size=20, max_iter=200, verbose=0
                #),
                #"AB": AdaBoostClassifier(),
                "XGB": XGBClassifier(objective="multi:softmax"),
            }

        else:
            clfs_pool = {
                "KNN": KNeighborsClassifier(),
                "DT": DecisionTreeClassifier(),
                "RF": RandomForestClassifier(),
                "MLP": MLPClassifier(
                    hidden_layer_sizes=(80, 100), activation="relu", batch_size=20, max_iter=200, verbose=0
                ),
                "AB": AdaBoostClassifier(),
                "XGB": XGBClassifier(eval_metric="logloss")
            }


        for classifier in self.clfs:
            self.clfs_ver1.append({classifier: clfs_pool[classifier]})
            self.clfs_ver2[classifier] = clfs_pool[classifier]

    def run_metrics(self):
        """
            Initialize metrics and set true values
        """
        from sklearn.metrics import (
            precision_score,
            f1_score,
            balanced_accuracy_score,
            average_precision_score,
            matthews_corrcoef,
            roc_auc_score,
            accuracy_score,
            fbeta_score,
            recall_score,
        )
        from imblearn.metrics import sensitivity_score, specificity_score

        self.ev = ws.evaluation.Evaluator(datasets=self.datasets, protocol2=(False, 2, None)).process(
            clfs=self.clfs_ver2, verbose=0
        )

        def true_positive_rate(y_true, y_pred):
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            return tp / (tp + fn)

        def false_positive_rate(y_true, y_pred):
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            tn = ((y_pred == 0) & (y_true == 0)).sum()
            return fp / (fp + tn)

        def precision_from_tpr_fpr(y_true, y_pred):
            # positive class prevalence
            self.y1 = self.datasets["all"][1]
            count1 = (self.y1 == 1).sum()
            N = self.y1.shape[0]
            prevalence = count1 / N

            if (true_positive_rate(y_true, y_pred) == 0) & (false_positive_rate(y_true, y_pred) == 0):
                print("0/0 case")
                return 0
            else:
                return (prevalence * true_positive_rate(y_true, y_pred)) / (
                    prevalence * true_positive_rate(y_true, y_pred)
                    + ((1 - prevalence) * false_positive_rate(y_true, y_pred))
                )

        def F2_score(y_true, y_pred):
            return fbeta_score(y_true, y_pred, beta=2)

        metrics_pool = {
            "precision": precision_score,
            "recall": recall_score,
            "sensitivity/recall": sensitivity_score,
            "F1": f1_score,
            "BAc": balanced_accuracy_score,
            "AP": average_precision_score,
            "specificity": specificity_score,
            "MCC": matthews_corrcoef,
            "ROC": roc_auc_score,
            "Acc": accuracy_score,
            "FPR": false_positive_rate,
            "TPR": true_positive_rate,
            "PTF": precision_from_tpr_fpr,
            "F2": F2_score,
        }

        for metric in self.eval_metrics:
            self.metrics[metric] = metrics_pool[metric]

        scores = self.ev.score(metrics=self.metrics)

    def permutation(self):
        """
            Run permutation tests to evalute the quality
        """
        self.a = np.shape(self.ev.scores.mean(axis=2)[:, :, 0])  # true result
        self.perm = np.zeros((self.nperm, len(self.perc), self.a[1]))
        self.corr = np.zeros((self.nperm, len(self.perc)))

        # Main testing loop
        # TODO improve paralel processing - each clasifier will be evaluted completely separately
        if __name__ == "__main__":
            with Bar("Evaluating Dataset Quality...", max=self.nperm * len(self.perc)) as bar:
                for i in range(self.nperm):
                    for j in range(len(self.perc)):
                        if self.verbose >= 1:
                            print("Iteration", i + j, "/", len(self.perc) * self.nperm)
                        t = 0
                        while True:
                            ind_pool = {}
                            nperc_pool = {}
                            for item in range(len(self.y1.value_counts())):
                                ind_pool[item] = np.where(self.y1 == item)
                                nperc_pool[item] = round(self.perc[j] * len(ind_pool[item][0]) / 100)
                            
                            tmp_list = []
                            for item in range(len(self.y1.value_counts())):
                                tmp_list.append(ind_pool[item][0][:nperc_pool[item]])
                            indP = np.random.permutation(np.concatenate(tmp_list))
                            #ind1 = np.where(self.y1 == 0)
                            #ind2 = np.where(self.y1 == 1)

                            #nperc1 = round(self.perc[j] * len(ind1[0]) / 100)
                            #nperc2 = round(self.perc[j] * len(ind2[0]) / 100)

                            #indP = np.random.permutation(np.concatenate((ind1[0][:nperc1], ind2[0][:nperc2])))
                            ind = np.sort(indP)

                            y1P = np.copy(self.y1)

                            y1P[ind] = self.y1[indP]

                            comparison = self.y1 == y1P

                            if not comparison.all() or t > 3:
                                if self.verbose >= 1:
                                    print("Too many permutations with the same result. Skipping this iteration...")
                                    print("Note: This usually hapends for small or suspicious datasets.")
                                break
                            t += 1

                        self.datasetsP = {"all": (self.X1, y1P)}

                        ## Non-paralel version of classifier evaluation
                        evP = ws.evaluation.Evaluator(datasets=self.datasetsP,protocol2=(False, 2, None)).process(clfs=self.clfs_ver2, verbose=0)
                        scores = evP.score(metrics=self.metrics)
                        self.perm[i,j,:] = evP.scores.mean(axis=2)[:, :, 0]
                        kk = np.corrcoef(y1P,self.y1)
                        self.corr[i,j] = kk[0,1]

                        ## Paralel version of classifier evaluation
                        #with Pool(self.cores) as p:
                        #    eval_scores = p.map(self.evaluate, self.clfs_ver1)
                        #tmp_scores = np.zeros((1, len(self.clfs_ver1)))
                        #for idx, item in enumerate(eval_scores):
                        #    tmp_scores[0][idx] = item[0]
                        #self.perm[i, j, :] = tmp_scores
                        #kk = np.corrcoef(y1P, self.y1)
                        #self.corr[i, j] = kk[0, 1]
                        #print("Evaluation done")
        else:
            cnt = 0
            for i in range(self.nperm):
                for j in range(len(self.perc)):
                    if self.verbose >= 1:
                        cnt += 1
                        print("Iteration", cnt, "/", len(self.perc) * self.nperm)
                    t = 0
                    while True:
                        ind_pool = {}
                        nperc_pool = {}
                        for item in range(len(self.y1.value_counts())):
                            ind_pool[item] = np.where(self.y1 == item)
                            nperc_pool[item] = round(self.perc[j] * len(ind_pool[item][0]) / 100)
                            
                        tmp_list = []
                        for item in range(len(self.y1.value_counts())):
                            tmp_list.append(ind_pool[item][0][:nperc_pool[item]])
                        indP = np.random.permutation(np.concatenate(tmp_list))
                        
                        #ind1 = np.where(self.y1 == 0)
                        #ind2 = np.where(self.y1 == 1)

                        #nperc1 = round(self.perc[j] * len(ind1[0]) / 100)
                        #nperc2 = round(self.perc[j] * len(ind2[0]) / 100)

                        #indP = np.random.permutation(np.concatenate((ind1[0][:nperc1], ind2[0][:nperc2])))
                        ind = np.sort(indP)

                        y1P = np.copy(self.y1)

                        y1P[ind] = self.y1[indP]

                        comparison = self.y1 == y1P

                        if not comparison.all() or t > 3:
                            if self.verbose >= 2:
                                print("Too many permutations with the same result. Skipping this iteration...")
                                print("Note: This usually hapends for small or suspicious datasets.")
                            break
                        t += 1

                    self.datasetsP = {"all": (self.X1, y1P)}

                    ## Non-paralel version of classifier evaluation
                    evP = ws.evaluation.Evaluator(datasets=self.datasetsP,protocol2=(False, 2, None)).process(clfs=self.clfs_ver2, verbose=0)
                    scores = evP.score(metrics=self.metrics)
                    self.perm[i,j,:] = evP.scores.mean(axis=2)[:, :, 0]
                    kk = np.corrcoef(y1P,self.y1)
                    self.corr[i,j] = kk[0,1]

                    ## Paralel version of classifier evaluation
                    #with Pool(self.cores) as p:
                    #    eval_scores = p.map(self.evaluate, self.clfs_ver1)
                    #tmp_scores = np.zeros((1, len(self.clfs_ver1)))
                    #for idx, item in enumerate(eval_scores):
                    #    tmp_scores[0][idx] = item[0]
                    #self.perm[i, j, :] = tmp_scores
                    #kk = np.corrcoef(y1P, self.y1)
                    #self.corr[i, j] = kk[0, 1]

    def evaluate(self, classifier):
        """
            Helper function for parallel processing
        """
        evP2 = ws.evaluation.Evaluator(datasets=self.datasetsP, protocol2=(False, 2, None)).process(
            clfs=classifier, verbose=0
        )
        scores = evP2.score(metrics=self.metrics)
        return evP2.scores.mean(axis=2)[:, :, 0][0]

    def print_results(self):
        """
            Generate output report files for dataset quality
        """
        classifiers = ()
        for i in self.clfs_ver2:
            classifiers = classifiers + (i,)
        pvalues = np.zeros((self.a[1], len(self.perc)))

        colors = cm.rainbow(np.linspace(0, 1, self.a[1]))

        for j in range(len(self.perc)):
            for i, c in zip(range(self.a[1]), colors):
                ind = np.where(self.perm[:, j, i] >= self.ev.scores.mean(axis=2)[:, i, 0])
                pvalues[i, j] = ((len(ind[0]) + 1) * 1.0) / (self.nperm + 1)

        pv = pd.DataFrame(data=pvalues, index=list(classifiers), columns=self.perc)
        if self.verbose >= 1:
            print("##### Results #####")
            print("P-value table")
            print(pv)

        # Get slope chart
        names = classifiers
        cor = []
        per = []
        slopes = []
        auc_scores = []
        max_perm = [0] * len(self.perc)  # List of values for maximal slopes across all models

        for i, c in zip(range(self.a[1]), colors):
            for j in range(len(self.perc)):
                # Find Maximal values for each correlation level
                if max_perm[j] < np.mean(self.perm[:, j, i]):
                    max_perm[j] = np.mean(self.perm[:, j, i])
            cor = np.mean(self.corr[:, :], axis=0)
            per = np.mean(self.perm[:, :, i], axis=0)
            auc_score = auc(cor, per)
            slope, intercept = np.polyfit(cor, per, 1)
            if self.verbose >= 1:
                print(names[i], "=", slope)
            slopes = np.append(slopes, slope)
            auc_scores = np.append(auc_scores, auc_score)

        maxind = np.argmax(abs(slopes))
        maxind_auc = np.argmax(abs(auc_scores))

        if self.verbose >= 1:
            print("Max Slope")
            print("Slope:", np.max(abs(slopes)), "-", names[maxind])
            print("AUC:", np.max(abs(auc_scores)), "-", names[maxind_auc])
            print(
                "Top AUC",
                auc(cor, max_perm),
                "- Max F1:",
                max_perm[-1],
                "- Final Metric:",
                (0.5 - auc(cor, max_perm) / max_perm[-1]) / 0.25,
            )
        self.raw_slope = np.max(abs(slopes))
        self.raw_auc = auc(cor, max_perm)
        self.auc_score = (0.5 - auc(cor, max_perm) / max_perm[-1]) / 0.25
        self.max_clf_metric_score = max_perm[-1]

    def get_score(self):
        """
            Get dataset quality result
        """
        classifiers = ()
        for i in self.clfs_ver2:
            classifiers = classifiers + (i,)
        pvalues = np.zeros((self.a[1], len(self.perc)))

        for j in range(len(self.perc)):
            for i in range(self.a[1]):
                ind = np.where(self.perm[:, j, i] >= self.ev.scores.mean(axis=2)[:, i, 0])
                pvalues[i, j] = ((len(ind[0]) + 1) * 1.0) / (self.nperm + 1)

        pv = pd.DataFrame(data=pvalues, index=list(classifiers), columns=self.perc)
        # def significant(v):
        #    return "font-weight: bold; color: red" if v > 0.01 else None
        tmp = 0
        status = None
        max_val = 0
        # print("classifiers",self.clfs_ver2)
        for j in range(len(pv)):
            if self.ev.scores.mean(axis=2)[:, j, 0] > max_val:
                max_val = self.ev.scores.mean(axis=2)[:, j, 0]

            for i in range(len(self.perc)):
                if pv.iloc[j, i] > 0.01:
                    tmp += 1
            if tmp == 0 and self.ev.scores.mean(axis=2)[:, j, 0] < max_val:
                status = "Repeat"
            elif tmp == 0:
                status = "Good"
            if tmp < len(self.perc) and status != "Good":
                status = "Mid"
            elif tmp == len(self.perc) and (status != "Good" or status != "Mid"):
                status = "Bad"

        # Get slope
        names = classifiers
        cor = []
        per = []
        slopes = []
        # auc_scores = []

        for i in range(self.a[1]):
            cor = np.mean(self.corr[:, :], axis=0)
            per = np.mean(self.perm[:, :, i], axis=0)
            slope, intercept = np.polyfit(cor, per, 1)
            slopes = np.append(slopes, slope)
        #   auc_score = auc(cor,per)
        #   auc_scores = np.append(auc_scores, auc_score)

        # score = {"qod":status,"slope":np.max(abs(slopes))}
        # self.auc_score = (0.5-auc(cor,max_perm)/max_perm[-1])/0.125
        score = self.auc_score
        final_result = {"Association":score,"Max Clf Score":self.max_clf_metric_score,"P-value status":status, "Raw Slope":self.raw_slope, "Raw AUC":self.raw_auc}
        return final_result  # {"Metric":score, "P-value":pv}

    def save_results(self):
        """
            Picle object with permuted evaluation data
        """
        try:
            with open(self.output + "/perqoda.obj", "wb") as f:
                pickle.dump(self, f)
        except Exception as err:
            print()
            print("Error: Failed to save PerQoDA object in " + self.output + " directory.")
            print("Full Error:", err)
            sys.exit(3)
        if self.verbose >= 1:
            print("PerQoDA object saved to pickle file - " + self.output + "/perqoda.obj")

    def run_evaluation(self):
        self.load_dataset()
        self.run_classifiers()
        self.run_metrics()
        self.permutation()
        self.print_results()
        return self.get_score()

def load_results(filename):
    """
        Load picled object of with permuted evalution data
    """
    try:
        with open(filename, "rb") as f:
            qod = pickle.load(f)
            return qod
    except Exception as err:
        print()
        print("Error: Unable to read the configuration file " + filename + ". Please check formating or file access.")
        print("Full Error Message", err)


def label_association(dataset, label):
    assoc = association(dataset, label)
    return assoc.run_evaluation()
