"""
    Dataset Redundacy Calculation
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from multiprocessing import Pool
import sklearn.metrics as metrics
from sklearn.metrics import auc
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import multiprocessing as mp
from multiprocessing import Process
from core import AbstractMetric
import os
import pickle
import sys

class Redundancy(AbstractMetric):
    def __init__(self, dataset, label, multiclass, verbose):
        self.runs = 5  # number of iterations
        self.alfa = 0.01  # Lift Value
        # Set if your dataset is multiclass or not
        self.MULTICLASS = multiclass
        self.X_1 = None
        self.y_1 = None
        self.max_score = 0
        self.clfs_set = {}
        self.ds_redundancy = 0
        self.dataset = dataset
        self.label = label
        self.verbose = verbose
        self.redundancy_metadata = {}

    def get_name(self):
        return "Redundancy"
    
    def get_details(self, output_dir_metadata_base):
        try:
            os.mkdir(output_dir_metadata_base+"-metric1")
            self.save_results(output_dir_metadata_base+"-metric1")
        except Exception as e:
            print(f"An error occurred: {e}")

    def eval_dataset(self, X_1, y_1, frac, clfs):
        """
            Evaluate specific dataset redundancy for the specific fraction between train and test part
        """
        tmp_results = {}
        name, clf = clfs
        l = []
        tmp_results[name] = [l.copy() for i in range(self.runs)]
        for i in range(0, self.runs):
            X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(
                X_1, y_1, test_size=1 - frac, stratify=y_1, shuffle=True
            )
            clf.fit(X_train_sub, y_train_sub)
            pred = clf.predict(X_test_sub)
            tmp_results[name][i].append(metrics.f1_score(y_test_sub, pred,average="weighted"))
        return tmp_results

    def save_results(self,outputdir):
        """
            Pickle object with permuted evaluation data
        """
        try:
            with open(outputdir + "/metric1.obj", "wb") as f:
                pickle.dump(self, f)
        except Exception as err:
            print()
            print("Error: Failed to save the metric1 object in " + outputdir + " directory.")
            print("Full Error:", err)
            sys.exit(3)
        if self.verbose >= 1:
            print("The metric1 object saved to pickle file - " + outputdir + "/metric1.obj")

    def calculate_redundancy(self, X_1, y_1, max_score, clfs):
        """
            Calculate redundancy score for selected classificators
        """
      
        limit = max_score * self.alfa
        low = 0.0
        high = 1.0
        mid = 0
        tmp_redundancy = 0.9
        redundancy_metadata = {}

        # Run divide and conquer finding of dataset redundancy
        while high - low > self.alfa:
            tmp_redundancy = (high + low) / 2
            tmp_score = self.eval_dataset(X_1, y_1, tmp_redundancy, clfs)
            tmp_high = []
            tmp_low = []
            name, clf = clfs
            tmp = []

            # Store metadata for visualization
            try:
                redundancy_metadata[name][str(tmp_redundancy)].append(tmp_score[name][0])
            except Exception as e:
                try:
                    redundancy_metadata[name][tmp_redundancy] = tmp_score[name] 
                except Exception as e:
                    redundancy_metadata[name] = {tmp_redundancy: tmp_score[name]}

            if self.verbose > 0:
                print("Testing", name, max_score, high, low)
            for item in range(self.runs):
                if (max_score - tmp_score[name][item][0]) < limit:
                    tmp.append(tmp_score[name][item][0])

            if len(tmp) == self.runs:
                tmp_high.append(tmp_redundancy)
            else:
                tmp_low.append(tmp_redundancy)

            # Check divide and conquer borders
            if len(tmp_high) > 0:
                high = tmp_redundancy
            else:
                low = tmp_redundancy
        
        return {"score": 1 - tmp_redundancy, "metadata": redundancy_metadata}

    def prepare_dataset(self):#, dataset, label):
        """
            Prepare X_1 and y_1 variables from the input dataset
        """
        self.dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.dataset = self.dataset.dropna()

        self.y_1 = self.dataset[self.label]
        self.X_1 = self.dataset.drop(columns=[self.label])
        self.y_1 = self.y_1.astype("category")
        self.y_1 = self.y_1.cat.codes

    def collect_result(self, result):
        """
            Get maximal redundancy across all models
        """
        print("Got", result)
        #print(int(result["score"]))

        self.ds_redundancy = max([item['score'] for item in result]) #max([result[0]["score"]])
        self.redundancy_metadata = [item['metadata'] for item in result] #result[0]["metadata"]

        #self.ds_redundancy = max([result[0]["score"]])
        #self.redundancy_metadata = result[0]["metadata"]
        #print("Got redundancy", self.ds_redundancy)
        #print("Got metadata", self.redundancy_metadata)
        return self.ds_redundancy

    def maximal_score(self, result):
        """
            Get maximal dataset score to find redudancy
        """
        for item in result:
            tmp = max(list(item.values())[0])[0]
            if tmp > self.max_score:
                self.max_score = tmp


    def run_evaluation(self):#, dataset, label):
        """
            Main method for computing the ds_redundancy metric
        """
        # Prepare dataset to requred format
        self.prepare_dataset()#(self.dataset, self.label)

        if self.MULTICLASS:
            self.clfs_set = {
                "DT": DecisionTreeClassifier(criterion="gini"),
                "RF": RandomForestClassifier(class_weight="balanced", criterion="gini"),
                "XGB": XGBClassifier(objective="multi:softmax"),
            }
        else:
            self.clfs_set = {
                "DT": DecisionTreeClassifier(),
                "RF": RandomForestClassifier(),
                "XGB": XGBClassifier(eval_metric="logloss"),
            }

        # Find max score in parallel
        pool = mp.Pool(len(self.clfs_set))
        pool.starmap_async(self.eval_dataset, [(self.X_1, self.y_1, 0.9, clfs) for clfs in self.clfs_set.items()], callback=self.maximal_score)
        pool.close()
        pool.join()

        if self.verbose > 0:
            print("found max score", self.max_score)
        # Find max dataset redundancy score in parallel
        pool = mp.Pool(len(self.clfs_set))
        pool.starmap_async(
            self.calculate_redundancy, [(self.X_1, self.y_1, self.max_score, clfs) for clfs in self.clfs_set.items()], callback=self.collect_result
        )
        pool.close()
        pool.join()

        return self.ds_redundancy
