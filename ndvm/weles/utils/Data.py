"""
Class for preparing datasets for the evaluator.
"""

# imports
import numpy as np
import os
import re
import collections

class Data():
    def __init__(self, selection, path="../../datasets/"):
        self.selection = selection
        self.path = path

    def load(self):
        """
        description

        Returns:
        datasets: dictionary containing dataset name and and a numpy array with
                  features and labels
                  "name" : numpy array (n_samples, n_features+1)
        """
        # load files
        files = self.__dir2files(path=self.path)
        datasets = {}

        if type(self.selection) == tuple:
            tag_filter = self.selection[1]
            for file in files:
                X, y, dbname, tags = self.__csv2Xy(file)
                if self.selection[0] == 'all':
                    is_good = all(elem in tags for elem in tag_filter)
                if self.selection[0] == 'any':
                    is_good = any(elem in tags for elem in tag_filter)
                if is_good:
                    datasets[dbname] = (X, y)

        # Load datasets by name
        elif type(self.selection) == list:
            for file in files:
                X, y, dbname, _ = self.__csv2Xy(file)
                is_good = dbname in self.selection
                if is_good:
                    datasets[dbname] = (X, y)
        else:
            print("Provide a list or a tuple")
        datasets_items = datasets.items()
        sorted_datasets = sorted(datasets_items)
        sorted_datasets = collections.OrderedDict(sorted_datasets)
        return sorted_datasets

    def multiply_dim(self, multiply=2):
        # load files
        files = self.__dir2files(path=self.path)
        datasets = {}

        # Load datasets by name
        if type(self.selection) == list:
            for file in files:
                X, y, dbname, _ = self.__csv2Xy(file)
                is_good = dbname in self.selection
                if is_good:
                    datasets[dbname + "_1"] = (X, y)
                    Xm = X
                    for i in range(multiply):
                        # for j in range(10):
                        rng = np.random.default_rng()
                        noise = rng.standard_normal((X.shape[0],X.shape[1]))
                        # print(X+noise)
                        # exit()
                        Xm = np.concatenate((Xm, X+noise), axis=1)
                        dbname_m = dbname + "_" + str(i+2)
                        datasets[dbname_m] = (Xm, y)
        else:
            print("Provide a list of dataset names")
        return datasets

    def __tags4Xy(self, X, y):
        tags = []
        numberOfFeatures = X.shape[1]
        # numberOfSamples = len(y)
        numberOfClasses = len(np.unique(y))
        if numberOfClasses == 2:
            tags.append("binary")
        else:
            tags.append("multi-class")
        if numberOfFeatures >= 8:
            tags.append("multi-feature")

        # Calculate ratio
        ratio = [0.0] * numberOfClasses
        for y_ in y:
            ratio[y_] += 1
        ratio = [int(round(i / min(ratio))) for i in ratio]
        if max(ratio) > 3:
            tags.append("imbalanced")
        else:
            tags.append("balanced")
        return tags

    def __csv2Xy(self, path):
        ds = np.genfromtxt(path, delimiter=",")
        X = ds[:, :-1]
        y = ds[:, -1].astype(int)
        dbname = path.split("/")[-1].split(".")[0]
        tags = self.__tags4Xy(X, y)
        return X, y, dbname, tags
        # return datasets

    def __dir2files(self, path, extension="csv"):
        return [
            path + x
            for x in os.listdir(path)
            if re.match("^([a-zA-Z0-9-_])+\.%s$" % extension, x)
        ]
