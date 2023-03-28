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

runs = 5 # number of iterations 
alfa = 0.01 # Lift Value
# Set if your dataset is multiclass or not
MULTICLASS=False
X_1 = None
y_1 = None
max_score = 0
clfs_set = {}
redundancy = 0


# Evaluate specific dataset redundancy for the specific fraction between train and test part
def eval_dataset(X_1, y_1, frac, clfs):
    global clfs_set
    global runs

    tmp_results = {}
    name, clf = clfs
    l = []
    tmp_results[name] = [l.copy() for i in range(runs)]
    for i in range(0,runs):
        X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(X_1, y_1, test_size=1-frac, stratify=y_1, shuffle=True)
        clf.fit(X_train_sub, y_train_sub)
        pred = clf.predict(X_test_sub)
        tmp_results[name][i].append(metrics.f1_score(y_test_sub,pred))
    return tmp_results

# Calculate redundancy score for selected classificators
def CalculateRedundancy(X_1, y_1, max_score, clfs):
    global runs
    global alfa
    
    limit = max_score*alfa
    low = 0.0
    high = 1.0
    mid = 0
    tmp_redundancy = 0.9
    
    # Run divide and conquer finding of dataset redundancy
    while high - low > alfa:
        tmp_redundancy = (high + low) / 2
        tmp_score = eval_dataset(X_1, y_1, tmp_redundancy, clfs)
        tmp_high = []
        tmp_low = []
        name, clf = clfs
        tmp = []
        print("Testing",name, max_score, high, low)
        for item in range(runs):
            if (max_score - tmp_score[name][item][0]) < limit:
                tmp.append(tmp_score[name][item][0])

        if len(tmp) == runs:
            tmp_high.append(tmp_redundancy)
        else:
            tmp_low.append(tmp_redundancy)

        # Check divide and conquer borders
        if len(tmp_high) > 0:
            high = tmp_redundancy
        else:
            low = tmp_redundancy
    return 1-tmp_redundancy

# Prepare X_1 and y_1 variables from the input dataset
def PrepareDataset(dataset, label):
    global X_1
    global y_1
    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
    dataset = dataset.dropna()
    
    y_1 = dataset [label]
    X_1 = dataset.drop(columns=[label])
    y_1 = y_1.astype('category')
    y_1 = y_1.cat.codes

# Get maximal redundancy across all models
def collect_result(result):
    global redundancy
    redundancy = max(result)
    return redundancy

# Get maximal dataset score to find redudancy
def maximal_score(result):
    global max_score
    for item in result:
        tmp = max(list(item.values())[0])[0]
        if tmp > max_score:
            max_score = tmp

# Main method for computing the redundancy metric
def Redundancy(dataset, label):
    global MULTICLASS
    global runs
    global max_score
    global clfs_set
    global X_1
    global y_1
    global redundancy

    PrepareDataset(dataset, label)

    if MULTICLASS:
        clfs_set = {
            "DT": DecisionTreeClassifier(criterion = "gini"),
            "RF": RandomForestClassifier(class_weight="balanced", criterion='gini'),
            "XGB": XGBClassifier(objective="multi:softmax"),
        }
    else:
        clfs_set = {
            "DT": DecisionTreeClassifier(),
            "RF": RandomForestClassifier(),
            "XGB": XGBClassifier(eval_metric="logloss"),
        }

    # Find max score in parallel
    pool = mp.Pool(len(clfs_set))
    pool.starmap_async(eval_dataset, [(X_1, y_1, 0.9,  clfs) for clfs in clfs_set.items()], callback=maximal_score)
    pool.close()
    pool.join()

    
    # Find max dataset redundancy score in parallel
    pool = mp.Pool(len(clfs_set))
    pool.starmap_async(CalculateRedundancy, [(X_1, y_1, max_score,  clfs) for clfs in clfs_set.items()], callback=collect_result)
    pool.close()
    pool.join()

    return redundancy
