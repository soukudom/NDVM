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
#metric =  f1_score
# Set label column name in the dataset -> it must be lowercase due to FET module
LABEL="is_doh"
# Set if your dataset is multiclass or not
MULTICLASS=False
X_1 = None
y_1 = None
max_score = 0
clfs_set = {}
redundancy = 0


def eval_dataset(X_1, y_1, frac, clfs):
    global clfs_set
    global runs

    tmp_results = {}

    name, clf = clfs
    #for name, clf in clfs_set.items():
    l = []
    tmp_results[name] = [l.copy() for i in range(runs)]

    #print(runs, len(y_1))
    #iprint(runs)
    for i in range(0,runs):
        X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(X_1, y_1, test_size=1-frac, stratify=y_1, shuffle=True)
        #for name, clf in clfs_set.items():
        clf.fit(X_train_sub, y_train_sub)
        pred = clf.predict(X_test_sub)
        tmp_results[name][i].append(metrics.f1_score(y_test_sub,pred))
    return tmp_results

def CalculateRedundancy(X_1, y_1, max_score, clfs):
    global runs
    global alfa
    #global max_score
#    global X_1
#    global y_1
    
    limit = max_score*alfa
    low = 0.0
    high = 1.0
    mid = 0
    tmp_redundancy = 0.9

    while high - low > alfa:
        print("Testing redundancy with",high, low, tmp_redundancy)
        tmp_redundancy = (high + low) / 2
        tmp_score = eval_dataset(X_1, y_1, tmp_redundancy, clfs)
        tmp_high = []
        tmp_low = []
        name, clf = clfs
        #for name, clf in clfs.items():
        tmp = []
        for item in range(runs):
            if (max_score - tmp_score[name][item][0]) < limit:
                tmp.append(tmp_score[name][item][0])

        print("comparing:",max_score, tmp_score[name][item][0])
        if len(tmp) == runs:
            tmp_high.append(tmp_redundancy)
        else:
            tmp_low.append(tmp_redundancy)

        # Check div and conq
        if len(tmp_high) > 0:
            high = tmp_redundancy
        else:
            low = tmp_redundancy
    return 1-tmp_redundancy

def PrepareDataset(dataset, label):
    global X_1
    global y_1
    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
    dataset = dataset.dropna()
    
    y_1 = dataset [label]
    X_1 = dataset.drop(columns=[label])
    y_1 = y_1.astype('category')
    y_1 = y_1.cat.codes

def collect_result(result):
    global redundancy
    redundancy = max(result)
    return redundancy

def maximal_score(result):
    global max_score
    print("max score")
    print(result,type(result))
    for item in result:
        print(">>>>>>>>>>", max(list(item.values())[0]))
        tmp = max(list(item.values())[0])[0]
        if tmp > max_score:
            max_score = tmp
    #print(result)
    print(max_score)

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
          #  "DT": DecisionTreeClassifier(criterion = "gini"),
            "RF": RandomForestClassifier(class_weight="balanced", criterion='gini'),
          #  "XGB": XGBClassifier(objective="multi:softmax"),
        }
    else:
        clfs_set = {
            "DT": DecisionTreeClassifier(),
            "RF": RandomForestClassifier(),
            "XGB": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
        }

    pool = mp.Pool(len(clfs_set))
    #results = eval_dataset(X_1, y_1, 0.9, clfs)
    pool.starmap_async(eval_dataset, [(X_1, y_1, 0.9,  clfs) for clfs in clfs_set.items()], callback=maximal_score)
    pool.close()
    pool.join()

    #max_score = 0
    #for name, clf in clfs_set.items(): 
    #    for i in range(runs):
    #        if max_score < results[name][i][0]:
    #            max_score = results[name][i][0]
    #max_score = 0.9

    
    #pool = mp.Pool(len(clfs_set))
    pool = mp.Pool(1)
    #results = [pool.apply(CalculateRedundancy, args=(X_1, y_1, clfs)) for clfs in clfs_set.items()]
    #results = pool.starmap(CalculateRedundancy, [(X_1, y_1, clfs) for clfs in clfs_set.items()])
    #for ii in range(runs):
    pool.starmap_async(CalculateRedundancy, [(X_1, y_1, max_score,  clfs) for clfs in clfs_set.items()], callback=collect_result)
    pool.close()
    pool.join()

    return redundancy
