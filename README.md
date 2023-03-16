# NDVM (Network Dataset Evaluation Metrics)
The Network Dataset Evaluation report is a tool used to analyze datasets to estimate trustworthiness. The provided solution can describe basic parameters such as the number of classes, imbalance ratio, and duplicated samples. Moreover, the report can provide PCA visualization and metrics for dataset optimization with a trustworthiness estimate. The implementation is available as a Jupyter notebook for direct visualization and providing a self-explanatory example. There is also a python module for proper deployment. 

# Metric Introduction
In the subsection, we describe the meaning of dataset quality metrics.
## Metric 1 - Dataset Redundancy
The first metric is focused on the level of dataset size redundancy. Using this measure, one can estimate what portion of the original dataset can be randomly removed while keeping the classification performance drop below the certain controlled level. Zero redundancy indicates not enough data for the classification task. For evaluation, we use the pool of classifiers and acceptance level $\alpha$ to generally assess the level of redundancy with a certain probability. 
The metric domain is $[0, 1]$, and it describes the percentage size of the dataset that is redundant. 

## Metric 2 - Dataset Association Quality
The Metric 2 evaluates the level of association between labels and respective data. Especially for the public datasets, we don't know how the dataset was collected and if it's meaningful to apply ML algorithm on such a dataset. The level of association is estimated based on permutation tests which are interpreted by this novel metric. As a result, we get an estimate of how strong is the connection between data and related labels.
The metric domain is $[0, 1]$ and it corresponds to the level of association between data and labels in the dataset.

## Metric 3 - Datset Class Similarity
TBA

# Paper
TBA
