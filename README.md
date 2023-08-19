# NDVM (Network Dataset Evaluation Metrics)
The Network Dataset Evaluation report is a tool used to analyze datasets to estimate trustworthiness. The provided solution can describe basic parameters such as the number of classes, imbalance ratio, and duplicated samples. Moreover, the report can provide PCA visualization and metrics for dataset optimization with a trustworthiness estimate. The implementation is available as a Jupyter notebook for direct visualization and providing a self-explanatory example. There is also a python module for proper deployment. Jupyter notebook is self-explanatory. In section below there is short howto for python module.

# Advanced Metric Introduction
In the subsection, we describe the meaning of more advanced dataset quality metrics.
## Metric 1 - Dataset Redundancy
The Metric 1 is focused on the level of dataset size redundancy. Using this measure, one can estimate what portion of the original dataset can be randomly removed while keeping the classification performance drop below the certain controlled level. Zero redundancy indicates not enough data for the classification task. For evaluation, we use the pool of classifiers and acceptance level $\alpha$ to generally assess the level of redundancy with a certain probability. 

The metric domain is $[0, 1]$, and it describes the percentage size of the dataset that is redundant. 

## Metric 2 - Dataset Association Quality
The Metric 2 evaluates the level of association between labels and respective data. Especially for the public datasets, we don't know how the dataset was collected and if it's meaningful to apply ML algorithm on such a dataset. The level of association is estimated based on permutation tests which are interpreted by this novel metric. As a result, we get an estimate of how strong is the connection between data and related labels.

The metric domain is $[0, 1]$ and it corresponds to the level of association between data and labels in the dataset.

## Metric 3 - Datset Class Similarity
The Metric 3 looks at the dataset classes. We propose a method to estimate how instances of different classes are similar to each other. In other words, how complex the classification task generally is on the input dataset. The metric measures relative class similarity using autoencoders and their respective reconstruction error. Calculated relative similarity lays out direct indicators of how prone are other machine learning models to misclassifications. 

Since this metric represents average relative reconstruction error over all instances of non-base classes, the domain are positive numbers -- multiples of reconstruction error on base class. $M_3 <= 1$ means that autoencoder performs similarly for all the classes and from this point of view, classes are similar, or even some classes might seem as a subset of the base class in the feature space. On the other hand, if $M_3 > 1 + K$, classes are different in the feature space and might be easier to separate from the base class.

# Python Module HOWTO
## Installation 
Clone this repo
* git clone https://github.com/soukudom/NDVM.git
Create virtual environemnt
* python3 -m venv venv
* . ./venv/bin/activate 
Install required python modules
* pip3 install -r requirements.txt
Run example
* python3 dataset_report.py

If everything runs without any issue you are ready for configuration of dataset evalution tool.

## Configuration
### config.py
The main configuration file is represented by a Python module. You can choose any name or location you want. The default name is "config.py," and it is located at the root of this repository. In this file, you can select (comment/uncomment) advanced metrics you would like to run. Also, there are additional parameters that globally affect the dataset evaluation. An explanation of each parameter is provided in the config.py file.

### Metric configuration
Each advanced metric selected in the main configuration file (section config.py above) is inherited from core.py, which defines the structure of each metric. This provides a unified interface for easier development. Each metric can have special parameters relevant just for the purposes of the metric. These parameters are part of class attributes in the metric module. Typically you can configure the pool of classifiers, number of repetitions, ... For further details, please check the comments in the metric module or read our papers with more detailed explanations and experiments. 

### dataset_report.py
This is the main file that allows you to start the dataset evaluation. From the configuration perspective, the only relevant part the main section (if __name__ == "__main__"). Here you can configure the path to your dataset, the name of the label column, and path to the main configuration file (section config.py)

### Typical Issues 
1. Wrong dataset format: You select a dataset that includes raw data (string, list, binary, ...). This tool requires a feature dataset that can be used with ML classifiers. 
2. Wrong configuration: You select the wrong delimiter or configure the wrong path 
3. Too big number: Some advanced metrics have higher computation complexity. For dataset evaluation, we typically don't need a complete dataset but just a sample. For the initial evaluation, it is recommended to start with ~10 0000 - 20 0000 samples per class. 

### Near Future Work
* Full expansion for binary and multiclass task
* Ability to preprocess some raw dataset format (pcap, pstats)
* Remove weles dependency
* Enable automatic configuration parameters tuning

## Typical Evaluation Scenario
1. Install this tool
2. Check config.py file
3. Modify the main section in datset_report.py -> path to your dataset, label column name, path to config.py 
4. Run evaluation: python3 datset_report.py
5. Check the output provided in json format
6. Modify configuration: increase verbose mode (gives you more detailed output and visualizations), adjust sampling, sensitivity, ...

# Paper
TBA
