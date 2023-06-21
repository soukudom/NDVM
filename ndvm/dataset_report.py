"""
    Network Dataset Evaluation Metrics Report
"""

# Import requried modules
#import metric1
#import metric2
#import metric3

import yaml
import pandas as pd
from pprint import pprint
from collections import OrderedDict
import numpy as np
import core
from config import metric_config
from importlib import import_module

# TODO
# - add more hidden data values that are used for metric calculation (f1 score, ..) -> good for metrics troubleshooting
class dataset_metrics:
    def __init__(self, filename):
        self.filename = filename
        self.classes = None
        self.samples = []
        self.cleared_samples = []
        self.labels = []
        self.features = None
        self.duplicated = None
        self.redundancy = None
        self.association = None
        self.similarity = None
        self._metrics = []
        self.scores = {}
        #self.dataset = None
        #self.label = None

    def create_metric(self, metric_path):
        """Imports metric class and creates metric object
        """
        mod_name, class_name = metric_path.rsplit('.', 1)
        metric_module = import_module(mod_name)
        metric_class = getattr(metric_module, class_name)
        return metric_class
        #mod_name, class_name = metric_path.rsplit('.', 1)

        #analyzer_module = import_module(mod_name)
        #analyzer_class = getattr(analyzer_module, class_name)

       # return analyzer_class(file_path)
    def load_metrics(self):
        """Loader of metrics."""
        for metric_path in metric_config:
            metric = self.create_metric(metric_path)
            self._metrics.append(metric)

   # def parse_config(self):
   #     with open(self.filename, "r") as stream:
   #         try:
   #             data = yaml.safe_load(stream)
   #         except yaml.YAMLError as err:
   #             print(err)

    def eval_metrics(self, dataset, label):
        """
            Main function to evaluate dataset metrics
        """
        #df_dataset = "sample_dataset/combined-doh-http.csv"
        df_dataset = pd.read_csv(dataset, delimiter=",")
        # Basic info
        self.classes = len(df_dataset[label].value_counts())
        for item in df_dataset[label].value_counts():
            self.samples.append(item)
        for item in df_dataset[label].value_counts().index.tolist():
            self.labels.append(item)
        self.features = len(df_dataset.drop(columns=[label]).columns)
        self.duplicated = df_dataset[df_dataset.drop(columns=[label]).duplicated()].shape[0]

        # TMP Testing section
        df_dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_dataset = df_dataset.dropna()
        df_dataset.reset_index(inplace=True)

        # Basic info part 2
        for item in df_dataset[label].value_counts():
            self.cleared_samples.append(item) 

        # Advanced metrics
        for metric in self._metrics:
            mx = metric(df_dataset, label)
            print("Running metric called",mx.get_name())
            score = mx.run_evaluation()
            self.scores[mx.get_name()] = score
            
    def get_report(self):
        """
            Get dataset report
        """
        report = OrderedDict(
            {
                "Classes": self.classes,
                "Samples": self.samples,
                "Cleared Samples": self.cleared_samples,
                "Features": self.features,
                "Duplicated Flows": self.duplicated,
            }
        )
        for key,item in self.scores.items():
            report[key] = item
        return report

    # print(report)


if __name__ == "__main__":
    print("Running Dataset Report Evaluation")
    dm = dataset_metrics("config.yml")
    dm.load_metrics()
    dm.eval_metrics("/home/dosoukup/Datasets/metrics/testing/netmon/datasets/tor/netisa/sampled-tor.csv","LABEL")
    report = dm.get_report()
    pprint(report)

