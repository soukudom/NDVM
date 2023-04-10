"""
    Network Dataset Evaluation Metrics Report
"""

# Import requried modules
import metric1
import metric2
import metric3

import yaml
import pandas as pd
from pprint import pprint
from collections import OrderedDict


class dataset_metrics:
    def __init__(self, filename):
        self.filename = filename
        self.classes = None
        self.samples = []
        self.labels = []
        self.features = None
        self.duplicated = None
        self.redundancy = None
        self.association = None
        self.similarity = None

    def parse_config(self):
        with open(self.filename, "r") as stream:
            try:
                data = yaml.safe_load(stream)
            except yaml.YAMLError as err:
                print(err)

    def eval_metrics(self):
        """
            Main function to evaluate dataset metrics
        """
        df_dataset = "sample_dataset/combined-doh-http.csv"
        df_dataset = pd.read_csv(df_dataset, delimiter=",")
        # Basic info
        self.classes = len(df_dataset["is_doh"].value_counts())
        for item in df_dataset["is_doh"].value_counts():
            self.samples.append(item)
        for item in df_dataset["is_doh"].value_counts().index.tolist():
            self.labels.append(item)
        self.features = len(df_dataset.drop(columns=["is_doh"]).columns)
        self.duplicated = df_dataset[df_dataset.drop(columns=["is_doh"]).duplicated()].shape[0]

        # Advanced metrics
        print("Running Metric 1 - Redundancy ...")
        self.redundancy = metric1.redundancy(df_dataset, "is_doh")
        print("Running Metric 2 - Association ...")
        self.association = metric2.label_association(df_dataset, "is_doh")
        print("Running Metric 3 - Class Similarity ...")
        self.similarity = metric3.class_similarity(df_dataset, df_dataset.drop(columns=["is_doh"]).columns, "is_doh")

    def get_report(self):
        """
            Get dataset report
        """
        report = OrderedDict(
            {
                "Classes": self.classes,
                "Samples": self.samples,
                "Features": self.features,
                "Duplicated Flows": self.duplicated,
                "Redundancy": self.redundancy,
                "Association": self.association,
                "Similarity": self.similarity,
            }
        )
        return report

    # print(report)


if __name__ == "__main__":
    print("Running Dataset Report Evaluation")
    dm = dataset_metrics("config.yml")
    dm.parse_config()
    dm.eval_metrics()
    report = dm.get_report()
    pprint(report)

#    parseConfig("config.yml")
#    df_dataset = "sample_dataset/combined-doh-http.csv"
#    df_dataset = pd.read_csv(df_dataset, delimiter=",")
#    df_dataset = df_dataset.sample(frac=1).reset_index(drop=True)
#    redundancy = metric1.Redundancy(df_dataset,"is_doh")
#    print("Redudancy", redundancy)
#    printReport()
