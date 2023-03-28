"""
    Network Dataset Evaluation Metrics Report
"""

# Import requried modules
import metric1

import yaml
import pandas as pd
from pprint import pprint

class DatasetMetrics:

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

    def parseConfig(self):
        with open(self.filename, "r") as stream:
            try:
                data = yaml.safe_load(stream)
            except yaml.YAMLError as err:
                print(err)
    def evalMetrics(self):
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
        print("Running Redundancy Metric ...")
        self.redundancy = metric1.Redundancy(df_dataset,"is_doh")

    def getReport(self):
        report = {
            "Classes": self.classes,
            "Samples": self.samples,
            "Features": self.features,
            "Duplicated Flows":  self.duplicated,
            "Redundancy": self.redundancy
        }
        return report
    #print(report)

if __name__ == "__main__":
    print("Running Dataset Report Evaluation")
    dm = DatasetMetrics("config.yml")
    dm.parseConfig()
    dm.evalMetrics()
    report = dm.getReport()
    pprint(report)

#    parseConfig("config.yml")
#    df_dataset = "sample_dataset/combined-doh-http.csv"
#    df_dataset = pd.read_csv(df_dataset, delimiter=",")
#    df_dataset = df_dataset.sample(frac=1).reset_index(drop=True)
#    redundancy = metric1.Redundancy(df_dataset,"is_doh")
#    print("Redudancy", redundancy)
#    printReport()
