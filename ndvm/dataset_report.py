"""
    Network Dataset Evaluation Metrics Report
"""

# Import requried modules
import metric1

import yaml
import pandas as pd

def parseConfig(filename):
    pass
    with open(filename, "r") as stream:
        try:
            print(yaml.safe_load(stream))
        except yaml.YAMLError as err:
            print(err)

def printReport():
    report = {
        "Classes": "2",
        "Samples": "5000:5000",
        "Features": 24,
        "Duplicated Flows":  "0",
        "Redundancy": {
            "Value": 0.23
        }
    }
    print(report)

if __name__ == "__main__":
    print("Running Dataset Report Evaluation")
    parseConfig("config.yml")
    df_dataset = "sample_dataset/combined-doh-http.csv"
    df_dataset = pd.read_csv(df_dataset, delimiter=",")
    df_dataset = df_dataset.sample(frac=1).reset_index(drop=True)
    redundancy = metric1.Redundancy(df_dataset,"is_doh")
    print("Redudancy", redundancy)
    printReport()
