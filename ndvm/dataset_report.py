"""
    Network Dataset Evaluation Metrics Report
"""

# Import requried modules

import yaml
import pandas as pd
from pprint import pprint
from collections import OrderedDict
import numpy as np
import core
from importlib import import_module
from datetime import date, datetime
import subprocess
import time
import argparse
import sys
import paramiko

# TODO
# - add more hidden data values that are used for metric calculation (f1 score, ..) -> good for metrics troubleshooting
class dataset_metrics:
    def __init__(self, sourceDir, filename):
        self.filename = filename
        self.sourceDir = sourceDir
        self.classes = None
        self.samples = []
        self.cleared_samples = []
        self.analyzed_samples = []
        self.labels = []
        self.features = None
        self.duplicated = None
        self.nan = None
        self.redundancy = None
        self.association = None
        self.similarity = None
        self._metrics = []
        self.scores = {}
        self.verbose = 0

        sys.path.append(self.sourceDir)
        self.cfg = __import__(self.filename)

    def create_metric(self, metric_path):
        """Imports metric class and creates metric object
        """
        mod_name, class_name = metric_path.rsplit('.', 1)
        metric_module = import_module(mod_name)
        metric_class = getattr(metric_module, class_name)
        return metric_class
        
    def load_metrics(self):
        """Loader of metrics."""
        for metric_path in self.cfg.metric_config:
            metric = self.create_metric(metric_path)
            self._metrics.append(metric)

    def eval_metrics(self, dataset, label):
        """
            Main function to evaluate dataset metrics
        """
        try:
            df_dataset = pd.read_csv(dataset, delimiter=self.cfg.delimiter)
        except Exception as e:
            # TODO include ipfixprobe for pcap processing and FET + add exception for unknow format
            raise ValueError('Error: Non CSV input detected. Please include featuredaset in csv.')
        
        # Basic info
        ## Get number of classes
        self.classes = len(df_dataset[label].value_counts())
        if self.classes != self.cfg.classes:
            raise ValueError('Error: Mismatch between input number of classes and detected.')

        ## Get amount of samples
        for item in df_dataset[label].value_counts():
            self.samples.append(item)
        ## Get labels
        for item in df_dataset[label].value_counts().index.tolist():
            self.labels.append(item)
        ## Get amount of features
        self.features = len(df_dataset.drop(columns=[label]).columns)
        ## Get amount of duplicated samples
        self.duplicated = len(df_dataset[df_dataset.duplicated])
        ### print duplicated rows
        if self.verbose >= 2 & self.duplicated > 0:
            print("Duplicated rows (Note: index is +1)")
            print(df_dataset[df_dataset.duplicated()].to_string(header=False))
        if self.cfg.delete_duplicated:
            df_dataset.drop_duplicates(inplace=True)

        ## Remove nan values
        df_dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.nan = df_dataset.isna().any(axis=1).sum()
        ### print N/A rows
        if self.verbose >= 2:
            print("N/A rows (Note: index is +1)")
            print(df_dataset[df_dataset.isna().any(axis=1)].to_string(header=False))
        if self.cfg.delete_nan:
            df_dataset = df_dataset.dropna()

        # Get reduces dataset + sample dataset
        dataset_merge = pd.DataFrame()
        for key,item in df_dataset[label].value_counts().items():
            if item > self.cfg.sampling_limit:
                df_dataset = df_dataset.sample(frac=1).reset_index(drop=True)
                class_tmp = df_dataset[df_dataset[label]==key][:self.cfg.sampling_limit]
                self.analyzed_samples.append(item)
            else:
                self.analyzed_samples.append(item)
                class_tmp = df_dataset[df_dataset[label]==key]
            dataset_merge = pd.concat([dataset_merge,class_tmp])
            
        # Udelat merge
        df_dataset = pd.DataFrame(dataset_merge)
        df_dataset = df_dataset.sample(frac=1).reset_index(drop=True)
        df_dataset.reset_index(inplace=True)
        df_dataset = df_dataset.drop(columns=['index'])

        # Advanced metrics
        for metric in self._metrics:
            mx = metric(df_dataset, label, self.cfg.multiclass, self.cfg.verbose)
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
                "Original Samples": self.samples,
                "Features": self.features,
                "Duplicated Feature Vectors": self.duplicated,
                "N/A Values": self.nan,
                "Sampling Limit": self.cfg.sampling_limit,
                "Analyzed Samples": self.analyzed_samples,
                "Date": date.today().strftime("%m/%d/%y")
            }
        )
        for key,item in self.scores.items():
            report[key] = item
        return report

    def sync(self,host,rsync_username,sourceDir,remoteDir):
        cmd = f"rsync -rz {sourceDir} {rsync_username}@{host}:{remoteDir}"
        # TODO implement two way sync
        p = subprocess.Popen(cmd, shell=True)
        p.wait()
        if p.returncode != 0:
            raise ValueError("Error during rsync command.")

    def run_remote(self,host,username,remoteDir, dataset, label):
        client = paramiko.SSHClient()
        try:
            client.connect(hostname=host, username=username)
        except:
            print("[!] Cannot connect to the SSH Server")
        client.exec_command(f"cd {remoteDir}")
        client.exec_command(f". ./venv/bin/activate")
        # TODO need to solve remote paths
        client.exec_command(f"nohup python3.9 dataset_report.py -d {dataset} -l {label} -s localhost -sd &")
        cmd = f"nohup python3.9  &"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s",
                        "--server",
                        default="localhost")
    parser.add_argument('--verbose',
                        '-v', action='count',
                        default=0)
    parser.add_argument("--config",
                        "-c",
                        default="config")
    parser.add_argument("--dataset",
                        "-d",
                        required=True)
    parser.add_argument("--label",
                        "-l",
                        required=True)
    parser.add_argument("--outputDir",
                        "-o",
                        default=".")
    parser.add_argument("--name",
                        "-n",
                        default="dataset")
    parser.add_argument("--remoteDir",
                        "-rd",
                        default=None)
    parser.add_argument("--sourceDir",
                        "-sd",
                        default=".")
    parser.add_argument("--rsyncUsername",
                        "-u",
                        default=None)
    args = parser.parse_args()
    input_data  = vars(args)
    # Generate timestamp of the report
    dt = datetime.today()  
    seconds = int(dt.timestamp())

    # run metacentrum
    if input_data["server"] == "metacentrum":
        pass
    # run locally
    elif input_data["server"] == "localhost":
        
        dm = dataset_metrics(input_data["sourceDir"],input_data["config"])
        if dm.verbose >= 1:
            print("Running Dataset Report Evaluation")
        dm.load_metrics()
        dm.eval_metrics(input_data["dataset"],input_data["label"])
        report = dm.get_report()
        try:
            output_file = input_data["outputDir"]+"/"+"report-"+input_data["name"]+"-"+str(seconds)+".csv"
            with open(output_file, "w") as log_file:
                pprint(dict(report), log_file)
        except Exception as e:
            raise ValueError("Error with output file. Wrong path or enough privileges.")
    # run 3rd party server
    else:
        if input_data["remoteDir"] == None or input_data["rsyncUsername"] == None:
            raise ValueError('Please define remote path with NDVM directory using "remoteDir" argument')
        # rsync files to remote server
        dm = dataset_metrics(input_data["sourceDir"],input_data["config"])
        dm.sync(input_data["server"], input_data["rsyncUsername"], input_data["sourceDir"], input_data["remoteDir"])
        # run remote process
        # reqularly check

        

    

