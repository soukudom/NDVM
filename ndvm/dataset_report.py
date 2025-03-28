"""
    Network Dataset Evaluation Metrics Report
"""

# Import requried modules

import yaml
import json
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
import yaml
import toml
import re

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class CustomTOMLEncoder(toml.TomlEncoder):
    """
    Custom TOML encoder that:
    1. Converts string values from single quotes to triple double quotes
    2. Formats dictionary values with specific formatting style:
       - Keys and values in double quotes
       - Each key-value pair on a new line
       - No trailing comma for the last item
    """
    
    def __init__(self, _dict=dict, preserve=False):
        super().__init__(_dict, preserve)
        self.dict_pattern = re.compile(r'{.*?}', re.DOTALL)
    
    def dump_value(self, v):
        """Override dump_value to handle special formatting for strings and dicts"""
        # Handle strings - convert single quotes to triple double quotes
        if isinstance(v, str):
            # Check if the string contains a dictionary representation
            if self.dict_pattern.search(v):
                return self._format_dict_string(v)
            # Use triple double quotes for all string values
            return '""" ' + v + ' """'
        
        # Handle dictionaries
        elif isinstance(v, dict):
            return self._format_dict(v)
        
        # For other types, use the parent class implementation
        return super().dump_value(v)
    
    def _format_dict(self, d):
        """Format a dictionary according to the specified style"""
        if not d:
            return '"""{  }"""'
        
        formatted = '{\n'
        items = list(d.items())
        
        for i, (key, value) in enumerate(items):
            # Ensure values are strings
            str_value = str(value)
            
            formatted += f'  "{key}": "{str_value}"'
            # Don't add comma after the last item
            if i < len(items) - 1:
                formatted += ','
            formatted += '\n'
        
        formatted += '}'
        
        # Wrap in triple double quotes
        return f'"""{formatted}"""'
    
    def _format_dict_string(self, s):
        """Process a string that contains a dictionary representation"""
        try:
            # Find the dictionary part in the string
            dict_match = self.dict_pattern.search(s)
            if not dict_match:
                return f'"""{s}"""'
            
            dict_str = dict_match.group(0)
            
            # Try to parse as a dictionary
            try:
                # First try as JSON
                d = json.loads(dict_str.replace("'", '"'))
            except json.JSONDecodeError:
                # If JSON fails, try to eval as Python dict with some cleanup
                cleaned = dict_str.replace("'", '"')
                # Replace single-quoted keys with double-quoted keys
                cleaned = re.sub(r'([{,]\s*)(\w+):', r'\1"\2":', cleaned)
                try:
                    d = json.loads(cleaned)
                except:
                    # If all parsing fails, just return the string wrapped in triple quotes
                    return f'"""{s}"""'
            
            # Format the dictionary
            formatted_dict = self._format_dict(d)
            
            # Replace the original dictionary string with the formatted version
            return f'"""{s.replace(dict_str, formatted_dict[3:-3])}"""'
            
        except Exception:
            # If any error occurs, return the original string wrapped in triple quotes
            return f'"""{s}"""'

# TODO
# - add more hidden data values that are used for metric calculation (f1 score, ..) -> good for metrics troubleshooting
class dataset_metrics:
    def __init__(self, sourceDir, filename):
        self.filename = filename
        self.sourceDir = sourceDir
        self.classes = None
        self.multiclass = None
        self.samples = []
        self.cleared_samples = []
        self.analyzed_samples = {}
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
        self.filtered_label = 0
        self.label_list = {}

        with open(sourceDir+"/"+filename, "r") as stream:
            try:
                self.cfg = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise("YAML Parsing Error",exc)

    def create_metric(self, metric_path):
        """Imports metric class and creates metric object
        """
        mod_name, class_name = metric_path.rsplit('.', 1)
        metric_module = import_module(mod_name)
        metric_class = getattr(metric_module, class_name)
        return metric_class
        
    def load_metrics(self):
        """Loader of metrics."""
        for metric_path in self.cfg["metric_config"]:
            metric = self.create_metric(metric_path)
            self._metrics.append(metric)

    def eval_metrics(self, dataset, label, output_dir_metadata_base):
        """
            Main function to evaluate dataset metrics
        """
        if type(dataset) == type(pd.DataFrame()):
            df_dataset = dataset
        elif type(dataset) == type(""):
            try:
                df_dataset = pd.read_csv(dataset, delimiter=self.cfg["delimiter"])
            except Exception as e:
                # TODO include ipfixprobe for pcap processing and FET + add exception for unknow format
                raise ValueError('Error: Non CSV input detected. Please include featuredaset in csv.')
        
        # Basic info
        ## Get number of classes
        self.classes = len(df_dataset[label].value_counts())
        if self.classes > 2:
            self.multiclass = True
        #if self.classes != self.cfg["classes"]:
        #    raise ValueError('Error: Mismatch between input number of classes and detected.')

        ## Get amount of samples
        self.samples = {str(k): v for k, v in dict(df_dataset[label].value_counts()).items()}

        ## Get labels
        for item in df_dataset[label].value_counts().index.tolist():
            self.labels.append(item)
        for key,val in df_dataset[label].value_counts().items():
            self.label_list[key] = val

        ## Get amount of features
        self.features = len(df_dataset.drop(columns=[label]).columns)
        ## Get list of features
        self.feature_list = dict(zip(df_dataset.columns, [''] * len(df_dataset.columns)))
        ## Get amount of duplicated samples
        self.duplicated = len(df_dataset[df_dataset.duplicated])
        ### print duplicated rows
        if self.verbose >= 5 & self.duplicated > 0:
            print("Duplicated rows (Note: index is +1)")
            print(df_dataset[df_dataset.duplicated()].to_string(header=False))
        if self.cfg["delete_duplicated"]:
            df_dataset.drop_duplicates(inplace=True)

        ## Remove nan values
        df_dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.nan = df_dataset.isna().any(axis=1).sum()
        ### print N/A rows
        if self.verbose >= 2:
            print("N/A rows (Note: index is +1)")
            print(df_dataset[df_dataset.isna().any(axis=1)].to_string(header=False))
        if self.cfg["delete_nan"]:
            df_dataset = df_dataset.dropna()

        # Get reduced dataset + sample dataset
        dataset_merge = pd.DataFrame()
        for key,item in df_dataset[label].value_counts().items():
            if item > self.cfg["sampling_limit"]:
                df_dataset = df_dataset.sample(frac=1).reset_index(drop=True)
                class_tmp = df_dataset[df_dataset[label]==key][:self.cfg["sampling_limit"]]
                self.analyzed_samples[str(key)] = self.cfg["sampling_limit"]
            else:
                self.analyzed_samples[str(key)] = item
                class_tmp = df_dataset[df_dataset[label]==key]
            dataset_merge = pd.concat([dataset_merge,class_tmp])
            
        # Finish merge of the sampled dataset
        df_dataset = pd.DataFrame(dataset_merge)
        df_dataset = df_dataset.sample(frac=1).reset_index(drop=True)
        df_dataset.reset_index(inplace=True)
        df_dataset = df_dataset.drop(columns=['index'])

        # Filter classes with samples less than minimal limit
        initial_state = len(df_dataset[label].value_counts())
        value_counts = df_dataset[label].value_counts()
        frequent_categories = value_counts[value_counts > self.cfg["min_sample_limit"] ].index
        df_dataset = df_dataset[df_dataset[label].isin(frequent_categories)]
        after_state = len(df_dataset[label].value_counts())
        if initial_state != after_state:
            self.filtered_label = initial_state - after_state
        df_dataset = df_dataset.sample(frac=1).reset_index(drop=True)
        df_dataset.reset_index(inplace=True)
        df_dataset = df_dataset.drop(columns=['index'])
       
        # Advanced metrics
        for metric in self._metrics:
            mx = metric(df_dataset, label, self.multiclass, self.cfg["verbose"])
            if self.cfg["verbose"] >= 1: 
                print("Running metric called",mx.get_name())
            score = mx.run_evaluation()
            self.scores[mx.get_name()] = score
            mx.get_details(output_dir_metadata_base)
        
    def get_katoda_report(self):
        """
            Get report for Katoda
        """
        report = {}
        insights = " The dataset has {} small classes, {} duplicated sampoles, {} nan value".format(self.filtered_label,self.duplicated, self.nan)
        for key,item in self.scores.items():
            report[key] = item
        # validate metrics
        try:
            keys = "Association", "Redundancy", "Similarity"
            for key in keys:
                self.scores[key]
        except Exception as e:
            self.scores[key] = ""

        try:
            self.scores["Association"]["Max Clf Score"]
        except Exception as e:
            self.scores["Association"] = {"Max Clf Score": ""}
        report = {
            "collection_workflow":{
                "data_collection_tool":"",
                "data_collection_year":"",
                "feature_extraction_tool_info":"Tool that converts dataset to feature dataset. If any.",
                "feature_extraction_tool":"",
                "feature_extraction_tool_description":"",
                "capture_config_parameters_info":"specific parameters that were used to capture dataset or feature dataset",
                "capture_config_parameters":"",
                "real_dataset_info":"Source of the dataset. E.g., real environment, testbed or generated.",
                "real_dataset":"",
                "annotation_info":"Description of the dataset annotation. E.g., manual, automatic",
                "annotation":""
            },
            "generic_info":{
                "classes": str(self.classes), 
                "features": str(self.features), 
                "f1-score_info":"F1-score calculated based on NDVM tool [https://github.com/soukudom/NDVM]", 
                "f1-score": str(self.scores["Association"]["Max Clf Score"]), 
                "performance_metric_info":"Perfomance metric defined by the author. Please define full specification e.g., F1-weighted", 
                "performance_metric_name":"", 
                "performance_metric_value":"", 
                "label_info":"Name of the field with label. In case this is unsupervised dataset, type None", 
                "label":"", 
                "key_observations_info":"List of known errors, drifts, limits, ... of the dataset", 
                "key_observations":"""* """+insights, 
                "known_issues_info": """Description of indentified issues in the dataset""",
                "known_issues": """ """,
                "key_observations_info": "List of known errors, drifts, limits, ... of the dataset",
                "key_observations": """ """,
                "dataset_organization_info":"Structure of the dataset. E.g., per day, per capture, per device", 
                "dataset_organization":"", 
                "dataset_organization_description_info": "Description of the content of the organization. Is there any metadata?",
                "dataset_organization_description": """ """,
                "dataset_documentation_info":"How to get started with the dataset. Ideally add example notebook.", 
                "dataset_documentation":""" """, 
                "used_dataset_info": "Script to get dataset for provided analysis",
                "used_dataset": "get-dataset.py",
                "dataset_application_info": "Where the dataset has been already applied.", 
                "dataset_application": """ """, 
                "per_class_data": str(self.label_list), 
                "per_feature_data": str(self.feature_list)
            },
            "dataset_drift_analysis":{

            },
            "advanced_metrics":{
                "description":"", 
                "perqoda_permutation_slope": str(self.scores["Association"]["Association"]), 
                "p_value_status": str(self.scores["Association"]["P-value status"]), 
                "redundancy": str(self.scores["Redundancy"]), 
                "similarity": str(self.scores["Similarity"]["metric"]), 
                "advanced_metrics_workflow":"dataset-metrics.json"
            },

            "dataset_comparison":{
                "description": "ML model comparison for this dataset",
                "use_case": """ """,
                "similar_dataset": """ """
            }

        }
        return report

    def get_report(self):
        """
            Get dataset report
        """
        report = OrderedDict(
            {
                "Classes": self.classes,
                "Feature List": self.feature_list, #dict(zip(df_dataset.columns, [''] * len(df_dataset.columns))),
                "Original Samples": self.samples,
                "Features": self.features,
                "Duplicated Feature Vectors": int(self.duplicated),
                "N/A Values": self.nan,
                "Sampling Limit": self.cfg["sampling_limit"],
                "Analyzed Samples": self.analyzed_samples,
                "Date": date.today().strftime("%m/%d/%y")
            }
        )
        for key,item in self.scores.items():
            report[key] = item
        return report

    def init_sync(self, host, username, remoteDir, dataset):
        #client = paramiko.SSHClient()
        #client.load_system_host_keys()
        #try:
        #    client.connect(hostname=host, username=username)
        #except Exception as e:
        #    print("[!] Cannot connect to the SSH Server")
        #    print("Error:",e)

        # Copy config directory and dataset to remote server
        cmds = [f"rsync -rz {self.sourceDir+'/'+self.filename} {username}@{host}:{remoteDir+'/tmp/'}", f"rsync -rz {dataset} {username}@{host}:{remoteDir+'/tmp/'}"]
        for cmd in cmds:
            if self.verbose > 1:
                print("Running remote cmd:",cmd)
            p = subprocess.Popen(cmd, shell=True)
            p.wait()
            if p.returncode != 0:
                raise ValueError("Error during rsync command.")

    def run_remote(self,host,username,remoteDir,dataset, label):
        # Run the experiment
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        try:
            client.connect(hostname=host, username=username)
        except:
            print("[!] Cannot connect to the SSH Server")
        client.exec_command(f"cd {remoteDir}; . ./venv/bin/activate")
        tmp_cmd = f"nohup python3.9 dataset_report.py -d {dataset} -l {label} -s localhost -sd ./tmp/ -c {self.filename} -o ./tmp/ >tmp/ot.log 2>&1 &"
        client.exec_command(f"cd {remoteDir}; echo '. ./venv/bin/activate' > tmp.sh")
        client.exec_command(f"cd {remoteDir}; echo '{tmp_cmd}' >> tmp.sh")
        client.exec_command(f"cd {remoteDir}; /bin/bash tmp.sh")

    def poll_remote_results(self, host, username, remoteDir, outputDir):
        exit_flag = None
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        try:
            client.connect(hostname=host, username=username)
        except Exception as e:
            print("[!] Cannot connect to the SSH Server")
            print("Error:", e)
        stdin, stdout, stderr = client.exec_command(f"cd {remoteDir}; ls tmp/ | grep report-*")
        output = stdout.readlines()
        if len(output) == 0:
            exit_flag = False
        else:
            exit_flag = True
            # Copy config directory and dataset to remote server
            cmds = [f"rsync -z {username}@{host}:{remoteDir+'/tmp/report*'} {outputDir}"]
            for cmd in cmds:
                p = subprocess.Popen(cmd, shell=True)
                p.wait()
                if p.returncode != 0:
                    raise ValueError("Error during rsync command.")
        return exit_flag

    def clean_remote(self, host, username, remoteDir, dataset):
        # Clean input files (config and dataset)
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        try:
            client.connect(hostname=host, username=username)
        except:
            print("[!] Cannot connect to the SSH Server")
        client.exec_command(f"cd {remoteDir}; rm -r tmp/; rm tmp.sh")


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
                        default="config.yml")
    parser.add_argument("--dataset",
                        "-d",
                        required=True)
    parser.add_argument("--label",
                        "-l",
                        required=True)
    parser.add_argument("--outputDir",
                        "-o",
                        default=".")
    parser.add_argument("--taskName",
                        "-tn",
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
        # TODO finalize metacentrum execution
        pass
    # run locally
    elif input_data["server"] == "localhost":
        
        dm = dataset_metrics(input_data["sourceDir"],input_data["config"])
        if dm.verbose >= 1:
            print("Running Dataset Report Evaluation")
        dm.load_metrics()
        output_dir_metadata_base = input_data["outputDir"]+"/"+"report-"+input_data["taskName"]+"-"+str(seconds)
        dm.eval_metrics(input_data["dataset"],input_data["label"],output_dir_metadata_base)
        report = dm.get_report()
        try:
            output_file = input_data["outputDir"]+"/"+"report-"+input_data["taskName"]+"-"+str(seconds)+".csv"
            with open(output_file, "w") as log_file:
                json.dump(report, log_file, cls=NumpyEncoder)
        except Exception as e:
            print("error e")
            raise ValueError("Error with output file. Wrong path or enough privileges.")
        # TODO add this option also for other running methods
        output_file_katoda = input_data["outputDir"]+"/"+"report-katoda-"+input_data["taskName"]+"-"+str(seconds)+".toml"
        with open(output_file_katoda, 'w') as f:
            toml.dump(dm.get_katoda_report(), f, encoder=CustomTOMLEncoder())
    # run 3rd party server
    else:
        tmp_flag = False
        if input_data["remoteDir"] == None or input_data["rsyncUsername"] == None:
            raise ValueError('Please define remote path with NDVM directory using "remoteDir" argument')
        # rsync files to remote server
        dm = dataset_metrics(input_data["sourceDir"],input_data["config"])
        dm.init_sync(input_data["server"], input_data["rsyncUsername"], input_data["remoteDir"], input_data["dataset"])
        # run remote process
        dm.run_remote(input_data["server"], input_data["rsyncUsername"], input_data["remoteDir"], input_data["dataset"], input_data["label"])
        # reqularly check
        while not tmp_flag:
            tmp_flag = dm.poll_remote_results(input_data["server"], input_data["rsyncUsername"], input_data["remoteDir"],input_data["outputDir"])
            time.sleep(600)
        dm.clean_remote(input_data["server"], input_data["rsyncUsername"], input_data["remoteDir"], input_data["dataset"])

        

    

