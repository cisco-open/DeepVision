
# Copyright 2022 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from .EthosightMediaAnalyzer import EthosightMediaAnalyzer, EthosightOutput 
from .LabelSpaceOptimization import SemanticRelationsOptimization, SemanticSimilarityOptimization
from ruamel.yaml import YAML
from pathlib import Path
import os
import shutil
from .utils import get_install_path
import datetime
import glob
import numpy as np
import time
import cv2
import csv
from tqdm import tqdm
from .ChatGPTReasoner import ChatGPTReasoner
import json
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
import traceback
import pandas as pd

class EthosightAppException(Exception):
    pass

class EthosightApp:
    def __init__(self, app_name, base_dir="ENV"):
        self.yaml = YAML()
        self.yaml.preserve_quotes = True
        self.yaml.default_flow_style = False

        # Load the base directory either from environment or from the provided value
        if base_dir == "ENV":
            base_dir = os.environ.get("ETHOSIGHT_APP_BASEDIR")
            if not base_dir:
                raise EthosightAppException("ETHOSIGHT_APP_BASEDIR environment variable is not defined.")
        if not os.path.isdir(base_dir):
            raise EthosightAppException(f"Base directory {base_dir} is not a valid directory. Please define a valid directory using ETHOSIGHT_APP_BASEDIR environment variable")
        
        # Ensure the app_dir is an absolute path 
        self.app_dir = os.path.join(base_dir, app_name)

        # Construct the path to the config file based on the app_dir
        self.config_file = os.path.join(self.app_dir, 'ethosightapp.yaml')
       
        # This loads absolute paths
        self.config = self.load_config(self.config_file)
        self.media_analyzer = EthosightMediaAnalyzer(self.config_file)
        self.ethosight = self.media_analyzer.ethosight

        # Load the default embeddings at initialization
        self.embeddings_path = self.config.get('embeddings_path')
        self.labels_path = self.config.get('labels_path')
        self.setActiveEmbeddingsFromFile(self.embeddings_path, self.labels_path)
        if self.config['benchmark']['enabled']:
            self.groundTruthEmbeddings(makeActive=False)
        self.reasoner = self.config['reasoner_type']

    def load_config(self, config_file):
        # Assuming the load_config function reads the YAML config file 
        # and returns a dictionary 
        with open(config_file, 'r') as file:
            config = self.yaml.load(file)

        # List of known keys that should contain paths
        path_keys = ['embeddings_path', 'labels_path', 
            'template_path',
            'image_dir', 
            'ground_trueh_path',
            'extracted_video_dir',
            ]  # Add all relevant keys here

        # Convert relative paths in the config to absolute paths 
        for key, value in config.items():
            if key in path_keys and isinstance(value, str) and not os.path.isabs(value):
                config[key] = os.path.join(self.app_dir, value)

        return config


    def setActiveEmbeddingsFromFile(self, embeddings_path, labels_path):
        if embeddings_path is None:
            raise ValueError('embeddings_path cannot be None')
        embeddings_path = os.path.join(self.app_dir, embeddings_path)

        # Check if embeddings_path points to an existing file
        if not os.path.isfile(embeddings_path):
            raise FileNotFoundError(f"The embeddings_path {embeddings_path} does not exist or is not a file.")

        self.activeLabelsPath = os.path.join(self.app_dir, labels_path)
        self.activeEmbeddings = self.media_analyzer.ethosight.load_embeddings_from_disk(embeddings_path)
        self.activeEmbeddingsPath = embeddings_path

    def setup_new_app(self):
        # Implement the logic to set up new app including creating necessary files
        pass

    def get_config_file(self, app_dir):
        # Code to determine the configuration file from the application directory
        # This will depend on your application's directory structure
        return Path(app_dir) / "config.yaml"

    @classmethod
    def create_app(cls, app_dir, config_file, base_dir="ENV"):
        if base_dir == "ENV":
            base_dir = os.environ.get("ETHOSIGHT_APP_BASEDIR", "")
            if not base_dir:
                raise EthosightAppException("Error: 'ETHOSIGHT_APP_BASEDIR' environment variable is not set.")
        
        if not os.path.isdir(base_dir):
            raise EthosightAppException(f"Error: The provided base directory '{base_dir}' is not a valid directory.")

        # Prepend the base directory to the app_dir if provided
        if base_dir:
            app_dir = os.path.join(base_dir, app_dir)

        app_dir_path = Path(os.path.abspath(app_dir))

        # If the directory already exists, return an error message
        if app_dir_path.exists() and app_dir_path.is_dir():
            return "Error: The directory specified already exists."

        # Check if the config_file exists
        if not Path(config_file).exists():
            # If not, try the EthosightYAMLDirectory environment variable
            yaml_dir = os.environ.get("EthosightYAMLDirectory", "")
            
            if yaml_dir:
                potential_paths = [
                    os.path.join(yaml_dir, config_file),
                    os.path.join(yaml_dir, config_file + ".yaml")
                ]
                
                for path in potential_paths:
                    if Path(path).exists():
                        config_file = path
                        break
                else:  # If none of the paths exist
                    return "Error: The configuration file specified does not exist and no alternatives were found."
            else:
                return "Error: The configuration file specified does not exist and 'EthosightYAMLDirectory' environment variable is not set."

        # Create the directory
        os.makedirs(app_dir, exist_ok=False)

        # Copy the configuration file to the new application directory
        new_config_file_path = os.path.join(app_dir, 'ethosightapp.yaml')
        shutil.copy(config_file, new_config_file_path)
        # Make a copy of the configuration file in the app directory as a template
        template_config_file_path = os.path.join(app_dir, 'ethosightapp_template.yaml')
        shutil.copy(config_file, template_config_file_path)

        yaml = YAML()
        yaml.preserve_quotes = True
        yaml.default_flow_style = False

        # Load config file
        with open(new_config_file_path, 'r') as f:
            config_data = yaml.load(f)

        ethosight_dir = get_install_path('Ethosight')
        embeddingsdir = os.path.join(ethosight_dir, 'embeddings')
        # Define the pairs of file keys and default dirs
        file_key_dir_pairs = [
            ('embeddings_path', embeddingsdir),
            ('labels_path', embeddingsdir)
        ]  

        # For each file key, copy the file to the new app directory and update the key in the config
        for key, default_dir in file_key_dir_pairs:
            if key in config_data:
                file_path = config_data[key]
                if default_dir is not None and not os.path.isabs(file_path):
                    file_path = os.path.join(default_dir, file_path)

                if Path(file_path).exists():
                    new_file_path = os.path.join(app_dir, os.path.basename(file_path))
                    shutil.copy(file_path, new_file_path)
                    #config_data[key] = new_file_path  # update the config with the absolute path

        # First, backup the original config file
        backup_config_file_path = os.path.join(app_dir, 'ethosightapp.yaml.bak')
        shutil.copy(new_config_file_path, backup_config_file_path)

        # Save the updated config back to the file
        with open(new_config_file_path, 'w') as f:
            yaml.dump(config_data, f)

        return cls(app_dir)  # return the created EthosightApp instance

    @classmethod
    def delete_app(cls, app_name, base_dir="ENV"):
        # Load the base directory either from environment or from the provided value
        if base_dir == "ENV":
            base_dir = os.environ.get("ETHOSIGHT_APP_BASEDIR")
            if not base_dir:
                raise EthosightAppException("ETHOSIGHT_APP_BASEDIR environment variable is not defined.")
        if not os.path.isdir(base_dir):
            raise EthosightAppException(f"Base directory {base_dir} is not a valid directory.")

        # Construct the full path to the app directory
        app_dir = os.path.join(base_dir, app_name)
        app_dir_path = Path(app_dir)

        # If the directory does not exist or it is not a directory, raise an exception
        if not app_dir_path.exists() or not app_dir_path.is_dir():
            raise EthosightAppException(f"The directory {app_dir} does not exist or is not a directory.")

        # Check if ethosightapp.yaml file exists in the directory
        ethosightapp_config_path = app_dir_path / 'ethosightapp.yaml'
        if not ethosightapp_config_path.exists():
            raise EthosightAppException(f"ethosightapp.yaml file does not exist in the directory {app_dir}. The directory might not be an EthosightApp directory.")

        # If the directory exists and contains ethosightapp.yaml, delete the directory and all its contents
        shutil.rmtree(app_dir)
        return f"Deleted EthosightApp named {app_name} in directory: {app_dir}"


    def create_label_space_optimizer(self):
        # Get the label_space_optimization configuration
        lso_config = self.config.get('label_space_optimization')
        
        # Check if label_space_optimization is in the config
        if lso_config is None:
            raise ValueError('label_space_optimization not found in the configuration')
        
        # Get the method and parameters from the configuration
        method = lso_config.get('method')
        params = lso_config.get('parameters')
        
        # Check if method is in the label_space_optimization config
        if method is None:
            raise ValueError('method not found in label_space_optimization configuration')

        # Check if parameters is in the label_space_optimization config
        if params is None:
            raise ValueError('parameters not found in label_space_optimization configuration')

        # Create and return an instance of the appropriate optimizer
        if method == 'semantic_similarity':
            return SemanticSimilarityOptimization(**params)
        elif method == 'semantic_relations':
            return SemanticRelationsOptimization(**params)
        elif method == 'another_strategy':
            return AnotherOptimizationStrategy(**params)
        else:
            raise ValueError(f'Unsupported label space optimization method: {method}')

    def processGeneralTemplates(self, template_path):
        """
        Processes general templates for label expansion.
        
        Args:
            template_path (str): Path to the file containing templates for label expansion.
        """

        # 1. Read the templates from the file
        with open(template_path, 'r') as f:
            templates = f.readlines()

        # Remove any newline characters
        templates = [template.strip() for template in templates]

        # 2. Expand the templates using the provided labels
        expanded_labels = []
        if len(self.gtLabels) == 0:
            raise ValueError('No active labels found.')

        for label in self.gtLabels:
            expanded_labels.append(label) #let's keep the original label
            for template in templates:
                newlabel = template.format(label)
                print(f"Expanded label: {newlabel}")
                expanded_labels.append(newlabel)

        # 3. Save the expanded labels to a file
        expanded_labels_path = os.path.join(self.app_dir, 'expanded_labels.labels')
        with open(expanded_labels_path, 'w') as f:
            for label in expanded_labels:
                f.write(f"{label}\n")

        # 4. Compute embeddings for the expanded labels
        self.ethosight.embed_labels_from_file(expanded_labels_path)

        # Update the paths
        expanded_embeddings_path = os.path.join(self.app_dir, 'expanded_labels.embeddings')
        
        # Load the embeddings from disk
        expanded_embeddings = self.ethosight.load_embeddings_from_disk(expanded_embeddings_path)
        
        # 5. Set the new embeddings and labels to active status
        self.activeEmbeddingsPath = expanded_embeddings_path
        self.activeEmbeddings = expanded_embeddings
        self.activeLabels = expanded_labels
        self.activeLabelsPath = expanded_labels_path

    def label_space_optimization(self, rerun=False):
        # Get the timestamped filenames for labels and embeddings
        timestamped_files = glob.glob(os.path.join(self.app_dir, "lso_optimized_*"))
        label_files = [file for file in timestamped_files if file.endswith(".labels")]
        embedding_files = [file for file in timestamped_files if file.endswith(".embeddings")]
        
        # Check if we have existing files and if rerun is not requested
        if not rerun and label_files and embedding_files:
            # Find the most recent file by timestamp
            latest_label_file = max(label_files, key=os.path.getctime)
            latest_embedding_file = max(embedding_files, key=os.path.getctime)
            
            # Load the latest labels and embeddings
            self.activeLabels = self.ethosight.read_labels_from_file(latest_label_file)
            self.activeEmbeddings = self.ethosight.load_embeddings_from_disk(latest_embedding_file)
            
            # Set the active paths to the latest files
            self.activeLabelsPath = latest_label_file
            self.activeEmbeddingsPath = latest_embedding_file
        else:
            # If rerun is requested or no previous files are found, perform the optimization
            # Use the factory method to create an instance of the optimizer
            label_space_optimizer = self.create_label_space_optimizer()
            
            # Use the optimizer to perform the label space optimization
            self.activeLabels = self.ethosight.read_labels_from_file(self.activeLabelsPath)
            optimized_labelset = label_space_optimizer.optimize(self.activeLabels)
            self.activeLabels = list(set(optimized_labelset))  # Ensure uniqueness of labels

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.activeLabelsPath = os.path.join(self.app_dir, f"lso_optimized_{timestamp}.labels")
            self.activeEmbeddingsPath = os.path.join(self.app_dir, f"lso_optimized_{timestamp}.embeddings")

            self.ethosight.write_labels_to_file(self.activeLabels, self.activeLabelsPath)
            self.activeEmbeddings = self.ethosight.embed_labels_from_file(self.activeLabelsPath)
            
        if self.config['general_templates']['enabled']:
            self.processGeneralTemplates(self.config['general_templates']['template_path'])

    def _read_ground_truth(self, ground_truth_path):
        """Reads the ground truth from a given path."""
        import csv
        import os

        file_extension = os.path.splitext(ground_truth_path)[1]

        # If the file is a .csv
        if file_extension == ".csv":
            with open(ground_truth_path, mode='r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                ground_truth = {rows[0]: rows[1] for rows in reader if len(rows) > 1}
            return ground_truth

        # If the file is a .txt
        elif file_extension == ".txt":
            with open(ground_truth_path, 'r') as f:
                # Assuming each line in the .txt is a label
                labels = [line.strip() for line in f]
            return dict(enumerate(labels))

        else:
            raise ValueError("Unsupported file type. Please provide a .csv or .txt file.")

    def groundTruthEmbeddings(self, makeActive=True):
        # Get the ground truth path from the config... normally this will be outside the app
        ground_truth_path = self.config['benchmark']['ground_truth_path']
        # Read the ground truth
        ground_truth = self._read_ground_truth(ground_truth_path)
        ground_truth_labels = set(ground_truth.values())
        # make a copy of the ground truth labels in the app directory
        ground_truth_path = os.path.join(self.app_dir, 'ground_truth.labels')
        # generate embeddings for the ground truth labels
        if not os.path.exists(ground_truth_path):
            self.ethosight.write_labels_to_file(ground_truth_labels, ground_truth_path)
            self.ethosight.embed_labels_from_file(ground_truth_path)

        self.gtEmbeddingsPath = os.path.join(self.app_dir, 'ground_truth.embeddings')   
        self.gtEmbeddings = self.ethosight.load_embeddings_from_disk(self.gtEmbeddingsPath)
        self.gtLabels = self.ethosight.read_labels_from_file(ground_truth_path) 
        self.gtLabelsPath = ground_truth_path

        if makeActive:
            self.activeEmbeddingsPath = self.gtEmbeddingsPath 
            self.activeEmbeddings = self.gtEmbeddings
            self.activeLabels = self.gtLabels
            self.activeLabelsPath = self.gtLabelsPath 

    def benchmark_batched(self, verbose=True):
        # Using configurations from YAML
        image_dir = self.config['benchmark']['image_dir']
        ground_truth_path = self.config['benchmark']['ground_truth_path']
        top_n = self.config['benchmark']['top_n']
        normallabel = self.config['benchmark']['normallabel']

        image_filetypes = [".jpg", ".jpeg", ".png"]
        image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir)
                    if any(filename.endswith(filetype) for filetype in image_filetypes)]

        # Read the ground truth
        ground_truth = self._read_ground_truth(ground_truth_path)

        # Convert ground truth to lowercase
        ground_truth = {k: v.lower() for k, v in ground_truth.items()}

        # Compute the top 1 and top n accuracy using the batched method
        results_batched = self.run_batched(image_paths)

        top1_correct = 0
        topn_correct = 0
        anomaly_correct = 0

        for idx, result in enumerate(results_batched):
            labels = result.output['labels']
            scores = result.output['scores']

            # Convert labels to lowercase
            labels = [label.lower() for label in labels]

            top1 = labels[0]
            topn = labels[:top_n]

            image_filename = os.path.basename(image_paths[idx])
            if image_filename in ground_truth:
                is_top1_correct = ground_truth[image_filename] == top1
                is_topn_correct = ground_truth[image_filename] in topn


                if( ((ground_truth[image_filename] == normallabel) and
                    (top1 == normallabel)) or 
                     (top1 != normallabel) ):
                     is_anomaly_correct = True

                if is_top1_correct:
                    top1_correct += 1
                if is_topn_correct:
                    topn_correct += 1
                if is_anomaly_correct:
                    anomaly_correct += 1

                # Verbose output
                if verbose:
                    status = "Correct" if is_topn_correct else "Wrong"
                    print(f"{status} prediction for '{image_filename}':")
                    print("Ground truth:", ground_truth[image_filename])
                    print("Top 1:", top1)
                    print(f"Top {top_n}:")
                    for i in range(top_n):
                        print(topn[i], scores[i])
                    print("\n")

        top1_acc = top1_correct / len(image_paths)
        topn_acc = topn_correct / len(image_paths)
        anomaly_acc = anomaly_correct / len(image_paths)

        # Log and return results
        print(f"Benchmarking results for directory '{image_dir}':")
        print(f"Top 1 accuracy: {top1_acc * 100:.2f}%")
        print(f"Top {top_n} accuracy: {topn_acc * 100:.2f}%")
        print(f"Anomaly accuracy: {anomaly_acc * 100:.2f}%")

        return top1_acc, topn_acc


    def benchmark(self):
        # Using configurations from YAML
        image_dir = self.config['benchmark']['image_dir']
        ground_truth_path = self.config['benchmark']['ground_truth_path']
        top_n = self.config['benchmark']['top_n']

        image_filetypes = [".jpg", ".jpeg", ".png"]
        image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir)
                       if any(filename.endswith(filetype) for filetype in image_filetypes)]

        # Read the ground truth
        ground_truth = self._read_ground_truth(ground_truth_path)

        # Compute the top 1 and top n accuracy
        top1_correct = 0
        topn_correct = 0

        for image_path in image_paths:
            result = self.run(image_path)  # Using the run method of EthosightApp
            labels = result.output['labels']
            scores = result.output['scores']

            top1 = labels[0]
            topn = labels[:top_n]

            image_filename = os.path.basename(image_path)
            if image_filename in ground_truth:
                if ground_truth[image_filename] in topn:
                    topn_correct += 1
                    if ground_truth[image_filename] == top1:
                        top1_correct += 1
                    else:
                        print("Wrong top1 prediction:", image_filename)
                        print("Ground truth:", ground_truth[image_filename])
                        print("Top 1:", top1)
                        print(f"top_{top_n}:{topn}")
                        print("\n")
                else:
                    print("Wrong prediction:", image_filename)
                    print("Ground truth:", ground_truth[image_filename])
                    print(f"Top {top_n}:")
                    for i in range(top_n):
                        print(topn[i], scores[i])
                    print("\n")

        top1_acc = top1_correct / len(image_paths)
        topn_acc = topn_correct / len(image_paths)

        # Log and return results
        print(f"Benchmarking results for directory '{image_dir}':")
        print(f"Top 1 accuracy: {top1_acc * 100:.2f}%")
        print(f"Top {top_n} accuracy: {topn_acc * 100:.2f}%")
        
        return top1_acc, topn_acc

    def _find_most_similar_label_gpt4mode2(self, label: str, affinities):
        # Find the most similar label in the ground truth
        default_prompt = ("Return a label we will use to help analyze an image. The context is to "
                  "identify crime behavior versus normal behavior. The camera angles may be "
                  "overhead or normal more horizontal security camera angles. Analyze "
                  "the following input label and return 'normal event' or 'crime event' "
                  "<<{label}>> return only 'normal event' or 'crime event' as the "
                  "label with no extra text or delimiters of any kind.")
        prompt = self.config['mapper'].get('prompt', default_prompt)
        formatted_prompt = prompt.format(label=label, affinityScores=affinities)
        new_label = self.reasoner.reason(prompt=formatted_prompt)
        new_label = new_label[0]
        if self.config['mapper'].get('verbose', False):
            print(f"New label for '{label}': {new_label}")
        if self.config['mapper'].get('submode', 'na') == 'normallabelonly':
            if new_label != 'normal event':
                new_label = label
        return new_label

    def _find_most_similar_label_gpt4(self, label: str):
        # Find the most similar label in the ground truth
        default_prompt = ("Return a label we will use to help analyze an image. The context is to "
                  "identify crime behavior versus normal behavior. The camera angles may be "
                  "overhead or normal more horizontal security camera angles. Analyze "
                  "the following input label <<{label}>> "
                  "return only 'normal event' or 'crime event' as the "
                  "label with no extra text or delimiters of any kind.")
        prompt = self.config['mapper'].get('prompt', default_prompt)
        formatted_prompt = prompt.format(label=label)
        new_label = self.reasoner.reason(prompt=formatted_prompt)
        new_label = new_label[0]
        if self.config['mapper'].get('verbose', False):
            print(f"New label for '{label}': {new_label}")
        if self.config['mapper'].get('submode', 'na') == 'normallabelonly':
            if new_label != 'normal event':
                new_label = label
        return new_label

    def filter_topn_labels(self, eo, ground_truth_labels, topn):
        """
        Filters the top-n affinity scores and labels that match any of the ground truth labels.
        
        Parameters:
        - eo: EthosightOutput object
        - ground_truth_labels: list of ground truth labels
        - topn: number of top affinity scores and labels to consider
        
        Returns:
        - topn_labels: list of top-n labels that match any of the ground truth labels
        - topn_scores: list of top-n scores corresponding to the labels that match any of the ground truth labels
        """
        
        # Filter the labels and scores based on whether they match any of the ground truth labels
        filtered_labels_and_scores = [
            (label, score) for label, score in zip(eo.output['labels'], eo.output['scores'])
            if label in ground_truth_labels
        ]
        
        # If there are fewer than topn matches, we will return them all
        topn = min(topn, len(filtered_labels_and_scores))
        
        # Unzip the filtered pairs into separate lists
        if filtered_labels_and_scores:
            topn_labels, topn_scores = zip(*filtered_labels_and_scores[:topn])
        else:
            topn_labels, topn_scores = [], []

        return topn_labels, topn_scores


    def mapper(self, eo: EthosightOutput):
        # 1. Extract top 5 labels from the EthosightOutput
        top5_expanded_labels = eo.output['labels'][:5]
        top5_expanded_scores = eo.output['scores'][:5]

        if self.config['mapper']['mode'] == 'passthrough':
            pass

        elif self.config['mapper']['mode'] == 'labelToLabelAffinity01':
            # 2. Map these to the most similar ground truth labels
            composite_labels = []
            for expanded_label in top5_expanded_labels:
                most_similar_gt_label = self._find_most_similar_label(expanded_label)
                composite_labels.append(most_similar_gt_label)

            # Replace the labels in the EthosightOutput with composite labels
            eo.output['labels'][:5] = composite_labels
            eo.output['scores'][:5] = top5_expanded_scores  # scores remain the same

        elif self.config['mapper']['mode'] == 'gpt4mode':
            top5_expanded_labels[0] = self._find_most_similar_label_gpt4(top5_expanded_labels[0])

            # Replace the labels in the EthosightOutput with composite labels
            eo.output['labels'][:5] = top5_expanded_labels 
            eo.output['scores'][:5] = top5_expanded_scores

        elif self.config['mapper']['mode'] == 'gpt4mode2':
            top5_expanded_labels[0] = self._find_most_similar_label_gpt4mode2(top5_expanded_labels[0], eo.output)

            if True:
                # 2. Map these to the most similar ground truth labels
                composite_labels = []
                for expanded_label in top5_expanded_labels:
                    most_similar_gt_label = self._find_most_similar_label(expanded_label)
                    composite_labels.append(most_similar_gt_label)

            # Replace the labels in the EthosightOutput with composite labels
            eo.output['labels'][:5] = top5_expanded_labels 
            eo.output['scores'][:5] = top5_expanded_scores

        elif self.config['mapper']['mode'] == 'hardmap01':
            topn = 5
            topn_expanded_labels, topn_expanded_scores = self.filter_topn_labels(eo, self.gtLabels, topn)

            # Replace the labels in the EthosightOutput with composite labels
            eo.output['labels'][:5] = topn_expanded_labels 
            eo.output['scores'][:5] = topn_expanded_scores

        else:
            print (f"Unsupported mapper mode: {self.config['mapper']['mode']}")
            raise NotImplementedError

        # 3. Apply the affinity min threshold if enabled
        if self.config['mapper']['affinity_minthreshold']['enabled']:
            # Check the top score against the threshold and default to the configured "normal" label if needed
            threshold = self.config['mapper']['affinity_minthreshold']['threshold']
            if top5_expanded_scores[0] < threshold:
                default_label = self.config['mapper']['affinity_minthreshold']['normallabel']
                eo.output['labels'][0] = default_label

        return eo

    def _find_most_similar_label(self, expanded_label):
        # We extract the embedding for the expanded label
        if expanded_label in self.activeEmbeddings:
            expanded_label_embedding = self.activeEmbeddings[expanded_label]
        else:
            print(f"'{expanded_label}' not found in activeEmbeddings")
            expanded_label_embedding = self.ethosight.compute_label_embeddings([expanded_label])
            self.activeEmbeddings.update(expanded_label_embedding)
            expanded_label_embedding = self.activeEmbeddings[expanded_label]

        # Use compute_affinity_scores_labels_to_label to get the affinity scores
        affinity_results = self.ethosight.compute_affinity_scores_labels_to_label(self.gtEmbeddings, {expanded_label: expanded_label_embedding})
        
        # Extract the label with the highest score
        top_label_index = np.argmax(affinity_results['scores'])
        most_similar_label = affinity_results['labels'][top_label_index]
        #print(f"New label for '{expanded_label}': '{most_similar_label}'")
        #exit()
        
        return most_similar_label


    def iterative_learning_loop(self, failure_cases):
        # Implement the logic for the iterative learning loop
        pass

    def get_failure_cases(self):
        # Implement the logic to gather failure cases from your benchmarking
        pass

    def optimize(self, input_data_mode='image', video_path=None, gt_label=None):
        # bootstrap mode starts with ground truth label embeddings
        if self.config['benchmark']['bootstrapMode']:
            self.groundTruthEmbeddings()

        if input_data_mode == "video" and (video_path is None or gt_label is None):
            raise ValueError("video_path and gt_label must be provided when input_data_mode is 'video'")

        if self.config['benchmark']['enabled'] and not self.config['benchmark']['skip_pre_optimization']:
            print("Benchmarking before optimization:")
            start_time = time.time()

            # Check the data mode
            if input_data_mode == "video":
                top1_acc, topn_acc = self.benchmark_video(video_path, gt_label, optmization=False)
            else:  # Assuming the default mode is image
                batch_mode = self.config['benchmark'].get('batch_mode', False)
                top1_acc, topn_acc = self.benchmark_batched() if batch_mode else self.benchmark()

            print(f"Benchmarking took {time.time() - start_time:.2f} seconds")
        else:
            top1_acc, topn_acc = None, None

        if self.config['label_space_optimization']['enabled']:
            # Perform label space optimization
            self.label_space_optimization(rerun=self.config['label_space_optimization']['rerun'])

            if self.config['benchmark']['enabled']:
                print("Benchmarking after optimization:")
                start_time = time.time()

                # Check the data mode again
                if input_data_mode == "video":
                    top1_acc, topn_acc = self.benchmark_video(video_path, gt_label, optmization=True)
                else:
                    batch_mode = self.config['benchmark'].get('batch_mode', False)
                    top1_acc, topn_acc = self.benchmark_batched() if batch_mode else self.benchmark()
                
                print(f"Benchmarking took {time.time() - start_time:.2f} seconds")
            else:
                top1_acc, topn_acc = None, None
        return top1_acc, topn_acc

    def run(self, image):
        affinities = self.media_analyzer.analyzeImage(image, self.activeEmbeddings)
        print(f"affinities pre-mapper: {affinities}")

        if self.config['mapper']['enabled']:
            affinities = self.mapper(affinities)
            print(f"affinities post-mapper: {affinities}")

        return affinities

    def run_batched(self, image_paths, optmization=False):
        """
        Analyze a batch of images and get their affinity scores.
        
        Args:
        image_paths (list of str): List of paths for images to analyze.
        
        Returns:
        List of affinities for each image.
        """

        affinities_batch = self.media_analyzer.analyzeImages_precompiled_batched(image_paths, self.activeEmbeddings)

        # If the mapper is enabled, apply it to each image's affinities
        if self.config['mapper']['enabled']:
            for idx, affinities in enumerate(affinities_batch):
                if self.config['benchmark']['verbose']:
                    print(f"affinities pre-mapper for image {idx} {image_paths[idx]}: {affinities}")
                affinities_batch[idx] = self.mapper(affinities)
                if self.config['benchmark']['verbose']:
                    print(f"affinities post-mapper for image {idx} {image_paths[idx]}: {affinities_batch[idx]}")

        # save the affinities to a file
        if self.config['benchmark']['save_affinities']:
            self.save_affinities(affinities_batch, image_paths, optmization=optmization)
        return affinities_batch

    def save_affinities(self, affinities_batch, image_paths, optmization=False):
        """
        Save all the affinities to a json file.

        Args:
        affinities_batch (list of list of dict): List of affinities for each image.
        image_paths (list of str): List of paths for images to analyze.
        """
        # todo: handle case for pure images
        if len(image_paths) == 0:
            return
        video_name = os.path.basename(os.path.dirname(image_paths[0]))
        affinities_save_dir = os.path.join(self.app_dir, 'affinities')
        if not os.path.exists(affinities_save_dir):
            os.makedirs(affinities_save_dir)
        affinities_json_file = os.path.join(affinities_save_dir, f'{video_name}_affinities_label_optimization_{optmization}.json')
        with open(affinities_json_file, 'w') as f:
            # key as image_path, value as affinities, append all in one file
            for idx, affinities in enumerate(affinities_batch):
                image_name = os.path.basename(image_paths[idx]).split('.')[0]
                json.dump({image_name: affinities.to_json()}, f, indent=4)
                f.write('\n')
        print(f"Saved affinities to {affinities_json_file}")

    def video_to_images(self, video_path):
        """
        Extract frames from a video and save them as images.
        """
        # if the video is already extracted, skip this step
        video_name = os.path.basename(video_path).split('.')[0]
        extracted_video_dir = self.config['benchmark']['extracted_video_dir']
        output_dir = os.path.join(extracted_video_dir, video_name)

        if os.path.exists(output_dir):
            print(f"Video {video_name} has already been extracted.")
            return output_dir
        else:
            os.makedirs(output_dir)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print("Error: Cannot open video file")
                return
        
            frame_num = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Save frame as image
                output_path = os.path.join(output_dir, f'frame_{frame_num + 1}.png')
                cv2.imwrite(output_path, frame)
                frame_num += 1
            cap.release()
        return output_dir

    def downsample_images(self, video_image_dir, skip_frames_param):
        if not os.path.exists(video_image_dir):
            print(f"Error: Cannot find directory {video_image_dir}")
            return
        output_paths = []
        for i in range(1, len(os.listdir(video_image_dir)) + 1):
            if i % (skip_frames_param + 1) == 0:
                image_path = os.path.join(video_image_dir, f'frame_{i}.png')
                output_paths.append(image_path)
        return output_paths


    def benchmark_video(self, video_path, gt_label, optmization=False, per_frame=False):
        """
        Benchmark a video against a ground truth label.

        Args:
        video_path (str): Path to the video to benchmark.
        gt_label (str): Ground truth label for the video.
        optmization (bool): Whether to use the optimized label space. Note that this is only used for the saving of the affinity scores.
        """
        skip_frames_count = self.config['video']['skip_frames']
        video_image_dir = self.video_to_images(video_path)
        frame_paths = self.downsample_images(video_image_dir, skip_frames_count)
        
        # Lowercase the ground truth label for the video for consistent comparison
        gt_label = gt_label.lower()
        
        # Analyze the frames
        results_batched = self.run_batched(frame_paths, optmization=optmization)

        if per_frame:
            return self._calculate_per_frame_accuracy(results_batched, gt_label, frame_paths, video_path)
        else:
            return self._calculate_video_accuracy(results_batched, gt_label, video_path)

    def _calculate_per_frame_accuracy(self, results_batched, gt_label, frame_paths, video_path):
        top1_correct = 0
        topn_correct = 0
        top_n = self.config['benchmark']['top_n']  # Take from the config like before

        # Here's the tqdm addition
        for result in tqdm(results_batched, desc="Processing frames", ncols=100):
            labels = result.output['labels']
            scores = result.output['scores']

            # Convert labels to lowercase
            labels = [label.lower() for label in labels]

            top1 = labels[0]
            topn = labels[:top_n]

            # Check against ground truth label
            is_top1_correct = gt_label == top1
            is_topn_correct = gt_label in topn

            if is_top1_correct:
                top1_correct += 1
            if is_topn_correct:
                topn_correct += 1

        total_frames = len(frame_paths)
        
        top1_acc = top1_correct / total_frames
        topn_acc = topn_correct / total_frames

        # Log and return results
        print(f"Benchmarking results for video '{video_path}':")
        print(f"Top 1 accuracy: {top1_acc * 100:.2f}%")
        print(f"Top {top_n} accuracy: {topn_acc * 100:.2f}%")

        return top1_acc, topn_acc

    def _calculate_video_accuracy(self, results_batched, gt_label, video_path):
        top_n = self.config['benchmark']['top_n']  # Take from the config like before

        # get top 1 label from all frames
        frame_top1_labels = []
        for result in results_batched:
            labels = result.output['labels']
            frame_top1_labels.append(labels[0])

        # map frame top1 labels to video label
        video_labels = self.map_frame_labels_to_video_label(frame_top1_labels)

        # calculate accuracy
        is_top1_correct = gt_label == video_labels[0]
        top1_acc = 1 if is_top1_correct else 0
        topn_acc = 1 if gt_label in video_labels[:top_n] else 0

        # Log and return results
        print(f"Benchmarking results for video '{video_path}':")
        print(f"Top 1: {top1_acc}; Top {top_n}: {topn_acc}")
        print(f"Grouth truth label: {gt_label}")
        print(f"Top 1 label: {video_labels[0]}")
        print(f"Top {top_n} labels: {video_labels[:top_n]}")

        return top1_acc, topn_acc

    def map_frame_labels_to_video_label(self, frame_labels):
        """
        Map a list of frame labels to a single video label.

        Args:
        frame_labels (list): List of frame labels.

        Returns:
        Sorted list of video labels.
        """

        if self.config["video"]["label_mapping"] == "majority":
            # frequency based mapping
            from collections import Counter

            label_count = Counter(frame_labels).most_common()
            print(f"Label count: {label_count}")
            video_labels, _ = zip(*label_count)
            return video_labels

        max_period_dict, periods_count_dict = self._get_period(frame_labels)
        if self.config["video"]["label_mapping"] == "longest_period":
            # post processing
            min_number_frames_as_period = 5
            # filter out periods that are too short
            max_period_dict = {k: v for k, v in max_period_dict.items() if v > min_number_frames_as_period}
            # sort by longest period
            sorted_by_longest_period = sorted(max_period_dict.keys(), key=lambda k: max_period_dict[k], reverse=True)

            # remove normal label if there are more than one label in sorted_by_longest_period
            normal_label_name = self.config["video"]["normal_label_name"]
            if len(sorted_by_longest_period) > 1 and normal_label_name in sorted_by_longest_period:
                sorted_by_longest_period.remove(normal_label_name)
            return sorted_by_longest_period

        if self.config["video"]["label_mapping"] == "periods_count":
            # post processing
            # sort by periods count
            sorted_by_periods_count = sorted(periods_count_dict.keys(), key=lambda k: periods_count_dict[k], reverse=True)
            # remove normal label if there are more than one label in sorted_by_periods_count
            normal_label_name = self.config["video"]["normal_label_name"]
            if len(sorted_by_periods_count) > 1 and normal_label_name in sorted_by_periods_count:
                sorted_by_periods_count.remove(normal_label_name)
            return sorted_by_periods_count

    def _get_period(self, frame_labels):
        """
        Get the longest period and number of periods for each existed label.
        """
        from collections import defaultdict
        # This dictionary keeps track of the longest period for each label
        max_period_dict = defaultdict(int)

        # This dictionary keeps track of the number of periods for each label
        periods_count_dict = defaultdict(int)

        # These variables keep track of the current label and its length
        current_label = frame_labels[0]
        current_length = 1

        # Iterate over the list from the second element
        for label in frame_labels[1:]:
            if label == current_label:
                # Increase the current length if the label is the same
                current_length += 1
            else:
                # Update the longest period for the current label
                max_period_dict[current_label] = max(max_period_dict[current_label], current_length)

                # Update the periods count for the current label
                periods_count_dict[current_label] += 1

                # Reset the current label and its length
                current_label = label
                current_length = 1

        # Make sure to update the longest period and periods count for the last label
        max_period_dict[current_label] = max(max_period_dict[current_label], current_length)
        periods_count_dict[current_label] += 1

        print(f"Max period dict: {max_period_dict}")
        print(f"Periods count dict: {periods_count_dict}")
        return max_period_dict, periods_count_dict

    def read_video_gt_csv(self, csv_filename):
        """
        Read video paths and ground truth labels from a CSV file.
        
        Args:
        csv_filename (str): Path to the CSV file.
        
        Returns:
        List of tuples containing video paths and corresponding ground truth labels.
        """

        video_gt_pairs = []
        
        with open(csv_filename, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            
            # Skip the header
            next(csvreader)
            
            for row in csvreader:
                video_path, gt_label = row
                print(f"Video path: {video_path}; Ground truth label: {gt_label}")
                video_gt_pairs.append((video_path, gt_label))
        
        return video_gt_pairs

    def benchmark_videos_from_csv(self, csv_filename):
        """
        Benchmark multiple videos by calling the optimize method for each video using video path and ground truth pairs from a CSV.
        
        Args:
        csv_filename (str): Path to the CSV file containing video paths and ground truth labels.
        
        Returns:
        List of tuples containing top 1 and top n accuracies for each video.
        """

        video_gt_pairs = self.read_video_gt_csv(csv_filename)
        
        return self.benchmark_videos(video_gt_pairs)

    def benchmark_videos(self, video_gt_pairs):
        """
        Benchmark multiple videos by calling the optimize method for each video.
        
        Args:
        video_gt_pairs (list of tuples): List of tuples containing video paths and corresponding ground truth labels.
        
        Returns:
        List of tuples containing top 1 and top n accuracies for each video.
        """

        top1_accs = []
        topn_accs = []

        for video_path, gt_label in video_gt_pairs:
            # Call the optimize function for each video
            top1_acc, topn_acc = self.optimize(input_data_mode='video', video_path=video_path, gt_label=gt_label)
            
            top1_accs.append(top1_acc)
            topn_accs.append(topn_acc)

        print(f"Results for {len(video_gt_pairs)} videos:")
        print(f"Top 1 accuracies: {top1_accs}")
        print(f"Top n accuracies: {topn_accs}")
        print(f"Average top 1 accuracy: {sum(top1_accs) / len(top1_accs)}")
        print(f"Average top n accuracy: {sum(topn_accs) / len(topn_accs)}")
        return top1_accs, topn_accs

    def rank_affinities(self, json_file_path):
        results = []
        buffer = ""
        depth = 0

        with open(json_file_path, 'r') as file:
            for line in file:
                buffer += line
                depth += line.count('{') - line.count('}')

                # When depth reaches zero, we have a complete JSON object
                if depth == 0:
                    frame_data = json.loads(buffer)
                    for frame_Id, affinities in frame_data.items():
                        for affinity, value in affinities.items():
                            item = (f"{frame_Id}:{affinity}", value)
                            results.append(item)
                    buffer = ""

        results.sort(key=lambda x: x[1], reverse=True)
        top_n = self.config['visualization']['top_n_affinity']
        isUnique = self.config['visualization']['show_distinct']

        if top_n > len(results):
            raise ValueError('The top parameter is greater than affinities length')

        if isUnique:
            results = self._get_unigue_labels_rank(results, top_n)
        else:
            results = results[:top_n]

        for result in results:
            print(f'{result}\n')

    def _get_unigue_labels_rank(self, sorted_affinity_tuples, top_n):
        s = set()
        unique_result = []
        for i in range(len(sorted_affinity_tuples)):
            affinity = sorted_affinity_tuples[i] 
            label_name = affinity[0].split(":")[1]
            if label_name not in s:
                unique_result.append(affinity)
                s.add(label_name)
            if len(unique_result) >= top_n:
                break
        return unique_result

    def read_video_gt_csv(self, csv_filename, debug=False):
        """
        Read video paths and ground truth labels from a CSV file.

        Args:
            csv_filename (str): Path to the CSV file.

        Returns:
            List of tuples containing video paths and corresponding ground truth labels.
        """

        video_gt_pairs = []

        try:
            with open(csv_filename, 'r') as csvfile:
                csvreader = csv.reader(csvfile)

                # Ensure there is a header
                next(csvreader)

                for row in csvreader:
                    if len(row) != 2:
                        if debug:
                            print(f"Skipping invalid row: {row}")
                        continue
                    video_path, gt_label = row
                    if debug:   
                        print(f"Video path: {video_path}; Ground truth label: {gt_label}")
                    video_gt_pairs.append((video_path, gt_label))

        except FileNotFoundError:
            print(f"Error: {csv_filename} not found.")
        except csv.Error as e:
            print(f"Error processing CSV file: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        return video_gt_pairs

    def rank_affinities2(self, json_file_path):
        results = []
        buffer = ""
        depth = 0

        with open(json_file_path, 'r') as file:
            for line in file:
                buffer += line
                depth += line.count('{') - line.count('}')

                # When depth reaches zero, we have a complete JSON object
                if depth == 0:
                    frame_data = json.loads(buffer)
                    for frame_Id, affinities in frame_data.items():
                        for affinity, value in affinities.items():
                            item = (f"{frame_Id}:{affinity}", value)
                            results.append(item)
                    buffer = ""

        results.sort(key=lambda x: x[1], reverse=True)
        top_n = self.config['visualization']['top_n_affinity']
        isUnique = self.config['visualization']['show_distinct']

        if top_n > len(results):
            raise ValueError('The top parameter is greater than affinities length')

        if isUnique:
            results = self._get_unigue_labels_rank(results, top_n)
        else:
            results = results[:top_n]

        labels = [label for label, _ in results]
        scores = [score for _, score in results]

        return EthosightOutput('affinityScores', {"labels": labels, "scores": scores})


    def analyzeVideoAffinities(self, ranked_affinities, debug=False):
        # Analyze the ranked affinities and determine the predicted ground truth label
        # For now, we'll assume the highest-ranked affinity is the predicted label
        predicted_label = ranked_affinities[0][0].split(':')[1]
        if debug:
            print(f"Predicted label: {predicted_label}") 
        return predicted_label

    def phase2Benchmark(self, video_path, gt_label, debug=True):
        """Benchmark a video against ground truth label."""
        ranked_affinities = self.rank_affinities2(video_path)
        predicted_label = self.analyzeVideoAffinities(ranked_affinities, debug)

        # Store both the ground truth and the predicted label
        if debug:
            print("#" * 80)
            print(f"Metrics for {video_path}:")
            print(f"Ground truth label: {gt_label}")
            print(f"Predicted label: {predicted_label}")
            
        return gt_label, predicted_label

    def compute_summary_statistics(self, y_true, y_pred, top1_correct, topn_correct, counter, top_n):
        # Compute anomaly metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label='anomaly')
        recall = recall_score(y_true, y_pred, pos_label='anomaly')
        f1 = f1_score(y_true, y_pred, pos_label='anomaly')
        
        # Compute top-1 and top-N accuracies
        top1_accuracy = top1_correct / counter
        topn_accuracy = topn_correct / counter

        # Print metrics
        print("\nAnomaly Detection Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Top-1 Accuracy: {top1_accuracy:.4f}")
        print(f"Top-{top_n} Accuracy: {topn_accuracy:.4f}")
        print(f"Total predictions: {counter}")

        return {
            "total_predictions": counter,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "top1_accuracy": top1_accuracy,
            "topn_accuracy": topn_accuracy
        }

    def phase2videobenchmarks(self, phase2_groundtruth_csv):
        progress_file = 'progress.json'
        affinityScores = []
        binaryLabels = []
    
        # Check for existing progress
        start_idx = 0
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
                start_idx = progress_data['last_processed_idx'] + 1

        print("Phase 2 video benchmarks")

        # Read video and ground truth pairs from the CSV file
        video_gt_pairs = self.read_video_gt_csv(phase2_groundtruth_csv)

        # Skip videos that have already been processed
        video_gt_pairs = video_gt_pairs[start_idx:]

        # Use configuration to get max_iterations and top_n
        max_iterations = self.config['phase2'].get('maxiterations', 1)
        top_n = self.config['phase2'].get('top_n', 5)

        # Initialize results storage, anomaly counters and top accuracies
        benchmark_results = []
        y_true, y_pred = [], []
        top1_correct, topn_correct = 0, 0

        # Loop counter
        counter = 0

        # Define the 'normal' label
        normal_label = "normal event"

        # Initialize tqdm progress bar
        pbar = tqdm(total=min(len(video_gt_pairs), max_iterations), desc="Processing videos")

        # Iterate through the video and ground truth pairs
        for video_path, gt_label in video_gt_pairs:
            if counter >= max_iterations:
                break

            try:
                print(f"Processing {video_path}...")
                #Generate an EthosightOutput for the predicted label from rank_affinities2
                predicted_output = self.rank_affinities2(video_path)
                predicted_output.output['labels'] = [label.split(":")[1].lower() for label in predicted_output.output['labels']]
                if self.config['mapper']['enabled']:
                    predicted_output = self.mapper(predicted_output)
                print(f"Predicted output data: {predicted_output.output}")
                predicted_labels = predicted_output.output['labels'] 
                

                # Step 1: Obtain the Top-1 Prediction Label and Score
                top1_label = predicted_output.output['labels'][0]
                top1_score = predicted_output.output['scores'][0]
                
                # Step 2: Convert to Binary Classification
                binary_label = 0 if top1_label == 'normal event' else 1
                
                # Step 3: Accumulate Scores
                affinityScores.append(top1_score)
                binaryLabels.append(binary_label)
                

                # Adding to the benchmark results
                benchmark_results.append({
                    "video_path": video_path,
                    "ground_truth": gt_label,
                    "predicted_affinities": predicted_output
                })

                print(f"Benchmark result for {video_path}: Ground truth - {gt_label}, Predicted - {predicted_labels[0]}")

                y_true.append(normal_label if gt_label == normal_label else 'anomaly')
                y_pred.append(normal_label if predicted_labels[0] == normal_label else 'anomaly')

                # Update top-1 and top-N accuracies
                if gt_label == predicted_labels[0]:
                    top1_correct += 1
                if gt_label in predicted_labels[:top_n]:
                    topn_correct += 1

                # Increment counter
                counter += 1
                pbar.update(1)

            except Exception as e:
                print(f"Error processing video {video_path}: {e}")
                traceback.print_exc()
                raise e

            # At the end of processing a video:
            with open(progress_file, 'w') as f:
                json.dump({'last_processed_idx': counter}, f)

        # If all videos processed, delete the progress file
        if counter >= len(video_gt_pairs):
            os.remove(progress_file)

        pbar.close()

        self.compute_summary_statistics(y_true, y_pred, top1_correct, topn_correct, counter, top_n)

        self.create_roc_auc_csv(phase2_groundtruth_csv, 
                                affinityScores,
                                'roc_auc.csv')

        roc_auc = self.compute_roc_auc('roc_auc.csv')
        print("ROC AUC Score:", roc_auc)

        return benchmark_results


    def create_roc_auc_csv(self, input_csv_path, affinity_scores, output_csv_path):
        """
        Create a new CSV file for ROC AUC computation.
        
        Parameters:
        input_csv_path (str): The path to the input CSV file containing video filenames and ground truth labels.
        affinity_scores (list): A list containing the affinity scores for each video.
        output_csv_path (str): The path to the new CSV file that will be created.
        """
        
        # Load the ground truth data from the input CSV file
        data = pd.read_csv(input_csv_path)

        # For development purposes: 
        # Restrict the data to the length of affinity_scores
        num_samples = len(affinity_scores)
        data = data.head(num_samples)
        print(f"Loaded {num_samples} samples from {input_csv_path}. data: {data}")
        print(f"Loaded {len(affinity_scores)} affinity scores.")

        if len(affinity_scores) != len(data):
            print("Error: Length of affinity_scores does not match number of rows in input CSV.")
            return
        
        # Normalize the affinity scores using Min-Max normalization
        min_score = min(affinity_scores)
        max_score = max(affinity_scores)
        if max_score == min_score:
            normalized_scores = [0.5 for _ in affinity_scores]  # or some default value
        else:
            normalized_scores = [(score - min_score) / (max_score - min_score) for score in affinity_scores]
 
        print(data)
        print(normalized_scores)
        print(len(normalized_scores))

        # Add a new column 'scores' to the DataFrame
        data['scores'] = normalized_scores
        
        # Convert the 'groundtruth' column to binary labels (0 if 'normal event' else 1)
        data['groundtruth'] = data['groundtruth'].apply(lambda x: 0 if x == 'normal event' else 1)
        
        # Write the new CSV file
        data.to_csv(output_csv_path, index=False)

    def compute_roc_auc(self, csv_path):
        """
        Compute the ROC AUC score.
        
        Parameters:
        csv_path (str): The path to the CSV file that contains 'groundtruth' and 'scores' columns.
        
        Returns:
        float: The ROC AUC score.
        """
        
        # Load the data from the CSV file
        data = pd.read_csv(csv_path)
        
        # Extract the true binary labels and the predicted probabilities
        true_labels = data['groundtruth']
        predicted_scores = data['scores']
        
        # Compute the ROC AUC score
        roc_auc = roc_auc_score(true_labels, predicted_scores)
        
        return roc_auc

    def add_labels(self, new_labels):
        """Adds new labels to the EthosightApp active embedding."""

        # Load the current labels from the file
        with open(self.labels_path, 'r') as file:
            current_labels = file.read().splitlines()
        
        # Add new unique labels that are not already in the current labels
        added_labels = []
        for label in new_labels:
            if label not in current_labels:
                current_labels.append(label)
                added_labels.append(label)
                print(f"Added label '{label}'")

        # Save the updated labels back to the file
        with open(self.labels_path, 'w') as file:
            for label in current_labels:
                file.write(f"{label}\n")

        if not added_labels:
            print("No new labels added.")
            return

        # compute the new embeddings
        new_embeddings = self.ethosight.compute_label_embeddings(added_labels)

        #load the original embeddings
        original_embeddings = self.ethosight.load_embeddings_from_disk(self.embeddings_path)

        # Append the new embeddings to the original embeddings
        for label, embedding in new_embeddings.items():
            original_embeddings[label] = embedding

        # Save the combined embeddings back to disk
        self.ethosight.save_embeddings_to_disk(original_embeddings, self.embeddings_path)
