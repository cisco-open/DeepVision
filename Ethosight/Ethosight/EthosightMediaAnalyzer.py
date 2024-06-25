
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

import os
import copy
import yaml
from Ethosight import Ethosight
from Ethosight.utils import get_install_path
from .ChatGPTReasoner import ChatGPTReasoner
from .langchain_reasoner import LangchainReasoner
from .llama_index_reasoner import LlamaIndexReasoner
from tqdm import tqdm
#need to handle the NARSGPT directory
#from NARSGPTReasoner import NARSGPTReasoner
#from NARSReasoner import NARSReasoner
#from .Ethosight import Ethosight

class EthosightOutput:
    def __init__(self, outputType, output):
        if outputType not in EthosightMediaAnalyzer.SUPPORTED_OUTPUT_TYPES:
            raise ValueError(f'Invalid output type: {outputType}. Must be one of {EthosightMediaAnalyzer.SUPPORTED_OUTPUT_TYPES}.')

        self.outputType = outputType
        self.original_output = copy.deepcopy(output)  # store the original output

    # Perform output-specific validation and formatting
        if outputType == 'affinityScores':
            if not isinstance(output, dict):
                raise ValueError('Output of type "affinityScores" must be a dictionary.')
            elif not 'labels' in output or not 'scores' in output:
                raise ValueError('Output of type "affinityScores" must be a dictionary with "labels" and "scores" keys.')
            elif len(output['labels']) != len(output['scores']):
                raise ValueError('The "labels" and "scores" arrays in the output must be of the same length.')
            # Convert arrays to lists for ease of use only if they are numpy arrays
            output['labels'] = output['labels'].tolist() if hasattr(output['labels'], 'tolist') else output['labels']
            output['scores'] = output['scores'].tolist() if hasattr(output['scores'], 'tolist') else output['scores']

        elif outputType == 'groundTruthClass':
            if not isinstance(output, str):
                raise ValueError('Output of type "groundTruthClass" must be a string.')

        self.output = output

    def __str__(self):
        # Format the output in a human-readable way
        if self.outputType == 'affinityScores':
            formatted_scores = ', '.join(f'{label}: {score:.3f}' for label, score in zip(self.output['labels'], self.output['scores']))
            return f'Affinity Scores: {formatted_scores}'
        elif self.outputType == 'groundTruthClass':
            return f'Ground Truth Class: {self.output}'

    def to_json(self):
        # Convert the output to JSON
        if self.outputType == 'affinityScores':
            # each label and score is a separate key-value pair in the JSON
            labels = self.output['labels']
            scores = self.output['scores']
            return {label: score for label, score in zip(labels, scores)}

        elif self.outputType == 'groundTruthClass':
            return {
                'groundTruthClass': self.output,
            }

class EthosightMediaAnalyzer:

    REASONER_CLASSES = {
        'chatgpt': ChatGPTReasoner,
        'langchain': LangchainReasoner,
        'llamaindex': LlamaIndexReasoner,
        '': ''
#        'narsgpt': NARSGPTReasoner,
#        'nars': NARSReasoner,
    }

    SUPPORTED_OUTPUT_TYPES = ["affinityScores", "groundTruthClass"]

    def validate_config(self, config):
        reasoner_type = config.get('reasoner_type')
        if reasoner_type is None:
            raise ValueError('reasoner_type not found in the configuration')
        if reasoner_type not in EthosightMediaAnalyzer.REASONER_CLASSES:
            raise ValueError(f'Invalid reasoner_type: {reasoner_type}. Must be one of {list(EthosightMediaAnalyzer.REASONER_CLASSES.keys())}.')

        # Validate analyzeImageMethod
        analyzeImageMethod = config.get('analyzeImageMethod')
        if analyzeImageMethod is None:
            raise ValueError('analyzeImageMethod not found in the configuration')
        if analyzeImageMethod not in self.analyzeImageMethods:
            raise ValueError(f'Invalid analyzeImageMethod: {analyzeImageMethod}. Must be one of {list(self.analyzeImageMethods.keys())}.')

        # Validate output_type
        output_type = config.get('output_type')
        if output_type is None:
            raise ValueError('output_type not found in the configuration')
        if output_type not in EthosightMediaAnalyzer.SUPPORTED_OUTPUT_TYPES:
            raise ValueError(f'Invalid output_type: {output_type}. Output type must be one of {EthosightMediaAnalyzer.SUPPORTED_OUTPUT_TYPES}.')

        # Return True if validation passes
        return True

    def validateConfigfile(self, configfile):
        # Load the config from the YAML file
        with open(configfile, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Call validate_config to check the loaded config data
        self.validate_config(config_data)

        # Return True if validation passes
        return True

    def load_config(self, filepath):
        with open(filepath, 'r') as f:
            config_data = yaml.safe_load(f)
        return config_data

    def load_reasoner(self, config):
        reasoner_type = config.get('reasoner_type')
        ReasonerClass = self.REASONER_CLASSES.get(reasoner_type)
        if ReasonerClass is None:
            raise ValueError(f"Invalid reasoner type: {reasoner_type}")
        if ReasonerClass == '':
            return ''
        return ReasonerClass()  # assuming reasoner classes have no-arg constructor


    def __init__(self, configfile):
        # initialize with config
        self.configfile = configfile
        self.config_data = self.load_config(configfile)

        self.analyzeImageMethods = {
            'analyzeImage_precompiled': self.analyzeImage_precompiled,
        }
        self.validate_config(self.config_data)    # Get package install path
        self.package_home = get_install_path('Ethosight')
        self.analyzeImageMethod = self.config_data.get('analyzeImageMethod')  

        self.reasoner = self.load_reasoner(self.config_data)
        self.ethosight = Ethosight(reasoner=self.reasoner)
    
    def analyzeImage(self, image_path, embeddings=None):
        # Call the analyzeImageMethod
        output = self.analyzeImageMethods[self.analyzeImageMethod](image_path, embeddings)
        return output

    def analyzeVideo(self, video_path):
        # Implement the logic of analyzing a video
        pass

    def analyzeImage_method1(self, image):
    # Implementation of analyzeImage method 1
        pass

    def analyzeImage_precompiled(self, image, embeddings=None):
        if embeddings is None:
            embeddings_path = self.config_data.get('embeddings_path')
            if embeddings_path is None:
                raise ValueError('embeddings_path not found in the configuration')
            # Build the absolute path to the embeddings
            embeddings_path = os.path.join(self.package_home, 'embeddings', embeddings_path)
            embeddings = self.ethosight.load_embeddings_from_disk(embeddings_path)

        # The rest of the code remains unchanged
        affinity_scores = self.ethosight.compute_affinity_scores(embeddings, image, verbose=False)

        # Check the output type and return the output accordingly
        output_type = self.config_data.get('output_type')
        if output_type == 'affinityScores':
            # Construct an EthosightOutput object
            output = EthosightOutput(outputType=output_type, output=affinity_scores)
            return output 
        else:
            raise ValueError(f'Unsupported output type: {output_type}.')


    def analyzeImages_precompiled_batched(self, image_paths, embeddings=None):
        """
        Analyzes a batch of images using precompiled embeddings.
        
        Args:
        image_paths (list of str): List of paths for images to analyze.
        embeddings (Optional): Preloaded embeddings to use. If None, embeddings will be loaded from disk.

        Returns:
        List of EthosightOutput objects for each image in the batch.
        """

        batch_size = self.config_data['benchmark']['batchsize']  # Fetch batch_size from config
        
        # If embeddings are not provided, load them from the specified path
        if embeddings is None:
            embeddings_path = self.config_data.get('embeddings_path')
            if embeddings_path is None:
                raise ValueError('embeddings_path not found in the configuration')
            
            # Build the absolute path to the embeddings
            embeddings_path = os.path.join(self.package_home, 'embeddings', embeddings_path)
            embeddings = self.ethosight.load_embeddings_from_disk(embeddings_path)

        # Get the affinity scores for the batch of images
        affinity_scores_batch = self.ethosight.compute_affinity_scores_batched(embeddings, image_paths, verbose=False, batch_size=batch_size)

        # Check the output type and return the output accordingly
        output_type = self.config_data.get('output_type')
        if output_type != 'affinityScores':
            raise ValueError(f'Unsupported output type: {output_type}.')

        outputs = []
        for affinity_scores in affinity_scores_batch:
            # Construct an EthosightOutput object for each image and add to the outputs list
            output = EthosightOutput(outputType=output_type, output=affinity_scores)
            outputs.append(output)

        return outputs


    def compute_accuracy_for_directory(self, image_dir, ground_truth, topnvalue):
        image_filetypes = [".jpg", ".jpeg", ".png"]
        image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir)
                    if any(filename.endswith(filetype) for filetype in image_filetypes)]

        # Adjust the path to the labels file
        labels_file_path = os.path.join(image_dir, ground_truth)

        # Read ground truth image_paths and labels
        import csv
        with open(labels_file_path, mode='r') as f:
            reader = csv.reader(f)
            next(reader)
            ground_truth = {os.path.basename(rows[0]): rows[1] for rows in reader}

        # Compute the top 1 and top 3 accuracy
        top1_correct = 0
        topn_correct = 0
        for image_path in image_paths:
            result = self.analyzeImage(image_path)  # Use self to call the method
            labels = result.output['labels']
            top1 = labels[0]
            topn = labels[:topnvalue]
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
                        print("\n")
                else:
                    print("Wrong prediction:", image_filename)
                    print("Ground truth:", ground_truth[image_filename])
                    print(f"Top {topnvalue}:")
                    scores = result.output['scores']  # Get the scores from the result
                    for i in range(topnvalue):
                        print(topn[i], scores[i])
                    print("\n")
        top1_acc = top1_correct / len(image_paths)
        topn_acc = topn_correct / len(image_paths)
        print("Top 1 accuracy:", top1_acc)
        print(f"Top {topnvalue} accuracy:", topn_acc)
        return top1_acc, topn_acc

