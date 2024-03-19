
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
import torch
from models import imagebind_model
from models.imagebind_model import ModalityType
import data
from tqdm import tqdm
import torchvision
from torchvision import transforms, datasets
import numpy as np
#from llama_index_reasoner import LlamaIndexReasoner
from .ChatGPTReasoner import ChatGPTReasoner
from .utils import get_install_path
import json
import torch.nn.functional as F

class EthosightCore:
    def __init__(self, model=None, reasoner='', gpu=0):
        #gpu=none means use cpu
        self.ethosight_dir = get_install_path('Ethosight')
        if model is None:
            # !!!DANGER DANGER!!! Monkey patching the data.BPE_PATH to point to the correct path
            data.BPE_PATH = os.path.join(self.ethosight_dir, data.BPE_PATH)
            self.model = imagebind_model.imagebind_huge(pretrained=True, imagebind_dir=self.ethosight_dir)
            self.model.eval()
            # If GPU parameter is provided, use it. Otherwise, use the first available GPU or CPU.
            self.device = torch.device(f"cuda:{gpu}" if gpu is not None and torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
        else:
            self.model = model
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        if reasoner is None:
            #self.reasoner = LlamaIndexReasoner()
            self.reasoner = ChatGPTReasoner()
        else:
            self.reasoner = reasoner

    def write_labels_to_file(self, labels, filepath):
        if not labels:
            raise ValueError("Labels list is empty.")
        labels = list(set(labels))  # ensure uniqueness of labels
        with open(filepath, 'w') as f:
            for label in labels:
                f.write(label + '\n')

    def compute_label_embeddings(self, labels, batch_size=1200):
        unique_labels = list(set(labels))

        def batched_data(data, batch_size):
            # Generator function for creating batches
            for i in range(0, len(data), batch_size):
                yield data[i:i+batch_size]
        
        unique_label_embeddings = []
        
        for i, batch in enumerate(tqdm(batched_data(unique_labels, batch_size), desc='Processing label batches', leave=False)):
            inputs = {}
            inputs[ModalityType.TEXT] = data.load_and_transform_text(batch, self.device)
            with torch.no_grad():
                batch_text_embeddings = self.model(inputs)[ModalityType.TEXT].cpu()
            unique_label_embeddings.append(batch_text_embeddings)
        
        # Concatenate all batch embeddings into a single tensor
        unique_label_embeddings = torch.cat(unique_label_embeddings, dim=0)

        # Instead of concatenating the embeddings, create a dictionary mapping labels to embeddings
        unique_label_embeddings = {label: embedding for label, embedding in zip(unique_labels, unique_label_embeddings)}
        
        return unique_label_embeddings

    def save_embeddings_to_disk(self, embeddings, filepath):
        if not embeddings:
            raise ValueError("Embeddings dictionary is empty.")
        torch.save(embeddings, filepath)
    
    def load_embeddings_from_disk(self, filepath):
        """
        Loads embeddings from disk from the specified filepath.
        
        Args:
        filepath (str): The path from which to load the embeddings.
        
        Returns:
        dict: The dictionary containing labels and their embeddings.
        """
        return torch.load(filepath)

    def read_labels_from_file(self, filepath):
        if not os.path.isfile(filepath):
            raise ValueError("File does not exist.")
        with open(filepath, 'r') as f:
            labels = f.readlines()
        labels = [label.strip() for label in labels]
        return labels

    def embed_labels_from_file(self, filename):
        # Remove extension to get base filename
        base_filename = os.path.splitext(filename)[0]

        # Construct the labels file and embeddings file paths
        labels_file_path = f"{base_filename}.labels"
        embeddings_file_path = f"{base_filename}.embeddings"

        # Read labels from file
        with open(labels_file_path, 'r') as f:
            labels = [line.strip() for line in f]

        # Compute embeddings
        embeddings = self.compute_label_embeddings(labels)

        # Save embeddings to file
        self.save_embeddings_to_disk(embeddings, embeddings_file_path)
        return embeddings 

    def compute_affinity_scores_for_directory(self, label_to_embeddings, image_dir):
        # Get all image paths
        image_filetypes = [".jpg", ".jpeg", ".png"]
        image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir)
                    if any(filename.endswith(filetype) for filetype in image_filetypes)]
        
        for image_path in image_paths:
            self.compute_affinity_scores(label_to_embeddings, image_path)

    def compute_accuracy_for_directory(self, label_to_embeddings, image_dir):
        image_filetypes = [".jpg", ".jpeg", ".png"]
        image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir)
                    if any(filename.endswith(filetype) for filetype in image_filetypes)]

        # read ground truth image_paths and labels
        import csv
        with open("./image_labels.csv", mode='r') as f:
            reader = csv.reader(f)
            next(reader)
            ground_truth = {rows[0]:rows[1] for rows in reader}

        #  compute the top 1 and top 3 accuracy
        top1_correct = 0
        top3_correct = 0
        for image_path in image_paths:
            result = self.compute_affinity_scores(label_to_embeddings, image_path, verbose=False)
            labels = result['labels']
            top1 = labels[0]
            top3 = labels[:3]
            if ground_truth[image_path] in top3:
                top3_correct += 1
                if ground_truth[image_path] == top1:
                    top1_correct += 1
                else:
                    print("Wrong top1 prediction:", image_path)
                    print("Ground truth:", ground_truth[image_path])
                    print("Top 1:", top1)
                    print("\n")
            else:
                print("Wrong prediction:", image_path)
                print("Ground truth:", ground_truth[image_path])
                print("Top 3:")
                for i in range(3):
                    print(top3[i], scores[i])
                print("\n")
        top1_acc = top1_correct / len(image_paths)
        top3_acc = top3_correct / len(image_paths)
        print("Top 1 accuracy:", top1_acc)
        print("Top 3 accuracy:", top3_acc)
        return top1_acc, top3_acc

    def compute_affinity_scores_labels_to_label(self, label_to_embeddings, target_label_to_embedding, normalize_fn='linear', scale=1):
        """
        Computes the affinity scores between a set of label embeddings and a single target label's embedding.

        Args:
        label_to_embeddings (dict): The dictionary containing labels and their embeddings.
        target_label_to_embedding (dict): The dictionary containing the target label and its embedding.
        normalize_fn (str): The normalization function to apply. Could be 'linear', 'softmax', or 'sigmoid'.
        scale (float): A scaling factor to apply to the scores when using linear normalization.

        Returns:
        dict: Labels and their corresponding affinity scores with the target label.
        """

        # Convert the dictionaries to separate lists for labels and embeddings
        unique_labels, unique_label_embeddings = zip(*label_to_embeddings.items())
        unique_label_embeddings = torch.stack(unique_label_embeddings).cpu()  # Convert the list of tensors to a single tensor

        target_label, target_embedding = next(iter(target_label_to_embedding.items()))  # Extract single label and its embedding
        target_embedding = target_embedding.cpu()

        # Ensure that target_embedding is a 2D array
        if len(target_embedding.shape) == 1:
            target_embedding = target_embedding.unsqueeze(0)

        # Compute the affinity scores between the target label and all unique labels
        raw_scores = target_embedding @ unique_label_embeddings.T
        if normalize_fn == 'linear':
            affinity_scores = (raw_scores * scale).numpy().flatten()
        elif normalize_fn == 'softmax':
            affinity_scores = torch.softmax(raw_scores, dim=-1).numpy().flatten()
        elif normalize_fn == 'sigmoid':
            min_score = torch.min(raw_scores)
            max_score = torch.max(raw_scores)
            affinity_scores = torch.sigmoid((raw_scores - min_score) / (max_score - min_score)).numpy().flatten()

        # Return the labels and scores as a dictionary
        return {'labels': unique_labels, 'scores': affinity_scores}

    def compute_affinity_scores(self, label_to_embeddings, image_path, normalize_fn='linear', scale=1, verbose=True):
        """
        Computes the affinity scores between an image and labels using the provided model.
        
        Args:
        label_to_embeddings (dict): The dictionary containing labels and their embeddings.
        image_path (str): The filepath of the image to process.
        normalize_fn (str): The normalization function to apply. Could be 'linear', 'softmax', or 'sigmoid'.
        scale (float): A scaling factor to apply to the scores when using linear normalization.
        verbose (bool): Whether to print verbose output.
        """

        # Convert the dictionary to separate lists for labels and embeddings
        unique_labels, unique_label_embeddings = zip(*label_to_embeddings.items())
        unique_label_embeddings = torch.stack(unique_label_embeddings).cpu()  # Convert the list of tensors to a single tensor
        
        # Use your model to get the embeddings for the selected image
        inputs = {
            ModalityType.VISION: data.load_and_transform_vision_data([image_path], self.device),  # Using a single image
        }

        with torch.no_grad():
            vision_embeddings = self.model(inputs)[ModalityType.VISION].cpu()

        # Ensure that vision_embeddings is a 2D array
        if len(vision_embeddings.shape) == 1:
            vision_embeddings = vision_embeddings.unsqueeze(0)

        # Compute the affinity scores between the image and all unique labels
        raw_scores = vision_embeddings @ unique_label_embeddings.T
        if normalize_fn == 'linear':
            affinity_scores_for_image = (raw_scores * scale).numpy().flatten()
        elif normalize_fn == 'softmax':
            affinity_scores_for_image = torch.softmax(raw_scores, dim=-1).numpy().flatten()
        elif normalize_fn == 'sigmoid':
            min_score = torch.min(raw_scores)
            max_score = torch.max(raw_scores)
            affinity_scores_for_image = torch.sigmoid((raw_scores - min_score) / (max_score - min_score)).numpy().flatten()

        # Sort the labels based on the affinity scores
        sorted_indices = np.argsort(affinity_scores_for_image)[::-1]  # get the indices that would sort the array in descending order
        sorted_labels = np.array(unique_labels)[sorted_indices]  # sort the labels based on the indices
        sorted_scores = affinity_scores_for_image[sorted_indices]

        if verbose:
            # Print the top 100 labels
            print(f"Top 100 labels for this image: {image_path}")
            for label, confidence in zip(sorted_labels[:100], sorted_scores[:100]):
                print(f'{label}: {confidence}')
            print("\n")

        top_labels = sorted_labels[:100]
        top_scores = sorted_scores[:100]

        # Return the labels and scores as a dictionary
        return {'labels': top_labels, 'scores': top_scores}


    def compute_affinity_scores_batched(self, label_to_embeddings, image_paths, normalize_fn='linear', scale=1, verbose=True, batch_size=32):
        """
        Computes the affinity scores between batches of images and labels using the provided model.
        
        Args:
        label_to_embeddings (dict): The dictionary containing labels and their embeddings.
        image_paths (list of str): The filepaths of the images to process in batches.
        normalize_fn (str): The normalization function to apply. Could be 'linear', 'softmax', or 'sigmoid'.
        scale (float): A scaling factor to apply to the scores when using linear normalization.
        verbose (bool): Whether to print verbose output.
        batch_size (int): The number of images to process in each batch.
        """

        # Convert the dictionary to separate lists for labels and embeddings
        unique_labels, unique_label_embeddings = zip(*label_to_embeddings.items())
        unique_label_embeddings = torch.stack(unique_label_embeddings).cpu()  # Convert the list of tensors to a single tensor

        all_results = []

        num_batches = len(image_paths) // batch_size + (1 if len(image_paths) % batch_size != 0 else 0)

        # Integrate tqdm into the loop
        for i in tqdm(range(num_batches), desc="Processing image batches", ncols=100, leave=True):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_image_paths = image_paths[start_idx:end_idx]

            # Use model to get the embeddings for the selected batch of images
            inputs = {
                ModalityType.VISION: data.load_and_transform_vision_data(batch_image_paths, self.device),
            }

            with torch.no_grad():
                vision_embeddings = self.model(inputs)[ModalityType.VISION].cpu()

            # Compute the affinity scores between the images in the batch and all unique labels
            raw_scores = vision_embeddings @ unique_label_embeddings.T

            # For each image in the batch...
            for idx, raw_scores_for_image in enumerate(raw_scores):
                # Normalize scores based on the given method
                if normalize_fn == 'linear':
                    affinity_scores_for_image = (raw_scores_for_image * scale).numpy()
                elif normalize_fn == 'softmax':
                    affinity_scores_for_image = torch.softmax(raw_scores_for_image, dim=-1).numpy()
                elif normalize_fn == 'sigmoid':
                    min_score = torch.min(raw_scores_for_image)
                    max_score = torch.max(raw_scores_for_image)
                    affinity_scores_for_image = torch.sigmoid((raw_scores_for_image - min_score) / (max_score - min_score)).numpy()

                # Sort the labels based on the affinity scores
                sorted_indices = np.argsort(affinity_scores_for_image)[::-1]  # sort in descending order
                sorted_labels = np.array(unique_labels)[sorted_indices]
                sorted_scores = affinity_scores_for_image[sorted_indices]

                # Extract the top 100 labels and scores
                top_labels = sorted_labels[:100]
                top_scores = sorted_scores[:100]

                # Add to results
                all_results.append({'labels': top_labels, 'scores': top_scores})

                if verbose:
                    print(f"Top 100 labels for image: {batch_image_paths[idx]}")
                    for label, confidence in zip(top_labels, top_scores):
                        print(f'{label}: {confidence}')
                    print("\n")

        return all_results



    def load_labels_from_file(self, file_path):
        """
        Load labels from a file. Each line in the file should contain one label.

        Args:
            file_path (str): The path to the file.

        Returns:
            list: The list of labels.
        """
        if not os.path.isfile(file_path):
            print(f"Warning: File '{file_path}' not found.")
            return []
        with open(file_path, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        return labels


    def manual_learning_loop(self, image_file: str, use_case_prompt: str, normalize_fn: str):
        self.reasoner.set_use_case_prompt(use_case_prompt)
        base_labels = self.reasoner.reason({}, "blank_slate")
        embeddings = self.compute_label_embeddings(base_labels)
        label_affinity_scores = self.compute_affinity_scores(embeddings, image_file, normalize_fn=normalize_fn)

        all_labels = base_labels.copy()
        while True:
            # Print the top 10 affinity scores.
            sorted_indices = np.argsort(label_affinity_scores['scores'])[::-1]  # get the indices that would sort the array in descending order
            sorted_labels = label_affinity_scores['labels'][sorted_indices]  # sort the labels based on the indices
            sorted_scores = label_affinity_scores['scores'][sorted_indices]
            
            print("Top 10 Affinity Scores:")
            for label, score in zip(sorted_labels[:10], sorted_scores[:10]):
                print(f"{label}: {score}")

            new_labels_input = input("\nPlease enter new labels (comma-separated), 'load <<label filenames>>', 'use <<image filename>>', 'clear' to reset labels, 'clearall' to reset all labels to 'danger', or 'quit' to finish: ")

            if new_labels_input.lower() == 'quit':
                break
            elif new_labels_input.lower() == 'clear':
                all_labels = base_labels
            elif new_labels_input.lower() == 'clearall':
                all_labels = ['zebra']
            elif new_labels_input.lower().startswith('load'):
                filename = new_labels_input.split(' ')[1]
                new_labels = self.load_labels_from_file(filename)
                all_labels += new_labels
            elif new_labels_input.lower().startswith('use'):
                new_image_file = new_labels_input.split(' ')[1]
                if not os.path.isfile(new_image_file):
                    print(f"Warning: File '{new_image_file}' not found.")
                    continue
                image_file = new_image_file
            else:
                new_labels = [label.strip() for label in new_labels_input.split(",")]
                all_labels += new_labels

            embeddings = self.compute_label_embeddings(all_labels)
            label_affinity_scores = self.compute_affinity_scores(embeddings, image_file, normalize_fn=normalize_fn)

        self.write_labels_to_file(all_labels, "final.labels")

        # Get only the base name (i.e., with extension) from the image_file
        base_name = os.path.basename(image_file)

        # Get the filename without extension
        filename = os.path.splitext(base_name)[0]

        # Form the affinities_filename using the filename without extension
        affinities_filename = f"{filename}_affinities.json"

        # Write the sorted label-score pairs to a file
        with open(affinities_filename, 'w') as f:
            json.dump({label: float(score) for label, score in zip(sorted_labels, sorted_scores)}, f)

        return affinities_filename

