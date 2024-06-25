
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

import numpy as np
import torch
import open_clip
from tqdm import tqdm
from open_clip import tokenizer
import os
from PIL import Image


class EthosightOpenclip:
    def __init__(self, model_name=None, pretrained=None, ethosight_dir="./"):
        self.model_name = 'ViT-H-14' if model_name is None else model_name
        self.pretrained = 'laion2b_s32b_b79k' if pretrained is None else pretrained
        self.ethosight_dir = ethosight_dir
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(self.model_name, pretrained=self.pretrained, device=self.device)

    def print_model_info(self):
        context_length = self.model.context_length
        vocab_size = self.model.vocab_size
        print("Model name:", self.model_name)
        print("Pretrained:", self.pretrained)
        print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in self.model.parameters()]):,}")
        print("Context length:", context_length)
        print("Vocab size:", vocab_size)

    def compute_label_embeddings(self, labels, batch_size=1200):
        unique_labels = list(set(labels))

        def batched_data(data, batch_size):
            # Generator function for creating batches
            for i in range(0, len(data), batch_size):
                yield data[i:i+batch_size]

        unique_label_embeddings = []
        
        for i, batch in enumerate(tqdm(batched_data(unique_labels, batch_size), desc='Processing label batches', leave=False)):
            texts = tokenizer.tokenize(batch).cuda() #tokenize

            with torch.no_grad():
                batch_text_embeddings = self.model.encode_text(texts) #embed with text encoder
                batch_text_embeddings /= batch_text_embeddings.norm(dim=-1, keepdim=True)
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
        print(f"Embeddings saved to disk: {filepath}")

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

        openclip_embeddings_dir = os.path.join(self.ethosight_dir, "openclip-embeddings")
        if not os.path.exists(openclip_embeddings_dir):
            os.makedirs(openclip_embeddings_dir)
        embeddings_file_path = os.path.join(openclip_embeddings_dir, f"{base_filename}_{self.model_name}_{self.pretrained}.embeddings")


        # Read labels from file
        with open(labels_file_path, 'r') as f:
            labels = [line.strip() for line in f]

        # Compute embeddings
        embeddings = self.compute_label_embeddings(labels)

        # Save embeddings to file
        self.save_embeddings_to_disk(embeddings, embeddings_file_path)
        return embeddings_file_path

    def compute_affinity_scores(self, label_to_embeddings, image_path):
        """
        Computes the affinity scores between an image and labels using the provided model.
        
        Args:
        label_to_embeddings (dict): The dictionary containing labels and their embeddings.
        image_path (str): The filepath of the image to process.
        """

        # Convert the dictionary to separate lists for labels and embeddings
        unique_labels, unique_label_embeddings = zip(*label_to_embeddings.items())
        unique_label_embeddings = torch.stack(unique_label_embeddings).cpu()  # Convert the list of tensors to a single tensor
        
        # Use your model to get the embeddings for the selected image
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
    
        with torch.no_grad():
            vision_embeddings = self.model.encode_image(image).cpu()

        # Ensure that vision_embeddings is a 2D array
        if len(vision_embeddings.shape) == 1:
            vision_embeddings = vision_embeddings.unsqueeze(0)
        # Compute the affinity scores between the image and all unique labels
        affinity_scores_for_image = torch.softmax(vision_embeddings @ unique_label_embeddings.T, dim=-1).numpy().flatten()

        # Sort the labels based on the affinity scores
        sorted_indices = np.argsort(affinity_scores_for_image)[::-1]  # get the indices that would sort the array in descending order
        sorted_labels = np.array(unique_labels)[sorted_indices]  # sort the labels based on the indices
        sorted_scores = affinity_scores_for_image[sorted_indices]

        # Print the top 100 labels
        print(f"Top 100 labels for this image: {image_path}")       
        for label, confidence in zip(sorted_labels[:100], sorted_scores[:100]):
            print(f'{label}: {confidence}')
        print("\n")

        top_labels = sorted_labels[:100]
        top_scores = sorted_scores[:100]

        # Return the labels and scores as a dictionary
        return {'labels': top_labels, 'scores': top_scores}

    def compute_affinity_scores_for_directory(self, label_to_embeddings, image_dir):
        # Get all image paths
        image_filetypes = [".jpg", ".jpeg", ".png"]
        image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir)
                    if any(filename.endswith(filetype) for filetype in image_filetypes)]
        
        for image_path in image_paths:
            self.compute_affinity_scores(label_to_embeddings, image_path)


if __name__ == "__main__":
    # Create an Ethosight instance
    ethosight = EthosightOpenclip()
    ethosight.print_model_info()

    # Specify the file name where the labels are stored
    labels_file_name = "general"

    # Compute the embeddings for the labels in the file and save the embeddings to a file
    embeddings_file_path = ethosight.embed_labels_from_file(labels_file_name)

    # Load the embeddings from the file
    embeddings = ethosight.load_embeddings_from_disk(embeddings_file_path)

    # Specify the image path
    image_path = "images/shoplifting.png"

    # Compute the affinity scores
    ethosight.compute_affinity_scores(embeddings, image_path)

# openclip model name, pretrained
#  [('RN50', 'openai'),
#  ('RN50', 'yfcc15m'),
#  ('RN50', 'cc12m'),
#  ('RN50-quickgelu', 'openai'),
#  ('RN50-quickgelu', 'yfcc15m'),
#  ('RN50-quickgelu', 'cc12m'),
#  ('RN101', 'openai'),
#  ('RN101', 'yfcc15m'),
#  ('RN101-quickgelu', 'openai'),
#  ('RN101-quickgelu', 'yfcc15m'),
#  ('RN50x4', 'openai'),
#  ('RN50x16', 'openai'),
#  ('RN50x64', 'openai'),
#  ('ViT-B-32', 'openai'),
#  ('ViT-B-32', 'laion400m_e31'),
#  ('ViT-B-32', 'laion400m_e32'),
#  ('ViT-B-32', 'laion2b_e16'),
#  ('ViT-B-32', 'laion2b_s34b_b79k'),
#  ('ViT-B-32-quickgelu', 'openai'),
#  ('ViT-B-32-quickgelu', 'laion400m_e31'),
#  ('ViT-B-32-quickgelu', 'laion400m_e32'),
#  ('ViT-B-16', 'openai'),
#  ('ViT-B-16', 'laion400m_e31'),
#  ('ViT-B-16', 'laion400m_e32'),
#  ('ViT-B-16', 'laion2b_s34b_b88k'),
#  ('ViT-B-16-plus-240', 'laion400m_e31'),
#  ('ViT-B-16-plus-240', 'laion400m_e32'),
#  ('ViT-L-14', 'openai'),
#  ('ViT-L-14', 'laion400m_e31'),
#  ('ViT-L-14', 'laion400m_e32'),
#  ('ViT-L-14', 'laion2b_s32b_b82k'),
#  ('ViT-L-14-336', 'openai'),
#  ('ViT-H-14', 'laion2b_s32b_b79k'),
#  ('ViT-g-14', 'laion2b_s12b_b42k'),
#  ('ViT-g-14', 'laion2b_s34b_b88k'),
#  ('ViT-bigG-14', 'laion2b_s39b_b160k'),
#  ('roberta-ViT-B-32', 'laion2b_s12b_b32k'),
#  ('xlm-roberta-base-ViT-B-32', 'laion5b_s13b_b90k'),
#  ('xlm-roberta-large-ViT-H-14', 'frozen_laion5b_s13b_b90k'),
#  ('convnext_base', 'laion400m_s13b_b51k'),
#  ('convnext_base_w', 'laion2b_s13b_b82k'),
#  ('convnext_base_w', 'laion2b_s13b_b82k_augreg'),
#  ('convnext_base_w', 'laion_aesthetic_s13b_b82k'),
#  ('convnext_base_w_320', 'laion_aesthetic_s13b_b82k'),
#  ('convnext_base_w_320', 'laion_aesthetic_s13b_b82k_augreg'),
#  ('convnext_large_d', 'laion2b_s26b_b102k_augreg'),
#  ('convnext_large_d_320', 'laion2b_s29b_b131k_ft'),
#  ('convnext_large_d_320', 'laion2b_s29b_b131k_ft_soup'),
#  ('convnext_xxlarge', 'laion2b_s34b_b82k_augreg'),
#  ('convnext_xxlarge', 'laion2b_s34b_b82k_augreg_rewind'),
#  ('convnext_xxlarge', 'laion2b_s34b_b82k_augreg_soup'),
#  ('coca_ViT-B-32', 'laion2b_s13b_b90k'),
#  ('coca_ViT-B-32', 'mscoco_finetuned_laion2b_s13b_b90k'),
#  ('coca_ViT-L-14', 'laion2b_s13b_b90k'),
#  ('coca_ViT-L-14', 'mscoco_finetuned_laion2b_s13b_b90k')]