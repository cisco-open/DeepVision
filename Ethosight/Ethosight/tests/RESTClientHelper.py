
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

import requests
import os
from Ethosight.EthosightCore import EthosightCore
from Ethosight.EthosightRESTServer import EthosightRESTServer
from fastapi.testclient import TestClient
import logging
import base64
import numpy as np
import torch
import json  # Ensure json is imported
from typing import Dict, Any

def tensor_to_base64(tensor):
    # Convert the tensor to a numpy array
    numpy_array = tensor.numpy()
    numpy_bytes = numpy_array.tobytes()

    # Encode the byte array to base64 string
    return base64.b64encode(numpy_bytes).decode('utf-8')


class RESTClientHelper(EthosightCore):

    def __init__(self, model=None, reasoner=''):
        model = "EthosightRESTClientHasNoModel"
        super().__init__(model, reasoner)
        self.url = ''
        server = EthosightRESTServer(mode='blocking', host='127.0.0.1', port=8000, consul_url='localhost', consul_port=8500,
                                     gpu=0, reasoner='')
        self.client = TestClient(server.app)
        logging.basicConfig(level=logging.DEBUG)

    def compute_label_embeddings(self, labels, batch_size=1200):
        payload = {
            "labels": labels,
            "batch_size": batch_size
        }

        logging.debug(f"Sending request to /compute_label_embeddings with payload: {payload}")

        response = self.client.post("/compute_label_embeddings", json=payload)

        # Log the response status code and content
        logging.debug(f"Received response with status code: {response.status_code}")

        serialized_embeddings = json.loads(response.json()["embeddings"])  # Deserialize the JSON string into a dictionary

        deserialized_embeddings = self._deserialize_embeddings(serialized_embeddings)

        return response.status_code, deserialized_embeddings

    def _deserialize_embeddings(self, serialized_embeddings):
        # Convert the serialized embeddings back to tensors
        deserialized_embeddings = {}
        for label, base64_str in serialized_embeddings.items():
            deserialized_embeddings[label] = self._base64_to_tensor(base64_str)
        return deserialized_embeddings

    def _base64_to_tensor(self, base64_str):
        # Decode the base64 string to a numpy byte array
        np_bytes = base64.b64decode(base64_str)
        # Convert the numpy byte array to a numpy array
        np_array = np.frombuffer(np_bytes, dtype=np.float32)
        # Convert the numpy array to a PyTorch tensor
        tensor = torch.from_numpy(np_array)
        return tensor


    def compute_affinity_scores(self, label_to_embeddings, image_path, normalize_fn='linear', scale=1, verbose=True):
        # Serialize the label_to_embeddings dictionary as before
        serializable_embeddings = {label: tensor_to_base64(embedding) for label, embedding in label_to_embeddings.items()}

        # Construct the data_content dictionary
        serialized_data = json.dumps({
            "label_to_embeddings": serializable_embeddings,
            "normalize_fn": normalize_fn,
            "scale": scale,
            "verbose": verbose,
        })

        # Prepare the files for multipart upload
        files = {
            "data": ("data.json", serialized_data, "application/json"),
            "image": (os.path.basename(image_path), open(image_path, 'rb'), "image/jpeg")
        }

        # Send the POST request
        # response = requests.post(f"{self.url}/compute_affinity_scores", files=files)
        response = self.client.post("/compute_affinity_scores", files=files)

        # Check for 422 response and print the server's error message
        if response.status_code == 422:
            print("Server validation error:", response.text)
            return None

        response.raise_for_status()

        # Parse the response to get the results in the desired format
        result = response.json()
        result_dict = {
            'labels': result['labels'],
            'scores': result['scores']
        }

        # If verbose, print the mock results
        if verbose:
            print("\nTop labels for the image:")
            for label, score in zip(result_dict['labels'], result_dict['scores']):
                print(f"{label}: {score}")
            print("\n")

        return response.status_code, result_dict


    def compute_affinity_scores_batched(self, label_to_embeddings, image_paths, normalize_fn='linear', scale=1, verbose=True, batch_size=32):
        # Serialize the label_to_embeddings dictionary as before
        serializable_embeddings = {label: tensor_to_base64(embedding) for label, embedding in label_to_embeddings.items()}
        # Serialize the data into a JSON string
        payload = json.dumps({
            "label_to_embeddings": serializable_embeddings,
            "normalize_fn": normalize_fn,
            "scale": scale,
            "verbose": verbose,
            "batch_size": batch_size
        })

        # Create a list of files to send (including the serialized data and the images)
        files = [
            ("data", ("data.json", payload, "application/json")),
            *[(f"image_paths", (os.path.basename(path), open(path, 'rb'))) for path in image_paths]
        ]

        #response = requests.post(f"{self.url}/compute_affinity_scores_batched", files=files)
        response = self.client.post("/compute_affinity_scores_batched", files=files)
        response.raise_for_status()

        # Don't forget to close the files after the request
        for _, (_, file) in files[1:]:
            file.close()

        response_data = response.json()

        processed_results = []
        for entry in response_data:
            processed_entry = {
                'labels': np.array(entry['labels']),
                'scores': np.array(entry['scores'], dtype=float)
            }
            processed_results.append(processed_entry)

        return response.status_code, processed_results
