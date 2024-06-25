
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

#!/usr/bin/env python3
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, status, Form
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import List, Dict
import threading
from Ethosight.EthosightCore import EthosightCore
import os
from .shared_models import LabelEmbeddingInput
import logging
import functools
import uuid
import base64
import io
import json
import torch
import sys
import numpy as np
import consul

logging.basicConfig(level=logging.DEBUG)

def tensor_to_base64(tensor):
    # Convert the tensor to a numpy array
    numpy_array = tensor.numpy()
    numpy_bytes = numpy_array.tobytes()

    # Encode the byte array to base64 string
    return base64.b64encode(numpy_bytes).decode('utf-8')

class ComputeAffinityScoresInput(BaseModel): 
    label_to_embeddings: dict 
    normalize_fn: str = 'linear' 
    scale: float = 1 
    verbose: bool = True 
    batch_size: int = 32 

class ComputeAffinityScoresBatchedInput(BaseModel): 
    label_to_embeddings: Dict[str, List[float]]
    image_paths: List[str]
    normalize_fn: str = 'linear'
    scale: float = 1
    verbose: bool = True
    batch_size: int = 32   

class EthosightRESTServer: 
    def __init__(self, mode="blocking", host="0.0.0.0", port=8000, consul_url="localhost", consul_port=8500, gpu=0, reasoner=""):
        self.core = EthosightCore(reasoner=reasoner, gpu=gpu)
        self.app = FastAPI()
        self.lock = threading.Lock()
        self.mode = mode
        self.host = host
        self.port = port
        self.consul_url = consul_url
        self.consul_port = consul_port
        self._initialize_routes()

        # Register with Consul
        c = consul.Consul(host=consul_url, port=consul_port)

        check = {
            "http": f"http://{host}:{port}/health",
            "interval": "10s"
        }

        c.agent.service.register("EthosightRESTServer", service_id=f"ethosight_{host}_{port}", address=host, port=port, check=check)
        self.app.add_event_handler("shutdown", self.deregister_from_consul)

    def deregister_from_consul(self):
        c = consul.Consul(host=self.consul_url, port=self.consul_port)
        c.agent.service.deregister(service_id=f"ethosight_{self.host}_{self.port}")


    def _base64_to_tensor(self, base64_str):
        # Decode the base64 string to a numpy byte array
        np_bytes = base64.b64decode(base64_str)
        # Convert the numpy byte array to a numpy array
        np_array = np.frombuffer(np_bytes, dtype=np.float32)
        # Convert the numpy array to a PyTorch tensor
        tensor = torch.from_numpy(np_array)
        return tensor

    def process_request(self, func): 
        @functools.wraps(func)
        def wrapper(*args, **kwargs): 
            if self.lock.locked(): 
                if self.mode == "non-blocking": 
                    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Server is currently processing another request.") 
                elif self.mode == "blocking": 
                    with self.lock: 
                        return func(*args, **kwargs) 
            else: 
                with self.lock: 
                    return func(*args, **kwargs) 
        return wrapper 

    def _initialize_routes(self): 
        self._initialize_compute_label_embeddings_route() 
        self._initialize_compute_affinity_scores_route() 
        self._initialize_compute_affinity_scores_batched_route()
        self._initialize_health_check_route()

    def _initialize_health_check_route(self):
        @self.app.get("/health")
        def health_check():
            return {"status": "healthy"}

    def _initialize_compute_label_embeddings_route(self): 
        @self.app.post("/compute_label_embeddings") 
        @self.process_request 
        def compute_label_embeddings_endpoint(data: LabelEmbeddingInput):
            import sys  # Make sure sys is imported
            
            def test_tensor_to_base64(tensor):
                # Convert the tensor to a numpy array
                numpy_array = tensor.numpy()
                numpy_bytes = numpy_array.tobytes()
                numpy_size = sys.getsizeof(numpy_bytes)

                # Encode the byte array to base64 string
                encoded = base64.b64encode(numpy_bytes)
                encoded_size = sys.getsizeof(encoded)

                return numpy_size, encoded_size

            # Rest of your method
            logging.debug(f"Received request with data: {data}")
            embeddings_dict = self.core.compute_label_embeddings(data.labels, data.batch_size)

            # Test with a few sample tensors
            for label, tensor in list(embeddings_dict.items())[:5]:
                buffer_size, encoded_size = test_tensor_to_base64(tensor)
                logging.debug(f"Label: {label}, Buffer size: {buffer_size} bytes, Base64 size: {encoded_size} bytes")

            # Continue with your serialization process
            serializable_embeddings = {label: tensor_to_base64(embedding) for label, embedding in embeddings_dict.items()}
            serialized_result = json.dumps(serializable_embeddings)
            logging.debug("Computed embeddings")
            return {"embeddings": serialized_result}

    def _initialize_compute_affinity_scores_route(self):
        @self.app.post("/compute_affinity_scores")
        @self.process_request
        def compute_affinity_scores_endpoint(data: UploadFile = File(...), image: UploadFile = File(...)):
            #logging.debug(f"Incoming image filename: {image.filename}")

            # Extract and parse JSON data
            data_content_str = data.file.read().decode('utf-8')
            data_content = json.loads(data_content_str)

            #logging.debug(f"Incoming data: {data_content}")

            # Read the image content 
            image_temp_path = f"/tmp/temp_{uuid.uuid4()}_{image.filename}" 
            with open(image_temp_path, "wb") as buffer: 
                buffer.write(image.file.read())

            label_to_embeddings = {label: self._base64_to_tensor(embedding_b64) 
                                for label, embedding_b64 in data_content['label_to_embeddings'].items()}

            # Compute the affinity scores
            try:
                affinity_scores_result = self.core.compute_affinity_scores(
                    label_to_embeddings, 
                    image_temp_path, 
                    data_content['normalize_fn'], 
                    data_content['scale'], 
                    data_content['verbose']
                )
                
                # Convert numpy arrays to lists if they are numpy arrays
                if isinstance(affinity_scores_result['labels'], np.ndarray):
                    affinity_scores_result['labels'] = affinity_scores_result['labels'].tolist()
                if isinstance(affinity_scores_result['scores'], np.ndarray):
                    affinity_scores_result['scores'] = affinity_scores_result['scores'].tolist()

            except Exception as e:
                os.remove(image_temp_path)
                logging.error(f"Error computing affinity scores: {e}")
                raise e  # Raise a custom error or message to the client
            
            # Clean up the temp image file
            os.remove(image_temp_path) 

            logging.debug("Computed affinity scores")
            return affinity_scores_result

    def _initialize_compute_affinity_scores_batched_route(self):
        @self.app.post("/compute_affinity_scores_batched", response_model=List[Dict[str, List[str]]])
        @self.process_request 
        def compute_affinity_scores_batched_endpoint(data: UploadFile = File(...), image_paths: List[UploadFile] = File(...)):
            logging.debug(f"Received batched request with data: {data.filename}")

            # Step 2: Extract and parse JSON data
            data_content_str = data.file.read().decode('utf-8')
            data_content = json.loads(data_content_str)

            # Step 3: Decode the base64 encoded tensors
            decoded_embeddings = {label: self._base64_to_tensor(embedding_base64) for label, embedding_base64 in data_content['label_to_embeddings'].items()}

            # Step 4: Save the images to temporary paths
            saved_image_paths = [] 
            for image_file in image_paths: 
                image_temp_path = f"temp_{uuid.uuid4()}_{image_file.filename}" 
                with open(image_temp_path, "wb") as buffer: 
                    buffer.write(image_file.file.read()) 
                saved_image_paths.append(image_temp_path) 

            try:
                # Step 5: Process the data with the core functionality
                result = self.core.compute_affinity_scores_batched(
                    label_to_embeddings=decoded_embeddings,
                    image_paths=saved_image_paths,
                    normalize_fn=data_content["normalize_fn"],
                    scale=data_content["scale"],
                    verbose=data_content["verbose"],
                    batch_size=data_content["batch_size"]
                )
                
                # Convert the result to a serializable format
                result_serialized = []
                for entry in result:
                    serialized_entry = {
                        'labels': entry['labels'].tolist()  # Assuming labels are always numpy arrays
                    }
                    if isinstance(entry['scores'], np.ndarray):
                        serialized_entry['scores'] = entry['scores'].tolist()
                    else:
                        serialized_entry['scores'] = entry['scores']
                    result_serialized.append(serialized_entry)

            except Exception as e:
                logging.error(f"Error occurred during computation or serialization: {e}")
                raise e  # Raising the exception so it can be caught and handled upstream, if necessary.

            finally:
                # Step 6: Clean up the temporary image files
                for path in saved_image_paths: 
                    os.remove(path)

            logging.debug(f"Serialized batched affinity scores: {result_serialized}")
            return result_serialized
