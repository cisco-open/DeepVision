
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

from locust import HttpUser, between, task
import json
import os


class EthosightUser(HttpUser):
    wait_time = between(1, 2)  # Simulated users will wait 1-2 seconds between tasks

    embeddings = None

    def on_start(self):
        """On start, get the label embeddings to use in other requests."""
        self.get_label_embeddings()

    def get_label_embeddings(self):
        data = {
            "labels": ["Electronics", "Unauthorized Vehicle", "Unattended Item", "Clothing", "Person", "Alcohol", "Tools"],  # Example labels
            "batch_size": 32
        }
        with self.client.post("/compute_label_embeddings", json=data, name="/compute_label_embeddings",
                              catch_response=True) as response:
            if response.ok:
                result = response.json()
                self.embeddings = json.loads(result["embeddings"])

    @task
    def compute_label_embeddings(self):
        data = {
            "labels": ["label1", "label2"],
            "batch_size": 32
        }
        self.client.post("/compute_label_embeddings", json=data, name="/compute_label_embeddings")

    @task
    def compute_affinity_scores(self):

        image_path = os.path.join(os.path.dirname(__file__), 'img', 'bus.jpg')
        data = {
            "label_to_embeddings": self.embeddings,
            "image_path": image_path,
            "normalize_fn": "linear",
            "scale": 1,
            "verbose": True,
            "batch_size": 32
        }
        files = {
            'data': ("data.json", json.dumps(data), 'application/json'),
            'image': (os.path.basename(image_path), open(image_path, 'rb'), 'image/jpeg')
        }
        self.client.post("/compute_affinity_scores", files=files, name="/compute_affinity_scores")

    @task
    def health_check(self):
        self.client.get("/health", name="/health")
