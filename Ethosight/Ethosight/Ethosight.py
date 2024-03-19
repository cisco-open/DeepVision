
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

# Ethosight environment variables
# EthosightBackend: "core" or "client"
# EthosightBackendURL: URL for EthosightRESTClient backend e.g. http://localhost:8000
import os 
from .EthosightCore import EthosightCore
from .EthosightRESTClient import EthosightRESTClient

class Ethosight: 
    def __init__(self, backend_type="dynamic", url=None, model=None, reasoner=None): 
        """ 
        Initialize the Ethosight instance with the specified backend. 
 
        Args: 
        - backend_type (str): "core" for EthosightCore, "client" for EthosightRESTClient, "dynamic" for selecting based on the environment variable. 
        - url (str): The URL for the EthosightRESTServer (only required for "client" backend_type). 
        - model, reasoner: Passed to the backend's initializer. 
        """ 
        if backend_type == "dynamic": 
            backend_type = os.environ.get("EthosightBackend", "core") 
 
        if backend_type == "core": 
            self.backend = EthosightCore(model, reasoner) 
        elif backend_type == "client": 
            if url is None:
                url = os.environ.get("EthosightBackendURL")
                if url is None:
                    raise ValueError("URL is required for EthosightRESTClient backend. Neither 'url' parameter nor 'EthosightBackendURL' environment variable is set.")
            self.backend = EthosightRESTClient(url, model, reasoner) 
        else: 
            raise ValueError(f"Unsupported backend_type: {backend_type}") 
 
    def __getattr__(self, name): 
        """ 
        Route attribute access to the backend. 
        """ 
        return getattr(self.backend, name) 
