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
