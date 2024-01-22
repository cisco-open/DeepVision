import requests 
import os 
from Ethosight.EthosightCore import EthosightCore
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

class EthosightRESTClient(EthosightCore): 

    def __init__(self, url, model=None, reasoner=None): 
        model="EthosightRESTClientHasNoModel"
        super().__init__(model, reasoner) 
        self.url = url 
        logging.basicConfig(level=logging.DEBUG)

    def compute_label_embeddings(self, labels, batch_size=1200):
        payload = { 
            "labels": labels, 
            "batch_size": batch_size 
        }
        
        logging.debug(f"Sending request to {self.url}/compute_label_embeddings with payload: {payload}")
        
        response = requests.post(f"{self.url}/compute_label_embeddings", json=payload)
        
        # Log the response status code and content
        logging.debug(f"Received response with status code: {response.status_code}")
        #logging.debug(f"Response content: {response.text}")
        
        response.raise_for_status()
        
        serialized_embeddings = json.loads(response.json()["embeddings"])  # Deserialize the JSON string into a dictionary
        deserialized_embeddings = self._deserialize_embeddings(serialized_embeddings)
        
        return deserialized_embeddings
    
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



#    def compute_affinity_scores(self, label_to_embeddings, image_path, normalize_fn='linear', scale=1, verbose=True): 
#        payload = { 
#            "label_to_embeddings": label_to_embeddings, 
#            "normalize_fn": normalize_fn, 
#            "scale": scale, 
#            "verbose": verbose 
#        } 
#
#        files = {"image_path": (os.path.basename(image_path), open(image_path, 'rb'))} 
#        response = requests.post(f"{self.url}/compute_affinity_scores", data=payload, files=files) 
#        response.raise_for_status() 
#        return response.json() 
#    def compute_affinity_scores(self, label_to_embeddings, image_path, normalize_fn='linear', scale=1, verbose=True): 
#        payload = { 
#            "label_to_embeddings": label_to_embeddings, 
#            "normalize_fn": normalize_fn, 
#            "scale": scale, 
#            "verbose": verbose 
#        } 
#
#        files = {"image_path": (os.path.basename(image_path), open(image_path, 'rb'))} 
#        response = requests.post(f"{self.url}/compute_affinity_scores", data=payload, files=files) 
#        response.raise_for_status() 
#
#        # Parse the response to get the results in the desired format
#        result = response.json()
#        result_dict = {
#            'labels': result['labels'],
#            'scores': result['scores']
#        }
#        
#        # If verbose, print the results
#        if verbose:
#            print("\nTop labels for the image:")
#            for label, score in zip(result_dict['labels'], result_dict['scores']):
#                print(f"{label}: {score}")
#            print("\n")
#
#        return result_dict
#    def compute_affinity_scores(self, label_to_embeddings, image_path, normalize_fn='linear', scale=1, verbose=True): 
#        # Serialize the label_to_embeddings dictionary
#        serializable_embeddings = {label: tensor_to_base64(embedding) for label, embedding in label_to_embeddings.items()}
#        serialized_embeddings = json.dumps(serializable_embeddings)
#        
#        # Construct the payload dictionary
#        payload = {
#            "data": json.dumps({
#                "label_to_embeddings": serialized_embeddings,
#                "normalize_fn": normalize_fn, 
#                "scale": scale, 
#                "verbose": verbose 
#            })
#        }
#
#        files = {"image_path": (os.path.basename(image_path), open(image_path, 'rb'))} 
#        response = requests.post(f"{self.url}/compute_affinity_scores", data=payload, files=files)
#        response.raise_for_status() 
#        
#        # Parse the response to get the results in the desired format
#        result = response.json()
#        result_dict = {
#            'labels': result['labels'],
#            'scores': result['scores']
#        }
#        
#        # If verbose, print the results
#        if verbose:
#            print("\nTop labels for the image:")
#            for label, score in zip(result_dict['labels'], result_dict['scores']):
#                print(f"{label}: {score}")
#            print("\n")
#
#        return result_dict
#
#
#    def compute_affinity_scores(self, label_to_embeddings, image_path, normalize_fn='linear', scale=1, verbose=True): 
#        # Serialize the label_to_embeddings dictionary
#        serializable_embeddings = {label: tensor_to_base64(embedding) for label, embedding in label_to_embeddings.items()}
#        serialized_embeddings = json.dumps(serializable_embeddings)
#        
#        # Construct the payload without additional serialization
#        data_content = {
#            "label_to_embeddings": serializable_embeddings,
#            "normalize_fn": normalize_fn, 
#            "scale": scale, 
#            "verbose": verbose 
#        }
#
#        files = {"image_path": (os.path.basename(image_path), open(image_path, 'rb'))} 
#
#        response = requests.post(f"{self.url}/compute_affinity_scores", data=data_content, files=files)
#
#        # Handle the response
#        if response.status_code == 422:
#            print("Validation error:", response.text)
#        response.raise_for_status() 
#
#        # Parse the response to get the results in the desired format
#        result = response.json()
#        result_dict = {
#            'labels': result['labels'],
#            'scores': base64_to_tensor(result['scores'])
#        }
#        
#        # If verbose, print the results
#        if verbose:
#            print("\nTop labels for the image:")
#            for label, score in zip(result_dict['labels'], result_dict['scores']):
#                print(f"{label}: {score}")
#            print("\n")
#
#        return result_dict
#    def compute_affinity_scores(self, label_to_embeddings, image_path, normalize_fn='linear', scale=1, verbose=True, batch_size=32): 
#        # Serialize the label_to_embeddings dictionary
#        serializable_embeddings = {label: tensor_to_base64(embedding) for label, embedding in label_to_embeddings.items()}
#        
#        # Construct the data_content dictionary
#        data_content = {
#            "label_to_embeddings": serializable_embeddings,
#            "normalize_fn": normalize_fn, 
#            "scale": scale, 
#            "verbose": verbose,
#            "batch_size": batch_size
#        }
#
#        # The data parameter expects a stringified JSON, so we'll serialize data_content
#        wrapped_data_content = {
#            "data": json.dumps(data_content)
#        }
#
#        # Attach the image as a file
#        files = {"image_path": (os.path.basename(image_path), open(image_path, 'rb'))} 
#
#        # Send the POST request
#        response = requests.post(f"{self.url}/compute_affinity_scores", data=wrapped_data_content, files=files)
#        
#        # Check for 422 response and print the server's error message
#        if response.status_code == 422:
#            print("Server validation error:", response.text)
#            return
#        
#        response.raise_for_status() 
#        
#        # Parse the response to get the results in the desired format
#        result = response.json()
#        result_dict = {
#            'labels': result['labels'],
#            'scores': result['scores']
#        }
#        
#        # If verbose, print the results
#        if verbose:
#            print("\nTop labels for the image:")
#            for label, score in zip(result_dict['labels'], result_dict['scores']):
#                print(f"{label}: {score}")
#            print("\n")
#
#        return result_dict
#
    def compute_affinity_scores_orig(self, label_to_embeddings, image_path, normalize_fn='linear', scale=1, verbose=True, batch_size=32): 
        # Serialize the label_to_embeddings dictionary
        serializable_embeddings = {label: tensor_to_base64(embedding) for label, embedding in label_to_embeddings.items()}
        
        # Construct the data_content dictionary
        data_content = {
            "label_to_embeddings": serializable_embeddings,
            "normalize_fn": normalize_fn, 
            "scale": scale, 
            "verbose": verbose,
            "batch_size": batch_size
        }

        # Attach the image as a file
        files = {"image_path": (os.path.basename(image_path), open(image_path, 'rb'))} 

        # Send the POST request with data_content directly as the data parameter
        response = requests.post(f"{self.url}/compute_affinity_scores", data=data_content, files=files)
        
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
        
        # If verbose, print the results
        if verbose and result_dict:
            print("\nTop labels for the image:")
            for label, score in zip(result_dict['labels'], result_dict['scores']):
                print(f"{label}: {score}")
            print("\n")

        return result_dict

# still with the 422
    def compute_affinity_scores_mock1(self, label_to_embeddings, image_path, normalize_fn='linear', scale=1, verbose=True, batch_size=32):
        # Mock serialization of the label_to_embeddings dictionary
        serializable_embeddings = {label: str(embedding) for label, embedding in label_to_embeddings.items()}
        
        # Construct the data_content dictionary
        data_content = {
            "label_to_embeddings": serializable_embeddings,
            "normalize_fn": normalize_fn, 
            "scale": scale, 
            "verbose": verbose,
            "batch_size": batch_size
        }

        # Attach the image as a file
        files = {"image_path": (os.path.basename(image_path), open(image_path, 'rb'))} 

        # Send the POST request with data_content directly as the data parameter
        response = requests.post(f"{self.url}/compute_affinity_scores", data=data_content, files=files)
        
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
        
        # If verbose, print the results
        if verbose and result_dict:
            print("\nTop labels for the image:")
            for label, score in zip(result_dict['labels'], result_dict['scores']):
                print(f"{label}: {score}")
            print("\n")

        return result_dict

# this works
    def compute_affinity_scores_mock2(self, label_to_embeddings, image_path, normalize_fn='linear', scale=1, verbose=True, batch_size=32):
        serializable_embeddings = {label: tensor_to_base64(embedding) for label, embedding in label_to_embeddings.items()}

        # Mock data_content for testing
        data_content = {
            #"label_to_embeddings": {"mock_label_1": "mock_embedding_1", "mock_label_2": "mock_embedding_2"},
            "label_to_embeddings": serializable_embeddings,
            "normalize_fn": 'linear', 
            "scale": 1, 
            "verbose": True,
            "batch_size": 32
        }

        # Send the POST request with mock data_content
        response = requests.post(f"{self.url}/compute_affinity_scores", json=data_content)
        
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

        return result_dict

    def compute_affinity_scores_mock3(self, label_to_embeddings, image_path, normalize_fn='linear', scale=1, verbose=True, batch_size=32):
        # Serialize the label_to_embeddings dictionary
        serializable_embeddings = {label: tensor_to_base64(embedding) for label, embedding in label_to_embeddings.items()}
        
        # Construct the data_content dictionary
        data_content = {
            "label_to_embeddings": serializable_embeddings,
            "normalize_fn": normalize_fn, 
            "scale": scale, 
            "verbose": verbose,
            "batch_size": batch_size
        }

        # Attach the image as a file
        files = {"image_path": (os.path.basename(image_path), open(image_path, 'rb'))} 

        # Send the POST request with data_content directly as the data parameter
        response = requests.post(f"{self.url}/compute_affinity_scores", json=data_content, files=files)
        
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
        
        # If verbose, print the results
        if verbose and result_dict:
            print("\nTop labels for the image:")
            for label, score in zip(result_dict['labels'], result_dict['scores']):
                print(f"{label}: {score}")
            print("\n")

        return result_dict

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
        response = requests.post(f"{self.url}/compute_affinity_scores", files=files)
        
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

        return result_dict

# produces 422
    def compute_affinity_scores_batched_orig(self, label_to_embeddings, image_paths, normalize_fn='linear', scale=1, verbose=True, batch_size=32): 
        payload = { 
            "label_to_embeddings": label_to_embeddings, 
            "normalize_fn": normalize_fn, 
            "scale": scale, 
            "verbose": verbose, 
            "batch_size": batch_size 
        } 

        files = [("image_paths", (os.path.basename(path), open(path, 'rb'))) for path in image_paths] 
        response = requests.post(f"{self.url}/compute_affinity_scores_batched", data=payload, files=files) 
        response.raise_for_status() 
        return response.json() 

    def compute_affinity_scores_batched_bug2(self, label_to_embeddings, image_paths, normalize_fn='linear', scale=1, verbose=True, batch_size=32): 
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

        response = requests.post(f"{self.url}/compute_affinity_scores_batched", files=files) 
        response.raise_for_status() 

        # Don't forget to close the files after the request
        for _, (_, file, _) in files[1:]:
            file.close()

        return response.json()


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

        response = requests.post(f"{self.url}/compute_affinity_scores_batched", files=files) 
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

        return processed_results
