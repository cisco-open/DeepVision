
#!/usr/bin/env python3
import os
import requests
import tarfile
import zipfile
import torch
import torchvision
from torchvision.datasets import ImageFolder
from Ethosight import Ethosight

class Caltech101:
    def __init__(self, root):
        self.root = root
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor()
        ])
        self.files = [
            {
                "url": "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1",
                "path": os.path.join(self.root, "caltech-101.zip")
            },
            {
                "path": os.path.join(self.root, "caltech-101", "101_ObjectCategories.tar.gz")
            }
        ]

    def download_and_extract(self):
        for file in self.files:
            extraction_dir = os.path.join(self.root, os.path.splitext(os.path.basename(file["path"]))[0])
            if not os.path.exists(file["path"]):
                if "url" in file:  # If URL is provided, download the file
                    print(f"{file['path']} does not exist, downloading...")
                    r = requests.get(file["url"])
                    with open(file["path"], 'wb') as f:
                        f.write(r.content)
            
            # Only extract if file exists and extraction directory does not exist
            if os.path.exists(file["path"]) and not os.path.exists(extraction_dir):
                print(f"{file['path']} exists, extracting...")
                if file["path"].endswith(".zip"):
                    with zipfile.ZipFile(file["path"], 'r') as zip_ref:
                        zip_ref.extractall(self.root)
                elif file["path"].endswith(".tar.gz"):
                    with tarfile.open(file["path"], 'r:gz') as tar_ref:
                        tar_ref.extractall(path=os.path.dirname(file["path"]))


    def load_dataset(self):
        return ImageFolder(os.path.join(self.root, "caltech-101", "101_ObjectCategories"), transform=self.transform)
    
    def get_labels(self):
        dataset = self.load_dataset()
        labels = [dataset.classes[label] for _, label in dataset.samples]
        return labels

    def process_dataset(self, ethosight, labels_file_path, embeddings_file_path):
        self.download_and_extract()
        labels = self.get_labels()
        ethosight.write_labels_to_file(labels, labels_file_path)
        embeddings = ethosight.compute_label_embeddings(labels)
        ethosight.save_embeddings_to_disk(embeddings, embeddings_file_path)
