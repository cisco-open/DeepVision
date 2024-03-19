
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

import torch

def load_embeddings(filename):
    with open(filename, 'rb') as f:
        return torch.load(f)

def compare_embeddings(file1, file2):
    embeddings1 = load_embeddings(file1)
    embeddings2 = load_embeddings(file2)

    # 1. Number of labels in each file
    print(f"Number of labels in {file1}: {len(embeddings1)}")
    print(f"Number of labels in {file2}: {len(embeddings2)}")

    # 2. Labels present in one file and not in the other
    missing_in_file2 = set(embeddings1.keys()) - set(embeddings2.keys())
    missing_in_file1 = set(embeddings2.keys()) - set(embeddings1.keys())
    print(f"Labels present in {file1} but not in {file2}: {len(missing_in_file2)}")
    print(f"Labels present in {file2} but not in {file1}: {len(missing_in_file1)}")

    # 3. & 4. Compare embeddings for labels that are present in both files
    differing_labels = []
    for label in embeddings1:
        if label in embeddings2 and not torch.allclose(embeddings1[label], embeddings2[label]):
            differing_labels.append(label)
    print(f"Number of labels with differing embeddings: {len(differing_labels)}")
    if differing_labels:
        print("Labels with differing embeddings:", differing_labels)

if __name__ == "__main__":
    compare_embeddings("general.embeddings.correct", "general.embeddings")

