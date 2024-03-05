from Ethosight.tests.RESTClientHelper import RESTClientHelper
import pytest
import os


client = RESTClientHelper()


@pytest.fixture(scope="session")
def compute_label_embeddings():
    labels = ["Electronics", "Unauthorized Vehicle", "Unattended Item", "Clothing", "Person", "Alcohol", "Tools"]
    batch_size = 2
    image_path = os.path.join(os.path.dirname(__file__), 'img', 'bus.jpg')
    status_code, label_to_embeddings = client.compute_label_embeddings(labels, batch_size)
    resource = {
        'embeddings': (status_code, label_to_embeddings),
        'image_path': image_path,
        'image_paths': [image_path]
    }

    return resource


def test_compute_label_embeddings_endpoint(compute_label_embeddings):
    status_code, data = compute_label_embeddings['embeddings']
    assert status_code == 200
    assert "Electronics" in data and "Unauthorized Vehicle" in data and "Unattended Item" in data

def test_compute_affinity_scores(compute_label_embeddings):
    label_to_embeddings = compute_label_embeddings["embeddings"][1]
    image_path = compute_label_embeddings["image_path"]
    status_code, data = client.compute_affinity_scores(label_to_embeddings=label_to_embeddings, image_path=image_path)
    assert status_code == 200
    labels = data["labels"]
    assert "Electronics" in labels and "Unauthorized Vehicle" in labels and "Unattended Item" in labels


def test_compute_affinity_scores_batched(compute_label_embeddings):
    label_to_embeddings = compute_label_embeddings["embeddings"][1]
    image_paths = compute_label_embeddings["image_paths"]
    status_code, data = client.compute_affinity_scores_batched(label_to_embeddings=label_to_embeddings,
                                                               image_paths=image_paths)
    assert status_code == 200
    labels = data[0]["labels"]
    assert "Electronics" in labels and "Unauthorized Vehicle" in labels and "Unattended Item" in labels


