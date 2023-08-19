from models import imagebind_model
from models.imagebind_model import ModalityType
import data
import torch
from tqdm import tqdm

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def batched_data(data, batch_size):
    # Generator function for creating batches
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]


def load_model():
    # Load the model
    model = imagebind_model.imagebind_huge(pretrained=True, imagebind_dir="../")
    model.eval()
    model.to(device)
    return model


def get_text_embeddings(classnames, templates, model):
    """
    Get text embeddings for a list of classnames using a list of templates
    Output: torch.Tensor of shape (embedding_dim, num_classes) (imagenet: torch.Size([512, 1000]))
    """
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            text_inputs = {
                ModalityType.TEXT: data.load_and_transform_text(texts, device),
            }
            text_embeddings = model(text_inputs)
            text_embeddings = text_embeddings[ModalityType.TEXT]
            text_embedding = text_embeddings.mean(dim=0)
            text_embedding /= text_embedding.norm()
            zeroshot_weights.append(text_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def get_batch_image_targets(dataset, batch_size):
    image_paths = dataset.fnames
    # Generator function for creating batches
    for i in range(0, len(image_paths), batch_size):
        if i + batch_size > len(image_paths):
            yield (image_paths[i:], [int(dataset.fnames[k].parent.name) for k in range(i, len(image_paths))])
        else:
            yield (image_paths[i:i+batch_size], [int(dataset.fnames[k].parent.name) for k in range(i, i+batch_size)])


def get_acc_image_embeddings(model, dataset, text_embeddings, image_batch_size=32):
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        vision_embeddings = None
        for i, (image_paths, target) in enumerate(tqdm(get_batch_image_targets(dataset, batch_size=image_batch_size))):
            vision_inputs = {
                ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
            }
            with torch.no_grad():
                if vision_embeddings is not None:
                    del vision_embeddings
                    torch.cuda.empty_cache()
                vision_embeddings = model(vision_inputs)
            vision_embeddings = vision_embeddings[ModalityType.VISION]

            logits = 100. * vision_embeddings @ text_embeddings

            # measure accuracy
            acc1, acc5 = accuracy(logits, torch.tensor(target).to(device), topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += len(image_paths)

    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100
    return top1, top5


def compute_affinity_scores_ethosight_batched(model, dataset, label_to_embeddings, image_batch_size=32):
    import time
    unique_labels, unique_label_embeddings = zip(*label_to_embeddings.items())
    unique_label_embeddings = torch.stack(unique_label_embeddings).to(device)  # Convert the list of tensors to a single tensor

    with torch.no_grad():
        vision_embeddings = None
        for i, (image_paths, target) in enumerate(tqdm(get_batch_image_targets(dataset, batch_size=image_batch_size))):
            st = time.time()
            vision_inputs = {
                ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
            }
            load_data_time = time.time()
            with torch.no_grad():
                if vision_embeddings is not None:
                    del vision_embeddings
                    torch.cuda.empty_cache()
                vision_embeddings = model(vision_inputs)
            vision_embeddings = vision_embeddings[ModalityType.VISION]

            # Compute the affinity scores between the image and all unique labels
            raw_scores = vision_embeddings @ unique_label_embeddings.T
            end_time = time.time()
            time_token = end_time - st
            print(f"*****Batch {i}*****")
            print(f"Time taken to load data and compute logits: {time_token} seconds")
            print(f"Time taken to compute logits only: {end_time - load_data_time} seconds")


if __name__ == "__main__":
    import time
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-batch-size", type=int, default=256, help="Image batch size")

    args = parser.parse_args()

    st = time.time()
    # Load the model
    model = load_model()
    print(f"Time taken to load model: {time.time() - st:.3f} seconds")

    # Load the data
    from imagenet import imagenet_classes, imagenet_templates
    print(f"{len(imagenet_classes)} classes, {len(imagenet_templates)} templates")

    # Get text embeddings
    # t1 = time.time()
    # text_embeddings = get_text_embeddings(imagenet_classes, imagenet_templates, model)
    # print(f"Time taken to get text embeddings: {time.time() - t1:.3f} seconds")
    
    # load ethosight gereral embeddings
    general_embeddings = torch.load("../embeddings/general.embeddings")

    from imagenet import get_imagenet_dataset
    dataset = get_imagenet_dataset()

    # Compute the affinity scores
    t1 = time.time()
    affinity_scores = compute_affinity_scores_ethosight_batched(model, dataset, general_embeddings, image_batch_size=args.image_batch_size)
    print(f"Time taken to compute affinity scores: {time.time() - t1:.3f} seconds")

    # top1, top5 = get_acc_image_embeddings(model, dataset, text_embeddings, image_batch_size=args.image_batch_size)
    # print(f"Top-1 accuracy: {top1:.2f}%, Top-5 accuracy: {top5:.2f}%")
    # print(f"Elapse time: {time.time() - st:.3f} seconds")
