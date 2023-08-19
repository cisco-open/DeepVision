import numpy as np
import torch
import open_clip
from tqdm import tqdm
from open_clip import tokenizer


def load_model(model_name, pretrained_name=None):
    # Load the model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_name, device=device)
    context_length = model.context_length
    vocab_size = model.vocab_size

    print("Model name:", model_name)
    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)
    return model, preprocess


def get_text_embeddings(classnames, templates, model):
    """
    Get text embeddings for a list of classnames using a list of templates
    Output: torch.Tensor of shape (embedding_dim, num_classes) (imagenet: torch.Size([512, 1000]))
    """
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            texts = tokenizer.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def get_acc_image_embeddings(model, loader, text_embeddings):
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for i, (images, target) in enumerate(tqdm(loader)):
            images = images.cuda()
            target = target.cuda()
            
            # predict
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100. * image_features @ text_embeddings

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100 
    return top1, top5


if __name__ == "__main__":
    import time
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, 
                        default="ViT-B-32", help="Model name", 
                        )
    parser.add_argument("--pretrained", type=str, 
                        default="openai")
    parser.add_argument("--image-batch-size", type=int, default=256, help="Image batch size")

    args = parser.parse_args()

    st = time.time()
    # Load the model
    model, preprocess = load_model(args.model, pretrained_name=args.pretrained)

    # Load the data
    from imagenet import imagenet_classes, imagenet_templates
    print(f"{len(imagenet_classes)} classes, {len(imagenet_templates)} templates")

    # Get text embeddings
    t1 = time.time()
    text_embeddings = get_text_embeddings(imagenet_classes, imagenet_templates, model)
    print(f"Time taken to get text embeddings: {time.time() - t1:.3f} seconds")

    from imagenet import get_data_loader
    loader = get_data_loader(args.image_batch_size, preprocess=preprocess)
    top1, top5 = get_acc_image_embeddings(model, loader, text_embeddings)
    print(f"Top-1 accuracy: {top1:.2f}%, Top-5 accuracy: {top5:.2f}%")
    print(f"Elapse time: {time.time() - st:.3f} seconds")

    # openclip model name, pretrained
#     [('RN50', 'openai'),
#  ('RN50', 'yfcc15m'),
#  ('RN50', 'cc12m'),
#  ('RN50-quickgelu', 'openai'),
#  ('RN50-quickgelu', 'yfcc15m'),
#  ('RN50-quickgelu', 'cc12m'),
#  ('RN101', 'openai'),
#  ('RN101', 'yfcc15m'),
#  ('RN101-quickgelu', 'openai'),
#  ('RN101-quickgelu', 'yfcc15m'),
#  ('RN50x4', 'openai'),
#  ('RN50x16', 'openai'),
#  ('RN50x64', 'openai'),
#  ('ViT-B-32', 'openai'),
#  ('ViT-B-32', 'laion400m_e31'),
#  ('ViT-B-32', 'laion400m_e32'),
#  ('ViT-B-32', 'laion2b_e16'),
#  ('ViT-B-32', 'laion2b_s34b_b79k'),
#  ('ViT-B-32-quickgelu', 'openai'),
#  ('ViT-B-32-quickgelu', 'laion400m_e31'),
#  ('ViT-B-32-quickgelu', 'laion400m_e32'),
#  ('ViT-B-16', 'openai'),
#  ('ViT-B-16', 'laion400m_e31'),
#  ('ViT-B-16', 'laion400m_e32'),
#  ('ViT-B-16', 'laion2b_s34b_b88k'),
#  ('ViT-B-16-plus-240', 'laion400m_e31'),
#  ('ViT-B-16-plus-240', 'laion400m_e32'),
#  ('ViT-L-14', 'openai'),
#  ('ViT-L-14', 'laion400m_e31'),
#  ('ViT-L-14', 'laion400m_e32'),
#  ('ViT-L-14', 'laion2b_s32b_b82k'),
#  ('ViT-L-14-336', 'openai'),
#  ('ViT-H-14', 'laion2b_s32b_b79k'),
#  ('ViT-g-14', 'laion2b_s12b_b42k'),
#  ('ViT-g-14', 'laion2b_s34b_b88k'),
#  ('ViT-bigG-14', 'laion2b_s39b_b160k'),
#  ('roberta-ViT-B-32', 'laion2b_s12b_b32k'),
#  ('xlm-roberta-base-ViT-B-32', 'laion5b_s13b_b90k'),
#  ('xlm-roberta-large-ViT-H-14', 'frozen_laion5b_s13b_b90k'),
#  ('convnext_base', 'laion400m_s13b_b51k'),
#  ('convnext_base_w', 'laion2b_s13b_b82k'),
#  ('convnext_base_w', 'laion2b_s13b_b82k_augreg'),
#  ('convnext_base_w', 'laion_aesthetic_s13b_b82k'),
#  ('convnext_base_w_320', 'laion_aesthetic_s13b_b82k'),
#  ('convnext_base_w_320', 'laion_aesthetic_s13b_b82k_augreg'),
#  ('convnext_large_d', 'laion2b_s26b_b102k_augreg'),
#  ('convnext_large_d_320', 'laion2b_s29b_b131k_ft'),
#  ('convnext_large_d_320', 'laion2b_s29b_b131k_ft_soup'),
#  ('convnext_xxlarge', 'laion2b_s34b_b82k_augreg'),
#  ('convnext_xxlarge', 'laion2b_s34b_b82k_augreg_rewind'),
#  ('convnext_xxlarge', 'laion2b_s34b_b82k_augreg_soup'),
#  ('coca_ViT-B-32', 'laion2b_s13b_b90k'),
#  ('coca_ViT-B-32', 'mscoco_finetuned_laion2b_s13b_b90k'),
#  ('coca_ViT-L-14', 'laion2b_s13b_b90k'),
#  ('coca_ViT-L-14', 'mscoco_finetuned_laion2b_s13b_b90k')]