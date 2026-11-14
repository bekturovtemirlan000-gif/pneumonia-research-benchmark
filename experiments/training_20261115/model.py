"""Model and deterministic preprocessing shared by training and standalone inference."""
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image, ImageOps

MEAN, STD = [.485, .456, .406], [.229, .224, .225]
SIZE = 320


def make_model(architecture, pretrained=False):
    if architecture == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
    elif architecture == 'efficientnet_v2_s':
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None)
    else:
        raise ValueError('Unsupported architecture: ' + architecture)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    return model, model.classifier


def image_tensor(path):
    with Image.open(path) as image:
        image = ImageOps.pad(image.convert('RGB'), (SIZE, SIZE), method=Image.Resampling.BICUBIC, color=0)
        return transforms.Normalize(MEAN, STD)(transforms.ToTensor()(image))


def load_config(folder):
    folder = Path(folder).resolve()
    config = json.loads((folder / 'inference_config.json').read_text())
    if config['image_size'] != SIZE or config['aggregation'] != 'mean_raw_sigmoid':
        raise ValueError('Unsupported inference configuration')
    for member in config['members']:
        path = (folder / member['checkpoint']).resolve()
        if not path.is_relative_to(folder):
            raise ValueError('Checkpoint escapes model folder')
        if hashlib.sha256(path.read_bytes()).hexdigest() != member['sha256']:
            raise ValueError('Checkpoint hash mismatch')
    return config


@torch.inference_mode()
def predict(folder, image, device='cpu'):
    folder = Path(folder)
    config = load_config(folder)
    torch.set_num_threads(2)
    x = image_tensor(image).unsqueeze(0).to(device)
    scores = []
    for member in config['members']:
        model, _ = make_model(member['architecture'])
        model.load_state_dict(torch.load(folder / member['checkpoint'], map_location='cpu', weights_only=True), strict=True)
        model.to(device).eval()
        scores.append(float(torch.sigmoid(model(x).flatten())[0]))
        del model
    score = sum(scores) / len(scores)
    return dict(score=score, threshold=config['threshold'], prediction=int(score >= config['threshold']),
                label='PNEUMONIA' if score >= config['threshold'] else 'NORMAL',
                interpretation='Research classification score, not a clinical probability')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    print(json.dumps(predict(args.model_dir, args.image, args.device)))
