"""Run the exported research classifier on an image. No training or downloads."""
import argparse
import json
import math
from pathlib import Path


def load_config(directory):
    directory = Path(directory).resolve()
    config = json.loads((directory / 'inference_config.json').read_text())
    if config.get('architecture') not in {'efficientnet_b0', 'densenet121', 'resnet18'}:
        raise ValueError('Unsupported architecture')
    if config.get('classes') != ['NORMAL', 'PNEUMONIA']:
        raise ValueError('Unexpected class mapping')
    if config.get('image_size', 224) not in {224, 320}:
        raise ValueError('Unsupported image size')
    threshold = config.get('threshold')
    if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError('Invalid threshold')
    names = config.get('selected_checkpoints', [])
    temperature = config.get('temperature', 1.0)
    if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Invalid temperature')
    if not names or len(names) != len(set(names)):
        raise ValueError('Missing or repeated checkpoints')
    paths = []
    for name in names:
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError('Checkpoint must exist inside the model directory')
        paths.append(path)
    return config, paths


def predict(directory, image_path, device_name='cpu'):
    import torch
    from PIL import Image, ImageOps
    from torchvision import models, transforms
    config, paths = load_config(directory)
    device = torch.device(device_name)
    image_size = config.get('image_size', 224)
    with Image.open(image_path) as image:
        image = ImageOps.pad(image.convert('RGB'), (image_size, image_size),
                             method=Image.Resampling.BICUBIC, color=0)
    transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    x = transform(image).unsqueeze(0).to(device)
    probabilities = []
    with torch.inference_mode():
        for path in paths:
            if config['architecture'] == 'efficientnet_b0':
                model = models.efficientnet_b0(weights=None)
                model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1)
            elif config['architecture'] == 'densenet121':
                model = models.densenet121(weights=None)
                model.classifier = torch.nn.Linear(model.classifier.in_features, 1)
            else:
                model = models.resnet18(weights=None)
                model.fc = torch.nn.Linear(model.fc.in_features, 1)
            model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
            model.to(device).eval()
            probabilities.append(float(torch.sigmoid(model(x).flatten())[0].cpu()))
            del model
    probability = sum(probabilities) / len(probabilities)
    raw_probability = probability
    if 'temperature' in config:
        p = min(1 - 1e-7, max(1e-7, probability))
        z = (math.log(p) - math.log1p(-p)) / config['temperature']
        probability = 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))
    return {'image': str(image_path), 'pneumonia_score': probability,
            'threshold': config['threshold'],
            'predicted_label': config['classes'][int(probability >= config['threshold'])],
            'member_scores': probabilities,
            'raw_mean_score': raw_probability,
            'temperature': config.get('temperature', 1.0),
            'image_size': image_size,
            'note': 'Research image classification score, not a calibrated clinical probability or diagnosis.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    print(json.dumps(predict(args.model_dir, args.image, args.device), indent=2))
