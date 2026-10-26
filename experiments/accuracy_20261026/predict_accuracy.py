"""Inference for an accuracy-selected mixed-architecture research ensemble."""
import argparse
import hashlib
import json
from pathlib import Path


def model_for(architecture):
    import torch
    from torchvision import models
    if architecture == 'efficientnet_b0':
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, 1)
    elif architecture == 'densenet121':
        m = models.densenet121(weights=None)
        m.classifier = torch.nn.Linear(m.classifier.in_features, 1)
    elif architecture == 'resnet18':
        m = models.resnet18(weights=None)
        m.fc = torch.nn.Linear(m.fc.in_features, 1)
    else:
        raise ValueError('Unsupported architecture')
    return m


def image_tensor(path, size):
    from PIL import Image, ImageOps
    from torchvision import transforms
    with Image.open(path) as image:
        image = ImageOps.pad(image.convert('RGB'), (size, size), method=Image.Resampling.BICUBIC, color=0)
    return transforms.Normalize([.485, .456, .406], [.229, .224, .225])(transforms.ToTensor()(image))


def predict(directory, image, device='cpu'):
    import torch
    import math
    root = Path(directory).resolve()
    config = json.loads((root/'accuracy_config.json').read_text())
    if config['classes'] != ['NORMAL', 'PNEUMONIA'] or config['aggregation'] != 'mean_raw_sigmoid':
        raise ValueError('Unsupported output mapping')
    threshold = config['threshold']
    if not math.isfinite(threshold) or not 0 <= threshold <= math.nextafter(1., math.inf):
        raise ValueError('Invalid threshold')
    members = config['members']
    if not members or len({m['checkpoint'] for m in members}) != len(members):
        raise ValueError('Missing or duplicate members')
    scores, cache = [], {}
    torch.set_num_threads(2)
    with torch.inference_mode():
        for member in members:
            path = (root/member['checkpoint']).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError('Checkpoint path must remain inside model directory')
            if hashlib.sha256(path.read_bytes()).hexdigest() != member['sha256']:
                raise ValueError('Checkpoint checksum mismatch')
            size = member['image_size']
            if size not in {224, 320}: raise ValueError('Unsupported resolution')
            if size not in cache: cache[size] = image_tensor(image, size).unsqueeze(0).to(device)
            model = model_for(member['architecture'])
            model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
            model.to(device).eval()
            scores.append(float(torch.sigmoid(model(cache[size]).flatten())[0].cpu()))
            del model
    score = sum(scores)/len(scores)
    return dict(pneumonia_score=score, threshold=threshold, predicted_label=config['classes'][int(score >= threshold)],
                member_scores=scores, note='Uncalibrated research score, not a diagnosis or clinical probability')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',required=True); parser.add_argument('--image',required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args()
    print(json.dumps(predict(args.model_dir,args.image,args.device)))
