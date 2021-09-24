import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.study.model import CovidModel
from src.study.utils import resolve_device, calculate_metrics, prepare_data

TEST_CONFIG = {
    "data_dir": "/home/tkrieger/var/aml-xrays",
    "checkpoint": "/home/tkrieger/var/aml-models/study/b35_lr0-0001/study_resnext_b35_lr0-0001_epoch_55.pt",
    "metrics_dir": "./metrics/b35_lr0-0001/",
    "random_state": 55,
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test argparser')
    parser.add_argument('--checkpoint', default=TEST_CONFIG['checkpoint'], type=str)
    parser.add_argument('--data_dir', default=TEST_CONFIG['data_dir'], type=str)
    parser.add_argument("--metrics_dir", default=TEST_CONFIG['metrics_dir'], type=str)
    args = parser.parse_args()
    Path(args.metrics_dir).mkdir(exist_ok=True)

    _, test_loader, _, _ = prepare_data(args.data_dir, 1)
    device = resolve_device()

    model = CovidModel().to(device)
    softmax = torch.nn.Softmax(dim=1).to(device)

    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint)

    all_probabilities = []
    all_targets = []

    model.eval()
    with torch.inference_mode(), torch.cuda.amp.autocast():
        for test_i, (images, targets) in enumerate(tqdm(test_loader, desc="Testing")):
            images = images.to(device)
            batch_logits = model(images)
            probabilities = softmax(batch_logits)
            all_probabilities.extend(probabilities.cpu().numpy())
            all_targets.extend(targets)

    metrics, all_predictions = calculate_metrics(all_targets, all_probabilities)

    checkpoint_name = Path(args.checkpoint).stem

    with open(f'{args.metrics_dir}/{checkpoint_name}_test_metrics.pickle', "wb") as pickle_file:
        pickle.dump(metrics, pickle_file)
    np.save(f'{args.metrics_dir}/{checkpoint_name}_test_predictions', all_predictions)
    np.save(f'{args.metrics_dir}/{checkpoint_name}_test_targets', all_targets)

    print(metrics)

