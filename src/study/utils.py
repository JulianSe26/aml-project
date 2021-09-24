from pathlib import Path

import numpy as np
import sklearn
import torch
from torch.utils.data import DataLoader

from src.study.dataset import split_dataset, StudyDataset


def resolve_device():
    if torch.cuda.is_available:
        found_device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
    else:
        found_device = torch.device("cpu")
        device_name = "cpu"
    print(f"Using device: {device_name}")
    return found_device


def to_one_hot(labels):
    one_hot = np.zeros((labels.size, labels.max() + 1))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot


def calculate_metrics(all_targets, all_probabilities):
    all_predictions = np.argmax(all_probabilities, axis=1)
    all_targets = np.asarray(all_targets)
    print(np.unique(all_predictions, axis=0, return_counts=True))
    print(np.unique(all_targets, axis=0, return_counts=True))
    metrics = {
        'precision': sklearn.metrics.precision_score(all_targets, all_predictions, average="macro"),
        'recall': sklearn.metrics.recall_score(all_targets, all_predictions, average="macro"),
        'f1': sklearn.metrics.f1_score(all_targets, all_predictions, average="macro"),
        'accuracy': sklearn.metrics.accuracy_score(all_targets, all_predictions),
    }
    hot_targets = to_one_hot(all_targets)
    hot_predictions = to_one_hot(all_predictions)
    aps = np.zeros(hot_targets.shape[1])
    for c in range(hot_targets.shape[1]):
        metrics[f'ap_{c}'] = sklearn.metrics.average_precision_score(hot_targets[:, c], hot_predictions[:, c])
        aps[c] = metrics[f'ap_{c}']
    metrics['map'] = np.mean(aps)
    return metrics, all_predictions


def prepare_data(data_dir, batch_size):
    data_path = Path(data_dir).expanduser()
    train_df, test_df = split_dataset(data_path, random_state=55)
    train = StudyDataset(train_df)
    test = StudyDataset(test_df, training=False)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=16)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=16)
    return train_loader, test_loader, train, test