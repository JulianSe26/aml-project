from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torchvision import transforms


class StudyDataset(Dataset):

    def __init__(self, merged_image_study, training=True):
        super(Dataset, self).__init__()

        self.train = training
        # Required transforms
        # https://pytorch.org/hub/pytorch_vision_resnext/

        self.merged_data = merged_image_study
        if self.train:
            self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(256, scale=(.5, 1.0)), # Randomly crop between half to all of the image
                    transforms.RandomHorizontalFlip(p=.5),
                    transforms.RandomAdjustSharpness(p=.3, sharpness_factor=.7), # Blur the image a little
                    transforms.RandomRotation(degrees=20), # Randomly rotate between -degrees and +degrees
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.RandomErasing(p=.3, scale=(.01, .1)), # Randomly erase between 1 and 10% percent of the image in a rectangular area
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        else:
            self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.merged_data)

    def __getitem__(self, idx):
        data_row = self.merged_data.iloc[idx]
        image = self.transform(Image.open(data_row["file_path"]).convert("RGB"))
        # return image, torch.from_numpy(np.squeeze((data_row.values[5:9])).astype(np.float32))
        return image, np.argmax(data_row.values[5:9])


def split_dataset(base_path: Path, train_size=0.66, random_state=42):
    study_level = pd.read_csv(base_path / "train_study_level.csv")
    image_level = pd.read_csv(base_path / "train_image_level.csv")
    image_level["study_id"] = image_level["StudyInstanceUID"] + "_study"
    merged = pd.merge(image_level, study_level, left_on="study_id", right_on="id", suffixes=("", "_o"))

    # Negative for Pneumonia, Typical Appearance, Indeterminate Appearance, Atypical Appearance
    y = np.squeeze((merged.values[:, [5, 6, 7, 8]]))

    merged = merged.drop(["id_o"], axis=1)

    def to_file_path(row):
        image_id = row["id"][:-6]
        return list(base_path.glob(f'data/train/{row["StudyInstanceUID"]}/**/{image_id}.png'))[0]

    merged["file_path"] = merged.apply(to_file_path, axis=1)
    train_X, test_X, _, _ = train_test_split(merged, y, train_size=train_size, random_state=random_state)
    return train_X, test_X


if __name__ == '__main__':
    base_path = Path('~/var/aml-xrays/').expanduser()
    train, test = split_dataset(base_path)
    dataset = StudyDataset(train, training=True)
    image, labels = dataset[1]

    print(labels)

    img = transforms.ToPILImage()(image)
    img.show()




