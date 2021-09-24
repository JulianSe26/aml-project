from torch.utils import data
from torch.utils.data import Dataset
import torch
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from torchvision import transforms
from PIL import Image


nih_classes = [
    'Atelectasis',
    'Cardiomegaly',
    'Effusion',
    'Infiltration',
    'Mass',
    'Nodule',
    'Pneumonia',
    'Pneumothorax',
    'Consolidation'
    'Edema',
    'Emphysema',
    'Fibrosis',
    'Pleural_Thickening',
    'Hernia',
    'No Finding',
]

class NIHDataset(Dataset):
    def __init__(self, base_dir=None, data=None, train=True) -> None:
        super(Dataset, self).__init__()

        self.train = train

        # Required transforms
        # https://pytorch.org/hub/pytorch_vision_resnext/

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



        self.base_dir = Path(base_dir)
        self.image_dir = self.base_dir.joinpath('data')

        self.image_dir.mkdir(parents=False, exist_ok=True)

        self.annotations = data #pd.read_csv(self.base_dir.joinpath('Data_Entry_2017.csv'))

        self.image_paths = sorted(self.base_dir.rglob("images*/*.png"))

        self.classes = {item: idx for idx, item in enumerate(nih_classes)}

        if len(self.image_paths) > 0:
            print("Initializing dataset and unpacking folders. This might take up to 1 minute if not done previously")
            for img in self.image_paths:
                img.rename(self.image_dir.joinpath(img.name))

        print(f"Training: {self.train} with {len(self.annotations)} entries")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):

        image = self.transform(Image.open(str(self.image_dir.joinpath(self.annotations.iloc[index]["Image Index"]).resolve())).convert("RGB"))

        labels = torch.zeros(len(nih_classes))
        labels[[self.classes.get(x) for x  in self.annotations.iloc[index]["Finding Labels"].split("|")]] = 1

        return image, labels
