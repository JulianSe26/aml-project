from torch.utils import data
from torch.utils.data import Dataset
import torch
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from torchvision import transforms
from PIL import Image

# ADJUST TO YOUR NEEDS
BASE_DIR = Path('../../nih')

#logging.basicConfig(level=logging.DEBUG)


nih_classes = [
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Effusion',
    'Emphysema',
    'Fibrosis',
    'Hernia',
    'Infiltration',
    'Mass',
    'No Finding',
    'Nodule',
    'Pleural_Thickening',
    'Pneumonia',
    'Pneumothorax',
]

class NIHDataset(Dataset):
    def __init__(self) -> None:
        super(Dataset, self).__init__()

        # Required transforms
        # https://pytorch.org/hub/pytorch_vision_resnext/
        self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self.base_dir = BASE_DIR
        self.image_dir = self.base_dir.joinpath('data')

        self.image_dir.mkdir(parents=False, exist_ok=True)

        self.annotations = pd.read_csv(self.base_dir.joinpath('Data_Entry_2017.csv'))

        self.image_paths = sorted(self.base_dir.rglob("images*/*.png"))

        self.classes = {item: idx for idx, item in enumerate(nih_classes)}

        if len(self.image_paths) > 0:
            print("Initializing dataset and unpacking folders. This might take up to 1 minute if not done previously")
            for img in self.image_paths:
                img.rename(self.image_dir.joinpath(img.name))

        print(self.annotations.head())

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):

        image = self.transform(Image.open(str(self.image_dir.joinpath(self.annotations.loc[index, "Image Index"]).resolve())).convert("RGB"))

        labels = torch.zeros(len(nih_classes))
        labels[[self.classes.get(x) for x  in self.annotations.loc[index, "Finding Labels"].split("|")]] = 1

        return image, labels
