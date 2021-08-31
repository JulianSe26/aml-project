from albumentations.augmentations.crops.transforms import RandomSizedBBoxSafeCrop
from albumentations.augmentations.geometric.transforms import ShiftScaleRotate
from albumentations.augmentations.transforms import Blur, Cutout, HorizontalFlip, RandomBrightnessContrast, Sharpen
from albumentations.core.composition import OneOf
import numpy as np
from torchvision.datasets import CocoDetection
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

class ChestCocoDetection(CocoDetection): 
    def __init__(self, root, ann_file, training=True): 
        super(ChestCocoDetection, self).__init__(root, ann_file) 
        self.training = training

        # We use albumentations here because they have support for the transformation
        # of bounding boxes per default unlike the torchvision transforms
        if self.training:
            self._transforms = A.Compose([
                A.RandomResizedCrop(height=512, width=512, scale=(.3, 1.0)),
                A.HorizontalFlip(p=.3),
                A.ShiftScaleRotate(rotate_limit=20, p=.3, border_mode=0, value=0),
                A.OneOf([A.Blur(blur_limit=5, p=.25), A.Sharpen(p=.25)], p=.5),
                A.RandomBrightnessContrast(p=.3),
                A.Cutout(num_holes=6, max_h_size=32, max_w_size=32, p=.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))
        else:
            self._transforms = A.Compose([
                    A.Resize(512, 512),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))
        

    def __getitem__(self, idx): 
        img, target_raw = super(ChestCocoDetection, self).__getitem__(idx) 
        image_id = self.ids[idx] 

        if len(target_raw) == 0:
            boxes = np.array([]).reshape(0,4)
            cats = np.array([])
        else:
            boxes = np.array([(a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]) for a in target_raw])
            cats = np.array([a['category_id'] for a in target_raw])

        transformed = self._transforms(image=np.array(img), bboxes=boxes, category_ids=cats) 

        target = {}
        target['image_id'] = torch.tensor([image_id], dtype=torch.int64)
        if len(transformed['bboxes']) == 0:
            target['boxes'] = torch.zeros((0,4), dtype=torch.float32)
            target['area'] = torch.zeros((0), dtype=torch.float32)
            target['labels'] = torch.ones_like(target['boxes'], dtype=torch.int64)
        else:
            boxes = np.array(transformed['bboxes'])
            target['boxes'] = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            target['area'] = torch.tensor((boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]), dtype=torch.float32)
            target['labels'] = torch.tensor(cats, dtype=torch.int64)
        return transformed['image'], target