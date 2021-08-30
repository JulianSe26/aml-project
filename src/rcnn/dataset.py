import numpy as np
from torchvision.datasets import CocoDetection
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

class ChestCocoDetection(CocoDetection): 
    def __init__(self, root, ann_file, transforms=None): 
        super(ChestCocoDetection, self).__init__(root, ann_file) 
        if transforms is None:
            transforms = A.Compose([
                    A.Resize(512, 512),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
            ])
        self._transforms = transforms 

    def __getitem__(self, idx): 
        img, target = super(ChestCocoDetection, self).__getitem__(idx) 
        #image_id = self.ids[idx] 
        #target = dict(image_id=image_id, annotations=target) 

        if len(target) == 0:
            boxes = np.array([]).reshape(0,4)
            cats = np.array([])
        else:
            boxes = np.array([(a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]) for a in target])
            cats = np.array([a['category_id'] for a in target])

        transformed = self._transforms(image=np.array(img), bboxes=boxes, category_ids=cats) 
        target = {}

        if len(transformed['bboxes']) == 0:
            target['boxes'] = torch.zeros((0,4), dtype=torch.float32)
            target['area'] = torch.zeros((0), dtype=torch.float32)
            target['labels'] = torch.ones_like(target['boxes'], dtype=torch.int64)
        else:
            target['boxes'] = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            target['area'] = torch.tensor((boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]), dtype=torch.float32)
            target['labels'] = torch.tensor(cats, dtype=torch.int64)
        return transformed['image'], target