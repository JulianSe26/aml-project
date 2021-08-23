import torch
import torch.nn as nn
import timm
from torch.cuda.amp import autocast
import torch.nn.functional as F

class BackboneModel(nn.Module):
    def __init__(self, training, in_features=2048, num_classes = 15):
        super(BackboneModel, self).__init__()
        self.architecture = 'resnext101_32x8d'
        self.backbone = timm.create_model(self.architecture, pretrained=False, num_classes = 0)
        self.training = training

        self.fc1 = nn.Linear(in_features, 1024, bias=True)
        self.fc2 = nn.Linear(1024, num_classes, bias=True)
    
    
    @autocast()
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(-1, self.in_features)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x