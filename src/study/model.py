import timm
import torch

from torch import nn
from torch.cuda.amp import autocast


def init_weights(model):
    if isinstance(model, nn.Linear):
        nn.init.xavier_uniform_(model.weight)
        if model.bias is not None:
            nn.init.constant_(model.bias, 0)


class CovidModel(nn.Module):

    def __init__(self, num_classes=4):
        super(CovidModel, self).__init__()
        # weights will be loaded initially from resnet.backbone
        self.backbone = timm.create_model("resnext101_32x8d", pretrained=False, num_classes=0)

        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes, bias=True)
        )

        self.classifier.apply(init_weights)

    @autocast()
    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)


