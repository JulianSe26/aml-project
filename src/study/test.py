import timm
import torch
import sys

sys.path.append('../')
from resnet.models import BackboneModel

if __name__ == '__main__':
    device = torch.device("cuda")
    model = timm.create_model("resnext101_32x8d", pretrained=False, num_classes=5)
    print(model)
