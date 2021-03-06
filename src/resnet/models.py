import torch.nn as nn
import timm
from torch.cuda.amp import autocast
import torch.nn.functional as F    

class BackboneModel(nn.Module):
    def __init__(self, in_features=2048, num_classes = 15):
        super(BackboneModel, self).__init__()
        self.architecture = 'resnext101_32x8d'
        self.backbone = timm.create_model(self.architecture, pretrained=True, num_classes = 0)
        self.backbone.inplanes = 2048
        self.in_features = in_features

        self.fc1 = nn.Linear(in_features, 1024, bias=True)
        self.fc2 = nn.Linear(1024, num_classes, bias=True)
        self.dropout = nn.Dropout(p=.5)
        
        # initalize
        for module in [self.fc1, self.fc2]:
            for m in module.modules():
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    
    @autocast()
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(-1, self.in_features)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        return self.fc2(x)
        