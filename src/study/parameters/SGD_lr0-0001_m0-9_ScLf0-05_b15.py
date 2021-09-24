import math

import torch
from torch import optim

NUMBER_EPOCHS = 35

lf = lambda x: (((1 + math.cos(x * math.pi / NUMBER_EPOCHS)) / 2) ** 1.0) * 0.95 + 0.05

CONFIG = {
    "epochs": NUMBER_EPOCHS,
    "scheduler_frequency": 2,
    "batch_size": 15,
    "optimizer": lambda params: optim.SGD(params, lr=0.0001, momentum=0.9, nesterov=True),
    "scheduler": lambda optimizer: torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf),
    "name": "SGD_lr0-0001_m0-9_cos0-05_b15"
}
