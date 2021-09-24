import torch
from torch import optim

NUMBER_EPOCHS = 56

CONFIG = {
    "epochs": NUMBER_EPOCHS,
    "scheduler_frequency": 2,
    "batch_size": 35,
    "optimizer": lambda params: optim.SGD(params, lr=0.0001, momentum=0.9, nesterov=True),
    "scheduler": lambda optimizer: torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, NUMBER_EPOCHS, 0.05),
    "name": "SGD_lr0-0001_m0-9_cos0-05_b35"
}
