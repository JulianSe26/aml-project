import torch
from torch import optim

LEARNING_RATE = 0.0005

NUMBER_EPOCHS = 58

CONFIG = {
    "epochs": NUMBER_EPOCHS,
    "scheduler_frequency": 2,
    "batch_size": 15,
    "optimizer": lambda params: optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, nesterov=True),
    "scheduler": lambda optimizer: torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, NUMBER_EPOCHS, 0.05*LEARNING_RATE),
    "name": "SGD_lr0-0005_m0-9_cos0-05_b15"
}
