import torch
from torch import optim

LEARNING_RATE = 0.0002

NUMBER_EPOCHS = 35

CONFIG = {
    "epochs": NUMBER_EPOCHS,
    "scheduler_frequency": 2,
    "batch_size": 15,
    "optimizer": lambda params: optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, nesterov=True),
    "scheduler": lambda optimizer: torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, NUMBER_EPOCHS, 0.25*LEARNING_RATE),
    "name": "SGD_lr0-0002_m0-9_cos0-25_b15"
}
