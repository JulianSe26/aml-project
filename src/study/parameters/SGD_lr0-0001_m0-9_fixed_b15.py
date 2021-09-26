import torch
from torch import optim

LEARNING_RATE = 0.0001

NUMBER_EPOCHS = 35

CONFIG = {
    "epochs": NUMBER_EPOCHS,
    "scheduler_frequency": 1000,
    "batch_size": 15, # no scheduling
    "optimizer": lambda params: optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, nesterov=True),
    "scheduler": lambda optimizer: torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: x),  # no scheduling
    "name": "SGD_lr0-0001_m0-9_fixed_b15"
}
