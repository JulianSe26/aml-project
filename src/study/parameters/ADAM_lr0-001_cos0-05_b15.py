import torch
from torch import optim

LEARNING_RATE = 0.001

NUMBER_EPOCHS = 35

CONFIG = {
    "epochs": NUMBER_EPOCHS,
    "scheduler_frequency": 2,
    "batch_size": 15,
    "optimizer": lambda params: optim.Adam(params, lr=LEARNING_RATE),
    "scheduler": lambda optimizer: torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, NUMBER_EPOCHS, LEARNING_RATE*0.05),
    "name": "ADAM_lr0-001_cos0-05_b15"
}
