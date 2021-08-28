  
import argparse
import logging
import math
import os
import random
from re import MULTILINE
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp, device
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import test  # import test.py to get mAP after each epoch
from yolo5_model import Model
from autoanchor import check_anchors
from datasets import create_dataloader
from general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_img_size, \
    check_requirements, set_logging, one_cycle, colorstr
from loss import ComputeLoss
from plots import plot_images, plot_labels, plot_results, plot_evolution
from torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel


'''
================= TRAIN CONFIGURATIONS ===============
'''
EPOCHS = 20
BATCH_SIZE = 24

MOMENTUM_SGD = .937
LR_SGD = .01
WEIGHT_DECAY = .0005
IMG_SIZE = 768


WARMUP_EPOCHS = 3
WARMUP_MOMENRUM = .8
WARMUP_BIAS_LR = .1


MULTI_SCALE = True

'''
======================================================
'''


if __name__ == '__main__':

    # check if GPU is available
    if torch.cuda.is_available:
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
    else:
        device = torch.device("cpu")
        device_name = "cpu"

    print(f'Using device: {device_name}')

    # Model init
    model = Model(cfg="yolo5l.yaml",ch=3,nc=1).to(device)

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    optimizer = optim.SGD(pg0, lr=LR_SGD, momentum=MOMENTUM_SGD, nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': WEIGHT_DECAY})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    del pg0, pg1, pg2


    #scheduler
    lf = one_cycle(1, .2, EPOCHS)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)


    # Image sizes
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz, imgsz_test = [check_img_size(x, gs) for x in [IMG_SIZE, IMG_SIZE]]  # verify imgsz are gs-multiples

    # Load data

    # Trainloader
    # TODO
    dataloader = None

    nb = len(dataloader)


    # Start training
    t0 = time.time()
    number_warmups = max(round(WARMUP_EPOCHS * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    maps = np.zeros(1)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = - 1  # do not move
    scaler = amp.GradScaler(enabled=True)
    compute_loss = ComputeLoss(model)  # init loss class

    print(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'Starting training for {EPOCHS} epochs...')


    # prepare for saving checkpoints during training
    os.makedirs("checkpoints", exist_ok=True)


    for epoch in range(EPOCHS):
        model.train()

        optimizer.zero_grad()

        for i,(imgs, targets) in tqdm(enumerate()):

            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

             # Warmup
            if ni <= number_warmups:
                xi = [0, number_warmups]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [WARMUP_BIAS_LR if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if MULTI_SCALE:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]] 
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward pass
            with amp.autocast(enabled=True):
                pred = model(imgs)
                loss, loss_items = compute_loss(pred, targets.to(device))

            # Backprop
            scaler.scale(loss).backward()

            # Gradient accumulations
            if ni % accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        scheduler.step()


        # TODO: Save logic and validation results


    # finalize
    torch.cuda.empty_cache()

