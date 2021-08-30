import math
import os
import random
import time
from PIL.Image import SAVE
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from torch.cuda import amp, device
import tqdm
from yolo5_model import Model
from datasets import create_dataloader
from general import labels_to_class_weights, xywh2xyxy, check_img_size, one_cycle, colorstr, non_max_suppression, scale_coords, box_iou
from metrics import ap_per_class
from loss import ComputeLoss
import pickle
from pathlib import Path


'''=========================PRETRAINING====================================='''
RSNA_TRAIN_PATH = "../../data/RSNA/rsna_pneumonia_yolov5_train.txt"
RSNA_VALIDATION_PATH = "../../data/RSNA/rsna_pneumonia_yolov5_valid.txt"
'''========================================================================='''

'''=========================TRAIN FOR COVID================================='''
SIIM_TRAIN_PATH = "tbd-"
SIIM_VALIDATION_PATH = "tbd-" 
'''=========================PRETRAINING====================================='''



'''
================= TRAIN CONFIGURATION ========================================
'''
VALDIATION_FREQUENCY = 1
SAVE_FREQUENCY = 2 # in epochs
LOSS_REPORT_FREQUENCY = 100
NUMBER_DATALOADER_WORKERS = 10
EPOCHS = 10
BATCH_SIZE = 5
IMG_SIZE = 640
NUMBER_OF_CLASSES = 1
MULTI_SCALE = True
HYPER_PARAMETERS= {
    "lr0": 0.01,  # initial learning rate (SGD=1E-2, Adam=1E-3)
    "lrf": 0.2,  # final OneCycleLR learning rate (lr0 * lrf)
    "momentum": 0.937,  # SGD momentum/Adam beta1
    "weight_decay": 0.0005,  # optimizer weight decay 5e-4
    "warmup_epochs": 3.0,  # warmup epochs (fractions ok)
    "warmup_momentum": 0.8,  # warmup initial momentum
    "warmup_bias_lr": 0.1,  # warmup initial bias lr
    "box": 0.05,  # box loss gain
    "cls": 0.5,  # cls loss gain
    "cls_pw": 1.0,  # cls BCELoss positive_weight
    "obj": 1.0,  # obj loss gain (scale with pixels)
    "obj_pw": 1.0,  # obj BCELoss positive_weight
    "iou_t": 0.20,  # IoU training threshold
    "anchor_t": 4.0,  # anchor-multiple threshold
    "anchors": 3 , # anchors per output layer (0 to ignore)
    "fl_gamma": 0.0,  # focal loss gamma (efficientDet default gamma=1.5)
    "hsv_h": 0.015,  # image HSV-Hue augmentation (fraction)
    "hsv_s": 0.7,  # image HSV-Saturation augmentation (fraction)
    "hsv_v": 0.4,  # image HSV-Value augmentation (fraction)
    "degrees": 0.0,  # image rotation (+/- deg)
    "translate": 0.1,  # image translation (+/- fraction)
    "scale": 0.5,  # image scale (+/- gain)
    "shear": 0.0,  # image shear (+/- deg)
    "perspective": 0.0,  # image perspective (+/- fraction), range 0-0.001
    "flipud": 0.0,  # image flip up-down (probability)
    "fliplr": 0.5,  # image flip left-right (probability)
    "mosaic": 1.0,  # image mosaic (probability)
    "mixup": 0.5,  # image mixup (probability)
    "conf_threshold": 0.001 ,# confidence threshold for nms in validation
    "iou_threshold": 0.6 # intersection over union threhsold for nms in validations
}

'''
===========================================================================
'''

'''=================Misc Configuration========================='''
model_folder = "./models"
loss_folder = "./losses"
checkpoint_folder = "./ckpt"
'''============================================================''' 


if __name__ == '__main__':

    # check if GPU is available
    if torch.cuda.is_available:
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
    else:
        device = torch.device("cpu")
        device_name = "cpu"

    print(f'Using device: {device_name}')

    # Create required directories
    Path(model_folder).mkdir(exist_ok=True)
    Path(checkpoint_folder).mkdir(exist_ok=True)
    Path(loss_folder).mkdir(exist_ok=True)

    # Model init
    model = Model(cfg="yolo5l.yaml",ch=3,nc=NUMBER_OF_CLASSES).to(device)

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay


    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / BATCH_SIZE), 1)  # accumulate loss before optimizing
    HYPER_PARAMETERS["weight_decay"] *= BATCH_SIZE * accumulate / nbs  # scale weight_decay

    optimizer = optim.SGD(pg0, lr=HYPER_PARAMETERS["lr0"], momentum=HYPER_PARAMETERS["momentum"], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': HYPER_PARAMETERS["weight_decay"]})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    del pg0, pg1, pg2


    #scheduler
    lf = one_cycle(1, .2, EPOCHS)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)


    # Image sizes
    gs = max(int(model.stride.max()), 32) 
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz, imgsz_test = [check_img_size(x, gs) for x in [IMG_SIZE, IMG_SIZE]]  # verify imgsz are gs-multiples

    # Load data
    train_loader, train_dataset = create_dataloader(RSNA_TRAIN_PATH, imgsz, BATCH_SIZE, gs, True,
                                            hyp=HYPER_PARAMETERS, augment=False, cache=True, workers= NUMBER_DATALOADER_WORKERS,
                                            prefix=colorstr('train: '))
    validation_loader, validation_dataset = create_dataloader(RSNA_VALIDATION_PATH, imgsz, BATCH_SIZE, gs, True,
                                            hyp=HYPER_PARAMETERS, augment=False, cache=True, workers= NUMBER_DATALOADER_WORKERS, rect=True,
                                            pad=.5, prefix=colorstr('val: '))

    print("data loading done..")


    mlc = np.concatenate(train_dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(train_loader)  # number of batches


    # Model parameters
    HYPER_PARAMETERS['box'] *= 3. / nl  # scale to layers
    HYPER_PARAMETERS['cls'] *= NUMBER_OF_CLASSES / 80. * 3. / nl  # scale to classes and layers
    HYPER_PARAMETERS['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    HYPER_PARAMETERS['label_smoothing'] = 0.0
    model.nc = NUMBER_OF_CLASSES  # attach number of classes to model
    model.hyp = HYPER_PARAMETERS  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(train_dataset.labels, NUMBER_OF_CLASSES).to(device) * NUMBER_OF_CLASSES  # attach class weights
    model.names = ['Pneumonia']


    # Start training
    t0 = time.time()
    number_warmups = max(round(HYPER_PARAMETERS["warmup_epochs"] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    maps = np.zeros(1)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = - 1  # do not move
    scaler = amp.GradScaler(enabled=True)
    compute_loss = ComputeLoss(model)  # init loss class

    print(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {train_loader.num_workers} dataloader workers\n'
                f'Starting training for {EPOCHS} epochs...\n'
                f'Doing Gradient accumulations: {accumulate}')

    losses_per_epoch = []
    val_losses_per_epoch = []
    general_test_results = []

    for epoch in range(EPOCHS):
        model.train()
        losses = []
        for i, (imgs, targets, paths, _) in enumerate(tqdm.tqdm(train_loader, desc=colorstr('train: '))):

            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0 # scale images

             # Warmup
            if ni <= number_warmups:
                xi = [0, number_warmups]  # x interp
                accumulate = max(1, np.interp(ni, xi, [1, nbs / BATCH_SIZE]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [HYPER_PARAMETERS["warmup_bias_lr"] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [HYPER_PARAMETERS["warmup_momentum"], HYPER_PARAMETERS["momentum"]])

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


            losses.append(loss.item())

            #Report loss
            if i % LOSS_REPORT_FREQUENCY == 0:
                print(f' current loss {np.mean(losses)}')

            # Backprop
            scaler.scale(loss).backward()

            # Gradient accumulations
            if ni % accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        scheduler.step()
        losses_per_epoch.append(np.mean(losses))

        if epoch % VALDIATION_FREQUENCY == 0:
            # CUDA support half precision, 
            model.half()
            model.eval()

            iouv = torch.linspace(0.5, 0.55, 1).to(device)  # iou vector for mAP@0.5:0.95
            niou = iouv.numel()

            seen = 0 
            p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
            loss = torch.zeros(3, device=device)
            jdict, stats, ap, ap_class = [], [], [], []

            for j, (imgs, targets, paths, shapes) in enumerate(tqdm.tqdm(validation_loader)):
                imgs = imgs.to(device, non_blocking=True)
                imgs = imgs.half()  
                imgs /= 255.0  # 0 - 255 to 0.0 - 1.0
                targets = targets.to(device)
                nb, _, height, width = imgs.shape  # batch size, channels, height, width

                with torch.no_grad(), torch.cuda.amp.autocast():
                    out, train_out = model(imgs, augment=False)  # inference and training outputs
                    loss += compute_loss([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls

                # Run NMS
                targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
                lb =  []
                out = non_max_suppression(out, HYPER_PARAMETERS["conf_threshold"], HYPER_PARAMETERS["iou_threshold"], labels=lb, multi_label=True, agnostic=True)

                # Statistics per image
                for si, pred in enumerate(out):
                    labels = targets[targets[:, 0] == si, 1:]
                    nl = len(labels)
                    tcls = labels[:, 0].tolist() if nl else []  # target class
                    path = Path(paths[si])
                    seen += 1

                    if len(pred) == 0:
                        if nl:
                            stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                        continue

                    # Predictions
                    pred[:, 5] = 0
                    predn = pred.clone()
                    scale_coords(imgs[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

                     # Assign all predictions as incorrect
                    correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
                    if nl:
                        detected = []  # target indices
                        tcls_tensor = labels[:, 0]

                        # target boxes
                        tbox = xywh2xyxy(labels[:, 1:5])
                        scale_coords(imgs[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                    
                        # Per target class
                        for cls in torch.unique(tcls_tensor):
                            ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                            pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                            # Search for detections
                            if pi.shape[0]:
                                # Prediction to target ious
                                ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                                # Append detections
                                detected_set = set()
                                for j in (ious > iouv[0]).nonzero(as_tuple=False):
                                    d = ti[i[j]]  # detected target
                                    if d.item() not in detected_set:
                                        detected_set.add(d.item())
                                        detected.append(d)
                                        correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                        if len(detected) == nl:  # all targets already located in image
                                            break

                    # Append statistics (correct, conf, pcls, tcls)
                    stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

            # Compute statistics
            stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
            if len(stats) and stats[0].any():
                p, r, ap, f1, ap_class = ap_per_class(*stats)
                ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
                mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
                nt = np.bincount(stats[3].astype(np.int64), minlength=NUMBER_OF_CLASSES)  # number of targets per class
            else:
                nt = torch.zeros(1)

            # Print results
            pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
            print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))


            # For training return back to float32
            model.float()

        if epoch % SAVE_FREQUENCY == 0:
            torch.save(model, f"./{model_folder}/yolov5_epoch_{epoch}_full.pt")
            torch.save(model.state_dict(), f"./{model_folder}/yolov5_epoch_{epoch}.pt")
            torch.save({"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict()}, f"./{checkpoint_folder}/yolov5_epoch_{epoch}_ckpt.pt")
            np.save(f"{loss_folder}/yolov5_train_loss_{epoch}.np", np.array(losses_per_epoch))
            np.save(f"{loss_folder}/yolov5_val_loss_{epoch}.np", np.array(val_losses_per_epoch))
            with open(f"{loss_folder}/yolov5_general_test_results_{epoch}.pickle", "wb") as p:
                pickle.dump(general_test_results, p)
            

    # finalize
    del model
    del optimizer
    torch.cuda.empty_cache()

