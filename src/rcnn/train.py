import argparse
import itertools
from pathlib import Path
import pickle
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from model import ChestRCNN
import math
from dataset import ChestCocoDetection
from utils import evaluate

from rcnn.model import ChestRCNN

'''================Train Configuration========================='''
number_epochs = 35
save_frequency = 2          # in epochs
test_frequency = 1          # in epochs
print_loss_frequency = 100  # in iterations
'''============================================================'''   

'''=================Misc Configuration========================='''
model_folder = "./models"
loss_folder = "./losses"
checkpoint_folder = "./ckpt"
'''============================================================''' 


def calculate_metrics(predictions, targets, threshold=.5):
        predictions = np.array(predictions > threshold, dtype=float)
        precision = precision_score(targets, predictions, average="macro")
        recall = recall_score(targets, predictions, average="macro")
        f1 = f1_score(targets, predictions, average="macro")
        accuarcy = accuracy_score(targets, predictions)
        return {"accuracy": accuarcy, "f1": f1, "recall": recall, "precision": precision}

def collate_fn(batch):
    imgs, targets = zip(*batch)
    imgs = torch.stack(imgs)
    return tuple((imgs, targets))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training argparser')
    parser.add_argument("--ckpt", default=None, type=str)
    parser.add_argument("--train_loss", default=None, type=str)
    parser.add_argument("--val_loss", default=None, type=str)
    parser.add_argument("--test_results", default=None, type=str)
    parser.add_argument("--start_epoch", default=0, type=int)

    args = parser.parse_args()

    train_data = ChestCocoDetection(root="D:\\Siim\\siim-covid19-detection", ann_file="D:\\Siim\\siim-covid19-detection\\train.json")
    test_data = ChestCocoDetection(root="D:\\Siim\\siim-covid19-detection", ann_file="D:\\Siim\\siim-covid19-detection\\test.json", training=False)

    # Create required directories
    Path(model_folder).mkdir(exist_ok=True)
    Path(checkpoint_folder).mkdir(exist_ok=True)
    Path(loss_folder).mkdir(exist_ok=True)


    train_loader = DataLoader(train_data, batch_size=4, shuffle=True, pin_memory=True, num_workers=4, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=8, shuffle=True, pin_memory=True, num_workers=4, collate_fn=collate_fn)

    if torch.cuda.is_available:
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
    else:
        device = torch.device("cpu")
        device_name = "cpu"

    print(f'Using device: {device_name}')

    model = ChestRCNN('../resnet/models/resnext101_32x8d_epoch_35.pt')
    model.to(device)

    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True)
    criterion = nn.BCEWithLogitsLoss().to(device)

    scaler = torch.cuda.amp.GradScaler()

    # training scheduler
    lf = lambda x: (((1 + math.cos(x * math.pi / number_epochs)) / 2) ** 1.0) * 0.95 + 0.05
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt)

        optimizer.load_state_dict(ckpt["optimizer"])
        model.load_state_dict(ckpt["model"])
        scaler.load_state_dict(ckpt["scaler"])
        scheduler.load_state_dict(ckpt["scheduler"])

        del ckpt

    test_results_general = []
    loss_per_epoch = []
    val_loss_per_epoch = []

    if args.train_loss is not None:
        loss_per_epoch = np.load(args.train_loss).tolist()
    if args.val_loss is not None:
        val_loss_per_epoch = np.load(args.val_loss).tolist()
    if args.test_results is not None:
        with open(args.test_results, 'rb') as f:
            test_results_general = pickle.load(f)

    for epoch in range(args.start_epoch, number_epochs+1):
        model.train()

        losses = []

        print(f"Training model on epoch {epoch} using lr={scheduler.get_last_lr()}")

        #for batch_i, (imgs, targets) in enumerate(tqdm(train_loader, desc="Training")):
        for (imgs, targets) in itertools.islice(tqdm(train_loader, desc="Training"), 50):

            imgs = imgs.to(device)
            targets = [{k: v.to(device).requires_grad_(False) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                out = model(imgs, targets)
                loss = sum(component_loss for component_loss in out.values())

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            losses.append(loss.item())

            # if batch_i % print_loss_frequency == 0 and batch_i != 0:
            #     tqdm.write(f'Epoch: {epoch}, step: {batch_i}; current loss: {np.mean(losses)}, improvement to previous loss: {np.mean(losses[:-print_loss_frequency]) - np.mean(losses[-print_loss_frequency:])}')
            # elif batch_i == 0:
            #     tqdm.write(f'Start loss: {np.mean(losses)}')

        # Perform operations after every epoch
        if epoch % 5 == 0:
             scheduler.step()


        loss_per_epoch.append(np.mean(losses))

        if epoch % test_frequency == 0:
                #model.eval()
                with torch.no_grad(), torch.cuda.amp.autocast():
                    predictions = []
                    val_targets = []
                    validation_loss = []
                    #for test_i, (imgs, labels) in enumerate(tqdm(test_loader, desc="Testing")):
                    #    imgs = imgs.to(device)
                    #    targets = [{k: v.to(device).requires_grad_(False) for k, v in t.items()} for t in targets]

                    #    batch_prediction = model(imgs)

                    evaluate(model, test_loader, device=device)

                        #val_loss = criterion(batch_prediction, labels)
                        #validation_loss.append(val_loss.item())
                        #predictions.extend(batch_prediction.cpu().numpy())
                        #val_targets.extend(labels.cpu().numpy())
                    
                    #test_results = calculate_metrics(np.array(predictions), np.array(val_targets))
                    #test_results_general.append(test_results)
                    #val_loss_per_epoch.append(np.mean(validation_loss))
                    #print(f'\nValidation loss: {val_loss_per_epoch[-1]}')
                    #print(f'\n{test_results}')
                    

        if epoch % save_frequency == 0 or epoch == number_epochs:
            torch.save(model, f"./{model_folder}/fasterrcnn_epoch_{epoch}_full.pt")
            torch.save(model.state_dict(), f"./{model_folder}/fasterrcnn_epoch_{epoch}.pt")
            torch.save({"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                        "scheduler": scheduler.state_dict()}, f"./{checkpoint_folder}/fasterrcnn_epoch_{epoch}_ckpt.pt")
            np.save(f"{loss_folder}/fasterrcnn_train_loss_{epoch}.np", np.array(loss_per_epoch))
            np.save(f"{loss_folder}/fasterrcnn_val_loss_{epoch}.np", np.array(val_loss_per_epoch))
            with open(f"{loss_folder}/fasterrcnn_general_test_results_{epoch}.pickle", "wb") as p:
                pickle.dump(test_results_general, p)


    del model
    del optimizer
    torch.cuda.empty_cache()
