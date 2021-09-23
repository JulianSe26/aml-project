from pathlib import Path
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import StudyDataset
from dataset import split_dataset
import pickle
from tqdm import tqdm
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import math
import argparse

from model import CovidModel


'''================Train Configuration========================='''
number_epochs = 55
save_frequency = 2          # in epochs
test_frequency = 1          # in epochs
scheduler_frequency = 2     # in epochs
print_loss_frequency = 500  # in iterations
batch_size = 10
'''============================================================'''

'''=================Misc Configuration========================='''
# ADJUST TO YOUR NEEDS
base_data_dir = "/home/tkrieger/var/aml-xrays"
model_folder = "/home/tkrieger/var/aml-models/study"
loss_folder = "./losses"
model_name = "study_resnext101_32x8d"
'''============================================================'''


def calculate_metrics(probabilities, targets):
        print(probabilities)
        prediction = np.argmax(probabilities, axis=1)
        print(np.unique(prediction, axis=0, return_counts=True))
        print(np.unique(targets, axis=0, return_counts=True))
        precision = precision_score(targets, prediction, average="macro")
        recall = recall_score(targets, prediction, average="macro")
        f1 = f1_score(targets, prediction, average="macro")
        accuarcy = accuracy_score(targets, prediction)
        return {"accuracy": accuarcy, "f1": f1, "recall": recall, "precision": precision}


def prepare_data(arguments):
    data_path = Path(arguments.data_dir).expanduser()
    train_df, test_df = split_dataset(data_path, random_state=55)
    train = StudyDataset(train_df)
    test = StudyDataset(test_df, training=False)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=16)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=16)
    return train_loader, test_loader, train, test


def resolve_device():
    if torch.cuda.is_available:
        found_device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
    else:
        found_device = torch.device("cpu")
        device_name = "cpu"
    print(f"Using device: {device_name}")
    return found_device

# todo structure
# overfitting?
# - super sample class in-balance with data loading

# --> calculate map


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training argparser')
    parser.add_argument("--ckpt", default=None, type=str)
    parser.add_argument("--train_loss", default=None, type=str)
    parser.add_argument("--val_loss", default=None, type=str)
    parser.add_argument("--test_results", default=None, type=str)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--initial_ckpt", default=False, type=bool)
    parser.add_argument("--data_dir", default=base_data_dir, type=str)
    parser.add_argument("--model_dir", default=model_folder, type=str)
    parser.add_argument("--loss_dir", default=loss_folder, type=str)
    parser.add_argument("--board_subdir", default=None, type=str)


    args = parser.parse_args()
    print(args)

    train_loader, test_loader, train_dataset, _ = prepare_data(args)
    # Create required directories
    Path(args.model_dir).mkdir(exist_ok=True)
    Path(args.loss_dir).mkdir(exist_ok=True)
    device = resolve_device()

    model = CovidModel()
    model.to(device)
    print(model)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True)
    # according to doi:10.1097/RTI.0000000000000541 classes are mutual exclusive
    criterion = nn.CrossEntropyLoss().to(device)
    scaler = torch.cuda.amp.GradScaler()
    softmax = torch.nn.Softmax(dim=1).to(device)

    # training scheduler
    lf = lambda x: (((1 + math.cos(x * math.pi / number_epochs)) / 2) ** 1.0) * 0.95 + 0.05
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt)
        if args.initial_ckpt:
            model.load_state_dict(ckpt, strict=False)
        else:
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

    board_log_dir = None
    if args.board_subdir is not None:
        board_log_dir = f"./runs/{args.board_subdir}"
    writer = SummaryWriter(log_dir=board_log_dir)

    for epoch in range(args.start_epoch, number_epochs+1):
        model.train()

        losses = []

        print(f"Training model on epoch {epoch} using lr={scheduler.get_last_lr()}")

        for batch_i, (imgs, labels) in enumerate(tqdm(train_loader, desc="Training")):

            imgs = imgs.to(device)
            labels = Variable(labels.to(device), requires_grad=False)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                out = model(imgs)
                loss = criterion(out, labels)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            losses.append(loss.item())
            writer.add_scalar(tag="Train/Loss/Batch", scalar_value=loss.item(), global_step=batch_i*(epoch+1))
            if batch_i % print_loss_frequency == 0 and batch_i != 0:
                tqdm.write(f'Epoch: {epoch}, step: {batch_i}; current loss: {np.mean(losses)}, improvement to previous loss: {np.mean(losses[:-print_loss_frequency]) - np.mean(losses[-print_loss_frequency:])}')
            elif batch_i == 0:
                tqdm.write(f'Start loss: {np.mean(losses)}')

        # Perform operations after every epoch
        if epoch % scheduler_frequency == 0:
             scheduler.step()

        loss_per_epoch.append(np.mean(losses))
        writer.add_scalar(tag="Train/Loss/Epoch", scalar_value=loss_per_epoch[-1], global_step=epoch)

        if epoch % test_frequency == 0:
                model.eval()
                with torch.no_grad(), torch.cuda.amp.autocast():
                    predictions = []
                    targets = []
                    validation_loss = []
                    for test_i, (imgs, labels) in enumerate(tqdm(test_loader, desc="Testing")):
                        imgs = imgs.to(device)
                        labels = Variable(labels.to(device), requires_grad=False)
                        batch_logits = model(imgs)
                        batch_predictions = softmax(batch_logits)
                        val_loss = criterion(batch_logits, labels)
                        validation_loss.append(val_loss.item())
                        predictions.extend(batch_predictions.cpu().numpy())
                        targets.extend(labels.cpu().numpy())
                        writer.add_scalar(tag="Validation/Loss/Batch", scalar_value=val_loss.item(),
                                          global_step=test_i * (epoch + 1))

                    test_results = calculate_metrics(np.array(predictions), np.array(targets))
                    test_results_general.append(test_results)
                    val_loss_per_epoch.append(np.mean(validation_loss))
                    writer.add_scalar(tag="Validation/Loss/Epoch", scalar_value=val_loss_per_epoch[-1], global_step=epoch)
                    writer.add_scalars(main_tag="Validation/Metrics/Epoch", tag_scalar_dict=test_results, global_step=epoch)
                    print(f'\nValidation loss: {val_loss_per_epoch[-1]}')
                    print(f'\n{test_results}')

        if epoch % save_frequency == 0 or epoch == number_epochs:
            torch.save(model.state_dict(), f"{args.model_dir}/{model_name}_epoch_{epoch}.pt")
            torch.save({"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                        "scheduler": scheduler.state_dict()}, f"{args.model_dir}/{model_name}_epoch_{epoch}_ckpt.pt")
            np.save(f"{args.loss_dir}/{model_name}_train_loss_{epoch}.np", np.array(loss_per_epoch))
            np.save(f"{args.loss_dir}/{model_name}_val_loss_{epoch}.np", np.array(val_loss_per_epoch))
            with open(f"{args.loss_dir}/{model_name}_general_test_results_{epoch}.pickle", "wb") as p:
                pickle.dump(test_results_general, p)

    writer.close()
    del model
    del optimizer
    torch.cuda.empty_cache()
