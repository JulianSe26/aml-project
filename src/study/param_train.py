import argparse
import importlib
import pickle
from pathlib import Path

import numpy as np
import torch.cuda.amp
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import CovidModel
from utils import prepare_data, resolve_device, calculate_metrics

base_data_dir = "/home/tkrieger/var/aml-xrays"
model_folder = "/home/tkrieger/var/aml-models/study"
metrics_folder = "./metrics"
summary_dir = "./runs/"

save_frequency = 2          # in epochs
base_metric_stub = None
base_model_stub = None


def parse_args():
    parser = argparse.ArgumentParser(description='Training command line argument parser')
    parser.add_argument("--config_module", default=None, type=str, required=True)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--initial_ckpt", default=False, type=bool)
    parser.add_argument("--data_dir", default=base_data_dir, type=str)
    parser.add_argument("--model_dir", default=model_folder, type=str)
    parser.add_argument("--metrics_dir", default=metrics_folder, type=str)
    parser.add_argument("--summary_dir", default=summary_dir, type=str)
    args = parser.parse_args()

    Path(args.model_dir).mkdir(exist_ok=True)
    Path(args.metrics_dir).mkdir(exist_ok=True)
    return args


def load_checkpoint(args, model, optimizer, scheduler, scaler):
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        if args.initial_ckpt:
            model.load_state_dict(checkpoint, strict=False)
            print("Initial checkpoint with pre-trained weights was loaded")
        else:
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint["scaler"])
            print("Checkpoint for continuing training was loaded")
        del checkpoint


def create_or_load_metric_buffers(args):
    train_loss_epoch = []
    validation_loss_epoch = []
    validation_metrics = []

    if args.start_epoch != 0:
        train_loss_epoch = np.load(base_metric_stub + f"{args.start_epoch-1}_train_loss.npy").tolist()
        validation_loss_epoch = np.load(base_metric_stub + f"{args.start_epoch-1}_validation_loss.npy").tolist()
        with open(f"{args.start_epoch-1}_validation_metrics.pickle", 'rb') as f:
            validation_metrics = pickle.load(f)

    return train_loss_epoch, validation_loss_epoch, validation_metrics


def create_dirs(args, config):
    global base_metric_stub, base_model_stub
    metrics_sub_dir = f'{args.metrics_dir}/{config["name"]}'
    Path(metrics_sub_dir).mkdir(exist_ok=True)
    base_metric_stub = f'{metrics_sub_dir}/study_resnext_{config["name"]}_epoch'
    models_sub_dir = f'{args.model_dir}/{config["name"]}'
    Path(models_sub_dir).mkdir(exist_ok=True)
    base_model_stub = f'{models_sub_dir}/study_resnext_{config["name"]}_epoch'


if __name__ == "__main__":
    args = parse_args()
    config = importlib.import_module(args.config_module).CONFIG
    create_dirs(args, config)
    print("Loaded config:", config)
    device = resolve_device()

    train_loader, test_loader, _, _ = prepare_data(args.data_dir, config['batch_size'])
    print("Data set was loaded")
    model = CovidModel().to(device)
    print("Model was loaded and moved to device")

    # create required training operators
    optimizer = config["optimizer"](model.parameters())
    scheduler = config["scheduler"](optimizer)
    loss_function = nn.CrossEntropyLoss().to(device)
    softmax = nn.Softmax(dim=1).to(device)
    scaler = torch.cuda.amp.GradScaler()

    load_checkpoint(args, model, optimizer, scheduler, scaler)
    train_loss_epoch, validation_loss_epoch, validation_metrics = create_or_load_metric_buffers(args)
    writer = SummaryWriter(log_dir=f"{args.summary_dir}/{config['name']}/")

    # TRAIN loop
    for epoch in range(args.start_epoch, config['epochs']):
        model.train()
        losses = []

        print(f"Training model on epoch {epoch} using lr={scheduler.get_last_lr()}")

        # start of a train epoch
        for batch_i, (images, targets) in enumerate(tqdm(train_loader, desc=f"Training - ep: {epoch}")):
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = loss_function(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # write loss infos
            losses.append(loss.item())
            writer.add_scalar(tag="Train/Loss/Batch", scalar_value=loss.item(), global_step=batch_i*(epoch+1))
            if batch_i == 0:
                tqdm.write(f'Start loss: {np.mean(losses)}')
        # end of a train epoch

        if epoch % config['scheduler_frequency'] == 0:
            scheduler.step()
        train_loss_epoch.append(np.mean(losses))
        writer.add_scalar(tag="Train/Loss/Epoch", scalar_value=train_loss_epoch[-1], global_step=epoch)

        # start of a VALIDATION epoch
        model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast():

            all_predictions = []
            all_targets = []
            all_validations_losses = []

            for test_i, (images, targets) in enumerate(tqdm(test_loader, desc=f"Testing - ep: {epoch}")):
                images = images.to(device)
                targets = targets.to(device)
                logits = model(images)
                batch_probabilities = softmax(logits)
                validation_loss = loss_function(logits, targets)

                # append to buffers
                all_validations_losses.append(validation_loss.item())
                all_predictions.extend(batch_probabilities.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                writer.add_scalar(tag="Validation/Loss/Batch", scalar_value=validation_loss.item(), global_step=test_i * (epoch + 1))

            validation_results, _ = calculate_metrics(np.array(all_targets), np.array(all_predictions))
            validation_metrics.append(validation_results)
            validation_loss_epoch.append(np.mean(all_validations_losses))
            writer.add_scalar(tag="Validation/Loss/Epoch", scalar_value=validation_loss_epoch[-1], global_step=epoch)
            writer.add_scalars(main_tag="Validation/Metrics/Epoch", tag_scalar_dict=validation_results, global_step=epoch)
            print(f'\nValidation loss: {all_validations_losses[-1]}')
            print(f'\n{validation_results}')
        # end of a VALIDATION epoch

        if epoch % save_frequency == 0 or epoch == config["epochs"]:
            torch.save(model.state_dict(), base_model_stub + f"{epoch}.pt")
            torch.save({"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                        "scheduler": scheduler.state_dict()}, base_model_stub + f"{epoch}_full.pt")
            np.save(base_metric_stub + f"{epoch}_train_loss", np.array(train_loss_epoch))
            np.save(base_metric_stub + f"{epoch}_val_loss", np.array(validation_loss_epoch))
            with open(base_metric_stub + f"{epoch}_validation_metrics.pickle", "wb") as p:
                pickle.dump(validation_metrics, p)

    writer.close()
    del model
    del optimizer
    torch.cuda.empty_cache()










