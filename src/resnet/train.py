from pathlib import Path
from numpy.lib.function_base import average
import torch
from torch import optim
import torch.nn as nn
from torch.utils import data
from torch.utils.data import DataLoader, random_split
from dataset import NIHDataset
import logging
import tqdm
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from models import BackboneModel
import math

#from torch.profiler import profile, record_function, ProfilerActivity


#logging.basicConfig(level=logging.DEBUG)

'''================Train Configuration========================='''
number_epochs = 35
save_frequency = 2          # in epochs
test_frequency = 1          # in epochs
print_loss_frequency = 500  # in iterations
'''============================================================'''   

'''=================Misc Configuration========================='''
model_folder = "./models"
loss_folder = "./losses"
checkpoint_folder = "./ckpt"
'''============================================================''' 


def calculate_metrics(predictions, targets, threshold=.5):
        predictions = np.array(predictions > threshold, dtype=float)
        print(predictions)
        print(targets)
        precision = precision_score(targets, predictions, average="macro")
        recall = recall_score(targets, predictions, average="macro")
        f1 = f1_score(targets, predictions, average="macro")
        accuarcy = accuracy_score(targets, predictions)
        return {"accuracy": accuarcy, "f1": f1, "recall": recall, "precision": precision}

if __name__ == '__main__':
    dataset = NIHDataset()

    train_len = int(.8 * len(dataset))
    test_len = len(dataset) - train_len
    train, test = random_split(dataset, [train_len, test_len])


    # Create required directories
    Path(model_folder).mkdir(exist_ok=True)
    Path(checkpoint_folder).mkdir(exist_ok=True)
    Path(loss_folder).mkdir(exist_ok=True)


    train_loader = DataLoader(train, batch_size=24, shuffle=True, pin_memory=True, num_workers=12)
    test_loader = DataLoader(test, batch_size=24, shuffle=True, pin_memory=True, num_workers=12)

    if torch.cuda.is_available:
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
    else:
        device = torch.device("cpu")
        device_name = "cpu"

    print(f'Using device: {device_name}')

    model = BackboneModel(training=True)
    model.to(device)

    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True)
    criterion = nn.BCEWithLogitsLoss().to(device)

    scaler = torch.cuda.amp.GradScaler()


    '''def trace_handler(p):
        output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
        print(output)
        p.export_chrome_trace("trace_" + str(p.step_num) + ".json")


for epoch in range(number_epochs):
    model.train()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=2),
        on_trace_ready=trace_handler
    ) as p:'''

    # training scheduler
    lf = lambda x: (((1 + math.cos(x * math.pi / number_epochs)) / 2) ** 1.0) * 0.95 + 0.05
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = 0

    test_results_general = []
    loss_per_epoch = []
    val_loss_per_epoch = []

    for epoch in range(number_epochs):
        model.train()

        losses = []

        print(f"Training model on epoch {epoch} using lr={scheduler.get_last_lr()}")

        for batch_i, (imgs, labels) in enumerate(tqdm.tqdm(train_loader, desc="Training")):

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

            if batch_i % print_loss_frequency == 0 and batch_i != 0:
                print(f'\ncurrent loss: {np.mean(losses)}, improvement to previous loss: {np.mean(losses[:-1]) - np.mean(losses)}')
            elif batch_i == 0:
                print(f'Start loss: {np.mean(losses)}')

        # Perform operations after every epoch
        if epoch % 5 == 0:
             scheduler.step()


        loss_per_epoch.append(np.mean(losses))

        if epoch % test_frequency == 0:
                model.eval()
                with torch.no_grad(), torch.cuda.amp.autocast():
                    predictions = []
                    targets = []
                    validation_loss = []
                    for test_i, (imgs, labels) in enumerate(tqdm.tqdm(test_loader, desc="Testing")):
                        imgs = imgs.to(device)
                        labels = Variable(labels.to(device), requires_grad=False)
                        batch_prediction = model(imgs)
                        val_loss = criterion(batch_prediction, labels)
                        validation_loss.append(val_loss.item())
                        predictions.extend(batch_prediction.cpu().numpy())
                        targets.extend(labels.cpu().numpy())
                    
                    test_results = calculate_metrics(np.array(predictions), np.array(targets))
                    test_results_general.append(test_results)
                    val_loss_per_epoch.append(np.mean(validation_loss))
                    print(f'\nValidation loss: {val_loss_per_epoch[-1]}')
                    print(f'\n{test_results}')
                    

        if epoch % save_frequency == 0:
            torch.save(model, f"./{model_folder}/resnext101_32x8d_epoch_{epoch}_full.pt")
            torch.save(model.state_dict(), f"./{model_folder}/resnext101_32x8d_epoch_{epoch}.pt")
            torch.save({"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict()}, f"./{checkpoint_folder}/resnext101_32x8d_epoch_{epoch}_ckpt.pt")
            np.save(f"{loss_folder}/resnext101_32x8d_train_loss_{epoch}.np", np.array(loss_per_epoch))
            np.save(f"{loss_folder}/resnext101_32x8d_val_loss_{epoch}.np", np.array(val_loss_per_epoch))


    del model
    del optimizer
    torch.cuda.empty_cache()
