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

#from torch.profiler import profile, record_function, ProfilerActivity


#logging.basicConfig(level=logging.DEBUG)

'''================Train Configuration========================='''
number_epochs = 35
save_frequency = 2          # in epochs
test_frequency = 1          # in epochs
print_loss_frequency = 100  # in iterations
'''============================================================'''    


def calculate_metrics(predictions, targets, threshold=.5):
        predictions = np.array(predictions > threshold, dtype=float)
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

    train_loader = DataLoader(train, batch_size=24, shuffle=True, pin_memory=True, num_workers=12)
    test_loader = DataLoader(test, batch_size=24, shuffle=True)

    dataset = NIHDataset()

    if torch.cuda.is_available:
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
    else:
        device = torch.device("cpu")
        device_name = "cpu"

    logging.info(f'Using device: {device_name}')

    model = BackboneModel(training=True)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss().to(device)

    scaler = torch.cuda.amp.GradScaler()


    '''def trace_handler(p):
        output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
        print(output)
        p.export_chrome_trace("trace_" + str(p.step_num) + ".json")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()


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
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, number_epochs-1)
    test_results_general = []
    loss_per_epoch = []

    for epoch in range(10):
        model.train()

        losses = []

        for batch_i, (imgs, labels) in enumerate(tqdm.tqdm(train_loader, desc="Training")):
            print(f"Training model on epoch {epoch} using lr={scheduler_cosine.get_last_lr()}")

            imgs = imgs.to(device)
            labels = Variable(labels.to(device), requires_grad=False)
            with torch.cuda.amp.autocast():
                    out = model(imgs)
                    loss = criterion(out, labels)


            losses.append(loss.item())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            #p.step()
            optimizer.step()
            optimizer.zero_grad()


            if batch_i % print_loss_frequency == 0:
                print(np.mean(losses))

        # Perform operations after every epoch
        scheduler_cosine.step()


        loss_per_epoch.append(np.mean(losses))

        if epoch % test_frequency == 0:
                print("Starting Testing..")
                model.eval()
                with torch.no_grad():
                    predictions = []
                    targets = []
                    for test_i, (imgs, labels) in enumerate(tqdm.tqdm(test_loader, desc="Testing")):
                        imgs = imgs.to(device)
                        batch_prediction = model(imgs)
                        predictions.extend(batch_prediction.cpu().numpy())
                        targets.extend(labels.cpu().numpy())
                    
                    test_results = calculate_metrics(np.array(predictions), np.array(targets))
                    test_results_general.append(test_results)
                    print(test_results)

        if epoch % save_frequency == 0:
            torch.save(model.state_dict(), f"./models/resnext101_32x8d_epoch_{epoch}.pt")
            np.save(f"loss_{epoch}.np", np.array(loss_per_epoch))


del model
del optimizer
torch.cuda.empty_cache()
