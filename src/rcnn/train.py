import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import DataLoader, random_split
from dataset import NIHDataset
import logging
import tqdm
from torch.autograd import Variable
import numpy as np

from models import BackboneModel

#from torch.profiler import profile, record_function, ProfilerActivity


#logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':
    dataset = NIHDataset()

    train_len = int(.8 * len(dataset))
    test_len = len(dataset) - train_len
    train, test = random_split(dataset, [train_len, test_len])

    train_loader = DataLoader(train, batch_size=24, shuffle=True, pin_memory=True, num_workers=12)
    test_loader = DataLoader(test, batch_size=24, shuffle=True)

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

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=2),
        on_trace_ready=trace_handler
    ) as p:'''
    for epoch in range(10):
        model.train()

        losses = []

        for batch_i, (imgs, labels) in enumerate(tqdm.tqdm(train_loader, desc="Training")):

            optimizer.zero_grad()

            imgs = imgs.to(device)
            labels = Variable(labels.to(device), requires_grad=False)
            with torch.cuda.amp.autocast():
                out = model(imgs)
                loss = criterion(out, labels)

            losses.append(loss.item())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if batch_i % 500 == 0:
                print(np.mean(losses))

        print(np.mean(losses))
        #p.step()

 
        
