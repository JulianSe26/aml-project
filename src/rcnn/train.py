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


#logging.basicConfig(level=logging.DEBUG)

dataset = NIHDataset()

train_len = int(.8 * len(dataset))
test_len = len(dataset) - train_len
train, test = random_split(dataset, [train_len, test_len])

train_loader = DataLoader(train, batch_size=24, shuffle=True, pin_memory=True)
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
criterion = nn.BCELoss()

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

        loss.backward()

        optimizer.step()

        if batch_i % 10 == 0:
            print(np.mean(losses))

    print(np.mean(losses))

 
        
