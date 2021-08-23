import torch
from torch.utils import data
from torch.utils.data import DataLoader, random_split
from dataset import NIHDataset
import logging
import tqdm
from torch.autograd import Variable

from models import BackboneModel


#logging.basicConfig(level=logging.DEBUG)

dataset = NIHDataset()

train_len = int(.8 * len(dataset))
test_len = len(dataset) - train_len
train, test = random_split(dataset, [train_len, test_len])

train_loader = DataLoader(train, batch_size=5, shuffle=True)
test_loader = DataLoader(test, batch_size=5, shuffle=True)

if torch.cuda.is_available:
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
else:
    device = torch.device("cpu")
    device_name = "cpu"

logging.info(f'Using device: {device_name}')

model = BackboneModel(training=True)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
optimizer
t = torch.cuda.get_device_properties(0).total_memory
r = torch.cuda.memory_reserved(0)
a = torch.cuda.memory_allocated(0)
f = r-a  # free inside reserved

print(t)
print(r)
print(a)
print(f)

for epoch in range(10):
    model.train()

    for batch_i, (imgs, labels) in enumerate(tqdm.tqdm(train_loader, desc="Training")):
        imgs = imgs.to(device)
        labels = Variable(labels.to(device), requires_grad=False)
        out = model(imgs)
