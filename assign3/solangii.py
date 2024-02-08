import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import tqdm
import matplotlib.pyplot as plt

'''
MLP
'''
class MLP(nn.Module):
    def __init__(self, D_out=10):
        super().__init__()
        self.linear1 = nn.Linear(3072, 200)
        self.linear2 = nn.Linear(200, 200)
        self.linear3 = nn.Linear(200, 200)
        self.linear4 = nn.Linear(200, D_out)

    def forward(self, x):
        x = x.view(-1,3*32*32)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.softmax(self.linear4(x))
        return x


def main():
    ## Create the dataset ##
    training_data = datasets.CIFAR100(root="data", train=True,
                                      download=True, transform=ToTensor())
    test_data = datasets.CIFAR100(root="data", train=False,
                                  download=True, transform=ToTensor())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using %s device" % device)

    model = MLP(100)
    model = model.to(device)

    # Define Loss, optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)

    for epoch in range(config.epochs):
        s = time.time()
        train(epoch, trainloader, config.DEVICE, model, optimizer, criterion)
        val(epoch, valloader, config.DEVICE, model, criterion, exp_name)
        f = time.time()
        rt = f - s
        run_time.append(rt)


if __name__ == "__main__":
    main()
