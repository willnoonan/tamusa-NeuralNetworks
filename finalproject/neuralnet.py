import torch
from torch import nn


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using %s device" % DEVICE)


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(9, 12)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(12, 9)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(9, 1)
        self.act_output = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act_output(self.output(x))
        return x


def train(X, y, model, loss_fn, optimizer, batch_size):
    model.train()
    for i in range(0, len(X), batch_size):
        Xbatch = X[i:i + batch_size].to(DEVICE)
        ybatch = y[i:i + batch_size].to(DEVICE)
        y_pred = model(Xbatch)
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model  # latest loss is returned


def test(X, y, model, loss_fn):
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
    accuracy = (y_pred.round() == y).float().mean()
    return loss, accuracy


