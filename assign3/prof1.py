import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import tqdm
import matplotlib.pyplot as plt

## Create the dataset ##
training_data = datasets.FashionMNIST(root="data", train=True,
                                      download=True, transform=ToTensor())
test_data = datasets.FashionMNIST(root="data", train=False,
                                  download=True, transform=ToTensor())
## Create data loader ##
batch_size = 32  # samples, how much data we want to load
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

##Downloader Some Data for Visulization##
# for data in train_dataloader:
#     break
#
# X = data[0]
# Y = data[1]

X, Y = next(iter(train_dataloader))

print("Shape of X [N, C, H, W]:", X.shape)  # [batch size, num channels (3 for RGB image),
print("Shape of Y:", Y.shape)

plt.figure(figsize=(8, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(X[i, 0, :, :], cmap="gray")
    plt.title(f"Label: {Y[i]}")

plt.tight_layout()
plt.show()

import sys


##Define MLP Model##
class MLP(nn.Module):
    # define the structure of the model
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 4096),  # first layer
            nn.ReLU(),  # non-linear activation
            nn.Linear(4096, 512),  # 2nd layer
            nn.ReLU(),
            nn.Linear(512, 10)  # 3rd layer
        )

    # define data passing flow
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


##Define the TRAIN operation##
def train(dataloader, model, loss_fn, optimizer, device):
    model.train()  # set model to train model
    for step, (X, y) in enumerate(dataloader):
        # send data to GPU or CPU
        X = X.to(device)
        y = y.to(device)
        # feed the data to model
        pred = model(X)
        # compute the loss
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()  # compute the gradients
        optimizer.step()  # update the parameters/weights
        if step % 100 == 0:
            loss = loss.item()
            print('current step:%d, loss:%.4f' % (step, loss))


##Define the TEST/Evaluation operation##
def test(dataloader, model, loss_fn, device):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            test_loss += loss.item()

            y_hat = pred.argmax(1)
            correct_batch = (y_hat == y).type(torch.float).sum().item()
            correct += correct_batch

    test_loss /= num_batches
    correct = correct / (num_batches * batch_size)
    print("Test Accuracy:%.4f" % correct)


##Train the model##
# get cpu or gpu for training
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using %s device" % device)
# create the model
model = MLP().to(device)
print(model)
# optimizing the model parameter
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
# train model in epochs
epochs = 5
for t in tqdm.tqdm(range(epochs)):
    print('Epoch %d \n---------------------' % t)
    train(train_dataloader, model, loss_fn, optimizer, device)
    test(test_dataloader, model, loss_fn, device)
print("Done!")

## Visualize the train result
# vis testing result
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Test the model for 1 batch
model.eval()
for X, y in test_dataloader:
    X = X.to(device)
    y = y.to(device)
    with torch.no_grad():
        pred = model(X)
        pred_labels = pred.argmax(1)
    break

# Visualize the result
plt.figure(figsize=(15, 18))
for i in range(25):
    y_hat = pred_labels[i].item()
    y_gt = y[i].item()
    plt.subplot(5, 5, i + 1)
    plt.imshow(X[i, 0, :, :].cpu().numpy(), cmap="gray")
    plt.title('Pred Label:%s\nGL Label:%s' % (classes[y_hat], classes[y_gt]))

plt.tight_layout()
plt.show()


##count_parameters##
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


count_parameters(model)
