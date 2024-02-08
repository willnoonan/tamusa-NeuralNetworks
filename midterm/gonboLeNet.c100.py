import numpy as np

from stdimports import *

BATCH_SIZE = 32
"""
The model will have three layers
    Input shape: 32 x 32 x 3
    Conv Layers
        1st kernels: (5x5)x6  # (5x5) is kernel size, 6 is num filters
        2nd kernels: (5x5)x16
    
    FC Layers
        1st: 16x4x4, 120
        2nd: 84
        
    Output layer shape: 10

"""

# MNIST dataset
train_dataset = torchvision.datasets.CIFAR100(root='data',
                                              train=True,
                                              transform=transforms.ToTensor(),
                                              download=True)

test_dataset = torchvision.datasets.CIFAR100(root='data',
                                             train=False,
                                             transform=transforms.ToTensor())

# Data loader
TRAIN_LOADER = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)

TEST_LOADER = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=False)


class LeNet(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        # 1st conv layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=6,  # num of kernels or channels or filters
                      kernel_size=5,  # means (5x5) kernel size
                      stride=1,
                      padding=2),
            nn.ReLU(),  # each layer followed by nonlinear activation
            nn.AvgPool2d(kernel_size=2, stride=2)  # 2x2 pooling layer
        )

        # 2nd conv layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6,
                      out_channels=16,  # how many filters
                      kernel_size=5,  # means (5x5) kernel size
                      stride=1,
                      padding=2),
            nn.ReLU(),  # each layer followed by nonlinear activation
            nn.AvgPool2d(kernel_size=2, stride=2)  # 2x2 pooling layer
        )

        # FC layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 8 * 8, 120),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )

        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        logits = self.fc3(x)
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
    correct = correct / (num_batches * BATCH_SIZE) * 100
    print("Test Accuracy:%.4f" % correct)


##Train the model##
# get cpu or gpu for training
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using %s device" % device)
# create the model
model = LeNet().to(device)
print(model)
# optimizing the model parameter
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
# train model in epochs
epochs = 5
for t in range(epochs):
    print('Epoch %d \n---------------------' % (t+1))
    train(TRAIN_LOADER, model, loss_fn, optimizer, device)
    test(TEST_LOADER, model, loss_fn, device)
print("Done!")