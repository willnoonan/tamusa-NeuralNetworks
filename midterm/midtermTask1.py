import numpy as np

from stdimports import *

"""
Author: William Noonan

Code for task 1, from-scratch CNN.

4 May 2023 Note: this is the file that I turned in
"""

BATCH_SIZE = 32

# CIFAR100 dataset
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


class NeuralNet(nn.Module):
    def __init__(self):
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
                      out_channels=12,  # how many filters
                      kernel_size=5,  # means (5x5) kernel size
                      stride=1,
                      padding=2),
            nn.ReLU(),  # each layer followed by nonlinear activation
            nn.AvgPool2d(kernel_size=2, stride=2)  # 2x2 pooling layer
        )

        # 3rd conv layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=12,
                      out_channels=24,  # how many filters
                      kernel_size=5,  # means (5x5) kernel size
                      stride=1,
                      padding=2),
            nn.ReLU(),  # each layer followed by nonlinear activation
        )

        # 4th conv layer
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=24,
                      out_channels=48,  # how many filters
                      kernel_size=3,  # 3x3 kernel size
                      stride=1,
                      padding=2),
            nn.ReLU(),  # each layer followed by nonlinear activation
        )

        # FC layers
        self.flatten = nn.Flatten()

        self.fc1 = nn.Sequential(
            nn.Linear(48 * 10 * 10, 120),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )

        self.fc3 = nn.Linear(84, 100)  # last FC layer not followed by ReLU

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


##Define the TRAIN operation##
def train(dataloader, model, loss_fn, optimizer, device, verbose=True):
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
        if verbose and step % 100 == 0:
            loss = loss.item()
            print('current step:%d, loss:%.4f' % (step, loss))
    return model


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

    accuracy = correct / (num_batches * BATCH_SIZE)
    print("Test Accuracy: {:.2%}".format(accuracy))
    return test_loss, accuracy


##Train the model##
# get cpu or gpu for training
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using %s device" % device)

# create the model
model = NeuralNet().to(device)
print(model)

# optimizing the model parameter
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

# train model in epochs
epochs = 10
train_hist = []
test_hist = []
for t in range(epochs):
    print('Epoch %d \n---------------------' % (t + 1))
    # Train
    model = train(TRAIN_LOADER, model, loss_fn, optimizer, device)

    # Test
    train_loss, train_accuracy = test(TRAIN_LOADER, model, loss_fn, device)
    test_loss, test_accuracy = test(TEST_LOADER, model, loss_fn, device)
    train_hist.append([train_loss, train_accuracy])
    test_hist.append([test_loss, test_accuracy])
print("Done!")

train_hist = np.asarray(train_hist)
test_hist = np.asarray(test_hist)

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(train_hist[:, 0], ls='-', marker='.')
plt.plot(test_hist[:, 0], ls='-', marker='.')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(('train', 'test'))

plt.subplot(1, 2, 2)
plt.plot(train_hist[:, 1], ls='-', marker='.')
plt.plot(test_hist[:, 1], ls='-', marker='.')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(('train', 'test'))

plt.tight_layout()
plt.show()
