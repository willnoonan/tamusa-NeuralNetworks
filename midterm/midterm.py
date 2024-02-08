import numpy as np
import torch.optim

from stdimports import *
import copy

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


"""
Load and transform the data
"""
# For loading data
BATCH_SIZE = 50

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


class NeuralNet(nn.Module):
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

        self.fc3 = nn.Linear(84, 100)

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



class CIFAR100_Test:
    def __init__(self, epochs, plot=True):
        self.train_loader = TRAIN_LOADER
        self.test_loader = TEST_LOADER

        self.plot = plot

        self.input_size = INPUT_SIZE = 32 * 32 * 3
        self.epochs = epochs
        self.device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = NeuralNet().to(device)
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer_func = torch.optim.SGD(self.model.parameters(), lr=1e-2)

    def run(self):
        print(f"Using device '{self.device}'")
        perf, epoch_loss = self.train()
        if self.plot:
            x = [i+1 for i in range(len(perf))]
            plt.subplot(1, 2, 1)
            plt.plot(x, perf, ls='-', marker='.')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy %')

            plt.subplot(1,2, 2)
            plt.plot(x, epoch_loss, ls='-', marker='.')
            # plt.plot(*zip(*step_loss))
            plt.xlabel('Epochs')
            plt.ylabel('Loss')

            plt.tight_layout()
            plt.show()

    # Train the model
    def train(self):
        model = copy.deepcopy(self.model)  # must deep copy to cut refs

        model = model.train()
        train_loader = self.train_loader
        device = self.device
        n_total_steps = len(train_loader)

        epoch_loss = []
        epoch_acc = []

        loss_item = None
        for epoch in range(self.epochs):
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = self.loss_func(outputs, labels)

                # Backward and optimize
                self.optimizer_func.zero_grad()
                loss.backward()
                self.optimizer_func.step()

                if (i + 1) % 100 == 0:
                    loss_item = loss.item()
                    print(f'Epoch [{epoch + 1}/{self.epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss_item:.4f}')

            # at end of epoch
            acc = self.test(model)
            epoch_acc.append(acc)
            epoch_loss.append(loss_item)
        return epoch_acc, epoch_loss


    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    def test(self, model):
        device = self.device
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            test_loader = self.test_loader
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                # max returns (value ,index)
                _, predicted = torch.max(outputs.train_data, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

            acc = 100.0 * n_correct / n_samples
            print(f'Accuracy of the network on the 10000 test images: {acc} %')
        return acc



exp = CIFAR100_Test(epochs=10)
exp.run()
