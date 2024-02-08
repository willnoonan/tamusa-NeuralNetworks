"""
William Noonan
MLP
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import copy


"""
Load and transform the data
"""
# For loading data
BATCH_SIZE = 100

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

# examples = iter(TEST_LOADER)
# example_data, example_targets = next(examples)
#
# print("Shape of X [N, C, H, W]:", example_data.shape)
# print("Shape of Y:", example_targets.shape)
#
# for i in range(6):
#     plt.subplot(2, 3, i + 1)
#     plt.imshow(example_data[i][0], cmap='gray')

# plt.show()


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, nn_layer_stack):
        super(NeuralNet, self).__init__()
        self.flatten = nn.Flatten()
        self.nn_layer_stack = nn_layer_stack

    def forward(self, x):
        x = self.flatten(x)
        out = self.nn_layer_stack(x)
        # no activation and no softmax at the end
        return out


"""
Train/Test methods
"""
class CIFAR100_Test:
    def __init__(self, input_size, nn_stack, epochs, optimizer_func, loss_func, learning_rate):
        self.train_loader = TRAIN_LOADER
        self.test_loader = TEST_LOADER

        self.input_size = input_size
        self.epochs = epochs
        self.device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = NeuralNet(nn_stack).to(device)
        self.optimizer_func = optimizer_func
        self.loss_func = loss_func
        self.learning_rate = learning_rate

    def run(self, plot=True):
        print(self)
        print(f"Using device '{self.device}'")
        perf, epoch_loss = self.train()
        if plot:
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
        optimizer = self.optimizer_func(model.parameters(), lr=self.learning_rate)
        loss_func = self.loss_func()
        n_total_steps = len(train_loader)

        epoch_loss = []
        perf = []

        loss_item = None
        for epoch in range(self.epochs):
            for i, (images, labels) in enumerate(train_loader):
                # origin shape: [100, 1, 28, 28]
                # resized: [100, 784]
                images = images.reshape(-1, self.input_size).to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = loss_func(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i + 1) % 100 == 0:
                    loss_item = loss.item()
                    print(f'Epoch [{epoch + 1}/{self.epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss_item:.4f}')

            # at end of epoch
            acc = self.test(model)
            perf.append(acc)
            epoch_loss.append(loss_item)
        return perf, epoch_loss


    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    def test(self, model):
        device = self.device
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            test_loader = self.test_loader
            for images, labels in test_loader:
                images = images.reshape(-1, self.input_size).to(device)
                labels = labels.to(device)
                outputs = model(images)
                # max returns (value ,index)
                _, predicted = torch.max(outputs.train_data, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

            acc = 100.0 * n_correct / n_samples
            print(f'Accuracy of the network on the 10000 test images: {acc} %')
        return acc


    def __str__(self):
        header = "*" * 100
        string = [header,
                  f"Loss Function: {self.loss_func.__name__}",
                  f"Optimizer Function: {self.optimizer_func.__name__}",
                  f"Epochs: {self.epochs}",
                  f"Learning Rate: {self.learning_rate}",
                  f"Model: {self.model}",
                  header]
        return "\n".join(string)

    def __repr__(self):
        pass


"""
Experiments
"""

# Hyper-parameters
INPUT_SIZE = 32 * 32 * 3
NUM_CLASSES = 100

linear_layer_stack_1 = nn.Sequential(
    nn.Linear(INPUT_SIZE, 500),  # first layer
    nn.ReLU(),  # non-linear activation
    nn.Linear(500, NUM_CLASSES),  # 2nd layer
)

linear_layer_stack_1b = nn.Sequential(
    nn.Linear(INPUT_SIZE, 1000),  # first layer
    nn.ReLU(),  # non-linear activation
    nn.Linear(1000, NUM_CLASSES),  # 2nd layer
)

linear_layer_stack_2 = nn.Sequential(
    nn.Linear(INPUT_SIZE, 500),  # first layer
    nn.ReLU(),  # non-linear activation
    nn.Linear(500, 500),  # 2nd layer
    nn.ReLU(),
    nn.Linear(500, NUM_CLASSES)  # 3rd layer
)

linear_layer_stack_2b = nn.Sequential(
    nn.Linear(INPUT_SIZE, 4096),  # first layer
    nn.ReLU(),  # non-linear activation
    nn.Linear(4096, 500),  # 2nd layer
    nn.ReLU(),
    nn.Linear(500, NUM_CLASSES)  # 3rd layer
)

linear_layer_stack_1_sigmoid = nn.Sequential(
    nn.Linear(INPUT_SIZE, 500),  # first layer
    nn.Sigmoid(),  # non-linear activation
    nn.Linear(500, NUM_CLASSES),  # 2nd layer
)

"""
Experiment: # Layers
"""
#
exp1 = CIFAR100_Test(input_size=INPUT_SIZE, nn_stack=linear_layer_stack_1, epochs=2, optimizer_func=torch.optim.SGD,
                     loss_func=nn.CrossEntropyLoss, learning_rate=0.001)

#
exp2 = CIFAR100_Test(input_size=INPUT_SIZE, nn_stack=linear_layer_stack_2, epochs=2, optimizer_func=torch.optim.SGD,
                     loss_func=nn.CrossEntropyLoss, learning_rate=0.001)

"""
# Neurons (layers fixed)
"""
exp3 = CIFAR100_Test(input_size=INPUT_SIZE, nn_stack=linear_layer_stack_2b, epochs=2, optimizer_func=torch.optim.SGD,
                     loss_func=nn.CrossEntropyLoss, learning_rate=0.001)

exp3b = CIFAR100_Test(input_size=INPUT_SIZE, nn_stack=linear_layer_stack_1b, epochs=2, optimizer_func=torch.optim.SGD,
                     loss_func=nn.CrossEntropyLoss, learning_rate=0.001)

"""
Non-Lin Activation Func
"""
exp4 = CIFAR100_Test(input_size=INPUT_SIZE, nn_stack=linear_layer_stack_1_sigmoid, epochs=2, optimizer_func=torch.optim.SGD,
                     loss_func=nn.CrossEntropyLoss, learning_rate=0.001)

"""
Epochs
"""
exp5 = CIFAR100_Test(input_size=INPUT_SIZE, nn_stack=linear_layer_stack_1, epochs=4, optimizer_func=torch.optim.SGD,
                     loss_func=nn.CrossEntropyLoss, learning_rate=0.001)

"""
Learning Rate
"""
exp6 = CIFAR100_Test(input_size=INPUT_SIZE, nn_stack=linear_layer_stack_1, epochs=2, optimizer_func=torch.optim.SGD,
                     loss_func=nn.CrossEntropyLoss, learning_rate=0.01)

exp7 = CIFAR100_Test(input_size=INPUT_SIZE, nn_stack=linear_layer_stack_1, epochs=2, optimizer_func=torch.optim.SGD,
                     loss_func=nn.CrossEntropyLoss, learning_rate=0.0001)

"""
Optimizer
"""
exp8 = CIFAR100_Test(input_size=INPUT_SIZE, nn_stack=linear_layer_stack_1, epochs=2, optimizer_func=torch.optim.Adam,
                     loss_func=nn.CrossEntropyLoss, learning_rate=0.001)


"""
Best experiment based on previous ones
"""
expBest = CIFAR100_Test(input_size=INPUT_SIZE, nn_stack=linear_layer_stack_1b, epochs=10, optimizer_func=torch.optim.Adam,
                     loss_func=nn.CrossEntropyLoss, learning_rate=0.0001)


# Run the experiments:
# expmts = [exp1, exp2, exp3, exp4, exp5, exp6, exp7, exp8]
expmts = [expBest]
for exp in expmts:
    exp.run()