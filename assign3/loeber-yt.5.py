import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device '{device}'")

# Hyper-parameters
INPUT_SIZE = 32 * 32 * 3
HIDDEN_SIZE = 500
NUM_CLASSES = 100
NUM_EPOCHS = 2
batch_size = 100
LEARNING_RATE = 0.001

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
                                           batch_size=batch_size,
                                           shuffle=True)

TEST_LOADER = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

examples = iter(TEST_LOADER)
example_data, example_targets = next(examples)

print("Shape of X [N, C, H, W]:", example_data.shape)
print("Shape of Y:", example_targets.shape)

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(example_data[i][0], cmap='gray')


# plt.show()


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, layer_stack):
        super(NeuralNet, self).__init__()
        self.flatten = nn.Flatten()
        self.layer_stack = layer_stack

    def forward(self, x):
        x = self.flatten(x)
        out = self.layer_stack(x)
        # no activation and no softmax at the end
        return out


"""
Train/Test methods
"""

# Train the model
def train(train_loader, model, loss_func, optimizer):
    n_total_steps = len(train_loader)
    for epoch in range(NUM_EPOCHS):
        for i, (images, labels) in enumerate(train_loader):
            # origin shape: [100, 1, 28, 28]
            # resized: [100, 784]
            images = images.reshape(-1, INPUT_SIZE).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = loss_func(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')


# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
def test(test_loader, model):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            images = images.reshape(-1, INPUT_SIZE).to(device)
            labels = labels.to(device)
            outputs = model(images)
            # max returns (value ,index)
            _, predicted = torch.max(outputs.train_data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network on the 10000 test images: {acc} %')


"""
Experiments
"""

linear_layer_stack = nn.Sequential(
    nn.Linear(INPUT_SIZE, HIDDEN_SIZE),  # first layer
    nn.ReLU(),  # non-linear activation
    nn.Linear(HIDDEN_SIZE, NUM_CLASSES),  # 2nd layer
    # nn.ReLU(),
    # nn.Linear(512, num_classes)  # 3rd layer
)

model = NeuralNet(linear_layer_stack).to(device)

print(f'Model: {model}')

# Loss and optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  # TODO try torch.optim.SGD

# train the model
train(TRAIN_LOADER, model, loss_func, optimizer)

# test the model
test(TEST_LOADER, model)