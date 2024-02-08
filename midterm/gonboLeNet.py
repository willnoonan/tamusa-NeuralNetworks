from stdimports import *
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


class LeNet(nn.Module):
    def __init__(self, num_classes=10):
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

