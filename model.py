"""
NeuralNetwork Model
Convolutional Neural Network:
                    28 x 28
Conv ->             32 * 28 x 28
MaxPool ->          32 * 14 x 14
Conv ->             64 * 14 x 14
MaxPool ->          64 * 7 x 7
Flatten ->          3136
Fully-connected ->  256
Fully-connected ->  10
"""

from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # Convolutional Layer
        self.conv = nn.Sequential(
            # Conv 28 x 28 -> 32 * 28 x 28
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(),
            # MaxPool 28 x 28 -> 14 x 14
            nn.MaxPool2d(2, 2),
            # Conv 32 * 14 x 14 -> 64 * 7 x 7
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            # MaxPool 14 x 14 -> 7 x 7
            nn.MaxPool2d(2, 2),
        )
        # Fully-connected Layer
        self.fc = nn.Sequential(
            # Flatten 64 * 7 x 7 -> 3136
            nn.Flatten(),
            # Fully-connected 3136 -> 256
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            # Fully-connected 256 -> 10
            nn.Linear(256, 10)
        )
    def forward(self, x):
        # 768 -> 28 x 28
        x = x.view(- 1, 1, 28, 28)
        x = self.conv(x)
        x = self.fc(x)
        return x
