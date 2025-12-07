import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # 32x32 -> 32x32
        self.pool1 = nn.MaxPool2d(2, 2)              # 32x32 -> 16x16
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # 16x16 -> 16x16
        self.pool2 = nn.MaxPool2d(2, 2)              # 16x16 -> 8x8
        self.fc1   = nn.Linear(64*8*8, 256)
        self.fc2   = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
