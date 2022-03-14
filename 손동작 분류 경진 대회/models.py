import torch
import torch.nn.functional as F
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # conv11
        self.conv11 = nn.Conv2d(1, 64, 2)
        self.bn11 = nn.BatchNorm2d(64)

        # conv12
        self.conv12 = nn.Conv2d(64, 16, 2)
        self.bn12 = nn.BatchNorm2d(16)

        # conv21
        self.conv21 = nn.Conv2d(16, 112, 2)
        self.bn21 = nn.BatchNorm2d(112)

        self.fc = nn.Linear(560, 4)

    def forward(self, x):
        x = F.relu(self.conv11(x))
        x = self.bn11(x)

        x = F.relu(self.conv12(x))
        x = self.bn12(x)

        x = F.relu(self.conv21(x))
        x = self.bn21(x)

        x = self.fc(x.view(x.shape[0], -1))
        x = F.softmax(x, dim = 1)
        return x