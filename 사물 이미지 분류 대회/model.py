import torch.nn as nn
import torchvision
import torch.nn.functional as F

class SRCNN_Resnet34(nn.Module):
    def __init__(self):
        super().__init__()
        resnet34 = torchvision.models.resnet34(pretrained=False)
        self.conv1 = nn.Conv2d(3, 64, 9, padding=2, padding_mode='replicate')
        self.conv2 = nn.Conv2d(64, 32, 1, padding=2, padding_mode='replicate')
        self.conv3 = nn.Conv2d(32, 3, 5, padding=2, padding_mode='replicate')
        self.layer1 = nn.Sequential(*list(resnet34.children())[:-1])
        self.fc = nn.Linear(512,10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.layer1(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x

class SRCNN_EFB0(nn.Module):
    def __init__(self):
        super().__init__()
        efb0 = torchvision.models.efficientnet_b0(pretrained=False)
        self.conv1 = nn.Conv2d(3, 64, 9, padding=2, padding_mode='replicate')
        self.conv2 = nn.Conv2d(64, 32, 1, padding=2, padding_mode='replicate')
        self.conv3 = nn.Conv2d(32, 3, 5, padding=2, padding_mode='replicate')
        self.layer1 = nn.Sequential(*list(efb0.children())[:-1])
        self.fc = nn.Linear(1280,10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.layer1(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x