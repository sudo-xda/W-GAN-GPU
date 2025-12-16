import torch.nn as nn
import torch


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()

        self.layer1 = nn.Conv2d(1, 64, 9, padding=4)
        self.layer2 = nn.Conv2d(64, 32, 5, padding=2)
        self.layer3 = nn.Conv2d(32, 1, 5, padding=2)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return torch.clamp(x, 0, 1)


