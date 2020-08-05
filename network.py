import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(784, 30)
        self.fc2 = nn.Linear(30, 10)

    def forward(self, x):
        x = F.torch.sigmoid(self.fc1(x))
        x = F.torch.sigmoid(self.fc2(x))
        return x
