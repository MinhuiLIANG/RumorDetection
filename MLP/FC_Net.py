import torch
import torch.nn as nn
import torch.nn.functional as F


class FC(nn.Module):

    def __init__(self, vec_size):
        super(FC, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(vec_size, 768), nn.BatchNorm1d(768))
        self.fc2 = nn.Sequential(nn.Linear(768, 64), nn.BatchNorm1d(64))
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = F.sigmoid(self.out(x))

        return out

