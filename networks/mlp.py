import sys
import torch
import torch.nn.functional as F
import torch.nn as nn


class MLP(nn.Module): 
    def __init__(self): 
        super(MLP, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.encoder = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            self.conv2,
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten(),
            self.fc1,
            nn.ReLU()
        )
    def forward(self, x):
        return self.encoder(x)


class SupConMLP(nn.Module):
    def __init__(self, name='mlp', feat_dim=500):
        super(SupConMLP, self).__init__()
        self.encoder = MLP()
        self.head = nn.Sequential(
                nn.Linear(500, 500),
                nn.ReLU(inplace=True),
                nn.Linear(500, feat_dim)
            )
    def forward(self, x):
        feat = self.encoder(x)
        logits = self.head(feat)
        return feat, logits 

class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self,name="mlp", num_classes=10):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(500, num_classes)

    def forward(self, features):
        return self.fc(features)