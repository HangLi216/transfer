import sys
import torch
import torch.nn.functional as F
import torch.nn as nn


class MLP(nn.Module): 
    def __init__(self): 
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 100)
        self.fc2 = nn.Linear(100, 100)

        self.encoder = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU(),
        ) 
    def forward(self, x):
        return self.encoder(x)


class SupConMLP(nn.Module):
    def __init__(self, name='mlp', feat_dim=100):
        super(SupConMLP, self).__init__()
        self.encoder = MLP()
        self.head = nn.Sequential(
                nn.Linear(100, feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim, feat_dim)
            )
    def forward(self, x):
        x = x.view(x.shape[0],-1)
        feat = self.encoder(x)
        logits = self.head(feat)
        return feat, logits 

class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self,name="mlp", num_classes=10):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(100, num_classes)

    def forward(self, features):
        return self.fc(features)