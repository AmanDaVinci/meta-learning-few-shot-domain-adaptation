import torch
import torch.nn as nn
from typing import Union, List, Dict


def accuracy(y_pred: torch.Tensor, y: torch.Tensor) -> float:
    return (y_pred.argmax(dim=1) == y).float().mean().item()


class Classifier(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.layers(x)