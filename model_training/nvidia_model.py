"""
CNN that processes duckiebot image data and returns the predicted
linear and angular velocity controls that should be applied.

For overall idea see:
    End to End Learning for Self-Driving Cars 2016
    https://arxiv.org/abs/1604.07316

Rather than use a single net or have dual output (like for modelv0) we use
a seperate net for each of linear and angular velocity.
"""
import torch
from torch import nn


class NvidiaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(24, 36, 5, 2),
            nn.LeakyReLU(),
            nn.Conv2d(36, 48, 5, 2),
            nn.LeakyReLU(),
            nn.Conv2d(48, 64, 3, 1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.MaxPool2d(3),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(1536, 100),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 50),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(50, 10),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(10, 1),
        )

    def forward(self, x):
        # x is in format (N, W, H, C), but pytorch conv layers expect (N, C, H, W), tranpose to fix it.
        x = x.transpose(3, 1)
        # Normalized to range [-1, 1], this is a standard trick to reduce noise.
        x = (x / 255. - 0.5) * 2

        # Get rid of last (,1) dimension so that tensors align for loss computation.
        out = self.net(x).squeeze(-1)

        return out
