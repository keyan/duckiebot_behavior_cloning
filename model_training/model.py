"""
A dual-output CNN that processes duckiebot image data and returns the predicted
linear and angular velocity controls that should be applied.

For overall idea see:
    End to End Learning for Self-Driving Cars 2016
    https://arxiv.org/abs/1604.07316
"""
import torch
from torch import nn


class VelocityModel(nn.Module):
    """
    This model uses a much smaller net than the NVIDA paper, and also shares
    the bulk of the net between both the linear and angular velocity outputs
    to reduce the training time.
    """
    def __init__(self):
        super().__init__()
        self.common = nn.Sequential(
            nn.Conv2d(3, 8, 3),
            nn.Conv2d(8, 16, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.lin_vel = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            # Don't want to go backwards ever, so prevent negative values.
            nn.ReLU(),
        )

        self.ang_vel = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x is in format (N, W, H, C), but pytorch conv layers expect (N, C, H, W), tranpose to fix it.
        x = x.transpose(3, 1)
        # Normalized to range [-1, 1], this is a standard trick to reduce noise.
        x = (x / 255.) * 2

        x = self.common(x)
        lin_vel = self.lin_vel(x)
        ang_vel = self.ang_vel(x)

        # Get rid of last (,1) dimension so that tensors align for loss computation.
        return lin_vel.squeeze(-1), ang_vel.squeeze(-1)
