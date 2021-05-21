"""
Almost the same as Modelv0 but uses LeakyReLU and centers linear velocity around 0
and angular velocity predicts the sine of angular velocity.

Results:
    - simulated+maserati+baselinesim dataset
"""
import torch
from torch import nn


class Modelv1(nn.Module):
    def __init__(self):
        super().__init__()
        self.common = nn.Sequential(
            nn.Conv2d(3, 8, 3),
            nn.Conv2d(8, 16, 3),
            nn.MaxPool2d(2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(16, 32, 3),
            nn.MaxPool2d(2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(32, 32, 3),
            nn.MaxPool2d(2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(32, 64, 3),
            nn.MaxPool2d(2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3),
            nn.MaxPool2d(2),
            nn.LeakyReLU(),
            nn.Flatten(),
        )

        self.lin_vel = nn.Sequential(
            nn.Linear(512, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
        )

        self.ang_vel = nn.Sequential(
            nn.Linear(512, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x is in format (N, W, H, C), but pytorch conv layers expect (N, C, H, W), tranpose to fix it.
        x = x.transpose(3, 1)
        # Normalized to range [-1, 1], this is a standard trick to reduce noise.
        x = (x / 255. - 0.5) * 2

        x = self.common(x)
        lin_vel = self.lin_vel(x)
        ang_vel = self.ang_vel(x)

        # Get rid of last (,1) dimension so that tensors align for loss computation.
        return lin_vel.squeeze(-1), ang_vel.squeeze(-1)
