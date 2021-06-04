"""
Almost the same as Modelv0 but uses LeakyReLU and centers linear velocity around 0
and angular velocity predicts the sine of angular velocity.

Results:
    - simulated+maserati+baselinesim dataset
"""
import torch
from torch import nn

class Model3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.common3d = nn.Sequential(
            nn.Conv3d(128, 128, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(128, 64, 3, padding=1),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Conv3d(64, 32, 3, padding=1),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Conv3d(32, 16, 3, padding=1),
            nn.LeakyReLU(),
        )
        self.common = nn.Sequential(
            nn.Conv2d(48, 48, 3),
            nn.LeakyReLU(),
            nn.Conv2d(48, 48, 3),
            nn.MaxPool2d(2),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(48, 48, 3),
            nn.MaxPool2d(2),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(48, 48, 3),
            nn.MaxPool2d(2),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(48, 64, 3),
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
        # # x is in format (N, S, W, H, C), but pytorch conv layers expect (N, S, C, H, W), tranpose to fix it.
        x = x.transpose(4, 2)
        # # Normalized to range [-1, 1], this is a standard trick to reduce noise.
        x = (x / 255. - 0.5) * 2

        x = self.common3d(x)
        x = x.view(x.shape[0], -1, x.shape[3], x.shape[4])
        x = self.common(x)
        lin_vel = self.lin_vel(x)
        ang_vel = self.ang_vel(x)

        # Get rid of last (,1) dimension so that tensors align for loss computation.
        return lin_vel.squeeze(-1), ang_vel.squeeze(-1)
