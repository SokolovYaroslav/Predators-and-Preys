import math

import torch
from torch import nn


class ImitationModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )
        # self.model = nn.Sequential(
        #     nn.Linear(input_dim, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 2),
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
        # xy = self.model(x)
        # return torch.atan2(xy[:, 0], xy[:, 1]) / math.pi
