import math
import os
from typing import List

import torch
from torch import nn


class ImitationModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        # self.model = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 1), nn.Tanh())
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return self.model(x)
        xy = self.model(x)
        return torch.atan2(xy[:, 0], xy[:, 1]) / math.pi


def prepare_xs(state: dict, predator: bool) -> List[List[float]]:
    features_list = []

    if predator:
        common_features = []
        for prey in state["preys"]:
            common_features.extend((prey["x_pos"], prey["y_pos"]))
        for pred in state["predators"]:
            features_list.append(common_features + [pred["x_pos"], pred["y_pos"], pred["speed"]])
    else:
        common_features = []
        for pred in state["predators"]:
            common_features.extend((pred["x_pos"], pred["y_pos"]))
        for prey in state["preys"]:
            features_list.append(common_features + [prey["x_pos"], prey["y_pos"], prey["speed"]])

    return features_list


class PredatorAgent:
    def __init__(self):
        state_dict = torch.load(os.path.join(__file__[:-13], "pred.pth"), map_location="cpu")
        self.model = ImitationModel(state_dict["model.0.weight"].size(1))
        self.model.load_state_dict(state_dict)

    def act(self, state_dict):
        xs = torch.tensor(prepare_xs(state_dict, predator=True), dtype=torch.float)
        return self.model(xs).cpu().detach().numpy()


class PreyAgent:
    def __init__(self):
        state_dict = torch.load(os.path.join(__file__[:-13], "prey.pth"), map_location="cpu")
        self.model = ImitationModel(state_dict["model.0.weight"].size(1))
        self.model.load_state_dict(state_dict)

    def act(self, state_dict):
        xs = torch.tensor(prepare_xs(state_dict, predator=False), dtype=torch.float)
        return self.model(xs).cpu().detach().numpy()
