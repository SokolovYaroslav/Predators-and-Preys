import itertools
from dataclasses import dataclass
from typing import Optional, List, Iterable, Union

import numpy as np
import torch


def clip_direction(directions: torch.Tensor) -> torch.Tensor:
    return torch.clamp(directions, -1, +1)


@dataclass
class EntityState:
    x_pos: float
    y_pos: float
    radius: float
    speed: Optional[float] = None
    is_alive: Optional[bool] = None

    def dist(self, other: "EntityState") -> float:
        return ((self.x_pos - other.x_pos) ** 2 + (self.y_pos - other.y_pos) ** 2) ** 0.5

    def direction(self, other: "EntityState") -> float:
        return np.arctan2(other.y_pos - self.y_pos, other.x_pos - self.x_pos) / np.pi


@dataclass
class GameState:
    predators: List[EntityState]
    preys: List[EntityState]
    obstacles: List[EntityState]

    def __init__(self, dict_state: dict):
        self.predators = []
        for pred in dict_state["predators"]:
            self.predators.append(EntityState(pred["x_pos"], pred["y_pos"], pred["radius"], pred["speed"]))
        self.preys = []
        for prey in dict_state["preys"]:
            self.preys.append(
                EntityState(prey["x_pos"], prey["y_pos"], prey["radius"], prey["speed"], prey["is_alive"])
            )
        self.obstacles = []
        for obstacle in dict_state["obstacles"]:
            self.obstacles.append(EntityState(obstacle["x_pos"], obstacle["y_pos"], obstacle["radius"]))

    @property
    def observation(self) -> np.ndarray:
        obs = np.array(
            list(
                itertools.chain(
                    (pred.speed for pred in self.predators),
                    (prey.speed for prey in self.preys),
                    GameState.dists(self.predators, self.preys),
                    GameState.directions(self.predators, self.preys),
                    GameState.dists(self.predators, self.obstacles),
                    GameState.directions(self.predators, self.obstacles),
                    GameState.dists(self.preys, self.obstacles),
                    GameState.directions(self.preys, self.obstacles),
                    (float(prey.is_alive) for prey in self.preys),
                )
            )
        )
        return obs

    @staticmethod
    def dists(xs: Iterable[EntityState], ys: Iterable[EntityState]) -> Iterable[float]:
        return (x.dist(y) for x in xs for y in ys)

    @staticmethod
    def directions(xs: Iterable[EntityState], ys: Iterable[EntityState]) -> Iterable[float]:
        return (x.direction(y) for x in xs for y in ys)
