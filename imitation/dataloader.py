import json
import random
from typing import List, Iterator, Tuple

import torch
from torch.utils.data.dataloader import get_worker_info
from torch.utils.data.dataset import IterableDataset


class ImitationDataset(IterableDataset):
    def __init__(self, predator: bool, path: str):
        self.predator = predator
        self.file_path = path
        self.len = 5217272
        self.rng = random.Random(42)

    def __len__(self):
        return self.len * (2 if self.predator else 5)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers

        with open(self.file_path) as f:
            for i, line in enumerate(f):
                if i % num_workers == worker_id:
                    yield from self.prepare_line(line)

    def prepare_line(self, line: str) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        json_dict = json.loads(line)
        state, pred_act, prey_act = json_dict["state"], json_dict["pred_act"], json_dict["prey_act"]
        xs = ImitationDataset.prepare_xs(state, self.predator)
        ys = pred_act if self.predator else prey_act
        return [(torch.tensor(x), torch.tensor(y)) for x, y in zip(xs, ys)]

    @staticmethod
    def prepare_xs(state: dict, predator: bool) -> List[List[float]]:
        features_list = []

        if predator:
            common_features = []
            for prey in state["preys"]:
                common_features.extend((prey["x_pos"], prey["y_pos"], prey["is_alive"]))
            for pred in state["predators"]:
                features_list.append(common_features + [pred["x_pos"], pred["y_pos"], pred["speed"]])
        else:
            common_features = []
            for pred in state["predators"]:
                common_features.extend((pred["x_pos"], pred["y_pos"]))
            for prey in state["preys"]:
                features_list.append(common_features + [prey["x_pos"], prey["y_pos"], prey["speed"], prey["is_alive"]])

        return features_list
