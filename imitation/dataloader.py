import json
from typing import List, Tuple

import torch
from torch.utils.data.dataset import IterableDataset
from tqdm.auto import tqdm


class ImitationDataset(IterableDataset):
    def __init__(self, predator: bool):
        self.predator = predator
        dicts = []
        with open("imitation/states.jsonl") as f:
            for line in tqdm(f, desc="Loading file to dataset..."):
                dicts.append(json.loads(line))
        self.json_dicts = dicts
        self.len = len(dicts)

    def __len__(self):
        return self.len * self._examples_per_dict

    @property
    def _examples_per_dict(self) -> int:
        return 2 if self.predator else 5

    def __getitem__(self, idx) -> Tuple[List[float], float]:
        dict_idx, obs_idx = divmod(idx, self._examples_per_dict)
        state, pred_act, prey_act = (
            self.json_dicts[dict_idx]["state"],
            self.json_dicts[dict_idx]["pred_act"],
            self.json_dicts[dict_idx]["prey_act"],
        )
        xs = ImitationDataset.prepare_xs(state, self.predator)
        ys = pred_act if self.predator else prey_act
        return xs[obs_idx], ys[obs_idx]

    @staticmethod
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
