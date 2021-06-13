import json
from typing import List, Tuple

import torch
from torch.nn import MSELoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from imitation.dataloader import ImitationDataset
from imitation.model import ImitationModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_fn(batch: List[Tuple[List[float], float]]) -> Tuple[List[List[float]], List[float]]:
    return zip(*batch)


def train(epochs=100, batch_size=2048, num_workers=8, lr=1e-4):
    dicts = []
    with open("imitation/states_small.jsonl") as f:
        for line in tqdm(f, desc="Loading file to dataset..."):
            dicts.append(json.loads(line))
    predator_dataset = ImitationDataset(predator=True, dicts=dicts)
    prey_dataset = ImitationDataset(predator=False, dicts=dicts)
    predator_dataloader = DataLoader(
        predator_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=collate_fn,
    )
    prey_dataloader = DataLoader(
        prey_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=collate_fn,
    )

    pred_shape = len(predator_dataset[0][0])
    predator_model = ImitationModel(pred_shape)
    predator_model = predator_model.to(device)
    predator_opt = AdamW(predator_model.parameters(), lr)

    prey_shape = len(prey_dataset[0][0])
    prey_model = ImitationModel(prey_shape)
    prey_model = prey_model.to(device)
    prey_opt = AdamW(prey_model.parameters(), lr)

    loss_fn = MSELoss()

    print(predator_model)
    print(prey_model)
    for i in range(epochs):
        # mse_sum = 0
        # cnt = 0
        bar = tqdm(predator_dataloader)
        for batch in bar:
            predator_opt.zero_grad()
            x, y = batch
            x, y = (
                torch.tensor(x, device=device, dtype=torch.float),
                torch.tensor(y, device=device, dtype=torch.float).unsqueeze(1),
            )
            y_pred = predator_model(x)
            loss = loss_fn(y, y_pred)
            loss.backward()
            predator_opt.step()
            bar.set_description(f"Epoch {i}: predator mse {loss.item():.3f}")
            # mse_sum += loss.item()
            # cnt += 1
        # print(f"Epoch {i} overall predator mse: {mse_sum / cnt}")
        torch.save(predator_model.state_dict(), f"pred_epoch_{i}.pth")

        # mse_sum = 0
        # cnt = 0
        bar = tqdm(prey_dataloader)
        for batch in bar:
            prey_opt.zero_grad()
            x, y = batch
            x, y = (
                torch.tensor(x, device=device, dtype=torch.float),
                torch.tensor(y, device=device, dtype=torch.float).unsqueeze(1),
            )
            y_pred = prey_model(x)
            loss = loss_fn(y, y_pred)
            loss.backward()
            prey_opt.step()
            bar.set_description(f"Epoch {i}: prey mse {loss.item():.3f}")
            # mse_sum += loss.item()
            # cnt += 1
        # print(f"Epoch {i} overall prey mse: {mse_sum / cnt}")
        torch.save(prey_model.state_dict(), f"prey_epoch_{i}.pth")

    predator_opt.zero_grad()
    prey_opt.zero_grad()


if __name__ == "__main__":
    train()
