from argparse import ArgumentParser

import torch
from torch.nn import MSELoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiplicativeLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from imitation.dataloader import ImitationDataset
from imitation.model import ImitationModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(path, suf: str,  pred_epochs=100, prey_epochs=3, batch_size=2048, num_workers=15, lr=1e-3):
    predator_dataset = ImitationDataset(predator=True, path=path)
    prey_dataset = ImitationDataset(predator=False, path=path)
    predator_dataloader = DataLoader(
        predator_dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers, drop_last=True
    )
    prey_dataloader = DataLoader(
        prey_dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers, drop_last=True
    )

    pred_shape = len(next(iter(predator_dataset))[0])
    predator_model = ImitationModel(pred_shape)
    predator_model = predator_model.to(device)
    predator_opt = AdamW(predator_model.parameters(), lr)
    predator_sched = MultiplicativeLR(predator_opt, lambda epoch: 0.95)

    prey_shape = len(next(iter(prey_dataset))[0])
    prey_model = ImitationModel(prey_shape)
    prey_model = prey_model.to(device)
    prey_opt = AdamW(prey_model.parameters(), lr)
    prey_sched = MultiplicativeLR(prey_opt, lambda epoch: 0.95)

    loss_fn = MSELoss()

    print(predator_model)
    print(prey_model)

    for i in range(max(pred_epochs, prey_epochs)):
        if i < pred_epochs:
            bar = tqdm(predator_dataloader)
            for batch in bar:
                predator_opt.zero_grad()
                x, y = batch
                x, y = x.to(device), y.to(device).unsqueeze(1)
                y_pred = predator_model(x)
                loss = loss_fn(y, y_pred)
                loss.backward()
                predator_opt.step()
                bar.set_description(f"Epoch {i}: predator mse {loss.item():.3f}")
            predator_sched.step()
            torch.save(predator_model.state_dict(), f"pred_epoch_{i}_{suf}.pth")

        if i < prey_epochs:
            bar = tqdm(prey_dataloader)
            for batch in bar:
                prey_opt.zero_grad()
                x, y = batch
                x, y = x.to(device), y.to(device).unsqueeze(1)
                y_pred = prey_model(x)
                loss = loss_fn(y, y_pred)
                loss.backward()
                prey_opt.step()
                bar.set_description(f"Epoch {i}: prey mse {loss.item():.3f}")
            prey_sched.step()
            torch.save(prey_model.state_dict(), f"prey_epoch_{i}_{suf}.pth")

    predator_opt.zero_grad()
    prey_opt.zero_grad()


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("suffix", type=str, help="Suffix for the saved path")
    args = argparser.parse_args()

    train("imitation/states_small.jsonl", args.suffix, pred_epochs=1, prey_epochs=1)
    train("imitation/states.jsonl", args.suffix, pred_epochs=100, prey_epochs=5)
