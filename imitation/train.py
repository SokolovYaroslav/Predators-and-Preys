import torch
from torch.nn import MSELoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from imitation.dataloader import ImitationDataset
from imitation.model import ImitationModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(path, pred_epochs=100, prey_epochs=3, batch_size=2048, num_workers=15, lr=1e-3):
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

    prey_shape = len(next(iter(prey_dataset))[0])
    prey_model = ImitationModel(prey_shape)
    prey_model = prey_model.to(device)
    prey_opt = AdamW(prey_model.parameters(), lr)

    loss_fn = MSELoss()

    for i in range(max(pred_epochs, prey_epochs)):
        print(predator_model)
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
            torch.save(predator_model.state_dict(), f"pred_epoch_{i}.pth")

        print(prey_model)
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
            torch.save(prey_model.state_dict(), f"prey_epoch_{i}.pth")

    predator_opt.zero_grad()
    prey_opt.zero_grad()


if __name__ == "__main__":
    train("imitation/states_small.jsonl", pred_epochs=1, prey_epochs=1)
    train("imitation/states.jsonl", pred_epochs=100, prey_epochs=5)
