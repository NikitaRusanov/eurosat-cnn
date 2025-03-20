import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import mlflow

import src.settings as settings
from src.data import train_dataloader, test_dataloader


def train_model(
    model: nn.Module,
    lr: float = 1e-3,
    max_epoch: int = 10,
    save_model: bool = True,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    mlflow.log_params(
        {
            "max_epoch": max_epoch,
            "learning_rate": lr,
            "optimizer": "Adam",
        }
    )

    for epoch in tqdm(range(max_epoch)):
        running_train_loss = 0.0
        train_correct_preds = 0
        for inputs, labels in tqdm(train_dataloader):
            optimizer.zero_grad()

            outp = model(inputs)
            loss = criterion(outp, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() / train_dataloader.batch_size

            _, preds = torch.max(outp, 1)
            train_correct_preds += (preds == labels).sum().item()
        running_train_loss /= len(train_dataloader)
        train_accuracy = train_correct_preds / (
            len(train_dataloader) * train_dataloader.batch_size
        )  # type: ignore

        mlflow.log_metrics(
            {"train loss": running_train_loss, "train accuracy": train_accuracy},
            step=epoch,
        )

        running_test_loss = 0.0
        test_correct_preds = 0
        with torch.no_grad():
            for inputs, labels in tqdm(test_dataloader):
                outp = model(inputs)
                loss = criterion(outp, labels)

                running_test_loss += loss.item() / test_dataloader.batch_size

                _, preds = torch.max(outp, 1)
                test_correct_preds += (preds == labels).sum().item()

        running_test_loss /= len(test_dataloader)
        test_accuracy = test_correct_preds / (
            len(test_dataloader) * test_dataloader.batch_size
        )  # type: ignore

        mlflow.log_metrics(
            {"test loss": running_test_loss, "test accuracy": test_accuracy},
            step=epoch,
        )

        print(f"\n\nEpoch number {epoch + 1}")
        print(
            f"train loss:\t{running_train_loss:.3f}\ttrain acc:\t{train_accuracy:.3f}"
        )
        print(f"test loss:\t{running_test_loss:.3f}\ttest acc:\t{test_accuracy:.3f}")

    if save_model:
        torch.save(model.state_dict(), settings.MODEL_SAVE_PATH)
