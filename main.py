from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision
import mlflow

from src import train_model, MyNet, get_sample, get_class_by_idx
import src.settings as settings


def _imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_sample(img, title: str):
    plt.title(title)
    _imshow(torchvision.utils.make_grid(img))


def main(exp_name: str = "Default"):
    mlflow.set_experiment(exp_name)

    with mlflow.start_run():
        model_path = Path(settings.MODEL_SAVE_PATH)
        model = MyNet()
        if model_path.exists():
            model.load_state_dict(torch.load(model_path, weights_only=True))
        else:
            train_model(model)
        mlflow.pytorch.log_model(model, "model")

    for _ in range(6):
        img, label = get_sample()
        img = img.view(1, 3, 64, 64)
        print(img.shape)
        _, pred = torch.max(model(img), 1)
        print(pred)
        title = f"pred: {get_class_by_idx(pred.item())}, true {get_class_by_idx(label)}"
        show_sample(img, title)


if __name__ == "__main__":
    mlflow.set_tracking_uri(settings.MLFLOW_URL)
    main()
