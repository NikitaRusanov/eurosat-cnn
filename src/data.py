import torch
import torchvision
import torchvision.transforms as trasnsforms
import ssl

import settings


ssl._create_default_https_context = ssl._create_unverified_context


transform = trasnsforms.Compose(
    [trasnsforms.ToTensor(), trasnsforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

_dataset = torchvision.datasets.EuroSAT(
    root="./data_raw", download=True, transform=transform
)

_train_dataset, _test_dataset = torch.utils.data.random_split(_dataset, (0.8, 0.2))

train_dataloader = torch.utils.data.DataLoader(
    _train_dataset, batch_size=settings.BATCH_SIZE, shuffle=True, num_workers=2
)

test_dataloader = torch.utils.data.DataLoader(
    _test_dataset, batch_size=settings.BATCH_SIZE, shuffle=False, num_workers=2
)

_idx_to_class = {idx: c for c, idx in _dataset.class_to_idx.items()}


def get_class_by_idx(idx: int) -> str:
    return _idx_to_class[idx]
