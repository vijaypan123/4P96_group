import random
from typing import Dict, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms

_DATASET_CACHE = {}


class CIFAR10SubsetWithOptionalLabels(Dataset):
    """
    A dataset wrapper for a subset of CIFAR-10.

    Supports:
    - selecting specific indices from a base dataset
    - optionally overriding labels (useful for pseudo-labeling)
    """

    def __init__(self, base_dataset, indices: List[int], override_labels: Dict[int, int] = None):
        self.base_dataset = base_dataset
        self.indices = indices
        self.override_labels = override_labels if override_labels is not None else {}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        image, true_label = self.base_dataset[base_idx]

        if base_idx in self.override_labels:
            label = self.override_labels[base_idx]
        else:
            label = true_label

        return image, label

def set_seed(seed: int = 42, seed_python: bool = False) -> None:
    if seed_python:
        random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_cifar10_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns train and test/validation transforms.
    Keep this simple first. You can add augmentation later.
    """
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return train_transform, test_transform


def load_cifar10(data_dir: str = "./data"):
    """
    Load CIFAR-10 train and test datasets.
    """
    cache_key = str(data_dir)
    if cache_key in _DATASET_CACHE:
        return _DATASET_CACHE[cache_key]

    train_transform, test_transform = get_cifar10_transforms()

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )

    _DATASET_CACHE[cache_key] = (train_dataset, test_dataset)
    return train_dataset, test_dataset


def build_dataloader(dataset, batch_size: int, shuffle: bool, num_workers: int):
    use_cuda = torch.cuda.is_available()
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": use_cuda
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    return DataLoader(dataset, **loader_kwargs)


def stratified_train_val_split(
    targets: List[int],
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[int], List[int]]:
    """
    Stratified split of training indices into train_pool and validation.
    Preserves class balance.
    """
    set_seed(seed)

    class_to_indices = {c: [] for c in range(10)}
    for idx, label in enumerate(targets):
        class_to_indices[label].append(idx)

    train_indices = []
    val_indices = []

    for c in range(10):
        indices = class_to_indices[c]
        random.shuffle(indices)

        val_count = int(len(indices) * val_ratio)
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    random.shuffle(train_indices)
    random.shuffle(val_indices)

    return train_indices, val_indices


def stratified_labeled_unlabeled_split(
    targets: List[int],
    candidate_indices: List[int],
    labeled_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[int], List[int]]:
    """
    Split the training pool into labeled and unlabeled subsets in a stratified way.
    """
    set_seed(seed)

    class_to_indices = {c: [] for c in range(10)}
    for idx in candidate_indices:
        label = targets[idx]
        class_to_indices[label].append(idx)

    labeled_indices = []
    unlabeled_indices = []

    for c in range(10):
        indices = class_to_indices[c]
        random.shuffle(indices)

        labeled_count = max(1, int(len(indices) * labeled_ratio))
        labeled_indices.extend(indices[:labeled_count])
        unlabeled_indices.extend(indices[labeled_count:])

    random.shuffle(labeled_indices)
    random.shuffle(unlabeled_indices)

    return labeled_indices, unlabeled_indices


def create_ssl_split(
    train_dataset,
    val_ratio: float = 0.1,
    labeled_ratio: float = 0.1,
    seed: int = 42
):
    """
    Create:
    - labeled training subset
    - unlabeled training subset
    - validation subset

    Returns index lists and wrapped datasets.
    """
    targets = train_dataset.targets

    train_pool_indices, val_indices = stratified_train_val_split(
        targets=targets,
        val_ratio=val_ratio,
        seed=seed
    )

    labeled_indices, unlabeled_indices = stratified_labeled_unlabeled_split(
        targets=targets,
        candidate_indices=train_pool_indices,
        labeled_ratio=labeled_ratio,
        seed=seed
    )

    labeled_dataset = CIFAR10SubsetWithOptionalLabels(train_dataset, labeled_indices)
    unlabeled_dataset = CIFAR10SubsetWithOptionalLabels(train_dataset, unlabeled_indices)
    val_dataset = CIFAR10SubsetWithOptionalLabels(train_dataset, val_indices)

    return {
        "labeled_indices": labeled_indices,
        "unlabeled_indices": unlabeled_indices,
        "val_indices": val_indices,
        "labeled_dataset": labeled_dataset,
        "unlabeled_dataset": unlabeled_dataset,
        "val_dataset": val_dataset
    }


def get_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    val_ratio: float = 0.1,
    labeled_ratio: float = 0.1,
    seed: int = 42,
    num_workers: int = 2
):
    """
    Main helper function for your project.

    Returns:
    - labeled train loader
    - unlabeled loader
    - validation loader
    - test loader
    - datasets and index info for later reuse
    """
    set_seed(seed, seed_python=False)

    train_dataset, test_dataset = load_cifar10(data_dir=data_dir)

    ssl_split = create_ssl_split(
        train_dataset=train_dataset,
        val_ratio=val_ratio,
        labeled_ratio=labeled_ratio,
        seed=seed
    )

    labeled_loader = build_dataloader(
        ssl_split["labeled_dataset"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    unlabeled_loader = build_dataloader(
        ssl_split["unlabeled_dataset"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    val_loader = build_dataloader(
        ssl_split["val_dataset"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = build_dataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return {
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "labeled_loader": labeled_loader,
        "unlabeled_loader": unlabeled_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "labeled_dataset": ssl_split["labeled_dataset"],
        "unlabeled_dataset": ssl_split["unlabeled_dataset"],
        "val_dataset": ssl_split["val_dataset"],
        "labeled_indices": ssl_split["labeled_indices"],
        "unlabeled_indices": ssl_split["unlabeled_indices"],
        "val_indices": ssl_split["val_indices"]
    }


if __name__ == "__main__":
    data = get_dataloaders(
        data_dir="./data",
        batch_size=64,
        val_ratio=0.1,
        labeled_ratio=0.1,
        seed=42,
        num_workers=2
    )

    print("Labeled samples   :", len(data["labeled_dataset"]))
    print("Unlabeled samples :", len(data["unlabeled_dataset"]))
    print("Validation samples:", len(data["val_dataset"]))
    print("Test samples      :", len(data["test_dataset"]))

    x, y = next(iter(data["labeled_loader"]))
    print("One labeled batch shape:", x.shape, y.shape)
