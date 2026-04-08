import copy
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from data import build_dataloader, get_dataloaders, CIFAR10SubsetWithOptionalLabels
from model import SimpleCNN


class WeightedCombinedDataset(Dataset):
    """
    Combined dataset for:
    - true labeled samples
    - pseudo-labeled samples

    Returns:
    - image
    - label
    - sample_weight

    True labeled samples get weight 1.0
    Pseudo-labeled samples get weight pseudo_weight
    """

    def __init__(
        self,
        base_dataset,
        labeled_indices: List[int],
        pseudo_indices: List[int],
        pseudo_label_dict: Dict[int, int],
        pseudo_weight: float
    ):
        self.samples = []

        for idx in labeled_indices:
            self.samples.append((idx, None, 1.0, False))

        for idx in pseudo_indices:
            self.samples.append((idx, pseudo_label_dict[idx], pseudo_weight, True))

        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        dataset_idx, pseudo_label, sample_weight, is_pseudo = self.samples[i]
        image, true_label = self.base_dataset[dataset_idx]

        if is_pseudo:
            label = pseudo_label
        else:
            label = true_label

        return image, label, torch.tensor(sample_weight, dtype=torch.float32)


def compute_macro_f1(all_preds, all_labels, num_classes=10):
    """
    Compute macro F1-score manually for multi-class classification.
    """
    f1_scores = []

    for cls in range(num_classes):
        tp = sum((p == cls and y == cls) for p, y in zip(all_preds, all_labels))
        fp = sum((p == cls and y != cls) for p, y in zip(all_preds, all_labels))
        fn = sum((p != cls and y == cls) for p, y in zip(all_preds, all_labels))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        f1_scores.append(f1)

    return sum(f1_scores) / num_classes


def train_one_epoch_weighted(model, dataloader, optimizer, device):
    """
    Weighted training epoch:
    - true labeled samples weight = 1.0
    - pseudo-labeled samples weight = pseudo_weight
    """
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    criterion = nn.CrossEntropyLoss(reduction="none")
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for images, labels, weights in dataloader:
        images = images.to(device, non_blocking=use_amp)
        labels = labels.to(device, non_blocking=use_amp)
        weights = weights.to(device, non_blocking=use_amp)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            per_sample_loss = criterion(outputs, labels)
            loss = (per_sample_loss * weights).mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


def evaluate_model(model, dataloader, device):
    """
    Standard evaluation on validation/test sets.
    Returns loss, accuracy, macro F1.
    """
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            use_amp = device.type == "cuda"
            images = images.to(device, non_blocking=use_amp)
            labels = labels.to(device, non_blocking=use_amp)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    macro_f1 = compute_macro_f1(all_preds, all_labels, num_classes=10)

    return avg_loss, accuracy, macro_f1


def train_for_epochs_weighted(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=5,
    lr=0.001,
    verbose: bool = True
):
    """
    Train for a few epochs using weighted pseudo-label loss.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch_weighted(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device
        )

        val_loss, val_acc, val_f1 = evaluate_model(model, val_loader, device)

        if verbose:
            print(
                f"    Epoch [{epoch}/{num_epochs}] | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro F1: {val_f1:.4f}"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return best_val_acc


def generate_pseudo_labels(
    model,
    unlabeled_loader,
    unlabeled_indices: List[int],
    device,
    threshold: float = 0.95,
    max_pseudo_labels: int = 1000
) -> Tuple[List[int], Dict[int, int]]:
    """
    Predict unlabeled samples, keep only confident predictions, then select top-k.
    """
    model.eval()

    selected = []
    pointer = 0

    with torch.no_grad():
        for images, _ in unlabeled_loader:
            images = images.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confidences, preds = probs.max(dim=1)

            batch_size = images.size(0)
            batch_indices = unlabeled_indices[pointer:pointer + batch_size]
            pointer += batch_size

            for dataset_idx, conf, pred in zip(batch_indices, confidences.cpu(), preds.cpu()):
                if conf.item() >= threshold:
                    selected.append((dataset_idx, pred.item(), conf.item()))

    selected.sort(key=lambda x: x[2], reverse=True)
    selected = selected[:max_pseudo_labels]

    selected_indices = [x[0] for x in selected]
    pseudo_label_dict = {x[0]: x[1] for x in selected}

    return selected_indices, pseudo_label_dict


def build_weighted_train_loader(
    base_train_dataset,
    labeled_indices: List[int],
    pseudo_indices: List[int],
    pseudo_label_dict: Dict[int, int],
    pseudo_weight: float,
    batch_size: int = 64,
    num_workers: int = 2
):
    dataset = WeightedCombinedDataset(
        base_dataset=base_train_dataset,
        labeled_indices=labeled_indices,
        pseudo_indices=pseudo_indices,
        pseudo_label_dict=pseudo_label_dict,
        pseudo_weight=pseudo_weight
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    return loader


def run_pseudo_labeling_ssl(
    data_dir: str = "./data",
    batch_size: int = 64,
    val_ratio: float = 0.1,
    labeled_ratio: float = 0.1,
    seed: int = 42,
    num_workers: int = 2,
    threshold: float = 0.95,
    max_pseudo_labels_per_round: int = 1000,
    pseudo_weight: float = 0.5,
    ssl_rounds: int = 3,
    epochs_per_round: int = 4,
    learning_rate: float = 0.001,
    save_model: bool = True,
    verbose: bool = True
):
    """
    Weighted pseudo-labeling SSL pipeline.
    """
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    if verbose:
        print(f"Using device: {device}")

    data = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        val_ratio=val_ratio,
        labeled_ratio=labeled_ratio,
        seed=seed,
        num_workers=num_workers
    )

    train_dataset = data["train_dataset"]
    val_loader = data["val_loader"]
    test_loader = data["test_loader"]

    labeled_indices = data["labeled_indices"]
    remaining_unlabeled_indices = list(data["unlabeled_indices"])

    model = SimpleCNN(num_classes=10).to(device)

    all_pseudo_indices = []
    all_pseudo_labels = {}

    best_overall_val_acc = 0.0
    best_overall_state = copy.deepcopy(model.state_dict())

    for round_idx in range(1, ssl_rounds + 1):
        if verbose:
            print(f"\n--- SSL Round {round_idx}/{ssl_rounds} ---")

        train_loader = build_weighted_train_loader(
            base_train_dataset=train_dataset,
            labeled_indices=labeled_indices,
            pseudo_indices=all_pseudo_indices,
            pseudo_label_dict=all_pseudo_labels,
            pseudo_weight=pseudo_weight,
            batch_size=batch_size,
            num_workers=num_workers
        )

        if verbose:
            print(f"  Original labeled samples: {len(labeled_indices)}")
            print(f"  Current pseudo-labeled  : {len(all_pseudo_indices)}")
            print(f"  Total training samples  : {len(train_loader.dataset)}")
            print(f"  Remaining unlabeled     : {len(remaining_unlabeled_indices)}")
            print(f"  Pseudo weight           : {pseudo_weight:.3f}")

        train_for_epochs_weighted(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=epochs_per_round,
            lr=learning_rate,
            verbose=verbose
        )

        _, current_val_acc, current_val_f1 = evaluate_model(model, val_loader, device)

        if verbose:
            print(f"  Validation accuracy after round {round_idx}: {current_val_acc:.4f}")
            print(f"  Validation macro F1 after round {round_idx}: {current_val_f1:.4f}")

        if current_val_acc > best_overall_val_acc:
            best_overall_val_acc = current_val_acc
            best_overall_state = copy.deepcopy(model.state_dict())

        if len(remaining_unlabeled_indices) == 0:
            if verbose:
                print("  No unlabeled samples left. Stopping early.")
            break

        current_unlabeled_dataset = CIFAR10SubsetWithOptionalLabels(
            base_dataset=train_dataset,
            indices=remaining_unlabeled_indices
        )

        current_unlabeled_loader = build_dataloader(
            current_unlabeled_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        new_pseudo_indices, new_pseudo_labels = generate_pseudo_labels(
            model=model,
            unlabeled_loader=current_unlabeled_loader,
            unlabeled_indices=remaining_unlabeled_indices,
            device=device,
            threshold=threshold,
            max_pseudo_labels=max_pseudo_labels_per_round
        )

        if verbose:
            print(f"  Newly selected pseudo-labels: {len(new_pseudo_indices)}")

        if len(new_pseudo_indices) == 0:
            if verbose:
                print("  No confident pseudo-labels found. Stopping early.")
            break

        all_pseudo_indices.extend(new_pseudo_indices)
        all_pseudo_labels.update(new_pseudo_labels)

        selected_set = set(new_pseudo_indices)
        remaining_unlabeled_indices = [
            idx for idx in remaining_unlabeled_indices if idx not in selected_set
        ]

    model.load_state_dict(best_overall_state)
    _, test_acc, test_f1 = evaluate_model(model, test_loader, device)

    if verbose:
        print("\n=== Final SSL Results ===")
        print(f"Best Validation Accuracy: {best_overall_val_acc:.4f}")
        print(f"Final Test Accuracy     : {test_acc:.4f}")
        print(f"Final Test Macro F1     : {test_f1:.4f}")
        print(f"Total Pseudo-Labeled    : {len(all_pseudo_indices)}")

    if save_model:
        torch.save(best_overall_state, "best_ssl_cnn.pt")
        if verbose:
            print("Saved best model to: best_ssl_cnn.pt")

    return {
        "best_val_acc": best_overall_val_acc,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "total_pseudo_labeled": len(all_pseudo_indices),
        "best_model_path": "best_ssl_cnn.pt" if save_model else None
    }


if __name__ == "__main__":
    run_pseudo_labeling_ssl(
        data_dir="./data",
        batch_size=64,
        val_ratio=0.1,
        labeled_ratio=0.2,
        seed=42,
        num_workers=2,
        threshold=0.97,
        max_pseudo_labels_per_round=500,
        pseudo_weight=1.0,
        ssl_rounds=3,
        epochs_per_round=4,
        learning_rate=0.001,
        save_model=True,
        verbose=True
    )
