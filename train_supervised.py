import copy
import torch
import torch.nn as nn
import torch.optim as optim

from data import get_dataloaders
from model import SimpleCNN


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


def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluate the model on a dataset.
    Returns average loss, accuracy, and macro F1.
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    macro_f1 = compute_macro_f1(all_preds, all_labels, num_classes=10)

    return avg_loss, accuracy, macro_f1


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    Returns average training loss and accuracy.
    """
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        preds = torch.argmax(outputs, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def main():
    data_dir = "./data"
    batch_size = 64
    val_ratio = 0.1
    labeled_ratio = 0.2
    seed = 42
    num_workers = 2

    num_epochs = 10
    learning_rate = 0.001

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )
    print(f"Using device: {device}")

    data = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        val_ratio=val_ratio,
        labeled_ratio=labeled_ratio,
        seed=seed,
        num_workers=num_workers
    )

    labeled_loader = data["labeled_loader"]
    val_loader = data["val_loader"]
    test_loader = data["test_loader"]

    print(f"Labeled training samples: {len(data['labeled_dataset'])}")
    print(f"Validation samples      : {len(data['val_dataset'])}")
    print(f"Test samples            : {len(data['test_dataset'])}")

    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = 0.0
    best_model_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, labeled_loader, criterion, optimizer, device
        )

        val_loss, val_acc, val_f1 = evaluate_model(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch [{epoch}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro F1: {val_f1:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_state)

    test_loss, test_acc, test_f1 = evaluate_model(
        model, test_loader, criterion, device
    )

    print("\nBest Validation Accuracy:", f"{best_val_acc:.4f}")
    print("Final Test Loss         :", f"{test_loss:.4f}")
    print("Final Test Accuracy     :", f"{test_acc:.4f}")
    print("Final Test Macro F1     :", f"{test_f1:.4f}")

    torch.save(best_model_state, "best_supervised_cnn.pt")
    print("Saved best model to: best_supervised_cnn.pt")


if __name__ == "__main__":
    main()