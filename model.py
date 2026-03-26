import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    A small CNN for CIFAR-10 classification.

    Architecture:
    - Conv -> ReLU -> MaxPool
    - Conv -> ReLU -> MaxPool
    - Flatten
    - Fully Connected -> ReLU
    - Output layer
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = SimpleCNN(num_classes=10)
    x = torch.randn(64, 3, 32, 32)
    y = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)