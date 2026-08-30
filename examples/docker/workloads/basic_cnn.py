# examples/docker/workloads/basic_cnn.py
"""Small convolutional classifier with training and evaluation phases."""
import torch
from torch import nn


class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(32, 10)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        return self.classifier(features.flatten(1))


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run the container with --gpus all")
    return torch.device("cuda")


def build_model(device: torch.device) -> tuple[SmallCNN, torch.optim.Optimizer]:
    model = SmallCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    return model, optimizer


def prepare_batch(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.randn((32, 3, 64, 64)).to(device)
    labels = torch.randint(0, 10, (32,)).to(device)
    return images, labels


def train_step(
    model: SmallCNN,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    loss = nn.functional.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    return loss.item()


def evaluate(model: SmallCNN, images: torch.Tensor) -> list[int]:
    model.eval()
    with torch.no_grad():
        predictions = model(images[:8]).argmax(dim=1)
    torch.cuda.synchronize()
    return predictions.cpu().tolist()


def main() -> None:
    torch.manual_seed(21)
    device = require_cuda()
    model, optimizer = build_model(device)
    images, labels = prepare_batch(device)
    loss = train_step(model, optimizer, images, labels)
    predictions = evaluate(model, images)
    print(
        f"basic_cnn: device={torch.cuda.get_device_name()} "
        f"loss={loss:.4f} predictions={predictions}"
    )


if __name__ == "__main__":
    main()
