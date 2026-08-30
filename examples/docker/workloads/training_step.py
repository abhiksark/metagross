# examples/docker/workloads/training_step.py
"""Tiny PyTorch training loop for function-attributed CUDA tracing."""
import torch
from torch import nn


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run the container with --gpus all")
    return torch.device("cuda")


def build_model(device: torch.device) -> tuple[nn.Module, torch.optim.Optimizer]:
    model = nn.Sequential(
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 10),
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    return model, optimizer


def make_batch(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.randn((64, 256)).to(device)
    labels = torch.randint(0, 10, (64,)).to(device)
    return features, labels


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    logits = model(features)
    loss = nn.functional.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    return loss.item()


def main() -> None:
    torch.manual_seed(11)
    device = require_cuda()
    model, optimizer = build_model(device)
    losses = []
    for _ in range(3):
        features, labels = make_batch(device)
        losses.append(train_step(model, optimizer, features, labels))
    print(f"training_step: device={torch.cuda.get_device_name()} final_loss={losses[-1]:.4f}")


if __name__ == "__main__":
    main()
