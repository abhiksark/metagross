# examples/docker/workloads/complex_pipeline.py
"""Multi-stage image pipeline with CPU prep, async upload, training, and eval."""
import dataclasses

import torch
from torch import nn


@dataclasses.dataclass
class CpuBatch:
    images: torch.Tensor
    labels: torch.Tensor


class HybridImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        embedding_dim = 64
        self.patch_stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, embedding_dim, kernel_size=8, stride=8),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=4,
            dim_feedforward=128,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.classifier = nn.Linear(embedding_dim, 10)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_stem(images).flatten(2).transpose(1, 2)
        encoded = self.encoder(tokens)
        return self.classifier(encoded.mean(dim=1))


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run the container with --gpus all")
    return torch.device("cuda")


def build_pipeline(
    device: torch.device,
) -> tuple[HybridImageClassifier, torch.optim.Optimizer, torch.cuda.Stream]:
    model = HybridImageClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    transfer_stream = torch.cuda.Stream(device=device)
    return model, optimizer, transfer_stream


def load_cpu_batch(seed: int) -> CpuBatch:
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn((24, 3, 64, 64), generator=generator)
    labels = torch.randint(0, 10, (24,), generator=generator)
    return CpuBatch(images=images, labels=labels)


def preprocess_cpu_batch(batch: CpuBatch) -> CpuBatch:
    images = batch.images.clamp(-2.5, 2.5).div(2.5)
    images = images.contiguous().pin_memory()
    labels = batch.labels.pin_memory()
    return CpuBatch(images=images, labels=labels)


def upload_batch(
    batch: CpuBatch, device: torch.device, transfer_stream: torch.cuda.Stream
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.cuda.stream(transfer_stream):
        images = batch.images.to(device, non_blocking=True)
        labels = batch.labels.to(device, non_blocking=True)
    torch.cuda.current_stream(device).wait_stream(transfer_stream)
    return images, labels


def forward_and_backward(
    model: HybridImageClassifier,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    logits = model(images)
    loss = nn.functional.cross_entropy(logits, labels)
    loss.backward()
    return loss.detach().item()


def optimizer_step(optimizer: torch.optim.Optimizer) -> None:
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()


def evaluate_pipeline(
    model: HybridImageClassifier, images: torch.Tensor, labels: torch.Tensor
) -> float:
    model.eval()
    with torch.no_grad():
        predictions = model(images).argmax(dim=1)
        accuracy = (predictions == labels).float().mean()
    torch.cuda.synchronize()
    return accuracy.cpu().item()


def main() -> None:
    torch.manual_seed(51)
    device = require_cuda()
    model, optimizer, transfer_stream = build_pipeline(device)
    optimizer.zero_grad(set_to_none=True)

    last_images = None
    last_labels = None
    losses = []
    for seed in (100, 101):
        cpu_batch = preprocess_cpu_batch(load_cpu_batch(seed))
        images, labels = upload_batch(cpu_batch, device, transfer_stream)
        losses.append(forward_and_backward(model, images, labels))
        last_images, last_labels = images, labels

    optimizer_step(optimizer)
    if last_images is None or last_labels is None:
        raise RuntimeError("pipeline did not produce an evaluation batch")
    accuracy = evaluate_pipeline(model, last_images, last_labels)
    print(
        f"complex_pipeline: device={torch.cuda.get_device_name()} "
        f"mean_loss={sum(losses) / len(losses):.4f} accuracy={accuracy:.3f}"
    )


if __name__ == "__main__":
    main()
