# examples/docker/workloads/basic_vit.py
"""Tiny vision transformer implemented with only core PyTorch modules."""
import torch
from torch import nn


class TinyViT(nn.Module):
    def __init__(self, image_size: int = 64, patch_size: int = 8) -> None:
        super().__init__()
        embedding_dim = 96
        patch_count = (image_size // patch_size) ** 2
        self.patch_embedding = nn.Conv2d(
            3, embedding_dim, kernel_size=patch_size, stride=patch_size
        )
        self.class_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.position_embedding = nn.Parameter(
            torch.randn(1, patch_count + 1, embedding_dim) * 0.02
        )
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=4,
            dim_feedforward=192,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=2, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, 10)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embedding(images).flatten(2).transpose(1, 2)
        class_tokens = self.class_token.expand(images.shape[0], -1, -1)
        tokens = torch.cat((class_tokens, patches), dim=1)
        encoded = self.encoder(tokens + self.position_embedding)
        return self.head(self.norm(encoded[:, 0]))


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run the container with --gpus all")
    return torch.device("cuda")


def build_model(device: torch.device) -> tuple[TinyViT, torch.optim.Optimizer]:
    model = TinyViT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, optimizer


def prepare_image_batch(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.randn((16, 3, 64, 64)).to(device)
    labels = torch.randint(0, 10, (16,)).to(device)
    return images, labels


def train_vit_step(
    model: TinyViT,
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


def classify(model: TinyViT, images: torch.Tensor) -> list[int]:
    model.eval()
    with torch.no_grad():
        predictions = model(images[:4]).argmax(dim=1)
    torch.cuda.synchronize()
    return predictions.cpu().tolist()


def main() -> None:
    torch.manual_seed(31)
    device = require_cuda()
    model, optimizer = build_model(device)
    images, labels = prepare_image_batch(device)
    loss = train_vit_step(model, optimizer, images, labels)
    predictions = classify(model, images)
    print(
        f"basic_vit: device={torch.cuda.get_device_name()} "
        f"loss={loss:.4f} predictions={predictions}"
    )


if __name__ == "__main__":
    main()
