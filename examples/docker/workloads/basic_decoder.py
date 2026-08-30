# examples/docker/workloads/basic_decoder.py
"""Tiny decoder-only language model with prefill and token-by-token decoding."""
import torch
from torch import nn


class TinyDecoderLM(nn.Module):
    def __init__(self, vocabulary_size: int = 256, context_length: int = 64) -> None:
        super().__init__()
        embedding_dim = 128
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_length, embedding_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=4,
            dim_feedforward=256,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer, num_layers=2, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.output = nn.Linear(embedding_dim, vocabulary_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        sequence_length = token_ids.shape[1]
        if sequence_length > self.context_length:
            raise ValueError("token sequence exceeds the model context length")
        positions = torch.arange(sequence_length, device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.full(
                (sequence_length, sequence_length),
                float("-inf"),
                device=token_ids.device,
            ),
            diagonal=1,
        )
        hidden = self.blocks(hidden, mask=causal_mask, is_causal=True)
        return self.output(self.norm(hidden))


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run the container with --gpus all")
    return torch.device("cuda")


def build_decoder(device: torch.device) -> TinyDecoderLM:
    return TinyDecoderLM().to(device).eval()


def prepare_prompt(device: torch.device) -> torch.Tensor:
    prompt_cpu = torch.tensor([[1, 17, 42, 9, 81, 3, 55, 12]], dtype=torch.long)
    return prompt_cpu.to(device)


def prefill(model: TinyDecoderLM, prompt: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        logits = model(prompt)
    torch.cuda.synchronize()
    return logits[:, -1]


def decode_tokens(
    model: TinyDecoderLM, prompt: torch.Tensor, token_count: int = 6
) -> torch.Tensor:
    generated = prompt
    with torch.no_grad():
        for _ in range(token_count):
            logits = model(generated)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)
    torch.cuda.synchronize()
    return generated


def main() -> None:
    torch.manual_seed(41)
    device = require_cuda()
    model = build_decoder(device)
    prompt = prepare_prompt(device)
    first_logits = prefill(model, prompt)
    generated = decode_tokens(model, prompt)
    score = first_logits.max().item()
    token_ids = generated[0].cpu().tolist()
    print(
        f"basic_decoder: device={torch.cuda.get_device_name()} "
        f"score={score:.4f} tokens={token_ids}"
    )


if __name__ == "__main__":
    main()
