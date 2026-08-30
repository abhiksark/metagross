# examples/docker/workloads/basic_tensor_ops.py
"""Small PyTorch workload with upload, compute, sync, and download phases."""
import torch


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run the container with --gpus all")
    return torch.device("cuda")


def upload_inputs(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    left_cpu = torch.randn((1024, 1024), generator=generator)
    right_cpu = torch.randn((1024, 1024), generator=generator)
    left = left_cpu.to(device)
    right = right_cpu.to(device)
    return left, right


def compute(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    result = torch.relu(left @ right)
    torch.cuda.synchronize()
    return result


def download_checksum(result: torch.Tensor) -> float:
    sample = result[:4, :4].cpu()
    return sample.sum().item()


def main() -> None:
    device = require_cuda()
    left, right = upload_inputs(device)
    result = compute(left, right)
    checksum = download_checksum(result)
    print(f"basic_tensor_ops: device={torch.cuda.get_device_name()} checksum={checksum:.4f}")


if __name__ == "__main__":
    main()
