# examples/docker/benchmarks/steady_state.py
"""Timed steady-state probes for Metagross overhead comparisons."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import torch


def python_step(value: int) -> int:
    return value + 1


def launch_step(tensor: torch.Tensor) -> None:
    tensor.add_(1.0)


def compute_step(left: torch.Tensor, right: torch.Tensor,
                 output: torch.Tensor) -> None:
    torch.mm(left, right, out=output)


def benchmark_python(iterations: int, repeats: int) -> tuple[list[float], int]:
    samples = []
    value = 0
    for _ in range(repeats):
        started = time.perf_counter()
        for _ in range(iterations):
            value = python_step(value)
        samples.append(time.perf_counter() - started)
        time.sleep(0.1)
    return samples, value


def benchmark_launches(iterations: int,
                       repeats: int) -> tuple[list[float], float]:
    tensor = torch.zeros(1 << 18, device="cuda")
    for _ in range(20):
        launch_step(tensor)
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        for _ in range(iterations):
            launch_step(tensor)
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - started)
        time.sleep(0.1)
    return samples, tensor[0].item()


def benchmark_compute(iterations: int,
                      repeats: int) -> tuple[list[float], float]:
    torch.manual_seed(404)
    left = torch.randn((4096, 4096), device="cuda")
    right = torch.randn((4096, 4096), device="cuda")
    output = torch.empty_like(left)
    for _ in range(5):
        compute_step(left, right, output)
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        for _ in range(iterations):
            compute_step(left, right, output)
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - started)
        time.sleep(0.1)
    return samples, output[0, 0].item()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pattern", choices=("python", "launch", "compute"), required=True
    )
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.iterations <= 0 or args.repeats <= 0:
        parser.error("iterations and repeats must be greater than zero")
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable")

    if args.pattern == "python":
        samples, check = benchmark_python(args.iterations, args.repeats)
    elif args.pattern == "launch":
        samples, check = benchmark_launches(args.iterations, args.repeats)
    else:
        samples, check = benchmark_compute(args.iterations, args.repeats)
    if isinstance(check, float) and not math.isfinite(check):
        raise RuntimeError("benchmark produced a non-finite check value")

    print(json.dumps({
        "schema_version": 1,
        "pattern": args.pattern,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "samples_seconds": samples,
        "median_seconds": statistics.median(samples),
        "check": check,
    }, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
