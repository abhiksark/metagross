# examples/docker/benchmark_overhead.py
"""Compare Dockerized PyTorch workloads with and without Metagross tracing."""
from __future__ import annotations

import argparse
import datetime
import json
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path


PORTFOLIO_WORKLOADS = (
    "basic_tensor_ops.py",
    "training_step.py",
    "basic_cnn.py",
    "basic_vit.py",
    "basic_decoder.py",
    "complex_pipeline.py",
)
STEADY_STATE_WORKLOADS = {
    # Enough records to overflow the 1 MiB profile pipe if the tracer lags.
    "python-calls": ("python", 200_000, 5),
    "rapid-launches": ("launch", 2_000, 5),
    "compute-heavy": ("compute", 100, 5),
}
WORKLOADS = (*PORTFOLIO_WORKLOADS, *STEADY_STATE_WORKLOADS)
MODES = ("bare", "full", "no-attribution", "launch-only")
_TRACE_MOUNTS = (
    ("/lib/modules", "/lib/modules", "ro"),
    ("/usr/src", "/usr/src", "ro"),
    ("/sys/kernel/debug", "/sys/kernel/debug", "rw"),
    ("/sys/kernel/tracing", "/sys/kernel/tracing", "rw"),
)
_LOST_RE = re.compile(r"metagross: lost (\d+) events")
_DROPPED_RE = re.compile(r"metagross: dropped (\d+) nested calls")
_PROFILE_LOSS_RE = re.compile(
    r"metagross: stats .*lost_profile=(\d+) refused=(\d+)")
_LOSS_KEYS = ("lost_events", "dropped_nested_calls", "lost_profile_records",
              "refused_attributions")


class BenchmarkError(RuntimeError):
    """A benchmark command could not produce a valid sample."""


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Metagross capture modes using its Docker workloads."
    )
    parser.add_argument("--image", default="metagross-pytorch")
    parser.add_argument(
        "--workload", action="append", choices=WORKLOADS,
        help="workload to run; repeat the option to select multiple",
    )
    parser.add_argument(
        "--mode", action="append", choices=MODES,
        help="capture mode to run; repeat the option to select multiple",
    )
    parser.add_argument("--repetitions", type=_positive_int, default=3)
    parser.add_argument("--warmups", type=_nonnegative_int, default=1)
    parser.add_argument("--timeout", type=_positive_int, default=180)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def _target_command(workload: str) -> tuple[str, list[str]]:
    steady = STEADY_STATE_WORKLOADS.get(workload)
    if steady is None:
        root = "/workspace/workloads"
        return root, [f"{root}/{workload}"]
    pattern, iterations, repeats = steady
    root = "/workspace/benchmarks"
    return root, [
        f"{root}/steady_state.py", "--pattern", pattern,
        "--iterations", str(iterations), "--repeats", str(repeats),
    ]


def _docker_command(image: str, workload: str, mode: str) -> list[str]:
    project_root, target_command = _target_command(workload)
    if mode == "bare":
        return [
            "docker", "run", "--rm", "--gpus", "all",
            "--user", "metagross-target",
            "--entrypoint", "/usr/bin/python3",
            image, *target_command,
        ]

    command = [
        "docker", "run", "--rm", "--gpus", "all", "--privileged",
        "--pid=host", "--tmpfs", "/benchmark-output:rw,mode=1777",
    ]
    for host, container, access in _TRACE_MOUNTS:
        command.extend(("-v", f"{host}:{container}:{access}"))
    command.append(image)
    command.extend((
        "--json", "--output", "/benchmark-output/trace.jsonl", "--stats",
        "--project-root", project_root,
    ))
    if mode == "no-attribution":
        command.append("--no-attribution")
    elif mode == "launch-only":
        command.extend(("--trace", "launch", "--no-attribution"))
    command.extend(target_command)
    return command


def _run_sample(image: str, workload: str, mode: str,
                timeout: int) -> dict:
    started = time.perf_counter()
    try:
        result = subprocess.run(
            _docker_command(image, workload, mode),
            capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BenchmarkError(
            f"{workload} in {mode} mode could not run: {exc}") from exc
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        raise BenchmarkError(
            f"{workload} in {mode} mode exited {result.returncode}:\n"
            f"{result.stderr.strip()}")
    lost = sum(int(value) for value in _LOST_RE.findall(result.stderr))
    dropped = sum(int(value) for value in _DROPPED_RE.findall(result.stderr))
    profile_loss = _PROFILE_LOSS_RE.search(result.stderr)
    lost_profile, refused = (
        map(int, profile_loss.groups()) if profile_loss else (0, 0))
    sample = {
        "elapsed_seconds": elapsed,
        "lost_events": lost,
        "dropped_nested_calls": dropped,
        "lost_profile_records": lost_profile,
        "refused_attributions": refused,
    }
    if workload in STEADY_STATE_WORKLOADS:
        try:
            sample["target_metrics"] = json.loads(
                result.stdout.strip().splitlines()[-1]
            )
        except (IndexError, json.JSONDecodeError) as exc:
            raise BenchmarkError(
                f"{workload} in {mode} mode emitted invalid metrics"
            ) from exc
    return sample


def _optional_command(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


def _environment(image: str) -> dict:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "kernel": platform.release(),
        "host_python": platform.python_version(),
        "image": image,
        "image_id": _optional_command([
            "docker", "image", "inspect", image, "--format", "{{.Id}}",
        ]),
        "gpu": _optional_command([
            "nvidia-smi", "--query-gpu=name,driver_version",
            "--format=csv,noheader",
        ]),
    }


def _summarize(samples: list[dict]) -> dict:
    elapsed = [sample["elapsed_seconds"] for sample in samples]
    summary = {
        "samples": samples,
        "median_seconds": statistics.median(elapsed),
        "min_seconds": min(elapsed),
        "max_seconds": max(elapsed),
    }
    for key in _LOSS_KEYS:
        summary[key] = sum(sample.get(key, 0) for sample in samples)
    target_elapsed = [
        sample["target_metrics"]["median_seconds"] for sample in samples
        if "target_metrics" in sample
    ]
    summary["target_median_seconds"] = (
        statistics.median(target_elapsed) if target_elapsed else None
    )
    return summary


def run_benchmark(args: argparse.Namespace) -> dict:
    workloads = list(dict.fromkeys(args.workload or PORTFOLIO_WORKLOADS))
    modes = list(dict.fromkeys(args.mode or MODES))
    results = {}

    for workload in workloads:
        mode_samples = {mode: [] for mode in modes}
        for mode in modes:
            for warmup in range(args.warmups):
                print(
                    f"warmup {warmup + 1}/{args.warmups}: {workload} {mode}",
                    file=sys.stderr, flush=True,
                )
                _run_sample(args.image, workload, mode, args.timeout)

        for repetition in range(args.repetitions):
            ordered_modes = modes if repetition % 2 == 0 else list(reversed(modes))
            for mode in ordered_modes:
                print(
                    f"sample {repetition + 1}/{args.repetitions}: "
                    f"{workload} {mode}",
                    file=sys.stderr, flush=True,
                )
                mode_samples[mode].append(
                    _run_sample(args.image, workload, mode, args.timeout)
                )

        summaries = {
            mode: _summarize(mode_samples[mode]) for mode in modes
        }
        bare_summary = summaries.get("bare", {})
        bare_median = bare_summary.get("median_seconds")
        bare_target_median = bare_summary.get("target_median_seconds")
        for summary in summaries.values():
            summary["ratio_to_bare"] = (
                summary["median_seconds"] / bare_median
                if bare_median is not None else None
            )
            target_median = summary["target_median_seconds"]
            summary["target_ratio_to_bare"] = (
                target_median / bare_target_median
                if target_median is not None and bare_target_median is not None
                else None
            )
        results[workload] = summaries

    return {
        "schema_version": 1,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "environment": _environment(args.image),
        "configuration": {
            "repetitions": args.repetitions,
            "warmups": args.warmups,
            "timeout_seconds": args.timeout,
            "workloads": workloads,
            "modes": modes,
        },
        "results": results,
    }


def _print_summary(report: dict) -> None:
    print(
        "WORKLOAD                 MODE                 E2E  E2E/BARE  "
        "TARGET TARGET/BARE LOST PROFILE-LOST REFUSED",
        file=sys.stderr,
    )
    for workload, modes in report["results"].items():
        for mode, result in modes.items():
            ratio = result["ratio_to_bare"]
            ratio_text = f"{ratio:.2f}x" if ratio is not None else "-"
            target = result["target_median_seconds"]
            target_text = f"{target:.4f}s" if target is not None else "-"
            target_ratio = result["target_ratio_to_bare"]
            target_ratio_text = (
                f"{target_ratio:.2f}x" if target_ratio is not None else "-"
            )
            print(
                f"{workload:24} {mode:17} "
                f"{result['median_seconds']:6.3f}s {ratio_text:>9} "
                f"{target_text:>8} {target_ratio_text:>11} "
                f"{result['lost_events']:4} "
                f"{result['lost_profile_records']:12} "
                f"{result['refused_attributions']:7}",
                file=sys.stderr,
            )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run_benchmark(args)
    except BenchmarkError as exc:
        print(f"benchmark: {exc}", file=sys.stderr)
        return 1
    _print_summary(report)
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output is None:
        sys.stdout.write(encoded)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
