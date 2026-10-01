# metagross/_viewer.py
"""Unprivileged streaming trace model and terminal dashboards."""

from __future__ import annotations

import argparse
import collections
import dataclasses
import ipaddress
import json
import math
import os
import shutil
import stat
import sys
import unicodedata
from pathlib import Path


_MAX_LINE_BYTES = 1 << 20
_MAX_SUMMARY_BYTES = 4 << 20
_MAX_TEXT = 500
_MAX_DETAIL_TEXT = 200
_MAX_DETAILS = 32
_MAX_APIS = 512
_MAX_FUNCTIONS = 4096
_MAX_KERNELS = 4096
_MAX_MEMORY_SAMPLES = 2000
_MAX_RECENT = 10_000
_DEFAULT_RECENT = 500
_MAX_INT = 2**63


class ViewerError(Exception):
    """A trace or summary cannot be viewed safely."""


@dataclasses.dataclass(frozen=True)
class ViewerEvent:
    timestamp: str
    pid: int
    tid: int
    function: str | None
    file: str | None
    line: int | None
    api: str
    kernel: str | None
    return_code: int
    duration_ns: int
    details: dict


@dataclasses.dataclass
class Aggregate:
    count: int = 0
    errors: int = 0
    total_duration_ns: int = 0
    max_duration_ns: int = 0

    def observe(self, event: ViewerEvent) -> None:
        self.count += 1
        self.errors += event.return_code != 0
        self.total_duration_ns += event.duration_ns
        self.max_duration_ns = max(self.max_duration_ns, event.duration_ns)


@dataclasses.dataclass
class TraceModel:
    recent_limit: int = _DEFAULT_RECENT
    events: int = 0
    attributed: int = 0
    cuda_errors: int = 0
    malformed_lines: int = 0
    total_api_duration_ns: int = 0
    synchronization_duration_ns: int = 0
    successful_copy_bytes: int = 0
    observed_peak_bytes: int = 0
    observed_outstanding_bytes: int = 0
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    summary: dict | None = None
    summary_mismatch: bool = False
    summary_expected_events: int | None = None

    def __post_init__(self) -> None:
        self.recent: collections.deque[ViewerEvent] = collections.deque(
            maxlen=self.recent_limit
        )
        self.memory_samples: collections.deque[tuple[str, int]] = collections.deque(
            maxlen=_MAX_MEMORY_SAMPLES
        )
        self.apis: dict[str, Aggregate] = {}
        self.functions: dict[tuple[str, str, int], Aggregate] = {}
        self.kernels: dict[str, Aggregate] = {}

    @staticmethod
    def _group(groups: dict, key, limit: int, overflow_key) -> Aggregate:
        aggregate = groups.get(key)
        if aggregate is not None:
            return aggregate
        if len(groups) >= limit:
            key = overflow_key
        return groups.setdefault(key, Aggregate())

    def observe(self, event: ViewerEvent) -> None:
        self.events += 1
        self.attributed += event.function is not None
        self.cuda_errors += event.return_code != 0
        self.total_api_duration_ns += event.duration_ns
        self.first_timestamp = self.first_timestamp or event.timestamp
        self.last_timestamp = event.timestamp
        self.recent.append(event)

        self._group(self.apis, event.api, _MAX_APIS, "<other>").observe(event)
        if event.function is not None:
            function_key = (event.function, event.file or "", event.line or 0)
            self._group(
                self.functions,
                function_key,
                _MAX_FUNCTIONS,
                ("<other>", "", 0),
            ).observe(event)
        if event.kernel is not None:
            self._group(self.kernels, event.kernel, _MAX_KERNELS, "<other>").observe(
                event
            )

        if event.api in (
            "cuStreamSynchronize",
            "cuCtxSynchronize",
            "cuEventSynchronize",
        ):
            self.synchronization_duration_ns += event.duration_ns
        if event.api.startswith("cuMemcpy") and event.return_code == 0:
            copied = event.details.get("bytes", 0)
            if _is_bounded_int(copied) and copied >= 0:
                self.successful_copy_bytes += copied
        gpu_total = event.details.get("gpu_total")
        if _is_bounded_int(gpu_total) and gpu_total >= 0:
            self.observed_outstanding_bytes = gpu_total
            self.observed_peak_bytes = max(self.observed_peak_bytes, gpu_total)
            self.memory_samples.append((event.timestamp, gpu_total))
        if self.summary_expected_events is not None:
            self.summary_mismatch = self.summary_expected_events != self.events

    def clear_summary(self) -> None:
        self.summary = None
        self.summary_expected_events = None
        self.summary_mismatch = False

    def load_summary(self, summary: dict) -> None:
        if not isinstance(summary, dict):
            raise ViewerError("summary root must be a JSON object")
        schema_version = summary.get("schema_version")
        if not _is_int(schema_version) or schema_version != 1:
            raise ViewerError(f"unsupported summary schema: {schema_version!r}")
        capture = summary.get("capture")
        events = capture.get("events") if isinstance(capture, dict) else None
        if not _is_int(events) or events < 0:
            raise ViewerError("summary capture.events must be a non-negative integer")
        delivery_dropped = capture.get("delivery_dropped", 0)
        if (
            not _is_int(delivery_dropped)
            or delivery_dropped < 0
            or delivery_dropped > events
        ):
            raise ViewerError(
                "summary capture.delivery_dropped must be between zero and events"
            )
        self.summary = summary
        self.summary_expected_events = events - delivery_dropped
        self.summary_mismatch = self.summary_expected_events != self.events

    @property
    def status(self) -> str:
        if self.summary is None:
            return "MALFORMED" if self.malformed_lines else "EVENTS ONLY"
        if self.summary_mismatch:
            status = "MISMATCH"
        elif self.summary_capture("delivery_dropped") > 0:
            status = "INCOMPLETE"
        elif self.summary.get("complete") is True:
            status = "COMPLETE"
        else:
            status = "INCOMPLETE"
        if self.malformed_lines:
            status += " / MALFORMED"
        return status

    def summary_capture(self, name: str, default=0):
        if self.summary is None:
            return default
        capture = self.summary.get("capture", {})
        value = capture.get(name, default) if isinstance(capture, dict) else default
        return value if _is_int(value) or isinstance(value, bool) else default


def live_status(
    model: TraceModel,
    *,
    waiting: bool = False,
    paused: bool = False,
    summary_error: str | None = None,
    error: str | None = None,
) -> str:
    """Return the shared status label for live terminal and web dashboards."""
    if error:
        return "ERROR"
    if paused:
        return "PAUSED"
    if waiting:
        return "WAITING"
    if summary_error:
        return "LIVE / SUMMARY ERROR"
    if model.summary is not None:
        return model.status
    if model.malformed_lines:
        return "LIVE / MALFORMED"
    return "LIVE"


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_bounded_int(value) -> bool:
    """Return whether value is a plain int within the accepted magnitude."""
    return _is_int(value) and abs(value) < _MAX_INT


def _reject_nonfinite(value: str):
    raise ValueError(f"non-finite JSON constant: {value}")


def _finite_float(text: str) -> float:
    """Parse a JSON number token, rejecting magnitudes that overflow to inf.

    A literal like ``1e400`` is valid JSON syntax but ``float()`` silently
    rounds it to ``inf``; without this guard such a value would pass
    ``_reject_nonfinite`` (which only sees the NaN/Infinity constant
    tokens) and later fail JSON re-encoding with ``allow_nan=False``.
    """
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"non-finite JSON number: {text}")
    return value


def _load_bounded_json(raw: bytes):
    """Decode one trust-boundary JSON payload without raising past the caller.

    Bounds recursion depth (translating RecursionError to ValueError) and
    rejects non-finite floats, whether spelled as a constant token
    (NaN, Infinity, -Infinity) or as a number literal that overflows to inf.
    """
    try:
        return json.loads(
            raw.decode("utf-8"),
            parse_float=_finite_float,
            parse_constant=_reject_nonfinite,
        )
    except RecursionError:
        raise ValueError("json nesting too deep") from None


def sanitize_text(value: str, limit: int = _MAX_TEXT) -> str:
    """Remove terminal controls and cap attacker-controlled display strings."""
    cleaned = "".join(
        character if not unicodedata.category(character).startswith("C") else "?"
        for character in value
    )
    return cleaned[:limit]


def _required_int(record: dict, name: str) -> int:
    value = record.get(name)
    if not _is_bounded_int(value):
        raise ValueError(f"{name} must be an integer within range")
    return value


def _optional_text(record: dict, name: str) -> str | None:
    value = record.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string or null")
    return sanitize_text(value)


def parse_event(record) -> ViewerEvent:
    if not isinstance(record, dict):
        raise ValueError("event must be a JSON object")
    timestamp = record.get("timestamp")
    api = record.get("api")
    details = record.get("details")
    if not isinstance(timestamp, str) or not timestamp:
        raise ValueError("timestamp must be a non-empty string")
    if not isinstance(api, str) or not api:
        raise ValueError("api must be a non-empty string")
    if not isinstance(details, dict):
        raise ValueError("details must be an object")
    line = record.get("line")
    if line is not None and not _is_bounded_int(line):
        raise ValueError("line must be an integer within range or null")
    # "span" is additive: older records omit it entirely, and a
    # present value must be an optional string like kernel/function. The
    # value itself is not carried onto ViewerEvent -- the viewer does not
    # render spans yet -- so this call exists purely to validate the type.
    _optional_text(record, "span")
    duration_ns = _required_int(record, "duration_ns")
    if duration_ns < 0:
        raise ValueError("duration_ns must not be negative")

    safe_details = {}
    for index, (key, value) in enumerate(details.items()):
        if index >= _MAX_DETAILS:
            break
        safe_key = sanitize_text(str(key), 100)
        if isinstance(value, str):
            safe_value = sanitize_text(value, _MAX_DETAIL_TEXT)
        elif value is None or isinstance(value, (bool, int, float)):
            safe_value = value
        else:
            try:
                encoded = json.dumps(value, separators=(",", ":"))
            except (TypeError, ValueError):
                encoded = "<complex>"
            safe_value = sanitize_text(encoded, _MAX_DETAIL_TEXT)
        safe_details[safe_key] = safe_value
    return ViewerEvent(
        timestamp=sanitize_text(timestamp, 100),
        pid=_required_int(record, "pid"),
        tid=_required_int(record, "tid"),
        function=_optional_text(record, "function"),
        file=_optional_text(record, "file"),
        line=line,
        api=sanitize_text(api),
        kernel=_optional_text(record, "kernel"),
        return_code=_required_int(record, "return_code"),
        duration_ns=duration_ns,
        details=safe_details,
    )


def observe_raw_line(model: TraceModel, raw: bytes) -> bool:
    """Validate and aggregate one bounded JSONL record.

    A malformed or hostile line (invalid JSON, too deeply nested, a
    non-finite float constant, or a record that fails schema validation)
    is counted in ``malformed_lines`` and never raises past this call.
    """
    try:
        record = _load_bounded_json(raw)
        model.observe(parse_event(record))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        model.malformed_lines += 1
        return False
    return True


def _open_regular_binary(path: Path, label: str):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
    except OSError as exc:
        raise ViewerError(f"cannot open {label} {str(path)!r}: {exc}") from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ViewerError(f"{label} {str(path)!r} is not a regular file")
        return os.fdopen(fd, "rb")
    except ViewerError:
        try:
            os.close(fd)
        except OSError:
            pass
        raise
    except OSError as exc:
        try:
            os.close(fd)
        except OSError:
            pass
        raise ViewerError(f"cannot open {label} {str(path)!r}: {exc}") from None


def load_trace(path: Path, recent_limit: int = _DEFAULT_RECENT) -> TraceModel:
    model = TraceModel(recent_limit=recent_limit)
    try:
        with _open_regular_binary(path, "trace") as stream:
            while True:
                raw = stream.readline(_MAX_LINE_BYTES + 1)
                if not raw:
                    break
                if len(raw) > _MAX_LINE_BYTES:
                    if not raw.endswith(b"\n"):
                        while raw and not raw.endswith(b"\n"):
                            raw = stream.readline(_MAX_LINE_BYTES + 1)
                    model.malformed_lines += 1
                    continue
                observe_raw_line(model, raw)
    except ViewerError:
        raise
    except OSError as exc:
        raise ViewerError(f"cannot read trace {str(path)!r}: {exc}") from None
    if model.events == 0 and model.malformed_lines:
        raise ViewerError(f"trace {str(path)!r} contains no valid event records")
    return model


def load_summary(path: Path) -> dict:
    try:
        with _open_regular_binary(path, "summary") as stream:
            raw = stream.read(_MAX_SUMMARY_BYTES + 1)
    except ViewerError:
        raise
    except OSError as exc:
        raise ViewerError(f"cannot read summary {str(path)!r}: {exc}") from None
    if len(raw) > _MAX_SUMMARY_BYTES:
        raise ViewerError(f"summary {str(path)!r} exceeds 4 MiB")
    try:
        return _load_bounded_json(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ViewerError(f"invalid summary JSON in {str(path)!r}: {exc}") from None


def _duration(ns: int) -> str:
    if ns < 1_000:
        return f"{ns}ns"
    if ns < 1_000_000:
        return f"{ns / 1_000:.1f}us"
    if ns < 1_000_000_000:
        return f"{ns / 1_000_000:.2f}ms"
    return f"{ns / 1_000_000_000:.2f}s"


def _bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if abs(amount) < 1024 or unit == units[-1]:
            return f"{amount:.1f}{unit}" if unit != "B" else f"{int(amount)}B"
        amount /= 1024
    return f"{value}B"


def _fit(value, width: int) -> str:
    text = sanitize_text(str(value))
    if len(text) > width:
        return text[: max(0, width - 1)] + "~"
    return text


def _wrap_fields(fields: list[str], width: int) -> list[str]:
    lines = []
    current = ""
    for field in fields:
        fitted = _fit(field, width)
        candidate = fitted if not current else f"{current}  {fitted}"
        if current and len(candidate) > width:
            lines.append(current)
            current = fitted
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _heading(title: str, width: int) -> str:
    label = f" {title} "
    return label + "-" * max(0, width - len(label))


def _top(groups: dict, limit: int = 8):
    return sorted(
        groups.items(),
        key=lambda item: (-item[1].count, -item[1].total_duration_ns, item[0]),
    )[:limit]


def _time_cell(timestamp: str) -> str:
    if "T" in timestamp:
        return timestamp.split("T", 1)[1][:12]
    return timestamp[:12]


def render_snapshot(model: TraceModel, width: int = 120) -> list[str]:
    width = max(60, min(width, 240))
    lines = []
    title = "METAGROSS TRACE"
    status = model.status
    gap = max(1, width - len(title) - len(status))
    lines.append(_fit(title + " " * gap + status, width))
    lines.append("=" * width)

    attributed_percent = (
        100.0 * model.attributed / model.events if model.events else 0.0
    )
    lost = model.summary_capture("lost_events")
    dropped = model.summary_capture("dropped_nested_calls")
    lines.extend(
        _wrap_fields(
            [
                f"Events {model.events:,}",
                f"Attributed {attributed_percent:.1f}%",
                f"CUDA errors {model.cuda_errors:,}",
                f"Lost {lost}",
                f"Dropped {dropped}",
                f"Malformed {model.malformed_lines}",
            ],
            width,
        )
    )
    lines.extend(
        _wrap_fields(
            [
                f"CPU API {_duration(model.total_api_duration_ns)}",
                f"Synchronization {_duration(model.synchronization_duration_ns)}",
                f"Copied {_bytes(model.successful_copy_bytes)}",
                f"Peak observed {_bytes(model.observed_peak_bytes)}",
                f"Outstanding {_bytes(model.observed_outstanding_bytes)}",
            ],
            width,
        )
    )
    if model.summary_mismatch and model.summary is not None:
        expected = model.summary["capture"]["events"]
        lines.append(
            _fit(
                f"WARNING: summary reports {expected} events but trace contains "
                f"{model.events}",
                width,
            )
        )

    lines.extend(("", _heading("TOP APIS (CPU duration)", width)))
    api_width = width - 41
    lines.append(
        f"{'API':<{api_width}} {'CALLS':>8} {'ERRORS':>7} {'TOTAL':>12} {'MAX':>10}"
    )
    for api, aggregate in _top(model.apis):
        lines.append(
            f"{_fit(api, api_width):<{api_width}} {aggregate.count:>8,} "
            f"{aggregate.errors:>7,} "
            f"{_duration(aggregate.total_duration_ns):>12} "
            f"{_duration(aggregate.max_duration_ns):>10}"
        )
    if not model.apis:
        lines.append("<no events>")

    lines.extend(("", _heading("TOP FUNCTIONS", width)))
    function_columns = width - 23
    location_width = min(24, max(16, function_columns // 2))
    name_width = function_columns - location_width
    lines.append(
        f"{'FUNCTION':<{name_width}} {'LOCATION':<{location_width}} "
        f"{'CALLS':>8} {'TOTAL CPU':>12}"
    )
    for (function, file, line), aggregate in _top(model.functions, 6):
        location = f"{os.path.basename(file)}:{line}" if file else "<unknown>"
        lines.append(
            f"{_fit(function, name_width):<{name_width}} "
            f"{_fit(location, location_width):<{location_width}} "
            f"{aggregate.count:>8,} "
            f"{_duration(aggregate.total_duration_ns):>12}"
        )
    if not model.functions:
        lines.append("<no attributed functions>")

    lines.extend(("", _heading("TOP KERNELS", width)))
    kernel_width = width - 24
    lines.append(f"{'KERNEL':<{kernel_width}} {'LAUNCHES':>10} {'TOTAL CPU':>12}")
    for kernel, aggregate in _top(model.kernels, 6):
        lines.append(
            f"{_fit(kernel, kernel_width):<{kernel_width}} "
            f"{aggregate.count:>10,} "
            f"{_duration(aggregate.total_duration_ns):>12}"
        )
    if not model.kernels:
        lines.append("<no resolved kernels>")

    lines.extend(("", _heading("RECENT EVENTS", width)))
    text_width = width - 31
    function_width = max(12, text_width // 2)
    api_width = text_width - function_width
    lines.append(
        f"{'TIME':<12} {'FUNCTION':<{function_width}} "
        f"{'API':<{api_width}} {'RET':>5} {'CPU':>10}"
    )
    for event in list(model.recent)[-8:]:
        function = event.function or "<unknown>"
        lines.append(
            f"{_fit(_time_cell(event.timestamp), 12):<12} "
            f"{_fit(function, function_width):<{function_width}} "
            f"{_fit(event.api, api_width):<{api_width}} "
            f"{event.return_code:>5} {_duration(event.duration_ns):>10}"
        )
    if not model.recent:
        lines.append("<no recent events>")
    return [_fit(line, width) for line in lines]


class _ViewerParser(argparse.ArgumentParser):
    def exit(self, status=0, message=None):
        if message:
            self._print_message(message, sys.stderr)
        raise SystemExit(status)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _recent_limit(value: str) -> int:
    parsed = _positive_int(value)
    if parsed > _MAX_RECENT:
        raise argparse.ArgumentTypeError(f"must not exceed {_MAX_RECENT}")
    return parsed


def _snapshot_width(value: str) -> int:
    parsed = _positive_int(value)
    if not 60 <= parsed <= 240:
        raise argparse.ArgumentTypeError("must be between 60 and 240 columns")
    return parsed


def _refresh_interval(value: str) -> float:
    parsed = float(value)
    if not 0.05 <= parsed <= 5.0:
        raise argparse.ArgumentTypeError("must be between 0.05 and 5.0 seconds")
    return parsed


def _web_port(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed <= 65_535:
        raise argparse.ArgumentTypeError("must be between 0 and 65535")
    return parsed


def _web_host(value: str) -> str:
    # Numeric IPv4 only: the dashboard's Host check rejects hostnames.
    try:
        address = ipaddress.IPv4Address(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "must be a numeric IPv4 address such as 0.0.0.0"
        ) from None
    if address.is_global:
        raise argparse.ArgumentTypeError(
            "must be an internal address; public addresses are refused"
        )
    if address.is_loopback and str(address) != "127.0.0.1":
        # Other 127.x binds print a URL the Host check would then refuse.
        raise argparse.ArgumentTypeError(
            "the only loopback address accepted is 127.0.0.1"
        )
    return str(address)


def build_parser() -> argparse.ArgumentParser:
    parser = _ViewerParser(
        prog="metagross view",
        description="View a Metagross trace without root, BCC, or CUDA.",
    )
    parser.add_argument("trace", nargs="?", type=Path)
    parser.add_argument("--summary", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--snapshot",
        action="store_true",
        help="print a static terminal dashboard and exit",
    )
    mode.add_argument(
        "--follow",
        action="store_true",
        help="open a live dashboard and wait for appended events",
    )
    mode.add_argument(
        "--web",
        action="store_true",
        help="serve a live browser dashboard (loopback by default; see --host)",
    )
    parser.add_argument(
        "--receive",
        action="store_true",
        help="receive one authenticated in-memory capture for --web",
    )
    parser.add_argument("--recent", type=_recent_limit, default=_DEFAULT_RECENT)
    parser.add_argument(
        "--width",
        type=_snapshot_width,
        help="snapshot width in columns (60-240)",
    )
    parser.add_argument(
        "--refresh",
        type=_refresh_interval,
        metavar="SECONDS",
        help="live dashboard refresh interval (default: 0.2)",
    )
    parser.add_argument(
        "--port",
        type=_web_port,
        help="web dashboard port (default: 8765; use 0 for any free port)",
    )
    parser.add_argument(
        "--host",
        type=_web_host,
        help=(
            "web dashboard bind address (default: 127.0.0.1); use 0.0.0.0 or "
            "an internal interface IPv4 address to let viewers on the internal "
            "network in; public addresses are refused"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)
    if args.receive and not args.web:
        parser.print_usage(sys.stderr)
        print(
            "metagross view: --receive is only valid with --web",
            file=sys.stderr,
        )
        return 2
    if not (args.snapshot or args.follow or args.web):
        parser.print_usage(sys.stderr)
        print(
            "metagross view: choose --snapshot, --follow, or --web",
            file=sys.stderr,
        )
        return 2
    if args.receive and args.trace is not None:
        parser.print_usage(sys.stderr)
        print(
            "metagross view: TRACE is not valid with --receive",
            file=sys.stderr,
        )
        return 2
    if args.receive and args.summary is not None:
        parser.print_usage(sys.stderr)
        print(
            "metagross view: --summary is not valid with --receive",
            file=sys.stderr,
        )
        return 2
    if not args.receive and args.trace is None:
        parser.print_usage(sys.stderr)
        print("metagross view: TRACE is required", file=sys.stderr)
        return 2
    if not args.snapshot and args.width is not None:
        parser.print_usage(sys.stderr)
        print(
            "metagross view: --width is only valid with --snapshot",
            file=sys.stderr,
        )
        return 2
    if args.snapshot and args.refresh is not None:
        parser.print_usage(sys.stderr)
        print(
            "metagross view: --refresh is only valid with --follow or --web",
            file=sys.stderr,
        )
        return 2
    if not args.web and args.port is not None:
        parser.print_usage(sys.stderr)
        print(
            "metagross view: --port is only valid with --web",
            file=sys.stderr,
        )
        return 2
    if not args.web and args.host is not None:
        parser.print_usage(sys.stderr)
        print(
            "metagross view: --host is only valid with --web",
            file=sys.stderr,
        )
        return 2
    if args.receive and args.host not in (None, "127.0.0.1", "0.0.0.0"):
        parser.print_usage(sys.stderr)
        print(
            "metagross view: --receive requires --host 127.0.0.1 or 0.0.0.0 "
            "(the tracer delivers captures only to 127.0.0.1)",
            file=sys.stderr,
        )
        return 2
    if args.web:
        from metagross import _web

        refresh_seconds = 0.2 if args.refresh is None else args.refresh
        port = _web._DEFAULT_PORT if args.port is None else args.port
        host = _web._DEFAULT_HOST if args.host is None else args.host
        if args.receive:
            from metagross import _publish

            try:
                ingest_token = _publish.take_dashboard_token(os.environ)
            except _publish.DashboardPublishError as exc:
                print(f"metagross view: {exc}", file=sys.stderr)
                return 2
            return _web.run_web_dashboard(
                None,
                None,
                args.recent,
                refresh_seconds,
                port,
                ingest_token=ingest_token,
                host=host,
            )
        return _web.run_web_dashboard(
            args.trace,
            args.summary,
            args.recent,
            refresh_seconds,
            port,
            host=host,
        )
    if args.follow:
        from metagross import _tui

        refresh_seconds = 0.2 if args.refresh is None else args.refresh
        return _tui.run_follow_dashboard(
            args.trace, args.summary, args.recent, refresh_seconds
        )
    try:
        model = load_trace(args.trace, recent_limit=args.recent)
        if args.summary is not None:
            model.load_summary(load_summary(args.summary))
    except ViewerError as exc:
        print(f"metagross view: {exc}", file=sys.stderr)
        return 1
    width = args.width or shutil.get_terminal_size((120, 24)).columns
    for line in render_snapshot(model, width=width):
        print(line)
    return 0
