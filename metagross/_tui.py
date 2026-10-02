# metagross/_tui.py
"""Dependency-free curses dashboard for live Metagross JSONL traces."""

from __future__ import annotations

import collections
import sys
import time
from pathlib import Path

from metagross import _follow, _viewer


_MIN_TERMINAL_WIDTH = 80
_MIN_TERMINAL_HEIGHT = 18


class RateTracker:
    def __init__(self, window_seconds: float = 5.0):
        self.window_seconds = window_seconds
        self.samples: collections.deque[tuple[float, int]] = collections.deque(
            maxlen=2048
        )

    def reset(self) -> None:
        self.samples.clear()

    def observe(self, now: float, total_events: int) -> None:
        if self.samples and total_events < self.samples[-1][1]:
            self.reset()
        self.samples.append((now, total_events))
        cutoff = now - self.window_seconds
        while len(self.samples) > 2 and self.samples[1][0] < cutoff:
            self.samples.popleft()

    @property
    def events_per_second(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        elapsed = self.samples[-1][0] - self.samples[0][0]
        if elapsed <= 0:
            return 0.0
        events = self.samples[-1][1] - self.samples[0][1]
        return max(0.0, events / elapsed)


def _panel(title: str, header: str, rows: list[str], width: int) -> list[str]:
    return [
        _viewer._heading(title, width),
        _viewer._fit(header, width),
        *(_viewer._fit(row, width) for row in rows),
    ]


def _api_panel(model: _viewer.TraceModel, width: int, limit: int) -> list[str]:
    name_width = max(1, width - 20)
    rows = [
        f"{_viewer._fit(api, name_width):<{name_width}} "
        f"{aggregate.count:>8,} "
        f"{_viewer._duration(aggregate.total_duration_ns):>10}"
        for api, aggregate in _viewer._top(model.apis, limit)
    ]
    if not rows:
        rows.append("<waiting for CUDA events>")
    return _panel(
        "TOP APIS",
        f"{'API':<{name_width}} {'CALLS':>8} {'CPU':>10}",
        rows,
        width,
    )


def _function_panel(model: _viewer.TraceModel, width: int, limit: int) -> list[str]:
    name_width = max(1, width - 20)
    rows = [
        f"{_viewer._fit(function, name_width):<{name_width}} "
        f"{aggregate.count:>8,} "
        f"{_viewer._duration(aggregate.total_duration_ns):>10}"
        for (function, _file, _line), aggregate in _viewer._top(model.functions, limit)
    ]
    if not rows:
        rows.append("<no attributed functions yet>")
    return _panel(
        "TOP FUNCTIONS",
        f"{'FUNCTION':<{name_width}} {'CALLS':>8} {'CPU':>10}",
        rows,
        width,
    )


def _kernel_panel(model: _viewer.TraceModel, width: int, limit: int) -> list[str]:
    name_width = max(1, width - 20)
    rows = [
        f"{_viewer._fit(kernel, name_width):<{name_width}} "
        f"{aggregate.count:>8,} "
        f"{_viewer._duration(aggregate.total_duration_ns):>10}"
        for kernel, aggregate in _viewer._top(model.kernels, limit)
    ]
    if not rows:
        rows.append("<no resolved kernels yet>")
    return _panel(
        "TOP KERNELS",
        f"{'KERNEL':<{name_width}} {'CALLS':>8} {'CPU':>10}",
        rows,
        width,
    )


def _recent_panel(model: _viewer.TraceModel, width: int, limit: int) -> list[str]:
    text_width = max(2, width - 12)
    function_width = text_width // 2
    api_width = text_width - function_width
    rows = []
    for event in list(model.recent)[-limit:]:
        marker = "!" if event.return_code else " "
        function = _viewer._fit(event.function or "<unknown>", function_width)
        api = _viewer._fit(event.api, api_width)
        rows.append(
            f"{function:<{function_width}} {api:<{api_width}} "
            f"{marker}{_viewer._duration(event.duration_ns):>9}"
        )
    if not rows:
        rows.append("<waiting for events>")
    return _panel(
        "RECENT EVENTS",
        f"{'FUNCTION':<{function_width}} {'API':<{api_width}} {'CPU':>10}",
        rows,
        width,
    )


def _combine(left: list[str], right: list[str], width: int) -> list[str]:
    gap = 3
    left_width = (width - gap) // 2
    right_width = width - gap - left_width
    rows = []
    for index in range(max(len(left), len(right))):
        left_row = left[index] if index < len(left) else ""
        right_row = right[index] if index < len(right) else ""
        rows.append(
            f"{_viewer._fit(left_row, left_width):<{left_width}}"
            f"{' ' * gap}{_viewer._fit(right_row, right_width):<{right_width}}"
        )
    return rows


def render_live_dashboard(
    model: _viewer.TraceModel,
    width: int,
    height: int,
    event_rate: float,
    *,
    waiting: bool = False,
    paused: bool = False,
    summary_error: str | None = None,
    refresh_seconds: float = 0.2,
) -> list[str]:
    """Render one terminal-sized overview frame without terminal controls."""
    width = max(20, min(width, 240))
    height = max(8, min(height, 200))
    status = _viewer.live_status(
        model,
        waiting=waiting,
        paused=paused,
        summary_error=summary_error,
    )

    title = "METAGROSS GPU TRACE"
    gap = max(1, width - len(title) - len(status))
    attributed = 100.0 * model.attributed / model.events if model.events else 0.0
    lost = model.summary_capture("lost_events")
    dropped = model.summary_capture("dropped_nested_calls")
    lines = [
        _viewer._fit(title + " " * gap + status, width),
        "=" * width,
    ]
    lines.extend(
        _viewer._wrap_fields(
            [
                f"Events {model.events:,}",
                f"Rate {event_rate:,.1f}/s",
                f"Attributed {attributed:.1f}%",
                f"Errors {model.cuda_errors:,}",
                f"Lost {lost}",
                f"Dropped {dropped}",
            ],
            width,
        )
    )
    lines.extend(
        _viewer._wrap_fields(
            [
                f"CPU API {_viewer._duration(model.total_api_duration_ns)}",
                f"Sync {_viewer._duration(model.synchronization_duration_ns)}",
                f"Copied {_viewer._bytes(model.successful_copy_bytes)}",
                f"GPU observed {_viewer._bytes(model.observed_outstanding_bytes)}",
                f"peak {_viewer._bytes(model.observed_peak_bytes)}",
            ],
            width,
        )
    )
    lines.extend(
        _viewer._wrap_fields(
            [
                f"Malformed {model.malformed_lines}",
                f"Last event {model.last_timestamp or '<none>'}",
            ],
            width,
        )
    )

    action = "resume" if paused else "pause"
    footer = f"q quit | p {action} | Ctrl-C stop | refresh {refresh_seconds:.2f}s"
    reasons = model.incomplete_reasons()
    reserved_rows = 1 + (1 if summary_error else 0) + (1 if reasons else 0)
    body_height = height - reserved_rows

    if width >= 79 and height >= _MIN_TERMINAL_HEIGHT:
        lines.append("")
        half_width = (width - 3) // 2
        right_width = width - half_width - 3
        top_limit = max(1, min(4, body_height - len(lines) - 6))
        lines.extend(
            _combine(
                _api_panel(model, half_width, top_limit),
                _function_panel(model, right_width, top_limit),
                width,
            )
        )
        lines.append("")
        bottom_limit = max(1, body_height - len(lines) - 2)
        lines.extend(
            _combine(
                _kernel_panel(model, half_width, bottom_limit),
                _recent_panel(model, right_width, bottom_limit),
                width,
            )
        )

    lines = lines[:body_height]
    while len(lines) < body_height:
        lines.append("")
    if reasons:
        lines.append(_viewer._fit("Incomplete: " + "; ".join(reasons), width))
    if summary_error:
        lines.append(_viewer._fit(f"Summary warning: {summary_error}", width))
    lines.append(_viewer._fit(footer, width))
    return [_viewer._fit(line, width) for line in lines]


def _color_attributes(curses_module) -> tuple[int, int]:
    if not curses_module.has_colors():
        return 0, 0
    try:
        curses_module.start_color()
        background = -1
        try:
            curses_module.use_default_colors()
        except curses_module.error:
            background = curses_module.COLOR_BLACK
        curses_module.init_pair(1, curses_module.COLOR_CYAN, background)
        curses_module.init_pair(2, curses_module.COLOR_YELLOW, background)
        return curses_module.color_pair(1), curses_module.color_pair(2)
    except curses_module.error:
        return 0, 0


def _run_curses(
    screen,
    curses_module,
    trace: Path,
    summary: Path | None,
    recent_limit: int,
    refresh_seconds: float,
) -> int:
    try:
        curses_module.curs_set(0)
    except curses_module.error:
        pass
    screen.keypad(True)
    screen.timeout(max(50, int(refresh_seconds * 1000)))
    title_attributes, warning_attributes = _color_attributes(curses_module)

    follower = _follow.TraceFollower(trace, recent_limit=recent_limit)
    summaries = _follow.SummaryFollower(summary)
    rates = RateTracker()
    waiting = True
    following = False
    paused = False
    try:
        while True:
            if not paused:
                update = follower.poll()
                waiting = update.waiting
                if waiting:
                    following = False
                elif update.reset or not following:
                    rates.reset()
                    summaries.reset(follower.trace_mtime_ns)
                    following = True
                if not waiting:
                    summaries.poll(follower.model)
                rates.observe(time.monotonic(), follower.model.events)

            height, width = screen.getmaxyx()
            canvas_width = max(1, width - 1)
            if width < _MIN_TERMINAL_WIDTH or height < _MIN_TERMINAL_HEIGHT:
                lines = [
                    "Terminal too small",
                    "Resize to at least 80x18",
                    "q quit",
                ]
            else:
                lines = render_live_dashboard(
                    follower.model,
                    canvas_width,
                    height,
                    rates.events_per_second,
                    waiting=waiting,
                    paused=paused,
                    summary_error=summaries.last_error,
                    refresh_seconds=refresh_seconds,
                )
            screen.erase()
            for row, line in enumerate(lines[:height]):
                attributes = title_attributes if row == 0 else 0
                if summaries.last_error and row == height - 2:
                    attributes = warning_attributes
                try:
                    screen.addnstr(
                        row, 0, line, min(len(line), canvas_width), attributes
                    )
                except curses_module.error:
                    pass
            screen.refresh()

            key = screen.getch()
            if key in (ord("q"), ord("Q")):
                return 0
            if key in (ord("p"), ord("P")):
                paused = not paused
                if not paused:
                    rates.reset()
    finally:
        follower.close()


def run_follow_dashboard(
    trace: Path,
    summary: Path | None,
    recent_limit: int,
    refresh_seconds: float,
) -> int:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print(
            "metagross view: --follow requires an interactive terminal; "
            "use --snapshot for redirected output",
            file=sys.stderr,
        )
        return 2
    try:
        import curses
    except ImportError as exc:
        print(
            f"metagross view: cannot start dashboard: {exc}",
            file=sys.stderr,
        )
        return 1
    try:
        return curses.wrapper(
            _run_curses, curses, trace, summary, recent_limit, refresh_seconds
        )
    except KeyboardInterrupt:
        return 130
    except _viewer.ViewerError as exc:
        print(f"metagross view: {exc}", file=sys.stderr)
        return 1
    except curses.error as exc:
        print(f"metagross view: cannot start dashboard: {exc}", file=sys.stderr)
        return 1
