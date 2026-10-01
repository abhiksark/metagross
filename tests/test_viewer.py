# tests/test_viewer.py
"""Unprivileged tests for the Metagross visual trace viewer."""

from __future__ import annotations

import contextlib
import fcntl
import io
import http.client
import json
import os
import pty
import select
import shutil
import struct
import signal
import subprocess
import socket
import sys
import tempfile
import termios
import threading
import time
import types
import urllib.parse
import urllib.request
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

import metagross
from metagross import _follow, _tui, _viewer, _web


def _record(**changes):
    record = {
        "timestamp": "2026-08-30T12:10:03.410000+00:00",
        "pid": 1234,
        "tid": 1234,
        "function": "compute",
        "file": "/project/train.py",
        "line": 10,
        "api": "cuLaunchKernel",
        "kernel": "vector_add",
        "return_code": 0,
        "duration_ns": 50_000,
        "details": {
            "grid": "8,1,1",
            "block": "128,1,1",
            "stream": "0x0",
        },
    }
    record.update(changes)
    return record


class ViewerModelTest(unittest.TestCase):
    def test_parse_and_aggregate_events(self):
        model = _viewer.TraceModel(recent_limit=2)
        records = [
            _record(),
            _record(
                timestamp="2026-08-30T12:10:04.000000+00:00",
                api="cuMemcpyHtoD",
                kernel=None,
                duration_ns=100_000,
                details={"bytes": 4096, "gpu_total": 8192},
            ),
            _record(
                timestamp="2026-08-30T12:10:05.000000+00:00",
                api="cuStreamSynchronize",
                kernel=None,
                function=None,
                file=None,
                line=None,
                return_code=2,
                duration_ns=200_000,
                details={"stream": "0x77", "gpu_total": 4096},
            ),
        ]
        for record in records:
            model.observe(_viewer.parse_event(record))

        self.assertEqual(model.events, 3)
        self.assertEqual(model.attributed, 2)
        self.assertEqual(model.cuda_errors, 1)
        self.assertEqual(model.successful_copy_bytes, 4096)
        self.assertEqual(model.synchronization_duration_ns, 200_000)
        self.assertEqual(model.observed_peak_bytes, 8192)
        self.assertEqual(model.observed_outstanding_bytes, 4096)
        self.assertEqual(len(model.recent), 2)
        self.assertEqual(model.apis["cuLaunchKernel"].count, 1)
        self.assertEqual(model.kernels["vector_add"].count, 1)

    def test_summary_status_and_event_mismatch(self):
        model = _viewer.TraceModel()
        model.observe(_viewer.parse_event(_record()))
        summary = {
            "schema_version": 1,
            "complete": True,
            "capture": {"events": 1, "lost_events": 0, "dropped_nested_calls": 0},
        }
        model.load_summary(summary)
        self.assertEqual(model.status, "COMPLETE")
        summary["capture"]["events"] = 2
        model.load_summary(summary)
        self.assertEqual(model.status, "MISMATCH")

    def test_declared_delivery_loss_is_incomplete_not_mismatch(self):
        model = _viewer.TraceModel()
        model.observe(_viewer.parse_event(_record()))
        model.load_summary(
            {
                "schema_version": 1,
                "complete": True,
                "capture": {"events": 2, "delivery_dropped": 1},
            }
        )
        self.assertFalse(model.summary_mismatch)
        self.assertEqual(model.status, "INCOMPLETE")

        model.load_summary(
            {
                "schema_version": 1,
                "complete": False,
                "capture": {"events": 3, "delivery_dropped": 1},
            }
        )
        self.assertTrue(model.summary_mismatch)
        self.assertEqual(model.status, "MISMATCH")

    def test_control_characters_are_sanitized(self):
        event = _viewer.parse_event(
            _record(
                function="safe\x1b[2Jname\nnext\u202ereversed",
                details={"stream": "bad\rvalue"},
            )
        )
        self.assertNotIn("\x1b", event.function)
        self.assertNotIn("\n", event.function)
        self.assertNotIn("\u202e", event.function)
        self.assertEqual(event.details["stream"], "bad?value")

    def test_invalid_event_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duration_ns"):
            _viewer.parse_event(_record(duration_ns=-1))
        with self.assertRaisesRegex(ValueError, "details"):
            _viewer.parse_event(_record(details=[]))

    def test_parse_event_accepts_additive_span_field_with_and_without(self):
        # "span" is an ADDITIVE key in the JSONL record. A record that
        # carries it and a record that omits it (the older shape) must both
        # still parse without error.
        with_span = _viewer.parse_event(_record(span="forward"))
        without_span = _viewer.parse_event(_record())
        self.assertEqual(with_span.function, "compute")
        self.assertEqual(without_span.function, "compute")

    def test_parse_event_rejects_non_string_span(self):
        with self.assertRaisesRegex(ValueError, "span"):
            _viewer.parse_event(_record(span=123))

    def test_summary_schema_requires_nonnegative_integers(self):
        model = _viewer.TraceModel()
        with self.assertRaisesRegex(_viewer.ViewerError, "unsupported"):
            model.load_summary(
                {
                    "schema_version": True,
                    "capture": {"events": 0},
                }
            )
        with self.assertRaisesRegex(_viewer.ViewerError, "non-negative"):
            model.load_summary(
                {
                    "schema_version": 1,
                    "capture": {"events": -1},
                }
            )
        for delivery_dropped in (-1, 2, True):
            with self.subTest(
                delivery_dropped=delivery_dropped
            ), self.assertRaisesRegex(_viewer.ViewerError, "delivery_dropped"):
                model.load_summary(
                    {
                        "schema_version": 1,
                        "capture": {
                            "events": 1,
                            "delivery_dropped": delivery_dropped,
                        },
                    }
                )

    def test_final_status_preserves_malformed_state(self):
        model = _viewer.TraceModel()
        model.observe(_viewer.parse_event(_record()))
        model.malformed_lines = 1
        self.assertEqual(model.status, "MALFORMED")

        for complete, events, expected in (
            (True, 1, "COMPLETE / MALFORMED"),
            (False, 1, "INCOMPLETE / MALFORMED"),
            (True, 2, "MISMATCH / MALFORMED"),
        ):
            with self.subTest(expected=expected):
                model.load_summary(
                    {
                        "schema_version": 1,
                        "complete": complete,
                        "capture": {"events": events},
                    }
                )
                self.assertEqual(model.status, expected)


class IngestBoundaryTest(unittest.TestCase):
    def test_deeply_nested_json_is_malformed_not_fatal(self):
        model = _viewer.TraceModel()
        # Nesting past the recursion limit used to crash the viewer. Exceed
        # the interpreter limit generously so the bomb is deterministic.
        depth = sys.getrecursionlimit() * 2
        raw = (b"[" * depth) + (b"]" * depth)
        self.assertFalse(_viewer.observe_raw_line(model, raw))
        self.assertEqual(model.malformed_lines, 1)  # counted, not raised

    def test_absurd_duration_is_malformed(self):
        model = _viewer.TraceModel()
        record = _record()
        record["duration_ns"] = 10**400
        line = json.dumps(record).encode("utf-8")
        self.assertFalse(_viewer.observe_raw_line(model, line))
        self.assertEqual(model.malformed_lines, 1)

    def test_non_finite_float_is_rejected(self):
        model = _viewer.TraceModel()
        self.assertFalse(_viewer.observe_raw_line(model, b'{"duration_ns": NaN}'))
        self.assertEqual(model.malformed_lines, 1)

    def test_overflowing_float_literal_is_malformed(self):
        # A JSON number literal like 1e400 is syntactically valid but
        # float() silently rounds it to inf; it must be rejected here, not
        # allowed through to poison a later allow_nan=False re-encoding.
        model = _viewer.TraceModel()
        record = _record(details={"x": "PLACEHOLDER"})
        line = json.dumps(record).replace('"PLACEHOLDER"', "1e400").encode("utf-8")
        self.assertFalse(_viewer.observe_raw_line(model, line))
        self.assertEqual(model.malformed_lines, 1)


class ViewerFileTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_loader_tolerates_one_malformed_line(self):
        path = self.root / "events.jsonl"
        path.write_text(json.dumps(_record()) + "\n{not-json}\n", encoding="utf-8")
        model = _viewer.load_trace(path)
        self.assertEqual(model.events, 1)
        self.assertEqual(model.malformed_lines, 1)
        self.assertEqual(model.status, "MALFORMED")

    def test_loader_rejects_non_regular_trace(self):
        with self.assertRaisesRegex(_viewer.ViewerError, "not a regular file"):
            _viewer.load_trace(self.root)

    def test_loader_rejects_file_with_no_valid_events(self):
        path = self.root / "bad.jsonl"
        path.write_text("{not-json}\n", encoding="utf-8")
        with self.assertRaisesRegex(_viewer.ViewerError, "no valid"):
            _viewer.load_trace(path)

    def test_loader_bounds_oversized_lines(self):
        path = self.root / "large.jsonl"
        with path.open("wb") as stream:
            stream.write(b"x" * (_viewer._MAX_LINE_BYTES + 100) + b"\n")
            stream.write(json.dumps(_record()).encode() + b"\n")
        model = _viewer.load_trace(path)
        self.assertEqual(model.events, 1)
        self.assertEqual(model.malformed_lines, 1)

    def test_summary_loader_rejects_unknown_schema(self):
        model = _viewer.TraceModel()
        with self.assertRaisesRegex(_viewer.ViewerError, "unsupported"):
            model.load_summary({"schema_version": 99, "capture": {"events": 0}})

    def test_summary_file_size_is_bounded(self):
        path = self.root / "large-summary.json"
        path.write_bytes(b" " * (_viewer._MAX_SUMMARY_BYTES + 1))
        with self.assertRaisesRegex(_viewer.ViewerError, "exceeds 4 MiB"):
            _viewer.load_summary(path)

    def test_post_open_read_failures_are_reported(self):
        class BrokenStream:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def readline(self, _limit):
                raise OSError("read failed")

            def read(self, _limit):
                raise OSError("read failed")

        with mock.patch.object(
            _viewer, "_open_regular_binary", return_value=BrokenStream()
        ):
            with self.assertRaisesRegex(_viewer.ViewerError, "cannot read trace"):
                _viewer.load_trace(self.root / "events.jsonl")
            with self.assertRaisesRegex(_viewer.ViewerError, "cannot read summary"):
                _viewer.load_summary(self.root / "summary.json")

    def test_fstat_failure_closes_descriptor(self):
        with (
            mock.patch.object(_viewer.os, "open", return_value=99),
            mock.patch.object(_viewer.os, "fstat", side_effect=OSError("fstat failed")),
            mock.patch.object(_viewer.os, "close") as close,
        ):
            with self.assertRaisesRegex(_viewer.ViewerError, "cannot open trace"):
                _viewer._open_regular_binary(self.root / "events.jsonl", "trace")
        close.assert_called_once_with(99)


class FollowReaderTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.trace = self.root / "events.jsonl"

    def test_rejects_non_regular_trace(self):
        follower = _follow.TraceFollower(self.root)
        self.addCleanup(follower.close)
        with self.assertRaisesRegex(_viewer.ViewerError, "not a regular file"):
            follower.poll()

    def test_waits_for_creation_and_holds_partial_line(self):
        follower = _follow.TraceFollower(self.trace)
        self.addCleanup(follower.close)
        self.assertTrue(follower.poll().waiting)

        encoded = json.dumps(_record()).encode()
        self.trace.write_bytes(encoded)
        update = follower.poll()
        self.assertFalse(update.waiting)
        self.assertEqual(follower.model.events, 0)

        with self.trace.open("ab") as stream:
            stream.write(b"\n")
        update = follower.poll()
        self.assertEqual(update.events, 1)
        self.assertEqual(follower.model.events, 1)

    def test_reads_appends_and_bounds_oversized_lines(self):
        self.trace.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
        follower = _follow.TraceFollower(self.trace)
        self.addCleanup(follower.close)
        self.assertEqual(follower.poll().events, 1)

        with self.trace.open("ab") as stream:
            stream.write(b"x" * (_viewer._MAX_LINE_BYTES + 1) + b"\n")
            stream.write(json.dumps(_record(api="cuMemAlloc")).encode() + b"\n")
        while follower.model.events < 2:
            follower.poll()
        self.assertEqual(follower.model.events, 2)
        self.assertEqual(follower.model.malformed_lines, 1)

    def test_resets_on_truncation_and_replacement(self):
        self.trace.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
        follower = _follow.TraceFollower(self.trace)
        self.addCleanup(follower.close)
        follower.poll()

        replacement = self.root / "replacement.jsonl"
        replacement.write_text(
            json.dumps(_record(api="cuMemFree")) + "\n", encoding="utf-8"
        )
        replacement.replace(self.trace)
        update = follower.poll()
        self.assertTrue(update.reset)
        self.assertEqual(follower.model.events, 1)
        self.assertIn("cuMemFree", follower.model.apis)
        self.assertNotIn("cuLaunchKernel", follower.model.apis)

        self.trace.write_text(
            json.dumps(_record(api="cuCtxSynchronize")) + "\n",
            encoding="utf-8",
        )
        update = follower.poll()
        self.assertTrue(update.reset)
        self.assertIn("cuCtxSynchronize", follower.model.apis)

    def test_summary_retries_partial_write_and_reconciles(self):
        self.trace.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
        follower = _follow.TraceFollower(self.trace)
        self.addCleanup(follower.close)
        follower.poll()
        summary_path = self.root / "summary.json"
        summaries = _follow.SummaryFollower(summary_path)

        summary_path.write_text("{", encoding="utf-8")
        self.assertFalse(summaries.poll(follower.model))
        self.assertIsNotNone(summaries.last_error)

        summary_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "complete": True,
                    "capture": {"events": 2},
                }
            ),
            encoding="utf-8",
        )
        self.assertTrue(summaries.poll(follower.model))
        self.assertEqual(follower.model.status, "MISMATCH")
        follower.model.observe(_viewer.parse_event(_record()))
        self.assertEqual(follower.model.status, "COMPLETE")

    def test_stat_failure_is_not_reported_as_waiting(self):
        follower = _follow.TraceFollower(self.trace)
        self.addCleanup(follower.close)
        with mock.patch.object(Path, "stat", side_effect=PermissionError(13, "denied")):
            with self.assertRaisesRegex(_viewer.ViewerError, "cannot inspect trace"):
                follower.poll()

    def test_disappearance_keeps_model_then_recreation_resets(self):
        self.trace.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
        follower = _follow.TraceFollower(self.trace)
        self.addCleanup(follower.close)
        follower.poll()

        self.trace.unlink()
        self.assertTrue(follower.poll().waiting)
        self.assertEqual(follower.model.events, 1)

        self.trace.write_text(
            json.dumps(_record(api="cuMemFree")) + "\n", encoding="utf-8"
        )
        update = follower.poll()
        self.assertTrue(update.reset)
        self.assertEqual(follower.model.events, 1)
        self.assertIn("cuMemFree", follower.model.apis)
        self.assertNotIn("cuLaunchKernel", follower.model.apis)

    def test_same_prefix_larger_rewrite_resets_from_tail_checkpoint(self):
        shared = (
            json.dumps(
                _record(api="cuMemAlloc", details={"padding": "x" * 400}),
                separators=(",", ":"),
            ).encode()
            + b"\n"
        )
        old_tail = (
            json.dumps(_record(api="cuLaunchKernel"), separators=(",", ":")).encode()
            + b"\n"
        )
        new_tail = (
            json.dumps(
                _record(
                    api="cuCtxSynchronize",
                    details={"padding": "y" * 400},
                ),
                separators=(",", ":"),
            ).encode()
            + b"\n"
        )
        old_content = shared + old_tail
        new_content = shared + new_tail
        self.assertEqual(
            old_content[: _follow._PREFIX_BYTES], new_content[: _follow._PREFIX_BYTES]
        )
        self.assertGreaterEqual(len(new_content), len(old_content))

        self.trace.write_bytes(old_content)
        original = self.trace.stat()
        follower = _follow.TraceFollower(self.trace)
        self.addCleanup(follower.close)
        follower.poll()

        self.trace.write_bytes(new_content)
        changed_mtime = max(time.time_ns(), original.st_mtime_ns + 1)
        os.utime(self.trace, ns=(changed_mtime, changed_mtime))
        self.assertEqual(self.trace.stat().st_ino, original.st_ino)
        update = follower.poll()

        self.assertTrue(update.reset)
        self.assertEqual(follower.model.events, 2)
        self.assertIn("cuCtxSynchronize", follower.model.apis)
        self.assertNotIn("cuLaunchKernel", follower.model.apis)

    def test_shorter_same_inode_truncation_resets(self):
        self.trace.write_text(
            "".join(
                json.dumps(_record(api=api)) + "\n"
                for api in ("cuMemAlloc", "cuLaunchKernel", "cuMemFree")
            ),
            encoding="utf-8",
        )
        original_inode = self.trace.stat().st_ino
        follower = _follow.TraceFollower(self.trace)
        self.addCleanup(follower.close)
        follower.poll()

        self.trace.write_text(
            json.dumps(_record(api="cuCtxSynchronize")) + "\n",
            encoding="utf-8",
        )
        self.assertEqual(self.trace.stat().st_ino, original_inode)
        update = follower.poll()

        self.assertTrue(update.reset)
        self.assertEqual(follower.model.events, 1)
        self.assertEqual(set(follower.model.apis), {"cuCtxSynchronize"})

    def test_summary_absence_empty_and_error_clear_stale_state(self):
        model = _viewer.TraceModel()
        model.observe(_viewer.parse_event(_record()))
        summary_path = self.root / "summary.json"
        summaries = _follow.SummaryFollower(summary_path)
        valid = {
            "schema_version": 1,
            "complete": True,
            "capture": {"events": 1},
        }

        summary_path.write_text(json.dumps(valid), encoding="utf-8")
        self.assertTrue(summaries.poll(model))
        self.assertEqual(model.status, "COMPLETE")

        summary_path.unlink()
        self.assertFalse(summaries.poll(model))
        self.assertIsNone(model.summary)
        self.assertIsNone(summaries.last_error)

        summary_path.write_bytes(b"")
        self.assertFalse(summaries.poll(model))
        self.assertIsNone(model.summary)
        self.assertIsNone(summaries.last_error)

        summary_path.write_text("{", encoding="utf-8")
        self.assertFalse(summaries.poll(model))
        self.assertIsNone(model.summary)
        self.assertIsNotNone(summaries.last_error)

        valid["complete"] = False
        summary_path.write_text(json.dumps(valid), encoding="utf-8")
        self.assertTrue(summaries.poll(model))
        self.assertEqual(model.status, "INCOMPLETE")
        self.assertIsNone(summaries.last_error)

    def test_summary_reset_ignores_older_document(self):
        model = _viewer.TraceModel()
        model.observe(_viewer.parse_event(_record()))
        summary_path = self.root / "summary.json"
        summary = {
            "schema_version": 1,
            "complete": True,
            "capture": {"events": 1},
        }
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        old_mtime = summary_path.stat().st_mtime_ns
        summaries = _follow.SummaryFollower(summary_path)
        summaries.reset(old_mtime + 1)

        self.assertFalse(summaries.poll(model))
        self.assertIsNone(model.summary)

        replacement = self.root / "replacement-summary.json"
        replacement.write_text(json.dumps(summary), encoding="utf-8")
        replacement.replace(summary_path)
        self.assertTrue(summaries.poll(model))
        self.assertEqual(model.status, "COMPLETE")


class LiveDashboardRendererTest(unittest.TestCase):
    def test_rate_tracker_handles_progress_and_reset(self):
        rates = _tui.RateTracker(window_seconds=5)
        rates.observe(10.0, 10)
        rates.observe(12.0, 30)
        self.assertEqual(rates.events_per_second, 10.0)
        rates.observe(13.0, 1)
        self.assertEqual(rates.events_per_second, 0.0)

    def test_live_dashboard_fits_terminal_and_shows_overview(self):
        model = _viewer.TraceModel()
        model.observe(_viewer.parse_event(_record()))
        lines = _tui.render_live_dashboard(model, width=100, height=24, event_rate=42.5)
        rendered = "\n".join(lines)
        self.assertEqual(len(lines), 24)
        self.assertTrue(all(len(line) <= 100 for line in lines))
        self.assertIn("LIVE", lines[0])
        self.assertIn("Rate 42.5/s", rendered)
        self.assertIn("TOP APIS", rendered)
        self.assertIn("TOP FUNCTIONS", rendered)
        self.assertIn("TOP KERNELS", rendered)
        self.assertIn("RECENT EVENTS", rendered)
        self.assertIn("q quit", lines[-1])

    def test_minimum_frame_reserves_summary_warning(self):
        model = _viewer.TraceModel()
        model.observe(_viewer.parse_event(_record()))
        lines = _tui.render_live_dashboard(
            model,
            width=79,
            height=18,
            event_rate=1.0,
            summary_error="invalid summary",
        )
        rendered = "\n".join(lines)
        self.assertEqual(len(lines), 18)
        self.assertTrue(all(len(line) <= 79 for line in lines))
        self.assertIn("TOP APIS", rendered)
        self.assertIn("TOP FUNCTIONS", rendered)
        self.assertIn("TOP KERNELS", rendered)
        self.assertIn("RECENT EVENTS", rendered)
        self.assertEqual(lines[-2], "Summary warning: invalid summary")
        self.assertIn("q quit", lines[-1])

    def test_color_setup_falls_back_without_failing_dashboard(self):
        class CursesError(Exception):
            pass

        curses_module = mock.Mock()
        curses_module.error = CursesError
        curses_module.COLOR_BLACK = 0
        curses_module.COLOR_CYAN = 6
        curses_module.COLOR_YELLOW = 3
        curses_module.has_colors.return_value = True
        curses_module.use_default_colors.side_effect = CursesError()
        curses_module.color_pair.side_effect = (11, 22)

        self.assertEqual(
            _tui._color_attributes(curses_module),
            (11, 22),
        )
        self.assertEqual(
            curses_module.init_pair.call_args_list,
            [mock.call(1, 6, 0), mock.call(2, 3, 0)],
        )

        curses_module.init_pair.reset_mock()
        curses_module.init_pair.side_effect = CursesError()
        self.assertEqual(_tui._color_attributes(curses_module), (0, 0))

    def test_curses_loop_pauses_resumes_and_closes(self):
        class CursesError(Exception):
            pass

        class FakeCurses:
            error = CursesError

            @staticmethod
            def curs_set(_visibility):
                return None

            @staticmethod
            def has_colors():
                return False

        class FakeScreen:
            def __init__(self):
                self.keys = [ord("p"), ord("p"), ord("q")]
                self.sizes = [(17, 79), (18, 80), (18, 80)]
                self.writes = []

            def keypad(self, _enabled):
                return None

            def timeout(self, _milliseconds):
                return None

            def getmaxyx(self):
                return self.sizes.pop(0)

            def erase(self):
                return None

            def addnstr(self, _row, _column, line, _length, _attributes):
                self.writes.append(line)

            def refresh(self):
                return None

            def getch(self):
                return self.keys.pop(0)

        class FakeFollower:
            def __init__(self):
                self.model = _viewer.TraceModel()
                self.trace_mtime_ns = 0
                self.polls = 0
                self.closed = False

            def poll(self):
                self.polls += 1
                return _follow.FollowUpdate(waiting=False)

            def close(self):
                self.closed = True

        class FakeSummaries:
            last_error = None

            def __init__(self):
                self.resets = 0

            def reset(self, _mtime_ns):
                self.resets += 1

            def poll(self, _model):
                return False

        class FakeRates:
            events_per_second = 0.0

            def __init__(self):
                self.resets = 0
                self.observations = 0

            def reset(self):
                self.resets += 1

            def observe(self, _now, _events):
                self.observations += 1

        screen = FakeScreen()
        follower = FakeFollower()
        summaries = FakeSummaries()
        rates = FakeRates()
        with (
            mock.patch.object(_follow, "TraceFollower", return_value=follower),
            mock.patch.object(_follow, "SummaryFollower", return_value=summaries),
            mock.patch.object(_tui, "RateTracker", return_value=rates),
        ):
            result = _tui._run_curses(
                screen, FakeCurses, Path("trace.jsonl"), None, 10, 0.2
            )

        self.assertEqual(result, 0)
        self.assertEqual(follower.polls, 2)
        self.assertTrue(follower.closed)
        self.assertEqual(summaries.resets, 1)
        self.assertEqual(rates.resets, 2)
        self.assertEqual(rates.observations, 2)
        self.assertIn("Resize to at least 80x18", screen.writes)


class WebDashboardTest(unittest.TestCase):
    TOKEN = "receiver-token-" + ("r" * 32)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def _start_ingest_server(self, lan_mode=False):
        state = _web.IngestDashboardState(recent_limit=50, refresh_seconds=0.05)
        server = _web._DashboardServer(
            ("127.0.0.1", 0),
            _web._DashboardRequestHandler,
        )
        # Exercise LAN-mode rules without opening every interface in the suite.
        server.lan_mode = lan_mode
        server.dashboard_state = state
        server.ingest_token = self.TOKEN
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def stop_server():
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.addCleanup(stop_server)
        return state, server

    def _closed_by_server(self, connection, within=2.0):
        connection.settimeout(within)
        try:
            return connection.recv(1) == b""
        except (socket.timeout, ConnectionResetError) as exc:
            return isinstance(exc, ConnectionResetError)

    def test_idle_connection_is_closed_after_the_timeout(self):
        self.assertGreater(_web._DashboardRequestHandler.timeout or 0, 0)
        with mock.patch.object(_web._DashboardRequestHandler, "timeout", 0.2):
            _, server = self._start_ingest_server()
            with socket.create_connection(server.server_address) as idle:
                self.assertTrue(self._closed_by_server(idle))

    def test_connections_beyond_the_cap_are_closed_at_once(self):
        with mock.patch.object(_web, "_MAX_CONNECTIONS", 2):
            _, server = self._start_ingest_server()
        held = [socket.create_connection(server.server_address)
                for _ in range(2)]
        try:
            for connection in held:
                connection.sendall(b"GET / HTTP/1.1\r\n")   # keep it mid-request
            deadline = time.monotonic() + 2.0
            while server._slots._value and time.monotonic() < deadline:
                time.sleep(0.01)
            with socket.create_connection(server.server_address) as extra:
                self.assertTrue(self._closed_by_server(extra))
        finally:
            for connection in held:
                connection.close()
        # A freed slot serves the next client normally.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with socket.create_connection(server.server_address) as client:
                client.sendall(b"GET / HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                               b"Connection: close\r\n\r\n")
                client.settimeout(2.0)
                if client.recv(12).startswith(b"HTTP/1.1 200"):
                    break
            time.sleep(0.02)
        else:
            self.fail("no connection was served after slots were freed")

    def _post_json(
        self,
        server,
        path,
        payload,
        *,
        token=None,
        content_type="application/json",
    ):
        body = json.dumps(payload, separators=(",", ":")).encode()
        connection = http.client.HTTPConnection(
            "127.0.0.1",
            server.server_port,
            timeout=2,
        )
        try:
            connection.request(
                "POST",
                path,
                body=body,
                headers={
                    "Authorization": f"Bearer {self.TOKEN if token is None else token}",
                    "Content-Type": content_type,
                },
            )
            response = connection.getresponse()
            raw = response.read()
            return response.status, response.headers, json.loads(raw)
        finally:
            connection.close()

    def _raw_http(self, server, request):
        with socket.create_connection(
            ("127.0.0.1", server.server_port),
            timeout=2,
        ) as connection:
            connection.sendall(request)
            connection.shutdown(socket.SHUT_WR)
            stream = connection.makefile("rb")
            status_line = stream.readline().decode("latin-1").rstrip()
            headers = {}
            while True:
                line = stream.readline()
                if line in (b"\r\n", b"\n", b""):
                    break
                name, value = line.decode("latin-1").split(":", 1)
                headers[name.lower()] = value.strip()
            length = int(headers.get("content-length", "0"))
            body = stream.read(length)
        return int(status_line.split()[1]), headers, body

    def _state_status(self, server, host_header):
        status, _headers, _body = self._raw_http(
            server,
            (f"GET /api/state HTTP/1.1\r\n"
             f"Host: {host_header}\r\n"
             f"Authorization: Bearer {server.viewer_token}\r\n"
             "Connection: close\r\n\r\n").encode(),
        )
        return status

    def test_lan_mode_accepts_ipv4_host_and_rejects_dns_names(self):
        state, server = self._start_ingest_server(lan_mode=True)
        state.start_capture("a" * 32, "private-trace.py")
        port = server.server_port
        self.assertEqual(self._state_status(server, f"10.1.2.3:{port}"), 200)
        self.assertEqual(self._state_status(server, f"127.0.0.1:{port}"), 200)
        # DNS names stay rejected: that is what defeats DNS rebinding.
        self.assertEqual(self._state_status(server, f"evil.example:{port}"), 403)
        self.assertEqual(self._state_status(server, f"10.1.2.3:{port + 1}"), 403)

    def test_default_mode_rejects_non_loopback_ipv4_host(self):
        state, server = self._start_ingest_server()
        state.start_capture("a" * 32, "private-trace.py")
        port = server.server_port
        self.assertEqual(self._state_status(server, f"10.1.2.3:{port}"), 403)
        self.assertEqual(self._state_status(server, f"127.0.0.1:{port}"), 200)

    def test_lan_mode_ingest_from_loopback_still_works(self):
        state, server = self._start_ingest_server(lan_mode=True)
        status, _headers, _body = self._post_json(
            server,
            "/api/capture/start",
            {"schema_version": 1, "capture_id": "c" * 32, "trace_name": "lan.py"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(state.payload()["trace_name"], "lan.py")

    def test_lan_mode_ingest_keeps_strict_loopback_host(self):
        # Ingest must not inherit the relaxed LAN Host rule (allow_lan=False).
        state, server = self._start_ingest_server(lan_mode=True)
        body = json.dumps(
            {"schema_version": 1, "capture_id": "d" * 32, "trace_name": "x.py"}
        ).encode()
        status, _headers, _body = self._raw_http(
            server,
            (f"POST /api/capture/start HTTP/1.1\r\n"
             f"Host: 10.1.2.3:{server.server_port}\r\n"
             f"Authorization: Bearer {self.TOKEN}\r\n"
             "Content-Type: application/json\r\n"
             f"Content-Length: {len(body)}\r\n"
             "Connection: close\r\n\r\n").encode() + body,
        )
        self.assertEqual(status, 403)
        self.assertNotEqual(state.payload().get("trace_name"), "x.py")

    def test_ingest_preflight_rejects_non_loopback_client(self):
        # The producer path stays loopback-only even when viewers are on the LAN.
        handler = types.SimpleNamespace(client_address=("10.1.2.3", 51515))
        with self.assertRaises(_web._IngestRequestError) as caught:
            _web._DashboardRequestHandler._post_preflight(handler)
        self.assertEqual(caught.exception.status, 403)

    def test_viewer_paths_refuse_public_clients_first(self):
        # A stub without headers or path proves the check runs before either.
        for method in ("_serve", "do_OPTIONS"):
            with self.subTest(method=method):
                sent = []
                handler = types.SimpleNamespace(
                    client_address=("8.8.8.8", 51515),
                    _send=lambda *args, status=200, **kwargs: sent.append(status),
                    _send_error=lambda status, message, **kwargs: sent.append(status),
                )
                getattr(_web._DashboardRequestHandler, method)(handler)
                self.assertEqual(sent, [403])

    def test_client_is_internal(self):
        cases = (
            (("127.0.0.1", 1), True),
            (("10.1.2.3", 1), True),
            (("172.16.5.10", 1), True),
            (("100.64.0.10", 1), True),  # shared (CGNAT) range, e.g. Tailscale
            (("8.8.8.8", 1), False),
            (("not-an-ip", 1), False),
        )
        for address, expected in cases:
            with self.subTest(address=address):
                self.assertEqual(_web._client_is_internal(address), expected)

    def test_lan_bind_warns_and_default_does_not(self):
        cases = (
            ("0.0.0.0", True, "http://<this-host-ip>:"),
            ("127.0.0.1", False, "http://127.0.0.1:"),
        )
        for host, warns, url in cases:
            with self.subTest(host=host):
                error = io.StringIO()
                with (
                    mock.patch.object(
                        _web._DashboardServer,
                        "serve_forever",
                        side_effect=KeyboardInterrupt,
                    ),
                    contextlib.redirect_stderr(error),
                ):
                    result = _web.run_web_dashboard(
                        None, None, 50, 0.05, 0,
                        ingest_token=self.TOKEN, host=host,
                    )
                self.assertEqual(result, 130)
                output = error.getvalue()
                self.assertEqual("reachable from the network" in output, warns)
                self.assertIn(url, output)

    def test_state_rejects_missing_wrong_and_duplicate_viewer_credentials(self):
        state, server = self._start_ingest_server()
        state.start_capture("a" * 32, "private-trace.py")
        # A missing auth check or accepting the first duplicate leaks trace data.
        token = server.viewer_token
        for method in ("GET", "HEAD"):
            for credentials in ([], ["wrong"], [self.TOKEN], [token, token],
                                [token, "wrong"], ["wrong", token],
                                [f"{token}, Bearer {token}"]):
                with self.subTest(method=method, credentials=credentials):
                    authorization = "".join(
                        f"Authorization: Bearer {value}\r\n"
                        for value in credentials
                    )
                    status, headers, body = self._raw_http(
                        server,
                        (f"{method} /api/state HTTP/1.1\r\n"
                         f"Host: 127.0.0.1:{server.server_port}\r\n"
                         f"{authorization}Connection: close\r\n\r\n").encode(),
                    )
                    self.assertEqual(status, 401)
                    self.assertEqual(headers["www-authenticate"], "Bearer")
                    self.assertNotIn(b"private-trace.py", body)
                    self.assertNotIn(token.encode(), body)
                    self.assertNotIn(self.TOKEN.encode(), body)

    def test_viewer_token_reads_state_but_cannot_publish_or_leak_into_assets(self):
        state, server = self._start_ingest_server()
        token = server.viewer_token
        self.assertNotEqual(token, self.TOKEN)
        state.start_capture("a" * 32, "private-trace.py")
        base = f"http://127.0.0.1:{server.server_port}"
        request = urllib.request.Request(
            base + "/api/state", headers={"Authorization": f"Bearer {token}"}
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            body = response.read()
            self.assertEqual(json.loads(body)["trace_name"], "private-trace.py")
            self.assertNotIn(token.encode(), body)
            self.assertNotIn(self.TOKEN.encode(), body)
        request = urllib.request.Request(
            base + "/api/state", method="HEAD",
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.read(), b"")
        for path in ("/", "/index.html", "/app.js", "/app.css",
                     "/logo.svg", "/favicon.svg"):
            with self.subTest(path=path):
                with urllib.request.urlopen(base + path, timeout=2) as response:
                    body = response.read()
                    self.assertNotIn(b"private-trace.py", body)
                    self.assertNotIn(token.encode(), body)
                    self.assertNotIn(self.TOKEN.encode(), body)
        status, _, body = self._post_json(
            server, "/api/capture/start",
            {"schema_version": 1, "capture_id": "b" * 32,
             "trace_name": "unauthorized.py"}, token=token,
        )
        self.assertEqual(status, 401)
        self.assertEqual(body, {"error": "unauthorized"})
        self.assertEqual(state.payload()["trace_name"], "private-trace.py")

    @unittest.skipUnless(shutil.which("node"), "browser-script check needs Node.js")
    def test_browser_fragment_bootstrap_refresh_and_single_bearer(self):
        # Execute the shipped script; deleting cleanup/storage/headers must fail.
        harness = r"""
const vm = require('node:vm');
const fs = require('node:fs');
const script = fs.readFileSync(0, 'utf8');
const storage = new Map();
async function load(fragment, blocked = false, navigation = null) {
  let location = new URL('http://127.0.0.1:8765/?view=timeline' + fragment);
  const requests = [];
  const elements = new Map();
  const listeners = new Map();
  let nextPoll;
  const context = {
    URLSearchParams, Headers,
    document: {
      getElementById(id) {
        if (!elements.has(id)) elements.set(id, {addEventListener() {}});
        return elements.get(id);
      },
      querySelectorAll() { return []; }
    },
    window: {
      get location() { return location; },
      history: {replaceState(state, title, url) {
        location = new URL(url, location);
      }},
      sessionStorage: {
        getItem(key) { if (blocked) throw Error('blocked'); return storage.get(key) || null; },
        setItem(key, value) { if (blocked) throw Error('blocked'); storage.set(key, value); },
        removeItem(key) { if (blocked) throw Error('blocked'); storage.delete(key); }
      },
      addEventListener(name, callback) { listeners.set(name, callback); },
      setTimeout(callback) { nextPoll = callback; return 1; },
      clearTimeout() {}
    },
    fetch(path, options) {
      requests.push({path, headers: [...new Headers(options.headers)],
        urlAtFetch: location.href});
      return Promise.resolve({ok: false, status: 401});
    }
  };
  vm.runInNewContext(script, context);
  await new Promise(setImmediate);
  if (navigation !== null) {
    requests.length = 0;
    location.hash = navigation;
    listeners.get('hashchange')?.();
    if (nextPoll) nextPoll();
    await new Promise(setImmediate);
  }
  return {url: location.href, stored: [...storage.values()], requests};
}
(async () => {
const token = 'v'.repeat(43);
const first = await load('#viewer_token=' + token);
const refresh = await load('');
const replacement = await load('#viewer_token=' + 'n'.repeat(43));
const duplicate = await load('#viewer_token=' + token + '&viewer_token=wrong');
storage.clear();
const fresh = await load('');
const blocked = await load('#viewer_token=' + token, true);
storage.clear();
const fromFresh = await load('', false, '#viewer_token=' + token);
const fromOld = await load('', false, '#viewer_token=' + 'n'.repeat(43));
const toDuplicate = await load('', false, '#viewer_token=x&viewer_token=y');
console.log(JSON.stringify({first, refresh, replacement, duplicate, fresh, blocked,
  fromFresh, fromOld, toDuplicate}));
})();
"""
        result = subprocess.run(
            [shutil.which("node"), "-e", harness], input=_web._APP_JS.decode(),
            capture_output=True, text=True, timeout=30, check=True,
        )
        observed = json.loads(result.stdout)
        clean_url = "http://127.0.0.1:8765/?view=timeline"
        for name in ("first", "refresh"):
            with self.subTest(load=name):
                self.assertEqual(observed[name]["url"], clean_url)
                self.assertEqual(observed[name]["stored"], ["v" * 43])
                self.assertEqual(observed[name]["requests"], [{
                    "path": "/api/state", "urlAtFetch": clean_url,
                    "headers": [["authorization", "Bearer " + "v" * 43]],
                }])
        self.assertEqual(observed["replacement"]["stored"], ["n" * 43])
        self.assertEqual(observed["replacement"]["requests"][0]["headers"],
                         [["authorization", "Bearer " + "n" * 43]])
        for name in ("duplicate", "fresh", "blocked"):
            with self.subTest(load=name):
                self.assertEqual(observed[name]["url"], clean_url)
                self.assertEqual(observed[name]["requests"], [])
        # Same-document navigation must update the running page, not reload JS.
        for name, token in (("fromFresh", "v" * 43), ("fromOld", "n" * 43)):
            with self.subTest(navigation=name):
                self.assertEqual(observed[name]["url"], clean_url)
                self.assertEqual(observed[name]["stored"], [token])
                self.assertEqual(observed[name]["requests"], [{
                    "path": "/api/state", "urlAtFetch": clean_url,
                    "headers": [["authorization", "Bearer " + token]],
                }])
        self.assertEqual(observed["toDuplicate"]["url"], clean_url)
        self.assertEqual(observed["toDuplicate"]["stored"], [])
        self.assertEqual(observed["toDuplicate"]["requests"], [])

    @unittest.skipUnless(shutil.which("node"), "browser-script check needs Node.js")
    def test_browser_pasted_token_uses_session_storage_not_the_url(self):
        # A tab opened without the fragment shows the form; pasting fixes it.
        harness = r"""
const vm = require('node:vm');
const fs = require('node:fs');
const script = fs.readFileSync(0, 'utf8');
async function paste(text, blocked = false) {
  const storage = new Map();
  const location = new URL('http://127.0.0.1:8765/?view=timeline');
  const requests = [];
  const elements = new Map();
  const element = (id) => {
    if (!elements.has(id)) elements.set(id, {listeners: {}, hidden: true,
      addEventListener(name, callback) { this.listeners[name] = callback; }});
    return elements.get(id);
  };
  const context = {
    URLSearchParams, Headers,
    document: {getElementById: element, querySelectorAll() { return []; }},
    window: {
      location,
      history: {replaceState() { throw Error('URL must not change'); }},
      sessionStorage: {
        getItem(key) { return storage.get(key) || null; },
        setItem(key, value) { if (blocked) throw Error('blocked'); storage.set(key, value); },
        removeItem(key) { storage.delete(key); }
      },
      addEventListener() {},
      setTimeout() { return 1; },
      clearTimeout() {}
    },
    fetch(path, options) {
      requests.push([...new Headers(options.headers)]);
      return Promise.resolve({ok: false, status: 401});
    }
  };
  vm.runInNewContext(script, context);
  await new Promise(setImmediate);
  const formShownBeforePaste = !element('token-form').hidden;
  element('token-input').value = text;
  element('token-form').listeners.submit({preventDefault() {}});
  await new Promise(setImmediate);
  return {formShownBeforePaste, url: location.href, stored: [...storage.values()],
    requests, inputCleared: element('token-input').value === ''};
}
(async () => {
  const token = 'v'.repeat(43);
  console.log(JSON.stringify({
    raw: await paste('  ' + token + '\n'),
    url: await paste('http://10.0.0.5:8765/#viewer_token=' + token),
    junk: await paste('not a token'),
    duplicate: await paste('http://10.0.0.5:8765/#viewer_token=a&viewer_token=b'),
    blocked: await paste(token, true),
  }));
})();
"""
        result = subprocess.run(
            [shutil.which("node"), "-e", harness], input=_web._APP_JS.decode(),
            capture_output=True, text=True, timeout=30, check=True,
        )
        observed = json.loads(result.stdout)
        clean_url = "http://127.0.0.1:8765/?view=timeline"
        bearer = [["authorization", "Bearer " + "v" * 43]]
        for name in ("raw", "url"):
            with self.subTest(paste=name):
                self.assertTrue(observed[name]["formShownBeforePaste"])
                self.assertEqual(observed[name]["stored"], ["v" * 43])
                self.assertEqual(observed[name]["requests"], [bearer])
                self.assertEqual(observed[name]["url"], clean_url)
                self.assertTrue(observed[name]["inputCleared"])
        for name in ("junk", "duplicate", "blocked"):
            with self.subTest(paste=name):
                self.assertEqual(observed[name]["stored"], [])
                self.assertEqual(observed[name]["requests"], [])
                self.assertTrue(observed[name]["inputCleared"])

    def test_ingest_state_lifecycle_idempotency_and_reset(self):
        state = _web.IngestDashboardState(recent_limit=50, refresh_seconds=0.05)
        first_id = "a" * 32
        second_id = "b" * 32

        initial = state.payload()
        self.assertEqual(initial["status"], "WAITING")
        self.assertEqual(initial["trace_name"], "Waiting for Docker capture")
        self.assertEqual(
            state.start_capture(first_id, "first.py"),
            {"schema_version": 1, "capture_id": first_id, "status": "live"},
        )
        event = _viewer.parse_event(_record())
        self.assertEqual(
            state.ingest_events(first_id, 0, [event]),
            {"schema_version": 1, "capture_id": first_id, "next_sequence": 1},
        )
        state.ingest_events(first_id, 0, [event])
        self.assertEqual(state.payload()["metrics"]["events"], 1)
        with self.assertRaisesRegex(_web.IngestConflict, "gap"):
            state.ingest_events(first_id, 2, [event])

        summary = {
            "schema_version": 1,
            "complete": True,
            "capture": {
                "events": 2,
                "delivery_dropped": 1,
                "lost_events": 0,
                "dropped_nested_calls": 0,
            },
        }
        finished = {
            "schema_version": 1,
            "capture_id": first_id,
            "next_sequence": 1,
            "status": "finished",
        }
        self.assertEqual(state.finish_capture(first_id, 1, summary), finished)
        self.assertEqual(state.finish_capture(first_id, 1, summary), finished)
        payload = state.payload()
        self.assertEqual(payload["status"], "INCOMPLETE")
        self.assertEqual(payload["metrics"]["delivery_dropped"], 1)
        with self.assertRaisesRegex(_web.IngestConflict, "terminal"):
            state.ingest_events(first_id, 1, [event])

        self.assertEqual(
            state.start_capture(first_id, "ignored.py")["status"],
            "live",
        )
        self.assertEqual(state.payload()["trace_name"], "first.py")
        state.start_capture(second_id, "second.py")
        reset = state.payload()
        self.assertEqual(reset["status"], "LIVE")
        self.assertEqual(reset["generation"], 1)
        self.assertEqual(reset["metrics"]["events"], 0)
        self.assertEqual(reset["metrics"]["delivery_dropped"], 0)
        with self.assertRaisesRegex(_web.IngestConflict, "not active"):
            state.ingest_events(first_id, 1, [event])

    def test_ingest_state_abort_is_idempotent_and_sanitized(self):
        state = _web.IngestDashboardState(recent_limit=50, refresh_seconds=0.05)
        capture_id = "c" * 32
        state.start_capture(capture_id, "abort.py")

        expected = {
            "schema_version": 1,
            "capture_id": capture_id,
            "status": "aborted",
        }
        self.assertEqual(state.abort_capture(capture_id, "lost\x1b[2J\nserver"), expected)
        self.assertEqual(state.abort_capture(capture_id, "different"), expected)
        payload = state.payload()
        self.assertEqual(payload["status"], "ERROR")
        self.assertEqual(payload["trace_error"], "lost?[2J?server")
        with self.assertRaisesRegex(_web.IngestConflict, "terminal"):
            state.finish_capture(
                capture_id,
                0,
                {"schema_version": 1, "complete": False, "capture": {"events": 0}},
            )

    def test_ingest_http_lifecycle_is_atomic_and_retry_safe(self):
        state, server = self._start_ingest_server()
        first_id = "d" * 32
        second_id = "e" * 32
        start = {
            "schema_version": 1,
            "capture_id": first_id,
            "trace_name": "first.py",
        }
        status, _headers, acknowledgement = self._post_json(
            server,
            "/api/capture/start",
            start,
        )
        self.assertEqual(status, 200)
        self.assertEqual(acknowledgement["status"], "live")

        invalid_batch = {
            "schema_version": 1,
            "capture_id": first_id,
            "sequence": 0,
            "events": [_record(), _record(duration_ns=-1)],
        }
        status, _headers, _body = self._post_json(
            server,
            "/api/capture/events",
            invalid_batch,
        )
        self.assertEqual(status, 400)
        self.assertEqual(state.payload()["metrics"]["events"], 0)
        self.assertEqual(state.expected_sequence, 0)

        event_batch = {
            **invalid_batch,
            "events": [_record(details={"gpu_total": 4096})],
        }
        status, _headers, first_ack = self._post_json(
            server,
            "/api/capture/events",
            event_batch,
        )
        self.assertEqual(status, 200)
        status, _headers, duplicate_ack = self._post_json(
            server,
            "/api/capture/events",
            event_batch,
        )
        self.assertEqual(status, 200)
        self.assertEqual(first_ack, duplicate_ack)
        self.assertEqual(state.payload()["metrics"]["events"], 1)

        gap = {**event_batch, "sequence": 2}
        status, _headers, _body = self._post_json(
            server,
            "/api/capture/events",
            gap,
        )
        self.assertEqual(status, 409)
        self.assertEqual(state.payload()["metrics"]["events"], 1)

        finish = {
            "schema_version": 1,
            "capture_id": first_id,
            "sequence": 1,
            "summary": {
                "schema_version": 1,
                "complete": True,
                "capture": {"events": 1},
            },
        }
        status, _headers, finish_ack = self._post_json(
            server,
            "/api/capture/finish",
            finish,
        )
        self.assertEqual(status, 200)
        status, _headers, duplicate_finish = self._post_json(
            server,
            "/api/capture/finish",
            finish,
        )
        self.assertEqual(status, 200)
        self.assertEqual(finish_ack, duplicate_finish)
        self.assertEqual(state.payload()["status"], "COMPLETE")

        terminal_batch = {**event_batch, "sequence": 1}
        status, _headers, _body = self._post_json(
            server,
            "/api/capture/events",
            terminal_batch,
        )
        self.assertEqual(status, 409)
        status, _headers, _body = self._post_json(
            server,
            "/api/capture/start",
            {**start, "capture_id": second_id, "trace_name": "second.py"},
        )
        self.assertEqual(status, 200)
        status, _headers, _body = self._post_json(
            server,
            "/api/capture/events",
            terminal_batch,
        )
        self.assertEqual(status, 409)
        self.assertEqual(state.payload()["generation"], 1)
        self.assertEqual(state.payload()["metrics"]["events"], 0)

    def test_ingest_http_validates_protocol_and_abort_idempotency(self):
        state, server = self._start_ingest_server()
        capture_id = "6" * 32
        malformed = (
            "POST /api/capture/start HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{server.server_port}\r\n"
            f"Authorization: Bearer {self.TOKEN}\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: 1\r\n\r\n{"
        ).encode()
        status, _headers, body = self._raw_http(server, malformed)
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body), {"error": "invalid request"})

        invalid_starts = (
            {
                "schema_version": 2,
                "capture_id": capture_id,
                "trace_name": "workload.py",
            },
            {
                "schema_version": 1,
                "capture_id": "A" * 32,
                "trace_name": "workload.py",
            },
            {
                "schema_version": 1,
                "capture_id": capture_id,
                "trace_name": "../workload.py",
            },
            {
                "schema_version": 1,
                "capture_id": capture_id,
                "trace_name": "workload.py",
                "extra": True,
            },
        )
        for request in invalid_starts:
            with self.subTest(request=request):
                status, _headers, _body = self._post_json(
                    server,
                    "/api/capture/start",
                    request,
                )
                self.assertEqual(status, 400)
        self.assertEqual(state.payload()["status"], "WAITING")

        status, _headers, acknowledgement = self._post_json(
            server,
            "/api/capture/start",
            {
                "schema_version": 1,
                "capture_id": capture_id,
                "trace_name": "workload.py",
            },
            content_type="application/json; charset=UTF-8",
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            set(acknowledgement),
            {"schema_version", "capture_id", "status"},
        )

        invalid_event_lists = ([], [_record()] * 129)
        for events in invalid_event_lists:
            with self.subTest(event_count=len(events)):
                status, _headers, _body = self._post_json(
                    server,
                    "/api/capture/events",
                    {
                        "schema_version": 1,
                        "capture_id": capture_id,
                        "sequence": 0,
                        "events": events,
                    },
                )
                self.assertEqual(status, 400)
        self.assertEqual(state.expected_sequence, 0)
        self.assertEqual(state.payload()["metrics"]["events"], 0)

        status, _headers, _body = self._post_json(
            server,
            "/api/capture/finish",
            {
                "schema_version": 1,
                "capture_id": capture_id,
                "sequence": 0,
                "summary": {
                    "schema_version": 1,
                    "complete": False,
                    "capture": {"events": 0, "delivery_dropped": 1},
                },
            },
        )
        self.assertEqual(status, 400)
        self.assertEqual(state.payload()["status"], "LIVE")

        abort = {
            "schema_version": 1,
            "capture_id": capture_id,
            "message": "receiver lost\x1b[2J",
        }
        status, _headers, first = self._post_json(
            server,
            "/api/capture/abort",
            abort,
        )
        self.assertEqual(status, 200)
        status, _headers, second = self._post_json(
            server,
            "/api/capture/abort",
            {**abort, "message": "retry"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(first, second)
        self.assertEqual(
            set(first),
            {"schema_version", "capture_id", "status"},
        )
        payload = state.payload()
        self.assertEqual(payload["status"], "ERROR")
        self.assertNotIn("\x1b", payload["trace_error"])

    def test_ingest_http_rejects_untrusted_request_boundaries(self):
        state, server = self._start_ingest_server()
        capture_id = "f" * 32
        start = {
            "schema_version": 1,
            "capture_id": capture_id,
            "trace_name": "secret.py",
        }

        status, headers, body = self._post_json(
            server,
            "/api/capture/start",
            start,
            token="wrong-" + self.TOKEN,
        )
        self.assertEqual(status, 401)
        self.assertEqual(headers["WWW-Authenticate"], "Bearer")
        self.assertEqual(body, {"error": "unauthorized"})
        self.assertNotIn(self.TOKEN, json.dumps(body))
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(headers["Referrer-Policy"], "no-referrer")
        self.assertIn("default-src 'none'", headers["Content-Security-Policy"])
        self.assertEqual(state.payload()["status"], "WAITING")

        encoded = json.dumps(start, separators=(",", ":")).encode()
        base_headers = (
            f"Host: 127.0.0.1:{server.server_port}\r\n"
            "Content-Type: application/json\r\n"
        )
        missing_auth = (
            "POST /api/capture/start HTTP/1.1\r\n"
            + base_headers
            + f"Content-Length: {len(encoded)}\r\n\r\n"
        ).encode() + encoded
        status, _headers, _body = self._raw_http(server, missing_auth)
        self.assertEqual(status, 401)

        bad_host = (
            "POST /api/capture/start HTTP/1.1\r\n"
            "Host: attacker.example\r\n"
            f"Authorization: Bearer {self.TOKEN}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(encoded)}\r\n\r\n"
        ).encode() + encoded
        status, _headers, _body = self._raw_http(server, bad_host)
        self.assertEqual(status, 403)

        status, _headers, _body = self._post_json(
            server,
            "/api/capture/start",
            start,
            content_type="text/plain",
        )
        self.assertEqual(status, 415)

        raw_cases = (
            (
                "duplicate authorization",
                (
                    "POST /api/capture/start HTTP/1.1\r\n"
                    + base_headers
                    + f"Authorization: Bearer {self.TOKEN}\r\n"
                    + f"Authorization: Bearer {self.TOKEN}\r\n"
                    + f"Content-Length: {len(encoded)}\r\n\r\n"
                ).encode()
                + encoded,
                401,
            ),
            (
                "missing content type",
                (
                    "POST /api/capture/start HTTP/1.1\r\n"
                    + f"Host: 127.0.0.1:{server.server_port}\r\n"
                    + f"Authorization: Bearer {self.TOKEN}\r\n"
                    + f"Content-Length: {len(encoded)}\r\n\r\n"
                ).encode()
                + encoded,
                415,
            ),
            (
                "missing length",
                (
                    "POST /api/capture/start HTTP/1.1\r\n"
                    + base_headers
                    + f"Authorization: Bearer {self.TOKEN}\r\n\r\n"
                ).encode(),
                411,
            ),
            (
                "invalid length",
                (
                    "POST /api/capture/start HTTP/1.1\r\n"
                    + base_headers
                    + f"Authorization: Bearer {self.TOKEN}\r\n"
                    + "Content-Length: nope\r\n\r\n"
                ).encode(),
                400,
            ),
            (
                "absurd length",
                (
                    "POST /api/capture/start HTTP/1.1\r\n"
                    + base_headers
                    + f"Authorization: Bearer {self.TOKEN}\r\n"
                    + "Content-Length: 999999999999999999999\r\n\r\n"
                ).encode(),
                413,
            ),
            (
                "duplicate length",
                (
                    "POST /api/capture/start HTTP/1.1\r\n"
                    + base_headers
                    + f"Authorization: Bearer {self.TOKEN}\r\n"
                    + "Content-Length: 2\r\nContent-Length: 2\r\n\r\n{}"
                ).encode(),
                400,
            ),
            (
                "transfer encoding",
                (
                    "POST /api/capture/start HTTP/1.1\r\n"
                    + base_headers
                    + f"Authorization: Bearer {self.TOKEN}\r\n"
                    + "Transfer-Encoding: chunked\r\n\r\n0\r\n\r\n"
                ).encode(),
                400,
            ),
            (
                "short body",
                (
                    "POST /api/capture/start HTTP/1.1\r\n"
                    + base_headers
                    + f"Authorization: Bearer {self.TOKEN}\r\n"
                    + "Content-Length: 10\r\n\r\n{}"
                ).encode(),
                400,
            ),
        )
        for name, request, expected in raw_cases:
            with self.subTest(name=name):
                status, _headers, _body = self._raw_http(server, request)
                self.assertEqual(status, expected)

        size_limits = (
            ("/api/capture/start", _web._MAX_CONTROL_BODY),
            ("/api/capture/events", _web._MAX_EVENT_BODY),
            ("/api/capture/finish", _web._MAX_FINISH_BODY),
            ("/api/capture/abort", _web._MAX_CONTROL_BODY),
        )
        for path, limit in size_limits:
            request = (
                f"POST {path} HTTP/1.1\r\n"
                + base_headers
                + f"Authorization: Bearer {self.TOKEN}\r\n"
                + "Expect: 100-continue\r\n"
                + f"Content-Length: {limit + 1}\r\n\r\n"
            ).encode()
            with self.subTest(path=path):
                status, _headers, _body = self._raw_http(server, request)
                self.assertEqual(status, 413)

        unauthorized_expect = (
            "POST /api/capture/events HTTP/1.1\r\n"
            + base_headers
            + "Authorization: Bearer wrong\r\n"
            + "Expect: 100-continue\r\n"
            + f"Content-Length: {_web._MAX_EVENT_BODY + 1}\r\n\r\n"
        ).encode()
        status, _headers, _body = self._raw_http(server, unauthorized_expect)
        self.assertEqual(status, 401)
        self.assertEqual(state.payload()["status"], "WAITING")

        options = (
            "OPTIONS /api/capture/start HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{server.server_port}\r\n\r\n"
        ).encode()
        status, headers, _body = self._raw_http(server, options)
        self.assertEqual(status, 405)
        self.assertNotIn("access-control-allow-origin", headers)
        self.assertEqual(headers["cache-control"], "no-store")

        bad_host_options = (
            "OPTIONS /api/capture/start HTTP/1.1\r\n"
            "Host: attacker.example\r\n\r\n"
        ).encode()
        status, _headers, _body = self._raw_http(server, bad_host_options)
        self.assertEqual(status, 403)

    def test_dashboard_state_response_can_exceed_ingest_ack_limit(self):
        _state, server = self._start_ingest_server()
        capture_id = "f" * 32
        status, _headers, _acknowledgement = self._post_json(
            server,
            "/api/capture/start",
            {
                "schema_version": 1,
                "capture_id": capture_id,
                "trace_name": "large.py",
            },
        )
        self.assertEqual(status, 200)
        status, _headers, _acknowledgement = self._post_json(
            server,
            "/api/capture/events",
            {
                "schema_version": 1,
                "capture_id": capture_id,
                "sequence": 0,
                "events": [
                    _record(timestamp_ns=index + 1) for index in range(50)
                ],
            },
        )
        self.assertEqual(status, 200)

        connection = http.client.HTTPConnection(
            "127.0.0.1",
            server.server_port,
            timeout=2,
        )
        try:
            connection.request(
                "GET", "/api/state",
                headers={"Authorization": f"Bearer {server.viewer_token}"},
            )
            response = connection.getresponse()
            body = response.read()
        finally:
            connection.close()

        self.assertEqual(response.status, 200)
        self.assertGreater(len(body), _web._MAX_RESPONSE_BODY)
        self.assertEqual(json.loads(body)["metrics"]["events"], 50)

    def test_file_backed_dashboard_disables_ingestion(self):
        class FixedState:
            @staticmethod
            def payload():
                return {"schema_version": 1, "status": "LIVE"}

        server = _web._DashboardServer(
            ("127.0.0.1", 0),
            _web._DashboardRequestHandler,
        )
        server.dashboard_state = FixedState()
        server.ingest_token = None
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def stop_server():
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.addCleanup(stop_server)
        status, headers, body = self._post_json(
            server,
            "/api/capture/start",
            {
                "schema_version": 1,
                "capture_id": "a" * 32,
                "trace_name": "ignored.py",
            },
        )
        self.assertEqual(status, 404)
        self.assertEqual(body, {"error": "not found"})
        self.assertEqual(headers["Cache-Control"], "no-store")

    def test_concurrent_ingest_snapshots_remain_consistent_and_bounded(self):
        state = _web.IngestDashboardState(recent_limit=5000, refresh_seconds=0.05)
        capture_id = "9" * 32
        state.start_capture(capture_id, "many.py")
        stop = threading.Event()
        errors = []

        def read_payloads():
            while not stop.is_set():
                try:
                    payload = state.payload()
                    if payload["metrics"]["events"] < 0:
                        errors.append("negative event count")
                except BaseException as exc:
                    errors.append(str(exc))
                    return

        reader = threading.Thread(target=read_payloads)
        reader.start()
        try:
            for sequence in range(1050):
                state.ingest_events(
                    capture_id,
                    sequence,
                    [
                        _viewer.parse_event(
                            _record(
                                timestamp=(
                                    "2026-08-30T12:10:03."
                                    f"{sequence:06d}+00:00"
                                ),
                                details={"gpu_total": sequence},
                            )
                        )
                    ],
                )
        finally:
            stop.set()
            reader.join(timeout=2)

        payload = state.payload()
        self.assertEqual(errors, [])
        self.assertEqual(payload["metrics"]["events"], 1050)
        self.assertEqual(len(payload["recent_events"]), 50)
        self.assertEqual(len(payload["memory_samples"]), 120)
        self.assertEqual(len(payload["timeline"]["events"]), 1000)
        self.assertEqual(payload["timeline"]["omitted"], 0)

    def test_dashboard_state_follows_trace_and_summary(self):
        trace = self.root / "events.jsonl"
        summary = self.root / "summary.json"
        trace.write_text(
            json.dumps(_record(details={"gpu_total": 4096})) + "\n",
            encoding="utf-8",
        )
        summary.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "complete": True,
                    "capture": {
                        "events": 1,
                        "lost_events": 0,
                        "dropped_nested_calls": 0,
                    },
                }
            ),
            encoding="utf-8",
        )
        state = _web.DashboardState(trace, summary, 50, 0.2)
        self.addCleanup(state.close)

        state.poll_once()
        payload = state.payload()

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["status"], "COMPLETE")
        self.assertFalse(payload["waiting"])
        self.assertEqual(payload["metrics"]["events"], 1)
        self.assertEqual(payload["metrics"]["observed"], "4.0KiB")
        self.assertEqual(payload["top_apis"][0]["name"], "cuLaunchKernel")
        self.assertEqual(payload["top_functions"][0]["name"], "compute")
        self.assertEqual(payload["top_kernels"][0]["name"], "vector_add")
        self.assertEqual(payload["recent_events"][0]["return_code"], 0)
        self.assertEqual(payload["memory_samples"][0]["bytes"], 4096)
        self.assertEqual(payload["timeline"]["events"][0]["family"], "launch")
        self.assertEqual(payload["timeline"]["events"][0]["lane"], "compute")
        self.assertEqual(payload["timeline"]["events"][0]["start_ns"], 0)
        self.assertEqual(payload["timeline"]["events"][0]["duration_ns"], 50_000)

    def test_dashboard_follower_thread_survives_recursion_bomb_line(self):
        # F5(a): a hostile line must not kill the background follower thread
        # while the dashboard keeps reporting LIVE. observe_raw_line()
        # returning False is necessary but not sufficient evidence; this
        # drives the real file follower thread end to end.
        trace = self.root / "events.jsonl"
        trace.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
        state = _web.DashboardState(trace, None, 50, 0.05)
        self.addCleanup(state.close)
        state.start()

        deadline = time.monotonic() + 2
        while state.follower.model.events == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(state.follower.model.events, 1)

        depth = sys.getrecursionlimit() * 2
        bomb = (b"[" * depth) + (b"]" * depth) + b"\n"
        with trace.open("ab") as handle:
            handle.write(bomb)

        deadline = time.monotonic() + 2
        while state.follower.model.malformed_lines == 0 and time.monotonic() < deadline:
            time.sleep(0.01)

        self.assertTrue(state._thread.is_alive())
        self.assertEqual(state.follower.model.malformed_lines, 1)
        self.assertEqual(state.payload()["status"], "LIVE / MALFORMED")

    def test_dashboard_payload_bounds_high_volume_sections(self):
        model = _viewer.TraceModel(recent_limit=500)
        for index in range(150):
            model.observe(
                _viewer.parse_event(
                    _record(
                        timestamp=f"2026-08-30T12:10:03.{index:06d}+00:00",
                        details={"gpu_total": index},
                    )
                )
            )

        payload = _web._model_payload(
            model,
            trace_name="events.jsonl",
            waiting=False,
            trace_error=None,
            summary_error=None,
            event_rate=25.0,
            refresh_seconds=0.2,
        )

        self.assertEqual(len(payload["recent_events"]), 50)
        self.assertEqual(len(payload["memory_samples"]), 120)
        self.assertEqual(payload["recent_events"][0]["details"]["gpu_total"], 149)
        self.assertEqual(payload["memory_samples"][0]["bytes"], 30)
        self.assertEqual(len(payload["timeline"]["events"]), 150)
        self.assertEqual(payload["timeline"]["omitted"], 0)
        self.assertEqual(payload["timeline"]["events"][-1]["id"], "150")

    def test_timeline_omits_invalid_timestamp_without_guessing(self):
        model = _viewer.TraceModel()
        model.observe(_viewer.parse_event(_record(timestamp="not-a-timestamp")))

        timeline = _web._timeline_payload(model)

        self.assertEqual(timeline["events"], [])
        self.assertEqual(timeline["omitted"], 1)
        self.assertIsNone(timeline["origin_timestamp"])

    def test_http_server_serves_assets_api_and_security_headers(self):
        class FixedState:
            @staticmethod
            def payload():
                return {"schema_version": 1, "status": "LIVE"}

        server = _web._DashboardServer(("127.0.0.1", 0), _web._DashboardRequestHandler)
        server.dashboard_state = FixedState()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def stop_server():
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.addCleanup(stop_server)
        base = f"http://127.0.0.1:{server.server_port}"

        with urllib.request.urlopen(base + "/", timeout=2) as response:
            html = response.read().decode()
            self.assertEqual(response.status, 200)
            self.assertEqual(response.headers["Cache-Control"], "no-store")
            self.assertNotIn(
                "unsafe-inline",
                response.headers["Content-Security-Policy"],
            )
        self.assertIn("METAGROSS", html)
        self.assertIn("/app.css", html)
        self.assertIn("/app.js", html)
        self.assertIn('src="/logo.svg" width="32" height="32" alt=""', html)
        self.assertIn("CUDA API timeline", html)
        self.assertIn('id="metric-delivery-dropped"', html)
        self.assertIn('aria-pressed="false">Pause', html)
        self.assertIn('role="tooltip"', html)
        self.assertIn(
            "Scroll horizontally for API, kernel, duration, and result.", html
        )
        with urllib.request.urlopen(base + "/app.css", timeout=2) as response:
            stylesheet = response.read().decode()
            self.assertIn("--dim: #969696", stylesheet)
            self.assertIn("--signal: #8ac926", stylesheet)
            self.assertIn(".timeline-tooltip", stylesheet)
            self.assertIn(".scroll-hint", stylesheet)

        with urllib.request.urlopen(base + "/app.js", timeout=2) as response:
            script = response.read().decode()
            self.assertIn("activeFamilies", script)
            self.assertIn("metric-delivery-dropped", script)
            self.assertIn("renderTimeline", script)
            self.assertIn("No timed CUDA events match the current filters.", script)
            self.assertIn("row.tabIndex = 0", script)
            self.assertIn('setAttribute("aria-selected"', script)
            self.assertIn("timelineSignature", script)

        master = Path(__file__).resolve().parent.parent / "assets/logo-mark.svg"
        path_tag = "{http://www.w3.org/2000/svg}path"
        expected_paths = [node.attrib["d"] for node in ET.parse(master).iter(path_tag)]
        for route in ("/logo.svg", "/favicon.svg"):
            with self.subTest(route=route):
                with urllib.request.urlopen(base + route, timeout=2) as response:
                    self.assertEqual(
                        response.headers["Content-Type"], "image/svg+xml; charset=utf-8"
                    )
                    self.assertEqual(response.headers["Cache-Control"], "no-store")
                    artwork = response.read().decode()
                self.assertIn("#8ac926", artwork)
                self.assertEqual(
                    [node.attrib["d"] for node in ET.fromstring(artwork).iter(path_tag)],
                    expected_paths,
                )

        request = urllib.request.Request(
            base + "/api/state",
            headers={"Authorization": f"Bearer {server.viewer_token}"},
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            self.assertEqual(
                json.loads(response.read()),
                {"schema_version": 1, "status": "LIVE"},
            )
            self.assertIsNone(response.headers["Access-Control-Allow-Origin"])

        connection = http.client.HTTPConnection(
            "127.0.0.1", server.server_port, timeout=2
        )
        self.addCleanup(connection.close)
        connection.putrequest("GET", "/", skip_host=True)
        connection.putheader("Host", "attacker.example")
        connection.endheaders()
        response = connection.getresponse()
        self.assertEqual(response.status, 403)
        response.read()

    def test_web_cli_starts_serves_current_trace_and_stops(self):
        trace = self.root / "events.jsonl"
        trace.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        process = subprocess.Popen(
            [
                "/usr/bin/python3",
                "-B",
                "-m",
                "metagross",
                "view",
                "--web",
                "--port",
                "0",
                str(trace),
            ],
            cwd=Path(__file__).resolve().parent.parent,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        def stop_process():
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()

        self.addCleanup(stop_process)

        url = None
        output = []
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and process.poll() is None:
            ready, _, _ = select.select([process.stderr], [], [], 0.1)
            if not ready:
                continue
            line = process.stderr.readline()
            output.append(line)
            marker = "dashboard available at "
            if marker in line:
                url = line.split(marker, 1)[1].strip()
                break
        self.assertIsNotNone(url, "".join(output))

        url, fragment = urllib.parse.urldefrag(url)
        viewer_tokens = urllib.parse.parse_qs(fragment).get("viewer_token", [])
        self.assertEqual(len(viewer_tokens), 1, "private URL needs a viewer fragment")
        viewer_token = viewer_tokens[0]
        self.assertNotIn(viewer_token, url)

        with urllib.request.urlopen(url, timeout=2) as response:
            self.assertIn("METAGROSS", response.read().decode())

        payload = None
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            request = urllib.request.Request(
                url + "api/state",
                headers={"Authorization": f"Bearer {viewer_token}"},
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                payload = json.loads(response.read())
            if payload["metrics"]["events"] == 1:
                break
            time.sleep(0.05)
        self.assertEqual(payload["metrics"]["events"], 1)
        self.assertEqual(payload["status"], "LIVE")
        self.assertEqual(len(payload["timeline"]["events"]), 1)
        self.assertEqual(payload["timeline"]["events"][0]["lane"], "compute")

        process.send_signal(signal.SIGINT)
        self.assertEqual(process.wait(timeout=5), 130)

        # A restarted process must reject the secret retained by an old tab.
        for stream in (process.stdout, process.stderr):
            stream.close()
        command = list(process.args)
        command[command.index("--port") + 1] = str(urllib.parse.urlsplit(url).port)
        process = subprocess.Popen(
            command, cwd=Path(__file__).resolve().parent.parent, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        restarted_url = None
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and process.poll() is None:
            ready, _, _ = select.select([process.stderr], [], [], 0.1)
            if ready:
                line = process.stderr.readline()
                if marker in line:
                    restarted_url = line.split(marker, 1)[1].strip()
                    break
        self.assertIsNotNone(restarted_url)
        restarted_base, fragment = urllib.parse.urldefrag(restarted_url)
        current_token = urllib.parse.parse_qs(fragment)["viewer_token"][0]
        self.assertEqual(restarted_base, url)
        self.assertNotEqual(current_token, viewer_token)
        for credential in (None, viewer_token, current_token):
            headers = {} if credential is None else {
                "Authorization": f"Bearer {credential}"
            }
            request = urllib.request.Request(url + "api/state", headers=headers)
            if credential == current_token:
                with urllib.request.urlopen(request, timeout=2) as response:
                    self.assertEqual(response.status, 200)
            else:
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(request, timeout=2)
                self.assertEqual(caught.exception.code, 401)
                caught.exception.close()
        process.send_signal(signal.SIGINT)
        self.assertEqual(process.wait(timeout=5), 130)


    def test_receive_cli_streams_fileless_capture_and_stops(self):
        token = "subprocess-token-" + ("s" * 32)
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["METAGROSS_DASHBOARD_TOKEN"] = token
        process = subprocess.Popen(
            [
                "/usr/bin/python3",
                "-B",
                "-m",
                "metagross",
                "view",
                "--web",
                "--receive",
                "--port",
                "0",
            ],
            cwd=Path(__file__).resolve().parent.parent,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        def stop_process():
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()

        self.addCleanup(stop_process)
        url = None
        output = []
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and process.poll() is None:
            ready, _, _ = select.select([process.stderr], [], [], 0.1)
            if not ready:
                continue
            line = process.stderr.readline()
            output.append(line)
            marker = "dashboard available at "
            if marker in line:
                url = line.split(marker, 1)[1].strip()
                break
        self.assertIsNotNone(url, "".join(output))

        self.assertNotIn(token, "".join(output))
        url, fragment = urllib.parse.urldefrag(url)
        viewer_tokens = urllib.parse.parse_qs(fragment).get("viewer_token", [])
        self.assertEqual(len(viewer_tokens), 1, "private URL needs a viewer fragment")
        viewer_token = viewer_tokens[0]
        self.assertNotEqual(viewer_token, token)
        self.assertNotIn(viewer_token, url)

        def state_payload():
            request = urllib.request.Request(
                url + "api/state",
                headers={"Authorization": f"Bearer {viewer_token}"},
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                return json.loads(response.read())

        def post(name, payload):
            request = urllib.request.Request(
                url + "api/capture/" + name,
                data=json.dumps(payload, separators=(",", ":")).encode(),
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                return json.loads(response.read())

        self.assertEqual(state_payload()["status"], "WAITING")
        capture_id = "7" * 32
        start = post(
            "start",
            {
                "schema_version": 1,
                "capture_id": capture_id,
                "trace_name": "container.py",
            },
        )
        self.assertEqual(start["status"], "live")
        self.assertEqual(state_payload()["status"], "LIVE")
        event_ack = post(
            "events",
            {
                "schema_version": 1,
                "capture_id": capture_id,
                "sequence": 0,
                "events": [_record()],
            },
        )
        self.assertEqual(event_ack["next_sequence"], 1)
        finish = post(
            "finish",
            {
                "schema_version": 1,
                "capture_id": capture_id,
                "sequence": 1,
                "summary": {
                    "schema_version": 1,
                    "complete": True,
                    "capture": {"events": 1, "delivery_dropped": 0},
                },
            },
        )
        self.assertEqual(finish["status"], "finished")
        final = state_payload()
        self.assertEqual(final["status"], "COMPLETE")
        self.assertEqual(final["metrics"]["events"], 1)
        self.assertNotIn(token, json.dumps(final))
        self.assertEqual(list(self.root.iterdir()), [])

        process.send_signal(signal.SIGINT)
        self.assertEqual(process.wait(timeout=5), 130)


class SnapshotRendererTest(unittest.TestCase):
    def test_fit_marks_truncated_values(self):
        self.assertEqual(_viewer._fit("abcdef", 5), "abcd~")

    def test_snapshot_is_bounded_and_contains_core_sections(self):
        model = _viewer.TraceModel()
        model.observe(_viewer.parse_event(_record()))
        lines = _viewer.render_snapshot(model, width=80)
        rendered = "\n".join(lines)
        self.assertTrue(all(len(line) <= 80 for line in lines))
        self.assertIn("METAGROSS TRACE", rendered)
        self.assertIn("TOP APIS", rendered)
        self.assertIn("TOP FUNCTIONS", rendered)
        self.assertIn("TOP KERNELS", rendered)
        self.assertIn("RECENT EVENTS", rendered)
        self.assertIn("cuLaunchKernel", rendered)

    def test_snapshot_displays_summary_mismatch_warning(self):
        model = _viewer.TraceModel()
        model.observe(_viewer.parse_event(_record()))
        model.load_summary(
            {
                "schema_version": 1,
                "complete": True,
                "capture": {"events": 2},
            }
        )
        rendered = "\n".join(_viewer.render_snapshot(model, width=100))
        self.assertIn("MISMATCH", rendered)
        self.assertIn("WARNING", rendered)

    def test_narrow_snapshot_preserves_numeric_columns(self):
        model = _viewer.TraceModel()
        model.observe(_viewer.parse_event(_record()))
        lines = _viewer.render_snapshot(model, width=60)
        rendered = "\n".join(lines)

        self.assertTrue(all(len(line) <= 60 for line in lines))
        self.assertIn("Outstanding 0B", rendered)
        self.assertTrue(
            next(line for line in lines if line.startswith("API")).endswith("MAX")
        )
        self.assertTrue(
            next(line for line in lines if line.startswith("FUNCTION")).endswith(
                "TOTAL CPU"
            )
        )
        self.assertTrue(
            next(line for line in lines if line.startswith("TIME")).endswith("CPU")
        )
        api_row = next(line for line in lines if line.startswith("cuLaunch"))
        self.assertTrue(api_row.endswith("50.0us"))


class ViewerRoutingTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.trace = Path(self.directory.name) / "events.jsonl"
        self.trace.write_text(json.dumps(_record()) + "\n", encoding="utf-8")

    def test_snapshot_routes_without_live_trace_validation(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = metagross.main(
                ["view", "--snapshot", "--width", "80", str(self.trace)]
            )
        self.assertEqual(result, 0)
        self.assertIn("METAGROSS TRACE", output.getvalue())

    def test_snapshot_loads_optional_summary(self):
        summary = Path(self.directory.name) / "summary.json"
        summary.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "complete": True,
                    "capture": {
                        "events": 1,
                        "lost_events": 0,
                        "dropped_nested_calls": 0,
                    },
                }
            ),
            encoding="utf-8",
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = metagross.main(
                [
                    "view",
                    "--snapshot",
                    "--summary",
                    str(summary),
                    str(self.trace),
                ]
            )
        self.assertEqual(result, 0)
        self.assertIn("COMPLETE", output.getvalue().splitlines()[0])

    def test_view_help_does_not_require_live_dependencies(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = metagross.main(["view", "--help"])
        self.assertEqual(result, 0)
        self.assertIn("without root, BCC, or CUDA", output.getvalue())

    def test_missing_view_mode_is_usage_error(self):
        error = io.StringIO()
        with contextlib.redirect_stderr(error):
            result = metagross.main(["view", str(self.trace)])
        self.assertEqual(result, 2)
        self.assertIn("choose --snapshot, --follow, or --web", error.getvalue())

    def test_live_limits_are_validated(self):
        for option, value in (("--refresh", "0.01"), ("--recent", "10001")):
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()):
                result = metagross.main(
                    ["view", "--follow", option, value, str(self.trace)]
                )
            self.assertEqual(result, 2)

    def test_snapshot_width_bounds_are_validated(self):
        for width in ("59", "241"):
            with self.subTest(width=width), contextlib.redirect_stderr(io.StringIO()):
                result = metagross.main(
                    ["view", "--snapshot", "--width", width, str(self.trace)]
                )
            self.assertEqual(result, 2)

    def test_web_port_bounds_are_validated(self):
        for port in ("-1", "65536"):
            with self.subTest(port=port), contextlib.redirect_stderr(io.StringIO()):
                result = metagross.main(
                    ["view", "--web", "--port", port, str(self.trace)]
                )
            self.assertEqual(result, 2)

    def test_web_host_must_be_numeric_ipv4(self):
        # A hostname URL would be rejected by the Host check in the browser.
        for host in ("localhost", "evil.example", "::1", "1.2.3", "256.1.1.1", ""):
            with self.subTest(host=host), contextlib.redirect_stderr(io.StringIO()):
                result = metagross.main(
                    ["view", "--web", "--host", host, str(self.trace)]
                )
            self.assertEqual(result, 2)

    def test_web_host_must_be_internal(self):
        cases = (
            ("8.8.8.8", "public addresses are refused"),
            ("1.1.1.1", "public addresses are refused"),
            ("127.0.0.2", "the only loopback address accepted is 127.0.0.1"),
        )
        for host, message in cases:
            error = io.StringIO()
            with self.subTest(host=host), contextlib.redirect_stderr(error):
                result = metagross.main(
                    ["view", "--web", "--host", host, str(self.trace)]
                )
            self.assertEqual(result, 2)
            self.assertIn(message, error.getvalue())

    def test_web_routes_internal_hosts(self):
        for host in ("172.16.5.10", "192.168.1.20", "169.254.1.1",
                     "100.64.0.10"):
            with (
                self.subTest(host=host),
                mock.patch.object(
                    _web, "run_web_dashboard", return_value=0
                ) as dashboard,
            ):
                result = metagross.main(
                    ["view", "--web", "--host", host, str(self.trace)]
                )
                self.assertEqual(result, 0)
                dashboard.assert_called_once_with(
                    self.trace, None, 500, 0.2, 8765, host=host
                )

    def test_mode_specific_options_are_rejected(self):
        cases = (
            (
                ["view", "--follow", "--width", "80", str(self.trace)],
                "--width is only valid with --snapshot",
            ),
            (
                ["view", "--web", "--width", "80", str(self.trace)],
                "--width is only valid with --snapshot",
            ),
            (
                ["view", "--snapshot", "--refresh", "0.2", str(self.trace)],
                "--refresh is only valid with --follow or --web",
            ),
            (
                ["view", "--snapshot", "--port", "8765", str(self.trace)],
                "--port is only valid with --web",
            ),
            (
                ["view", "--follow", "--port", "8765", str(self.trace)],
                "--port is only valid with --web",
            ),
            (
                ["view", "--snapshot", "--host", "0.0.0.0", str(self.trace)],
                "--host is only valid with --web",
            ),
            (
                ["view", "--follow", "--host", "0.0.0.0", str(self.trace)],
                "--host is only valid with --web",
            ),
            (
                # The tracer delivers captures only to 127.0.0.1, so a receiver
                # bound to one LAN address alone would silently refuse it.
                ["view", "--web", "--receive", "--host", "172.16.5.10"],
                "--receive requires --host 127.0.0.1 or 0.0.0.0",
            ),
            (
                ["view", "--snapshot", "--receive", str(self.trace)],
                "--receive is only valid with --web",
            ),
            (
                ["view", "--follow", "--receive", str(self.trace)],
                "--receive is only valid with --web",
            ),
            (
                ["view", "--web", "--receive", str(self.trace)],
                "TRACE is not valid with --receive",
            ),
            (
                [
                    "view",
                    "--web",
                    "--receive",
                    "--summary",
                    str(self.trace.with_name("summary.json")),
                ],
                "--summary is not valid with --receive",
            ),
        )
        for argv, message in cases:
            error = io.StringIO()
            with self.subTest(message=message), contextlib.redirect_stderr(error):
                result = metagross.main(argv)
            self.assertEqual(result, 2)
            self.assertIn(message, error.getvalue())

    def test_follow_routes_to_dashboard(self):
        with mock.patch.object(
            _tui, "run_follow_dashboard", return_value=0
        ) as dashboard:
            result = metagross.main(
                ["view", "--follow", "--refresh", "0.1", str(self.trace)]
            )
        self.assertEqual(result, 0)
        dashboard.assert_called_once_with(self.trace, None, 500, 0.1)

    def test_web_routes_to_loopback_dashboard(self):
        with mock.patch.object(_web, "run_web_dashboard", return_value=0) as dashboard:
            result = metagross.main(
                [
                    "view",
                    "--web",
                    "--refresh",
                    "0.1",
                    "--port",
                    "9000",
                    str(self.trace),
                ]
            )
        self.assertEqual(result, 0)
        dashboard.assert_called_once_with(
            self.trace, None, 500, 0.1, 9000, host="127.0.0.1"
        )

    def test_web_routes_lan_host(self):
        with mock.patch.object(_web, "run_web_dashboard", return_value=0) as dashboard:
            result = metagross.main(
                ["view", "--web", "--host", "0.0.0.0", "--port", "9000",
                 str(self.trace)]
            )
        self.assertEqual(result, 0)
        dashboard.assert_called_once_with(
            self.trace, None, 500, 0.2, 9000, host="0.0.0.0"
        )

    def test_receive_allows_wildcard_host(self):
        token = "route-token-" + ("y" * 32)
        with (
            mock.patch.object(_web, "run_web_dashboard", return_value=0) as dashboard,
            mock.patch.dict(
                os.environ,
                {"METAGROSS_DASHBOARD_TOKEN": token},
                clear=False,
            ),
        ):
            result = metagross.main(
                ["view", "--web", "--receive", "--host", "0.0.0.0",
                 "--port", "9000"]
            )
        self.assertEqual(result, 0)
        dashboard.assert_called_once_with(
            None, None, 500, 0.2, 9000, ingest_token=token, host="0.0.0.0"
        )

    def test_receive_routes_with_removed_token(self):
        token = "route-token-" + ("x" * 32)
        with (
            mock.patch.object(_web, "run_web_dashboard", return_value=0) as dashboard,
            mock.patch.dict(
                os.environ,
                {"METAGROSS_DASHBOARD_TOKEN": token},
                clear=False,
            ),
        ):
            result = metagross.main(
                [
                    "view",
                    "--web",
                    "--receive",
                    "--refresh",
                    "0.1",
                    "--port",
                    "9000",
                ]
            )
            self.assertNotIn("METAGROSS_DASHBOARD_TOKEN", os.environ)
        self.assertEqual(result, 0)
        dashboard.assert_called_once_with(
            None,
            None,
            500,
            0.1,
            9000,
            ingest_token=token,
            host="127.0.0.1",
        )

    def test_receive_rejects_missing_or_invalid_token_without_echoing_it(self):
        values = (None, "secret token with spaces")
        for value in values:
            environment = os.environ.copy()
            environment.pop("METAGROSS_DASHBOARD_TOKEN", None)
            if value is not None:
                environment["METAGROSS_DASHBOARD_TOKEN"] = value
            error = io.StringIO()
            with (
                self.subTest(value=value),
                mock.patch.dict(os.environ, environment, clear=True),
                contextlib.redirect_stderr(error),
            ):
                result = metagross.main(["view", "--web", "--receive"])
                self.assertNotIn("METAGROSS_DASHBOARD_TOKEN", os.environ)
            self.assertEqual(result, 2)
            if value is not None:
                self.assertNotIn(value, error.getvalue())

    def test_follow_requires_interactive_terminal(self):
        for stdin_tty, stdout_tty in ((False, True), (True, False)):
            fake_stdin = mock.Mock()
            fake_stdin.isatty.return_value = stdin_tty
            fake_stdout = mock.Mock()
            fake_stdout.isatty.return_value = stdout_tty
            error = io.StringIO()
            with (
                self.subTest(stdin_tty=stdin_tty, stdout_tty=stdout_tty),
                mock.patch.object(sys, "stdin", fake_stdin),
                mock.patch.object(sys, "stdout", fake_stdout),
                mock.patch.object(sys, "stderr", error),
            ):
                result = _tui.run_follow_dashboard(self.trace, None, 10, 0.2)
            self.assertEqual(result, 2)
            self.assertIn("interactive terminal", error.getvalue())

    def test_follow_reports_missing_curses_cleanly(self):
        terminal = mock.Mock()
        terminal.isatty.return_value = True
        error = io.StringIO()
        with (
            mock.patch.object(sys, "stdin", terminal),
            mock.patch.object(sys, "stdout", terminal),
            mock.patch.object(sys, "stderr", error),
            mock.patch.dict(sys.modules, {"curses": None}),
        ):
            result = _tui.run_follow_dashboard(self.trace, None, 10, 0.2)
        self.assertEqual(result, 1)
        self.assertIn("cannot start dashboard", error.getvalue())

    def test_follow_wrapper_preserves_exit_mapping(self):
        class CursesError(Exception):
            pass

        terminal = mock.Mock()
        terminal.isatty.return_value = True
        curses_module = mock.Mock()
        curses_module.error = CursesError

        for effect, expected in (
            (0, 0),
            (KeyboardInterrupt(), 130),
            (CursesError("failed"), 1),
        ):
            curses_module.wrapper.reset_mock()
            if isinstance(effect, int):
                curses_module.wrapper.return_value = effect
                curses_module.wrapper.side_effect = None
            else:
                curses_module.wrapper.side_effect = effect
            error = io.StringIO()
            with (
                self.subTest(expected=expected),
                mock.patch.object(sys, "stdin", terminal),
                mock.patch.object(sys, "stdout", terminal),
                mock.patch.object(sys, "stderr", error),
                mock.patch.dict(sys.modules, {"curses": curses_module}),
            ):
                result = _tui.run_follow_dashboard(self.trace, None, 10, 0.2)
            self.assertEqual(result, expected)
            curses_module.wrapper.assert_called_once_with(
                _tui._run_curses,
                curses_module,
                self.trace,
                None,
                10,
                0.2,
            )

    def test_snapshot_reports_malformed_lines_through_cli(self):
        with self.trace.open("a", encoding="utf-8") as stream:
            stream.write("{bad-json}\n")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = metagross.main(
                ["view", "--snapshot", "--width", "80", str(self.trace)]
            )
        self.assertEqual(result, 0)
        self.assertIn("MALFORMED", output.getvalue().splitlines()[0])
        self.assertIn("Malformed 1", output.getvalue())

    def test_missing_trace_is_reported(self):
        error = io.StringIO()
        with contextlib.redirect_stderr(error):
            result = metagross.main(
                ["view", "--snapshot", str(self.trace.with_name("missing.jsonl"))]
            )
        self.assertEqual(result, 1)
        self.assertIn("cannot open trace", error.getvalue())


class LiveDashboardPtyTest(unittest.TestCase):
    def test_follow_starts_and_quits(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "events.jsonl"
            trace.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
            master_fd, slave_fd = pty.openpty()
            process = None
            try:
                window_size = struct.pack("HHHH", 18, 80, 0, 0)
                fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, window_size)
                environment = os.environ.copy()
                environment["TERM"] = "xterm-256color"
                environment["PYTHONDONTWRITEBYTECODE"] = "1"
                process = subprocess.Popen(
                    [
                        "/usr/bin/python3",
                        "-B",
                        "-m",
                        "metagross",
                        "view",
                        "--follow",
                        str(trace),
                    ],
                    cwd=Path(__file__).resolve().parent.parent,
                    env=environment,
                    stdin=slave_fd,
                    stdout=slave_fd,
                    stderr=slave_fd,
                    close_fds=True,
                )
                os.close(slave_fd)
                slave_fd = -1

                output = bytearray()
                title = b"METAGROSS GPU TRACE"
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        break
                    ready, _, _ = select.select([master_fd], [], [], 0.1)
                    if not ready:
                        continue
                    try:
                        chunk = os.read(master_fd, 4096)
                    except OSError:
                        break
                    if not chunk:
                        break
                    output.extend(chunk)
                    if title in output:
                        break

                self.assertIn(title, output)
                os.write(master_fd, b"q")
                self.assertEqual(process.wait(timeout=5), 0)
            finally:
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=1)
                for descriptor in (master_fd, slave_fd):
                    if descriptor < 0:
                        continue
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass


if __name__ == "__main__":
    unittest.main()
