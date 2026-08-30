# test_viewer.py
"""Unprivileged tests for the Metagross visual trace viewer."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import metagross
from metagross import _viewer


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
                api="cuMemcpyHtoD", kernel=None, duration_ns=100_000,
                details={"bytes": 4096, "gpu_total": 8192},
            ),
            _record(
                timestamp="2026-08-30T12:10:05.000000+00:00",
                api="cuStreamSynchronize", kernel=None,
                function=None, file=None, line=None,
                return_code=2, duration_ns=200_000,
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
            "capture": {"events": 1, "lost_events": 0,
                        "dropped_nested_calls": 0},
        }
        model.load_summary(summary)
        self.assertEqual(model.status, "COMPLETE")
        summary["capture"]["events"] = 2
        model.load_summary(summary)
        self.assertEqual(model.status, "MISMATCH")

    def test_control_characters_are_sanitized(self):
        event = _viewer.parse_event(_record(
            function="safe\x1b[2Jname\nnext\u202ereversed",
            details={"stream": "bad\rvalue"},
        ))
        self.assertNotIn("\x1b", event.function)
        self.assertNotIn("\n", event.function)
        self.assertNotIn("\u202e", event.function)
        self.assertEqual(event.details["stream"], "bad?value")

    def test_invalid_event_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duration_ns"):
            _viewer.parse_event(_record(duration_ns=-1))
        with self.assertRaisesRegex(ValueError, "details"):
            _viewer.parse_event(_record(details=[]))


class ViewerFileTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_loader_tolerates_one_malformed_line(self):
        path = self.root / "events.jsonl"
        path.write_text(
            json.dumps(_record()) + "\n{not-json}\n", encoding="utf-8"
        )
        model = _viewer.load_trace(path)
        self.assertEqual(model.events, 1)
        self.assertEqual(model.malformed_lines, 1)
        self.assertEqual(model.status, "MALFORMED")

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
        model.load_summary({
            "schema_version": 1,
            "complete": True,
            "capture": {"events": 2},
        })
        rendered = "\n".join(_viewer.render_snapshot(model, width=100))
        self.assertIn("MISMATCH", rendered)
        self.assertIn("WARNING", rendered)


class ViewerRoutingTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.trace = Path(self.directory.name) / "events.jsonl"
        self.trace.write_text(json.dumps(_record()) + "\n", encoding="utf-8")

    def test_snapshot_routes_without_live_trace_validation(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = metagross.main([
                "view", "--snapshot", "--width", "80", str(self.trace)
            ])
        self.assertEqual(result, 0)
        self.assertIn("METAGROSS TRACE", output.getvalue())

    def test_snapshot_loads_optional_summary(self):
        summary = Path(self.directory.name) / "summary.json"
        summary.write_text(json.dumps({
            "schema_version": 1,
            "complete": True,
            "capture": {"events": 1, "lost_events": 0,
                        "dropped_nested_calls": 0},
        }), encoding="utf-8")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = metagross.main([
                "view", "--snapshot", "--summary", str(summary),
                str(self.trace),
            ])
        self.assertEqual(result, 0)
        self.assertIn("COMPLETE", output.getvalue().splitlines()[0])

    def test_view_help_does_not_require_live_dependencies(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = metagross.main(["view", "--help"])
        self.assertEqual(result, 0)
        self.assertIn("without root, BCC, or CUDA", output.getvalue())

    def test_missing_snapshot_flag_is_usage_error(self):
        error = io.StringIO()
        with contextlib.redirect_stderr(error):
            result = metagross.main(["view", str(self.trace)])
        self.assertEqual(result, 2)
        self.assertIn("use --snapshot", error.getvalue())

    def test_missing_trace_is_reported(self):
        error = io.StringIO()
        with contextlib.redirect_stderr(error):
            result = metagross.main([
                "view", "--snapshot", str(self.trace.with_name("missing.jsonl"))
            ])
        self.assertEqual(result, 1)
        self.assertIn("cannot open trace", error.getvalue())


if __name__ == "__main__":
    unittest.main()
