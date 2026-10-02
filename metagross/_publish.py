# metagross/_publish.py
"""Bounded authenticated delivery to a local Metagross dashboard."""

from __future__ import annotations

import copy
import dataclasses
import http.client
import json
import os
import queue
import secrets
import threading


_TOKEN_ENV = "METAGROSS_DASHBOARD_TOKEN"
_TOKEN_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.~"
)
_PROTOCOL_VERSION = 1
_MAX_EVENT_BYTES = 64 << 10
_MAX_BATCH_BYTES = 1 << 20
_MAX_BATCH_EVENTS = 128
_MAX_RESPONSE_BYTES = 4 << 10
_BATCH_SIZE_SEQUENCE = 99_999_999_999_999_999_999


class DashboardPublishError(Exception):
    """The requested dashboard delivery cannot be established."""


@dataclasses.dataclass(frozen=True)
class PublishResult:
    dropped_events: int
    error: str | None


def take_dashboard_token(environ) -> str:
    """Remove and validate the shared dashboard secret from an environment."""
    token = environ.pop(_TOKEN_ENV, None)
    if (
        not isinstance(token, str)
        or not 32 <= len(token) <= 128
        or any(character not in _TOKEN_CHARACTERS for character in token)
    ):
        raise DashboardPublishError(
            f"{_TOKEN_ENV} must contain 32-128 URL-safe ASCII characters"
        )
    return token


def listener_owners(port: int,
                    tables=("/proc/net/tcp", "/proc/net/tcp6")) -> set[int]:
    """Return the user IDs that own a listening TCP socket on `port`."""
    owners = set()
    for table in tables:
        try:
            with open(table, encoding="ascii") as stream:
                rows = stream.read().splitlines()[1:]
        except OSError:
            continue  # no IPv6, for example
        for row in rows:
            fields = row.split()
            # local address:port, remote, state (0A is LISTEN), ..., uid
            if (len(fields) > 7 and fields[3] == "0A"
                    and int(fields[1].rpartition(":")[2], 16) == port):
                owners.add(int(fields[7]))
    return owners


def require_own_receiver(port: int, allowed_uids) -> None:
    """Refuse to deliver to a port that another user is listening on.

    The first request carries the producer token and the events follow it.
    A local user who took the port before the receiver started would get
    both. Any address counts: an IPv6 wildcard listener also receives
    connections made to 127.0.0.1.
    """
    owners = listener_owners(port)
    if not owners:
        raise DashboardPublishError(
            f"no dashboard is listening on port {port}")
    foreign = owners - set(allowed_uids)
    if foreign:
        raise DashboardPublishError(
            f"port {port} is served by another user (uid {min(foreign)}); "
            "not sending the capture and its token there")


class DashboardPublisher:
    """Offer events without blocking and deliver them in one ordered worker."""

    def __init__(
        self,
        port: int,
        token: str,
        trace_name: str,
        *,
        request_timeout_s: float = 1.0,
        shutdown_timeout_s: float = 2.0,
        queue_size: int = 256,
    ):
        if not 1 <= port <= 65_535:
            raise DashboardPublishError("dashboard port must be between 1 and 65535")
        if request_timeout_s <= 0 or shutdown_timeout_s <= 0 or queue_size <= 0:
            raise DashboardPublishError("dashboard publisher limits must be positive")
        if (
            not isinstance(token, str)
            or not 32 <= len(token) <= 128
            or any(character not in _TOKEN_CHARACTERS for character in token)
        ):
            raise DashboardPublishError(
                "dashboard token must contain 32-128 URL-safe ASCII characters"
            )
        if not trace_name or os.path.basename(trace_name) != trace_name:
            raise DashboardPublishError("dashboard trace name must be a basename")
        self.port = port
        self.token = token
        self.trace_name = os.path.basename(trace_name)
        self.request_timeout_s = request_timeout_s
        self.shutdown_timeout_s = shutdown_timeout_s
        self.capture_id = secrets.token_hex(16)
        self._queue: queue.Queue[dict] = queue.Queue(maxsize=queue_size)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._accepting = False
        self._disabled = False
        self._started = False
        self._terminal = False
        self._next_sequence = 0
        self._offered_events = 0
        self._delivered_events = 0
        self._error: str | None = None
        self._error_reported = False

    def start(self) -> None:
        """Establish the capture before the traced target is released."""
        if self._started:
            return
        body = self._encode(
            {
                "schema_version": _PROTOCOL_VERSION,
                "capture_id": self.capture_id,
                "trace_name": self.trace_name,
            }
        )
        try:
            response = self._post("/api/capture/start", body)
            self._expect_ack(response, status="live")
        except BaseException:
            self._abort_best_effort("dashboard delivery could not start")
            raise
        self._started = True
        self._accepting = True
        worker = threading.Thread(
            target=self._run_guarded,
            name="metagross-dashboard-publisher",
            daemon=True,
        )
        self._worker = worker
        try:
            worker.start()
        except BaseException as exc:
            self._accepting = False
            self._disabled = True
            self._remember_error(f"dashboard publisher thread failed: {exc}")
            self._abort_best_effort("dashboard publisher thread failed")
            raise DashboardPublishError("cannot start dashboard publisher thread") from None

    def offer(self, record: dict) -> None:
        """Offer one normalized event in constant bounded time."""
        with self._lock:
            self._offered_events += 1
            if not self._accepting or self._disabled:
                return
            try:
                self._queue.put_nowait(record)
            except queue.Full:
                if self._error is None:
                    self._error = "dashboard delivery queue is full; events were dropped"

    def drop_event(self, message: str) -> None:
        """Count one event that could not be normalized for delivery."""
        with self._lock:
            self._offered_events += 1
            if self._error is None:
                self._error = message

    def pop_error(self) -> str | None:
        with self._lock:
            if self._error is None or self._error_reported:
                return None
            self._error_reported = True
            return self._error

    def finish(self, summary: dict) -> PublishResult:
        """Drain acknowledged events, then post the final capture summary."""
        with self._lock:
            self._accepting = False
        self._stop.set()
        worker = self._worker
        if worker is not None:
            worker.join(timeout=self.shutdown_timeout_s)
            if worker.is_alive():
                self._disable("dashboard publisher shutdown timed out")
                self._abort_best_effort("dashboard publisher shutdown timed out")

        with self._lock:
            dropped = self._offered_events - self._delivered_events
            disabled = self._disabled
            sequence = self._next_sequence

        if self._started and not disabled:
            remote_summary = copy.deepcopy(summary)
            capture = remote_summary.get("capture")
            if not isinstance(capture, dict):
                self._disable("dashboard summary has no capture object")
                self._abort_best_effort("dashboard summary was invalid")
            else:
                capture["delivery_dropped"] = dropped
                if dropped:
                    remote_summary["complete"] = False
                body = self._encode(
                    {
                        "schema_version": _PROTOCOL_VERSION,
                        "capture_id": self.capture_id,
                        "sequence": sequence,
                        "summary": remote_summary,
                    }
                )
                try:
                    response = self._post("/api/capture/finish", body)
                    self._expect_ack(
                        response,
                        status="finished",
                        next_sequence=sequence,
                    )
                    self._terminal = True
                except DashboardPublishError as exc:
                    self._disable(str(exc))
                    self._abort_best_effort("dashboard summary delivery failed")

        with self._lock:
            return PublishResult(
                self._offered_events - self._delivered_events,
                self._error,
            )

    def abort(self, message: str) -> None:
        """Best-effort terminalize a started capture without masking an error."""
        with self._lock:
            self._accepting = False
        self._stop.set()
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=self.shutdown_timeout_s)
        self._abort_best_effort(message)

    def close(self) -> None:
        """Stop the worker within the configured shutdown bound."""
        with self._lock:
            self._accepting = False
        self._stop.set()
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=self.shutdown_timeout_s)

    def _run_guarded(self) -> None:
        try:
            self._run()
        except BaseException as exc:
            self._disable(f"dashboard delivery failed: {exc}")
            self._abort_best_effort("dashboard event delivery failed")

    def _run(self) -> None:
        pending: bytes | None = None
        while pending is not None or not (self._stop.is_set() and self._queue.empty()):
            batch: list[bytes] = []
            if pending is not None:
                batch.append(pending)
                pending = None
            else:
                try:
                    record = self._queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                encoded = self._encode_event(record)
                if encoded is not None:
                    batch.append(encoded)
            if not batch:
                continue
            body_size = len(self._encode_batch(_BATCH_SIZE_SEQUENCE, [])) + len(
                batch[0]
            )
            while len(batch) < _MAX_BATCH_EVENTS:
                try:
                    record = self._queue.get_nowait()
                except queue.Empty:
                    break
                encoded = self._encode_event(record)
                if encoded is None:
                    continue
                candidate_size = body_size + 1 + len(encoded)
                if candidate_size > _MAX_BATCH_BYTES:
                    pending = encoded
                    break
                batch.append(encoded)
                body_size = candidate_size

            with self._lock:
                if self._disabled:
                    return
                sequence = self._next_sequence
            body = self._encode_batch(sequence, batch)
            response = self._post("/api/capture/events", body)
            self._expect_ack(response, next_sequence=sequence + 1)
            with self._lock:
                self._next_sequence += 1
                self._delivered_events += len(batch)

    def _encode_event(self, record: dict) -> bytes | None:
        try:
            encoded = self._encode(record)
        except (TypeError, ValueError):
            self._remember_error("dashboard event was not JSON serializable")
            return None
        if len(encoded) > _MAX_EVENT_BYTES:
            self._remember_error("dashboard event exceeded the 65536-byte limit")
            return None
        return encoded

    def _encode_batch(self, sequence: int, records: list[bytes]) -> bytes:
        prefix = self._encode(
            {
                "schema_version": _PROTOCOL_VERSION,
                "capture_id": self.capture_id,
                "sequence": sequence,
            }
        )
        return prefix[:-1] + b',"events":[' + b",".join(records) + b"]}"

    @staticmethod
    def _encode(value) -> bytes:
        return json.dumps(value, separators=(",", ":")).encode("utf-8")

    def _post(self, path: str, body: bytes) -> dict:
        last_error: str | None = None
        for attempt in range(2):
            connection = http.client.HTTPConnection(
                "127.0.0.1", self.port, timeout=self.request_timeout_s
            )
            try:
                connection.request(
                    "POST",
                    path,
                    body=body,
                    headers={
                        "Authorization": f"Bearer {self.token}",
                        "Content-Type": "application/json",
                    },
                )
                response = connection.getresponse()
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
                if response.status >= 500 and attempt == 0:
                    last_error = f"dashboard returned HTTP {response.status}"
                    continue
                if len(raw) > _MAX_RESPONSE_BYTES:
                    raise DashboardPublishError("dashboard acknowledgement is too large")
                if response.status != 200:
                    raise DashboardPublishError(
                        f"dashboard rejected delivery with HTTP {response.status}"
                    )
                try:
                    decoded = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    raise DashboardPublishError(
                        "dashboard returned an invalid acknowledgement"
                    ) from None
                if not isinstance(decoded, dict):
                    raise DashboardPublishError(
                        "dashboard returned an invalid acknowledgement"
                    )
                return decoded
            except DashboardPublishError:
                raise
            except (OSError, TimeoutError, http.client.HTTPException) as exc:
                last_error = f"dashboard request failed: {type(exc).__name__}"
                if attempt == 0:
                    continue
            finally:
                connection.close()
        raise DashboardPublishError(last_error or "dashboard request failed")

    def _expect_ack(
        self,
        response: dict,
        *,
        status: str | None = None,
        next_sequence: int | None = None,
    ) -> None:
        expected_fields = {"schema_version", "capture_id"}
        valid = (
            type(response.get("schema_version")) is int
            and response.get("schema_version") == _PROTOCOL_VERSION
            and response.get("capture_id") == self.capture_id
        )
        if status is not None:
            expected_fields.add("status")
            valid = valid and response.get("status") == status
        if next_sequence is not None:
            expected_fields.add("next_sequence")
            valid = (
                valid
                and type(response.get("next_sequence")) is int
                and response.get("next_sequence") == next_sequence
            )
        if set(response) != expected_fields:
            valid = False
        if not valid:
            raise DashboardPublishError("dashboard acknowledgement did not match request")

    def _disable(self, message: str) -> None:
        with self._lock:
            self._accepting = False
            self._disabled = True
            if self._error is None:
                self._error = message
        self._stop.set()

    def _remember_error(self, message: str) -> None:
        with self._lock:
            if self._error is None:
                self._error = message

    def _abort_best_effort(self, message: str) -> None:
        if self._terminal:
            return
        body = self._encode(
            {
                "schema_version": _PROTOCOL_VERSION,
                "capture_id": self.capture_id,
                "message": message[:500],
            }
        )
        try:
            response = self._post("/api/capture/abort", body)
            self._expect_ack(response, status="aborted")
            self._terminal = True
        except BaseException:
            return
