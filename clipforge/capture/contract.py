"""The files Layer A writes and ingest reads — defined exactly once.

Until now this contract lived in three places: prose in `capture/README.md`,
a writer in `tests/fixtures/make_fixture.py`, and a validator in
`ingest/register.py`. The README said so itself — *"change one and you must
change all three"* — which is an invitation to drift in the one place where
drift cannot be detected. `record_start_epoch_ms` is §4.1's entire sync
solution; a wrong or misread anchor shifts every marker in a stream by a
constant and **nothing downstream can tell**. There is no checksum for "these
timestamps are 40 seconds late".

So the shapes live here, and the daemons, the fixture generator and ingest all
import them.

**This module imports nothing from the rest of clipforge, and nothing outside
the standard library.** §2.1 is explicit that Layer A "must be independently
runnable and must not depend on the application being installed or functional",
so `clipforge/capture/` can be copied to the streaming PC on its own and run
with a bare Python plus `pynput`. A test enforces that.

A8: **epoch milliseconds everywhere.** Never local time, never a formatted
string. Conversion to VOD-relative seconds happens at ingest and only there.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

# --------------------------------------------------------------------------
# anchor.json — §4.1
# --------------------------------------------------------------------------

#: Contract version. Ingest refuses anything it does not recognise rather than
#: reading a field that has moved.
ANCHOR_SCHEMA = 1

ANCHOR_FILENAME = "anchor.json"

#: How the anchor was obtained. §4.1 offers the first two; fixtures use the third.
SOURCE_OBS_WEBSOCKET = "obs_websocket"
SOURCE_HOTKEY_SCRIPT = "hotkey_script"
SOURCE_SYNTHETIC = "synthetic"
ANCHOR_SOURCES = (SOURCE_OBS_WEBSOCKET, SOURCE_HOTKEY_SCRIPT, SOURCE_SYNTHETIC)

#: Earliest epoch-ms worth believing (2020-01-01). Catches the classic mistake
#: of writing seconds where milliseconds belong, which would put every marker
#: roughly fifty-five years before the recording started.
MIN_PLAUSIBLE_EPOCH_MS = 1_577_836_800_000

# --------------------------------------------------------------------------
# markers.jsonl — §4.3
# --------------------------------------------------------------------------

MARKERS_FILENAME = "markers.jsonl"

MARKER_MAYBE = "marker_maybe"          # F1 — "something might have happened"
MARKER_DEFINITE = "marker_definite"    # F2 — "that was good"
MARKER_KINDS = (MARKER_MAYBE, MARKER_DEFINITE)

# --------------------------------------------------------------------------
# input.jsonl — §4.4
# --------------------------------------------------------------------------

INPUT_FILENAME = "input.jsonl"

#: §4.4: "10 Hz aggregate rather than per-event to keep files small".
INPUT_HZ = 10.0


def now_epoch_ms() -> int:
    """Wall-clock epoch milliseconds (A8).

    Wall clock, deliberately, even though it can step under NTP: the anchor and
    every marker have to be comparable to each other across processes that
    started at different times, and only the wall clock is. Anything measuring
    an *interval* — the input logger's aggregation window — must use
    `time.monotonic()` instead.
    """
    return int(time.time() * 1000)


def anchor_payload(
    record_start_epoch_ms: int, source: str, written_at_epoch_ms: int | None = None
) -> dict:
    """The anchor as it is written to disk.

    `written_at_epoch_ms` differs from the anchor for the hotkey fallback, and
    the gap between them is that fallback's error bar — worth recording, since
    it is the only evidence of how good the number is.
    """
    if source not in ANCHOR_SOURCES:
        raise ValueError(f"unknown anchor source {source!r}; expected one of {ANCHOR_SOURCES}")
    if not isinstance(record_start_epoch_ms, int) or isinstance(record_start_epoch_ms, bool):
        raise ValueError(f"record_start_epoch_ms must be an int, got {record_start_epoch_ms!r}")
    if record_start_epoch_ms < MIN_PLAUSIBLE_EPOCH_MS:
        raise ValueError(
            f"record_start_epoch_ms is {record_start_epoch_ms}, which is before 2020. "
            f"A8 requires epoch MILLISECONDS — this looks like seconds."
        )
    return {
        "schema": ANCHOR_SCHEMA,
        "record_start_epoch_ms": record_start_epoch_ms,
        "source": source,
        "written_at_epoch_ms": (
            now_epoch_ms() if written_at_epoch_ms is None else written_at_epoch_ms
        ),
    }


def write_anchor(
    path: Path, record_start_epoch_ms: int, source: str,
    written_at_epoch_ms: int | None = None,
) -> Path:
    """Write `anchor.json`, atomically.

    Atomic because `register` may be reading it — and because a half-written
    anchor that still parses is the worst possible artifact.
    """
    payload = anchor_payload(record_start_epoch_ms, source, written_at_epoch_ms)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temp, path)
    return path


def marker_record(epoch_ms: int, kind: str) -> dict:
    """One §4.3 marker press.

    Only these two keys. No VOD times, no local timestamps, no metadata — the
    daemon has to stay dumb enough that it cannot fail in an interesting way.
    """
    if kind not in MARKER_KINDS:
        raise ValueError(f"unknown marker kind {kind!r}; expected one of {MARKER_KINDS}")
    return {"epoch_ms": int(epoch_ms), "kind": kind}


def input_record(
    epoch_ms: int, keys_per_s: float, mouse_vel_px_s: float, clicks_per_s: float
) -> dict:
    """One §4.4 aggregate window.

    Rates only. There is no field here for *which* key was pressed and there
    must never be one — see input_logger.py.
    """
    return {
        "epoch_ms": int(epoch_ms),
        "keys_per_s": round(float(keys_per_s), 3),
        "mouse_vel_px_s": round(float(mouse_vel_px_s), 3),
        "clicks_per_s": round(float(clicks_per_s), 3),
    }


class JsonlSink:
    """Append-only JSONL, flushed per line.

    §4.5's failure isolation, made concrete: a crash mid-stream should cost the
    presses that never happened, not the ones already written. Opened in append
    mode so a restarted daemon continues a file rather than truncating a
    session's worth of judgment calls.
    """

    def __init__(self, path, *, fsync: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8", newline="\n")
        self._fsync = fsync

    def write(self, record: dict) -> None:
        self._handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._handle.flush()
        if self._fsync:
            os.fsync(self._handle.fileno())

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()

    def __enter__(self) -> JsonlSink:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
