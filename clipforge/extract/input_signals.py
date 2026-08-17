"""`input_signals` — §4.4's activity log into signal_series (§5.1 stage 9).

§4.4 calls this "arguably the second-strongest gameplay signal after markers",
and it is the cheapest thing in the pipeline: a JSONL file of 10 Hz aggregates,
one subtraction per record to convert it (§4.1), and three arrays out.

**The file is DAILY, not per-recording.** `input_logger.input_path` writes
`input-YYYY-MM-DD.jsonl` into a capture directory, because §4.5 forbids the
capture daemons from depending on OBS to learn where it is writing. A day with
three streams has one file covering all three, so this stage slices by the
anchor and the duration rather than assuming the file is its own.

**A GAP IS NOT A ZERO, AND THIS IS THE WHOLE POINT OF `input_coverage`.**

The logger can be started late, crash, or be restarted mid-session, and any of
those leaves a stretch with no records. Reading that as "no input activity" is
wrong in the most damaging possible direction: §6.4's AFK penalty fires on *no
input AND no speech*, so a quiet stretch of a stream whose logger was down —
or a stream with no log at all — would be penalised end to end for something
that was never observed.

So the rates are NaN where nothing was recorded and `input_coverage` is 0 there.
NaN is what `score/grid.py` already knows how to carry; the coverage mask is what
`score/derived.py`'s `input_stillness` and §6.4's AFK gate require before they
will claim anything.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from clipforge import db, signals
from clipforge.capture import contract
from clipforge.extract.markers import to_vod_seconds
from clipforge.pipeline.context import StageContext

#: §5.4.1's two input signals, plus the mask that says when they were observed.
#: `input_rate` is "keys+clicks per second", so the two recorded fields are
#: summed here rather than stored apart — §5.4.1 names one signal, not two.
RATE_KIND = "input_rate"
VELOCITY_KIND = "mouse_velocity"
COVERAGE_KIND = "input_coverage"
KINDS = (RATE_KIND, VELOCITY_KIND, COVERAGE_KIND)


class InputSignalError(RuntimeError):
    pass


@dataclass
class ParsedInput:
    """What one log yielded for one recording."""

    times: list[float] = field(default_factory=list)
    rate: list[float] = field(default_factory=list)
    velocity: list[float] = field(default_factory=list)
    before_start: int = 0
    after_end: int = 0
    malformed: list[str] = field(default_factory=list)

    @property
    def kept(self) -> int:
        return len(self.times)


def parse(
    path: Path, *, record_start_epoch_ms: int, duration_s: float | None
) -> ParsedInput:
    """Read §4.4's JSONL into VOD-relative samples for ONE recording.

    Records outside the recording are dropped rather than clamped — the same
    rule `marker_events` follows, and for the same reason: a sample from
    yesterday's stream is not a sample of this one, and clamping it would invent
    activity at second zero of every stream that shares a log with another.
    """
    result = ParsedInput()

    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            result.malformed.append(f"line {number}: {exc}")
            continue

        epoch_ms = record.get("epoch_ms")
        if not isinstance(epoch_ms, int) or isinstance(epoch_ms, bool):
            result.malformed.append(
                f"line {number}: epoch_ms is {epoch_ms!r}, expected an integer"
            )
            continue

        t = to_vod_seconds(epoch_ms, record_start_epoch_ms)
        if t < 0:
            result.before_start += 1
            continue
        if duration_s is not None and t > duration_s:
            result.after_end += 1
            continue

        keys = _number(record.get("keys_per_s"))
        clicks = _number(record.get("clicks_per_s"))
        velocity = _number(record.get("mouse_vel_px_s"))
        if keys is None or clicks is None or velocity is None:
            result.malformed.append(f"line {number}: a rate field is not a number")
            continue

        result.times.append(t)
        # §5.4.1: `input_rate` is "keys+clicks per second". One signal.
        result.rate.append(keys + clicks)
        result.velocity.append(velocity)

    return result


def _number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def to_series(
    parsed: ParsedInput, *, signal_hz: float, duration_s: float, source: str,
    log_hz: float = contract.INPUT_HZ,
) -> dict[str, signals.Series]:
    """Bin the samples onto the standard grid, with a coverage mask.

    Samples land in the bucket their timestamp falls in and are averaged, which
    is the right reduction for a rate: two 10 Hz records inside one 100 ms bucket
    describe the same tenth of a second twice, not twice as much input.

    Buckets with no record are NaN in the rates and 0.0 in the coverage. See the
    module docstring — this is the distinction §6.4's AFK penalty rests on.
    """
    count = max(int(np.floor(duration_s * signal_hz)), 0)
    if count == 0:
        raise InputSignalError(
            f"stream duration {duration_s}s is shorter than one sample at "
            f"{signal_hz} Hz"
        )

    totals = np.zeros((2, count), dtype=np.float64)
    seen = np.zeros(count, dtype=np.float64)

    if parsed.times:
        times = np.asarray(parsed.times, dtype=np.float64)
        index = np.floor(times * signal_hz).astype(np.int64)
        inside = (index >= 0) & (index < count)
        index = index[inside]

        np.add.at(totals[0], index, np.asarray(parsed.rate)[inside])
        np.add.at(totals[1], index, np.asarray(parsed.velocity)[inside])
        np.add.at(seen, index, 1.0)

    observed = seen > 0
    values = np.full((2, count), np.nan)
    # `where=` and a preallocated `out`, NOT `out=values[:, observed]`: boolean
    # indexing produces a COPY, so that form writes every quotient into a
    # temporary and discards it, leaving the array untouched at NaN. The tests
    # caught it as "a real zero rate is not kept"; nothing about the code looks
    # wrong.
    np.divide(totals, np.maximum(seen, 1.0), out=values, where=observed)

    params = {
        "source": source,
        "unit": "",
        "log_hz": log_hz,
        "samples_in_log": parsed.kept,
        "buckets_observed": int(observed.sum()),
        "dropped_before_start": parsed.before_start,
        "dropped_after_end": parsed.after_end,
    }
    return {
        RATE_KIND: signals.Series(
            kind=RATE_KIND, values=values[0], sample_rate_hz=signal_hz,
            t0=0.5 / signal_hz,
            params={**params, "note": "§5.4.1 keys+clicks per second"},
        ),
        VELOCITY_KIND: signals.Series(
            kind=VELOCITY_KIND, values=values[1], sample_rate_hz=signal_hz,
            t0=0.5 / signal_hz,
            params={**params, "unit": "px/s"},
        ),
        # NOT NaN where absent: "the logger was not running here" is an
        # observation about the logger, and it is always known.
        COVERAGE_KIND: signals.Series(
            kind=COVERAGE_KIND, values=observed.astype(np.float64),
            sample_rate_hz=signal_hz, t0=0.5 / signal_hz,
            params={**params, "note": "1 where §4.4's logger was running"},
        ),
    }


def coverage_fraction(series: signals.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(np.mean(series.values > 0.5))


# --------------------------------------------------------------------------
# stage hooks
# --------------------------------------------------------------------------


def _input_path(ctx: StageContext) -> Path:
    return ctx.paths.input_jsonl


def available(ctx: StageContext) -> tuple[bool, str]:
    """Whether this stage can run here at all.

    DEFERRED rather than failed, and deliberately so: §4.4's logger has never run
    on a real stream, so this is the state of every recording that exists today.
    A hard failure would stop `execute` at the first error and `score` would
    never produce candidates — the trap `whisperx` documented in Phase 2.
    """
    path = _input_path(ctx)
    if not path.is_file():
        return False, (
            "no input.jsonl for this stream (§4.4's logger writes "
            "input-YYYY-MM-DD.jsonl to its capture directory; register with "
            "--input <file> to attach one)"
        )
    if ctx.stream["record_start_epoch_ms"] is None:
        return False, (
            "input.jsonl carries epoch milliseconds (§4.4) and this stream has no "
            "record_start_epoch_ms to convert them against (§4.1). Re-register "
            "with --anchor <anchor.json> --force."
        )
    return True, ""


def params(ctx: StageContext) -> dict:
    path = _input_path(ctx)
    row = ctx.stream
    return {
        # The file's own identity: a logger that appended more records must
        # re-run, and nothing else needs to.
        "input_size": path.stat().st_size if path.is_file() else None,
        "input_present": path.is_file(),
        "anchor": row["record_start_epoch_ms"],
        "signal_hz": ctx.cfg.get("extract.signal_hz"),
    }


def outputs(ctx: StageContext) -> list[Path]:
    return []  # writes signal_series rows


def verify(ctx: StageContext) -> tuple[bool, str]:
    present = set(signals.kinds(ctx.conn, ctx.stream_id))
    missing = sorted(set(KINDS) - present)
    if missing:
        return False, f"no signal_series for {', '.join(missing)}"
    return True, ""


def run(ctx: StageContext) -> None:
    path = _input_path(ctx)
    row = ctx.stream
    anchor = row["record_start_epoch_ms"]
    duration = float(row["duration_s"] or 0.0)
    if not duration:
        raise InputSignalError("stream has no duration; run probe first")

    parsed = parse(
        path, record_start_epoch_ms=int(anchor), duration_s=duration
    )
    if parsed.malformed:
        raise InputSignalError(
            f"{path} has {len(parsed.malformed)} unparseable line(s):\n  "
            + "\n  ".join(parsed.malformed[:5])
        )

    signal_hz = float(ctx.cfg.get("extract.signal_hz"))
    produced = to_series(
        parsed, signal_hz=signal_hz, duration_s=duration, source=path.name
    )

    with db.transaction(ctx.conn):
        for series in produced.values():
            signals.store(ctx.conn, ctx.stream_id, series)

    covered = coverage_fraction(produced[COVERAGE_KIND])
    ctx.log(
        f"    {parsed.kept} record(s) over {duration:.0f}s, "
        f"{100 * covered:.0f}% of the stream covered"
    )
    for kind in (RATE_KIND, VELOCITY_KIND):
        stats = signals.summarize(produced[kind])
        if stats.get("observed"):
            ctx.log(
                f"    {kind:<14} median {stats['median']:.2f}, p95 {stats['p95']:.2f}"
            )
    ctx.metric("input_coverage_fraction", round(covered, 4))

    if parsed.before_start or parsed.after_end:
        # Expected, not a problem: the log is daily and this is one recording.
        ctx.log(
            f"    dropped {parsed.before_start} record(s) before the recording and "
            f"{parsed.after_end} after it — the log covers the whole day (§4.4)"
        )
    if covered < 1.0:
        ctx.log(
            f"    note: {100 * (1 - covered):.0f}% of the stream has no input "
            f"records. Those samples are absent rather than idle, so §6.4's AFK "
            f"penalty will not fire across them."
        )
