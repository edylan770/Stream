"""`marker_events` — marker JSONL into the events table (§5.1 stage 10, §4.3).

§4.1 calls the anchor "the entire sync solution", and this is where it gets
used: one subtraction per marker, no drift, no alignment hunting.

    vod_time_s = (event_epoch_ms - record_start_epoch_ms) / 1000.0

**`events.t` holds the raw press time.** §4.3 says to anchor a marker's window
at `t − 20s`, and it is tempting to write that shifted value here. It would be
a mistake: `marker_retro_offset_s`, `pre_s`, `post_s` and `shoulder_s` are all
§17 tunables, so baking the offset into an extraction row freezes it into every
stream ever parsed and makes retuning it cost a re-parse of the whole
catalogue. The row records what was observed — a button was pressed at this
moment — and scoring decides what that implies. Same boundary as storing
absolute rather than baselined RMS (C3).

`meta` carries the offset that was configured at parse time anyway, so it is
possible to tell afterwards what the operator's settings were, without any
stage depending on it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from clipforge import db
from clipforge.pipeline.context import StageContext

#: §4.3's two hotkeys. Anything else in the file is a typo or a future kind,
#: and either way is worth reporting rather than silently importing.
MARKER_KINDS = ("marker_maybe", "marker_definite")

SOURCE = "marker"


class MarkerError(RuntimeError):
    pass


@dataclass
class ParsedMarkers:
    events: list[dict] = field(default_factory=list)
    skipped_before_start: int = 0
    skipped_after_end: int = 0
    malformed: list[str] = field(default_factory=list)
    unknown_kinds: dict[str, int] = field(default_factory=dict)

    @property
    def dropped(self) -> int:
        return self.skipped_before_start + self.skipped_after_end


def to_vod_seconds(epoch_ms: int, record_start_epoch_ms: int) -> float:
    """§4.1's conversion. One number per stream, no drift."""
    return (epoch_ms - record_start_epoch_ms) / 1000.0


def parse(
    path: Path,
    *,
    record_start_epoch_ms: int,
    duration_s: float | None,
    retro_offset_s: float,
) -> ParsedMarkers:
    """Read §4.3 JSONL into event dicts, in VOD seconds."""
    result = ParsedMarkers()
    text = path.read_text(encoding="utf-8")

    for number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            result.malformed.append(f"line {number}: {exc}")
            continue

        epoch_ms = record.get("epoch_ms")
        kind = record.get("kind")
        if not isinstance(epoch_ms, int) or isinstance(epoch_ms, bool):
            result.malformed.append(f"line {number}: epoch_ms is {epoch_ms!r}, expected an integer")
            continue
        if kind not in MARKER_KINDS:
            result.unknown_kinds[str(kind)] = result.unknown_kinds.get(str(kind), 0) + 1
            continue

        t = to_vod_seconds(epoch_ms, record_start_epoch_ms)

        # Dropped, not clamped: a press before the recording started refers to
        # something that is not in the file, and clamping it to zero would
        # invent a marker at the very beginning of every such stream.
        if t < 0:
            result.skipped_before_start += 1
            continue
        if duration_s is not None and t > duration_s:
            result.skipped_after_end += 1
            continue

        result.events.append({
            "t": t,
            "kind": kind,
            "meta": {
                "epoch_ms": epoch_ms,
                "line": number,
                # Provenance only. Nothing reads this back; the kernel is built
                # from live config at score time.
                "retro_offset_s_at_parse": retro_offset_s,
            },
        })

    result.events.sort(key=lambda e: e["t"])
    return result


# --------------------------------------------------------------------------
# stage hooks
# --------------------------------------------------------------------------


def _markers_path(ctx: StageContext) -> Path:
    return ctx.paths.markers_jsonl


def params(ctx: StageContext) -> dict:
    path = _markers_path(ctx)
    row = ctx.stream
    return {
        # The file's own identity: re-parsing after the daemon appended more
        # presses must re-run, and nothing else needs to.
        "markers_size": path.stat().st_size if path.is_file() else None,
        "markers_present": path.is_file(),
        "anchor": row["record_start_epoch_ms"],
        "time_base": row["marker_time_base"],
    }


def outputs(ctx: StageContext) -> list[Path]:
    return []  # writes events rows


def verify(ctx: StageContext) -> tuple[bool, str]:
    """A stream with no marker file legitimately has no marker events.

    So this cannot check "rows exist" — absence is a valid result. What it can
    check is that the stage's conclusion still matches the file on disk.
    """
    path = _markers_path(ctx)
    count = ctx.conn.execute(
        "SELECT count(*) FROM events WHERE stream_id = ? AND source = ?",
        (ctx.stream_id, SOURCE),
    ).fetchone()[0]

    if not path.is_file():
        return (count == 0), ("marker events exist but markers.jsonl is gone" if count else "")
    return True, ""


def run(ctx: StageContext) -> None:
    row = ctx.stream
    path = _markers_path(ctx)

    if not path.is_file():
        # Not an error. Most Phase 1 test streams have no markers, and a stream
        # where the operator never pressed anything is perfectly normal.
        _replace_events(ctx, [])
        ctx.log("    no markers.jsonl — nothing to parse")
        return

    anchor = row["record_start_epoch_ms"]
    if row["marker_time_base"] == "epoch" and anchor is None:
        raise MarkerError(
            "markers.jsonl carries epoch milliseconds (§4.3) but this stream has no "
            "record_start_epoch_ms, so there is nothing to convert them against (§4.1). "
            "Re-register with --anchor <anchor.json> --force."
        )

    parsed = parse(
        path,
        record_start_epoch_ms=int(anchor or 0),
        duration_s=row["duration_s"],
        retro_offset_s=float(ctx.cfg.get("score.markers.retro_offset_s")),
    )

    if parsed.malformed:
        raise MarkerError(
            f"{path} has {len(parsed.malformed)} unparseable line(s):\n  "
            + "\n  ".join(parsed.malformed[:5])
        )

    _replace_events(ctx, parsed.events)
    _report(ctx, parsed)


def _replace_events(ctx: StageContext, events: list[dict]) -> None:
    """Delete then insert.

    `events` has no natural key — §3.2 gives it an autoincrement id precisely so
    that a new sensor needs no schema change — so re-running would otherwise
    double every marker in the stream.
    """
    with db.transaction(ctx.conn):
        ctx.conn.execute(
            "DELETE FROM events WHERE stream_id = ? AND source = ?", (ctx.stream_id, SOURCE)
        )
        ctx.conn.executemany(
            "INSERT INTO events (stream_id, t, source, kind, value, meta) "
            "VALUES (?, ?, ?, ?, 1.0, ?)",
            [
                (ctx.stream_id, event["t"], SOURCE, event["kind"], json.dumps(event["meta"]))
                for event in events
            ],
        )


def _report(ctx: StageContext, parsed: ParsedMarkers) -> None:
    counts: dict[str, int] = {}
    for event in parsed.events:
        counts[event["kind"]] = counts.get(event["kind"], 0) + 1

    summary = ", ".join(f"{kind}={counts.get(kind, 0)}" for kind in MARKER_KINDS)
    ctx.log(f"    {len(parsed.events)} markers ({summary})")
    for kind, count in counts.items():
        ctx.metric(f"marker_count.{kind}", float(count))

    if parsed.skipped_before_start:
        ctx.log(
            f"    note: dropped {parsed.skipped_before_start} press(es) before record start "
            f"— the daemon was running before OBS was"
        )
    if parsed.skipped_after_end:
        ctx.log(
            f"    note: dropped {parsed.skipped_after_end} press(es) after the recording ended"
        )
    for kind, count in parsed.unknown_kinds.items():
        ctx.log(f"    WARNING  {count} line(s) with unrecognised kind {kind!r}, ignored")

    if not parsed.events:
        ctx.log("    WARNING  no usable markers. Scoring will run on audio alone.")
