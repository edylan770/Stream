"""`scene_events` — §5.1 stage 11, "parse OBS log -> events".

The text parsing lives in `obs_log.py`, which carries the UNVALIDATED banner and
the checklist for correcting it. This module is the stage around it: find the
log, choose the recording, write spans, and — most of the time, today — defer
with a reason.

WHY THIS IS LOW-VALUE AND BUILT ANYWAY

§16 rejects scene changes as a scorer outright ("Low value. Retained as a
chapter-boundary tie-breaker only") and §9.3 makes them the fourth of four
boundary inputs, explicitly "weak signal, tie-breaker only". So nothing here
moves a candidate, and `feature_schema.yaml` gives `scene_change` no profile
weight in either §6.5 profile.

It is built because §9.3 wants it and because, unlike §4.3's markers and §4.4's
input log, **OBS writes this file whether or not anyone asked it to.** Marker and
input data that was not captured is gone (C6); scene data is sitting in
`%APPDATA%` waiting. That inverts the usual urgency: this stage is the one piece
of Phase 3 that loses nothing by being wrong today and corrected later.

SPANS, NOT POINTS

One row per scene, `t` to `t_end`, which is the reconciliation commit 36 already
settled for `menu_screen`: `events.t_end` has meant "NULL for instantaneous"
since `0001_init.sql`, and §9.3's consumer wants to know how long a scene was up
rather than only when it changed. A chapter boundary is the *edge* of a span, and
that is derivable from a span; a span is not derivable from an edge.

WHAT IT REFUSES TO DO

**It never guesses which recording in a log is this stream.** See
`obs_log.select_recording`. A wrong choice offsets every event by a constant and
nothing downstream can detect a constant offset — §4.1's entire reason for the
anchor.

**It never reads a wall clock.** Not the log's filename, not `Current
Date/Time`. The offset comes from the recording-start line inside the same file,
so the stage is timezone-free and DST-proof by construction rather than by care.

NOT BUILT, BUT WORTH KNOWING

A scene named "BRB" or "Starting Soon" is exactly §6.4's `menu_screen` case,
available with no Phase 7 vision at all — `gates.py` already reads `menu_screen`
events and is inert only because nothing writes them. Wiring scene names to that
gate is a genuinely cheap win **and it is deliberately not taken here**: it would
be scoring on a parser that has never seen a real log (C5), and §16's rejection
of scene changes as a scorer is precisely about that. Revisit once `--check` has
run against real text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from clipforge import db
from clipforge.extract import obs_log
from clipforge.pipeline.context import StageContext

SOURCE = "scene"
KIND = "scene_change"


class SceneEventError(RuntimeError):
    pass


@dataclass
class Result:
    spans: list[obs_log.SceneSpan] = field(default_factory=list)
    parsed: obs_log.ParsedLog | None = None
    recording: obs_log.Recording | None = None
    scenes: dict[str, float] = field(default_factory=dict)


def _cfg(ctx: StageContext, key: str, default=None):
    return ctx.cfg.get(f"extract.scene_events.{key}", default)


def log_path(ctx: StageContext) -> Path:
    """Where the OBS log for this stream lives, once `register` has copied it.

    Beside the markers and the input log, for the same reason those are there:
    inside a stream directory the file that matters is the one belonging to THIS
    recording, and the daemon's own naming is irrelevant by then.
    """
    return ctx.paths.raw_dir / "obs-log.txt"


def patterns(ctx: StageContext) -> obs_log.Patterns:
    return obs_log.Patterns.from_config(_cfg(ctx, "patterns") or {})


def available(ctx: StageContext) -> tuple[bool, str]:
    """Deferred, not failed — and today this is every stream in existence.

    A hard failure would stop `execute` at the first error and `score` would
    never produce candidates: the trap `whisperx` documented in Phase 2 and
    `input_signals` follows.
    """
    if not _cfg(ctx, "enabled", True):
        return False, "extract.scene_events.enabled is false"

    path = log_path(ctx)
    if not path.is_file():
        return False, (
            "no OBS log for this stream. OBS writes one per session to "
            "%APPDATA%\\obs-studio\\logs\\; attach it with "
            "`clipforge register --obs-log <file> --force`. There is no "
            "auto-discovery on purpose -- picking the wrong log is silent."
        )

    try:
        compiled = patterns(ctx)
    except obs_log.ObsLogError as exc:
        return False, str(exc)

    parsed = obs_log.parse(obs_log.read(path), compiled)
    if not parsed.looks_parsed:
        return False, (
            f"the OBS log at {path.name} produced no parseable lines. Run "
            f"`clipforge scene-events --check \"{path}\"` -- the `elapsed` "
            f"pattern is the first suspect, and every pattern is UNVALIDATED."
        )
    if not parsed.recordings:
        return False, (
            f"no recording start found in {path.name}. Either the "
            f"`recording_start` pattern does not match this OBS version -- run "
            f"`clipforge scene-events --check` -- or nothing was recorded in "
            f"that session."
        )

    try:
        obs_log.select_recording(parsed, _master_name(ctx))
    except obs_log.ObsLogError as exc:
        return False, str(exc)
    return True, ""


def _master_name(ctx: StageContext) -> str | None:
    master = ctx.stream["master_path"]
    return Path(str(master)).name if master else None


def params(ctx: StageContext) -> dict:
    """Patterns are in here, so correcting one re-runs the stage.

    That is the intended coupling: a different regex is a different set of
    events, which is a different extraction.
    """
    path = log_path(ctx)
    return {
        "patterns": _cfg(ctx, "patterns"),
        "log_bytes": path.stat().st_size if path.is_file() else None,
    }


def outputs(ctx: StageContext) -> list[Path]:
    return []


def verify(ctx: StageContext) -> tuple[bool, str]:
    """Rows, not files — so deleting them must not leave the stage looking done."""
    count = ctx.conn.execute(
        "SELECT COUNT(*) FROM events WHERE stream_id = ? AND source = ?",
        (ctx.stream_id, SOURCE),
    ).fetchone()[0]
    if count:
        return True, ""
    # Zero rows is legitimate: a session with one scene up throughout switches
    # never. Distinguished from "did not run" by the stage row itself.
    return True, ""


def run(ctx: StageContext) -> None:
    path = log_path(ctx)
    compiled = patterns(ctx)
    parsed = obs_log.parse(obs_log.read(path), compiled)
    recording = obs_log.select_recording(parsed, _master_name(ctx))

    duration = float(ctx.stream["duration_s"] or 0.0)
    if duration <= 0:
        raise SceneEventError("stream has not been probed")

    spans = obs_log.scene_spans(parsed, recording, duration)

    with db.transaction(ctx.conn):
        ctx.conn.execute(
            "DELETE FROM events WHERE stream_id = ? AND source = ?",
            (ctx.stream_id, SOURCE),
        )
        ctx.conn.executemany(
            "INSERT INTO events (stream_id, t, t_end, source, kind, value, meta) "
            "VALUES (?, ?, ?, ?, ?, NULL, ?)",
            [
                (ctx.stream_id, span.t, span.t_end, SOURCE, KIND,
                 json.dumps({"scene": span.scene, "from": span.previous}))
                for span in spans
            ],
        )

    _report(ctx, parsed, recording, spans)


def _report(ctx: StageContext, parsed: obs_log.ParsedLog,
            recording: obs_log.Recording, spans: list[obs_log.SceneSpan]) -> None:
    held: dict[str, float] = {}
    for span in spans:
        held[span.scene] = held.get(span.scene, 0.0) + span.duration_s

    ctx.log(f"    {len(spans)} scene span(s) from {len(parsed.switches)} switch(es)")
    for scene, seconds in sorted(held.items(), key=lambda kv: -kv[1]):
        ctx.log(f"      {scene:<24} {seconds / 60.0:6.1f} min")

    if parsed.unmatched_suspects:
        # THE line that matters while the patterns are unvalidated: a log full of
        # scene-looking lines that nothing claimed is a dead pattern, and it
        # would otherwise present as a stream that simply had no scene changes.
        ctx.log(
            f"    NOTE  {len(parsed.unmatched_suspects)} line(s) mention a scene "
            f"or a recording and matched no pattern. Every pattern here is "
            f"UNVALIDATED -- run `clipforge scene-events --check` on this log."
        )

    ctx.metric("scene_spans", float(len(spans)), json.dumps({
        "switches": len(parsed.switches),
        "recordings_in_log": len(parsed.recordings),
        "unmatched_suspects": len(parsed.unmatched_suspects),
        "scenes": sorted(held),
    }))
