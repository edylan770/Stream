"""Parsing OBS's own log file — the text half of `scene_events` (§5.1 stage 11).

§4.2's configuration list ends with *"Scene switch logging (OBS log file already
contains this; parse it)"*, which is the whole of the spec's guidance.

**NO OBS LOG HAS EVER BEEN SEEN BY THIS CODE.** There is no OBS install on the
build machine, so every pattern below is written from the documented log format
and **none of it is validated**. That is not a footnote: a regex that does not
match produces zero events, silently, and looks exactly like a stream with no
scene switches in it.

So this module is built to be *corrected cheaply* rather than to be right:

════════════════════════════════════════════════════════════════════════
WHEN THE REAL LOG ARRIVES — the whole checklist
────────────────────────────────────────────────────────────────────────
1.  clipforge scene-events --check "<path to any OBS log>"

    Prints which patterns fired, the recording spans found, the scene
    timeline, and every line that LOOKS like it should have matched and
    did not. Run it on the streaming PC; send back the report, not the
    log — OBS logs carry machine paths and hardware details.

2.  Fix whichever patterns it reports as dead, in ONE place:
        clipforge/config/clipforge.yaml  ->  extract.scene_events.patterns
    Nothing else in the codebase contains a regex for OBS's text.

3.  Drop the log into  tests/fixtures/obs_logs/  and run pytest.
    That directory is parametrized, so a real log is picked up with no
    test changes at all. Delete synthetic-UNVALIDATED.txt once a real
    one is in there.

4.  Move this stage's rows in spec/GUESSES.md from `arbitrary` to
    `grounded`, and drop the UNVALIDATED banner from this docstring.
════════════════════════════════════════════════════════════════════════

WHAT IS ACTUALLY CERTAIN HERE, AND WHAT IS NOT

The *arithmetic* is certain and is tested for real. The *text matching* is the
guess, and the two are deliberately kept apart so replacing the guess cannot
disturb the part that is known to be right.

**Certain — and the reason there is no wall clock anywhere in this file.** OBS
prefixes every line with elapsed time since OBS launched, and the log marks its
own recording start. So

    t = elapsed(scene line) - elapsed(recording start)

is recording-relative seconds computed entirely from *inside* the file. The
obvious alternative — reading the local timestamp out of the log's FILENAME —
would drag in a timezone and a DST discontinuity, and would be wrong by a whole
hour twice a year with nothing downstream able to notice. That is precisely the
constant-offset failure §4.1 calls unrecoverable and A8 exists to prevent.

**Guessed:** the elapsed-time prefix format, the scene-switch wording, the
recording start/stop wording, and whether the log names its output file at all.

TWO REFUSALS, BECAUSE A WRONG GUESS MUST BE LOUD

**A log holding several recordings with no way to tell which one this is gets
REFUSED, not guessed.** One OBS session can start and stop recording repeatedly,
so a log legitimately holds several spans. If a configured output-path pattern
names the master's own filename, that span wins — exact, and needs no clock. If
there is exactly one span, that wins. Otherwise this reports the ambiguity and
the stage defers, because picking the wrong span shifts every scene event by a
constant.

**There is no auto-discovery from `%APPDATA%\\obs-studio\\logs`.** Choosing the
right log out of a directory needs the wall clock this module refuses to use,
and choosing wrong is silent. The log is attached explicitly, with
`register --obs-log <path>`, exactly as §4.4's activity log already is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

#: Pattern keys this module understands. A config file naming anything else is
#: reporting a typo rather than adding a feature, so it is refused by name.
PATTERN_KEYS = ("elapsed", "scene_switch", "recording_start", "recording_stop",
                "recording_path")

#: Lines worth reporting as "looked like it should have matched". Used only by
#: the diagnostic — never to parse anything, because a heuristic that guessed at
#: content would be a second, undocumented parser.
SUSPECT_WORDS = ("scene", "recording", "record")


class ObsLogError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# the pattern set
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Patterns:
    """Every regex that touches OBS's text, compiled once.

    Built from config and nowhere else. If a future OBS renames a line, the fix
    is a YAML edit — that is the entire point of routing all of this through one
    object.
    """

    elapsed: re.Pattern
    scene_switch: re.Pattern
    recording_start: re.Pattern
    recording_stop: re.Pattern
    recording_path: re.Pattern | None = None

    @classmethod
    def from_config(cls, raw: dict) -> Patterns:
        unknown = sorted(set(raw or {}) - set(PATTERN_KEYS))
        if unknown:
            raise ObsLogError(
                f"unknown scene_events pattern(s): {', '.join(unknown)}. "
                f"Known: {', '.join(PATTERN_KEYS)}"
            )

        compiled: dict[str, re.Pattern] = {}
        for key in PATTERN_KEYS:
            source = (raw or {}).get(key)
            if not source:
                if key == "recording_path":
                    continue
                raise ObsLogError(
                    f"extract.scene_events.patterns.{key} is missing. Every "
                    f"pattern except `recording_path` is required."
                )
            try:
                compiled[key] = re.compile(str(source))
            except re.error as exc:
                raise ObsLogError(
                    f"extract.scene_events.patterns.{key} is not a valid "
                    f"regex: {exc}"
                ) from exc

        _require_group(compiled["elapsed"], "elapsed", ("h", "m", "s"))
        _require_group(compiled["scene_switch"], "scene_switch", ("scene",))
        if "recording_path" in compiled:
            _require_group(compiled["recording_path"], "recording_path", ("path",))
        return cls(**compiled)


def _require_group(pattern: re.Pattern, key: str, names: tuple[str, ...]) -> None:
    """A pattern that matches but captures nothing is the worst failure mode.

    It would parse every line, find no scene name, and write nothing — with no
    error anywhere. Checked at load so a bad edit fails at `clipforge config`
    rather than at the end of a forty-minute run.
    """
    missing = [name for name in names if name not in pattern.groupindex]
    if missing:
        raise ObsLogError(
            f"extract.scene_events.patterns.{key} must capture "
            f"{', '.join(f'(?P<{n}>...)' for n in missing)} — it matched text "
            f"but had nowhere to put it, which writes zero events silently."
        )


# --------------------------------------------------------------------------
# the clock: entirely internal to the file
# --------------------------------------------------------------------------


def elapsed_seconds(line: str, patterns: Patterns) -> float | None:
    """Seconds since OBS launched, from the line's own prefix.

    None when the line has no timestamp — a continuation line, a banner, a blank.
    """
    match = patterns.elapsed.match(line)
    if match is None:
        return None
    parts = match.groupdict()
    try:
        total = (int(parts["h"]) * 3600.0
                 + int(parts["m"]) * 60.0
                 + float(parts["s"]))
    except (TypeError, ValueError):
        return None
    return total


# --------------------------------------------------------------------------
# what a parse produces
# --------------------------------------------------------------------------


@dataclass
class Recording:
    """One `Recording Start` .. `Recording Stop` span, in OBS-elapsed seconds."""

    start_elapsed: float
    stop_elapsed: float | None = None
    #: Output path, when the log names one. This is what lets a span be matched
    #: to a master with no clock involved at all.
    path: str | None = None

    @property
    def duration_s(self) -> float | None:
        if self.stop_elapsed is None:
            return None
        return self.stop_elapsed - self.start_elapsed


@dataclass
class SceneSwitch:
    """A `Switched to scene` line, in OBS-elapsed seconds."""

    elapsed: float
    scene: str


@dataclass
class ParsedLog:
    """Everything the text half found, before any recording is chosen."""

    recordings: list[Recording] = field(default_factory=list)
    switches: list[SceneSwitch] = field(default_factory=list)
    #: pattern key -> how many lines it claimed. The diagnostic's headline: a
    #: zero here is a dead pattern, which is the failure this module expects.
    hits: dict[str, int] = field(default_factory=dict)
    #: Lines mentioning a scene or a recording that NO pattern claimed. The
    #: single most useful thing to look at when a pattern is wrong.
    unmatched_suspects: list[tuple[int, str]] = field(default_factory=list)
    lines: int = 0
    #: Lines carrying no parseable elapsed prefix. A large number here means the
    #: `elapsed` pattern is wrong, and nothing else can be trusted.
    without_timestamp: int = 0

    @property
    def looks_parsed(self) -> bool:
        """Whether this log looks like it was understood at all.

        Deliberately conservative: an `elapsed` pattern that fails makes every
        other number meaningless, so it is checked first and separately.
        """
        return bool(self.lines) and self.without_timestamp < self.lines


def parse(text: str, patterns: Patterns) -> ParsedLog:
    """Read a log into spans and switches, in OBS-elapsed seconds.

    Deliberately does no interpretation: it does not decide which recording is
    which, and it does not convert to recording-relative time. Those are
    `select_recording` and `to_recording_seconds`, so the diagnostic can show
    what was found before anything is chosen.
    """
    result = ParsedLog(hits={key: 0 for key in PATTERN_KEYS})
    open_recording: Recording | None = None
    pending_path: str | None = None

    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        result.lines += 1

        elapsed = elapsed_seconds(line, patterns)
        if elapsed is None:
            result.without_timestamp += 1
        else:
            result.hits["elapsed"] += 1

        claimed = False

        match = patterns.scene_switch.search(line)
        if match is not None:
            result.hits["scene_switch"] += 1
            claimed = True
            if elapsed is not None:
                scene = (match.groupdict().get("scene") or "").strip()
                if scene:
                    result.switches.append(SceneSwitch(elapsed, scene))

        if patterns.recording_start.search(line) is not None:
            result.hits["recording_start"] += 1
            claimed = True
            if elapsed is not None:
                # An unterminated previous span stays unterminated rather than
                # being closed here: OBS crashing mid-recording is a real thing
                # and inventing a stop time would invent a duration.
                open_recording = Recording(start_elapsed=elapsed,
                                           path=pending_path)
                pending_path = None
                result.recordings.append(open_recording)

        if patterns.recording_stop.search(line) is not None:
            result.hits["recording_stop"] += 1
            claimed = True
            if elapsed is not None and open_recording is not None:
                open_recording.stop_elapsed = elapsed
                open_recording = None

        if patterns.recording_path is not None:
            match = patterns.recording_path.search(line)
            if match is not None:
                result.hits["recording_path"] += 1
                claimed = True
                path = (match.groupdict().get("path") or "").strip()
                if path:
                    # Attached to the open span if one is running, otherwise
                    # HELD for the next start. Which side of the start banner
                    # OBS logs the path on is one of the things this module does
                    # not know, and holding covers both without inventing a span
                    # that has no recording in it.
                    if open_recording is not None:
                        open_recording.path = path
                    else:
                        pending_path = path

        if not claimed and _is_suspect(line):
            result.unmatched_suspects.append((number, line.strip()))

    return result


def _is_suspect(line: str) -> bool:
    lowered = line.lower()
    return any(word in lowered for word in SUSPECT_WORDS)


# --------------------------------------------------------------------------
# choosing which recording this stream is
# --------------------------------------------------------------------------


class Ambiguous(ObsLogError):
    """Several recordings and nothing to tell them apart. Deliberately fatal.

    Picking one at random shifts every scene event by a constant, and nothing
    downstream can detect a constant offset — §4.1's own reason for the anchor.
    """


def select_recording(parsed: ParsedLog, master_name: str | None) -> Recording:
    """Which span in this log is the recording being registered.

    Three rules, in order, and the first two need no clock:

    1. The span whose logged output path ends with the master's filename. Exact.
    2. The only span, when there is exactly one.
    3. Refuse.
    """
    usable = [r for r in parsed.recordings if r.start_elapsed is not None]
    if not usable:
        raise ObsLogError(
            "no recording start found in this log. Either the "
            "`recording_start` pattern does not match this OBS version — run "
            "`clipforge scene-events --check` to see — or this log covers a "
            "session in which nothing was recorded."
        )

    if master_name:
        wanted = master_name.replace("\\", "/").rsplit("/", 1)[-1].lower()
        matched = [r for r in usable
                   if r.path and r.path.replace("\\", "/").rsplit("/", 1)[-1].lower() == wanted]
        if len(matched) == 1:
            return matched[0]
        if len(matched) > 1:
            raise Ambiguous(
                f"{len(matched)} recordings in this log name {wanted!r}. That "
                f"should not happen; check the log by hand."
            )

    if len(usable) == 1:
        return usable[0]

    raise Ambiguous(
        f"this log holds {len(usable)} recordings and none of them names "
        f"{master_name or 'the master'}, so there is no way to tell which one "
        f"this stream is. Picking one would shift every scene event by a "
        f"constant that nothing downstream can detect. Either attach the log "
        f"for the right session, or set "
        f"`extract.scene_events.patterns.recording_path` so spans can be "
        f"matched by filename."
    )


def to_recording_seconds(elapsed: float, recording: Recording) -> float:
    """OBS-elapsed to seconds-into-this-recording. The whole clock, in one line."""
    return float(elapsed) - float(recording.start_elapsed)


# --------------------------------------------------------------------------
# scenes as spans
# --------------------------------------------------------------------------


@dataclass
class SceneSpan:
    """A scene, from the switch that selected it to the switch that replaced it.

    A SPAN rather than a point, which is the same reconciliation `menu_screen`
    needed in commit 36: `events.t_end` has meant "NULL for instantaneous" since
    `0001_init.sql`, and every consumer of this — §9.3's chapter boundaries
    above all — wants to know how long a scene was up, not only when it changed.
    """

    t: float
    t_end: float
    scene: str
    previous: str | None = None

    @property
    def duration_s(self) -> float:
        return self.t_end - self.t


def scene_spans(parsed: ParsedLog, recording: Recording,
                duration_s: float) -> list[SceneSpan]:
    """Scene switches, clipped to this recording and turned into spans.

    Two details that decide whether the output is honest:

    **The scene that was already up when recording started is included**, running
    from t=0. OBS logs a switch when it happens, not when recording begins, so
    the first minutes of a stream belong to a switch that appears EARLIER in the
    log than the recording start — dropping it would leave the opening of every
    stream with no scene at all.

    **The last span runs to the end of the recording**, not to the last switch.
    """
    switches = sorted(parsed.switches, key=lambda s: s.elapsed)
    if not switches:
        return []

    stop_elapsed = recording.stop_elapsed
    if stop_elapsed is None:
        stop_elapsed = recording.start_elapsed + max(0.0, float(duration_s))
    span_end = min(float(duration_s),
                   to_recording_seconds(stop_elapsed, recording))
    if span_end <= 0:
        return []

    # The scene in force at t=0: the last switch at or before the start.
    before = [s for s in switches if s.elapsed <= recording.start_elapsed]
    inside = [s for s in switches
              if recording.start_elapsed < s.elapsed <= stop_elapsed]

    points: list[tuple[float, str]] = []
    if before:
        points.append((0.0, before[-1].scene))
    for switch in inside:
        points.append((to_recording_seconds(switch.elapsed, recording),
                       switch.scene))

    if not points:
        return []

    spans: list[SceneSpan] = []
    for index, (start, scene) in enumerate(points):
        end = points[index + 1][0] if index + 1 < len(points) else span_end
        start = max(0.0, min(start, span_end))
        end = max(0.0, min(end, span_end))
        if end <= start:
            continue
        # `previous` is whatever this span replaced ON SCREEN, which is the
        # previous span's scene. None for the first: the scene before the
        # recording began is either the same one (it is `points[0]`) or was
        # never logged, and claiming either would be inventing history.
        spans.append(SceneSpan(
            t=start, t_end=end, scene=scene,
            previous=spans[-1].scene if spans else None,
        ))
    return spans


def read(path: Path) -> str:
    """OBS logs are UTF-8 but a scene name can be anything the operator typed.

    `errors="replace"` rather than a hard failure: one unrepresentable byte in a
    scene name must not cost every event in the file.
    """
    return path.read_text(encoding="utf-8", errors="replace")
