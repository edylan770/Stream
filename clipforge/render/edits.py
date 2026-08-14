"""§8.6's profanity muting and §8.2's filler removal. Both off by default.

> **Toggle. OFF by default.** (Explicit operator decision.) Store mute ranges
> as data (derived from word timestamps + a profanity list); apply only at
> export time when the toggle is on.

§16 lists "profanity muting by default" as an explicitly rejected feature, so
`render.mute.enabled` and `render.filler.enabled` are both `false` and a render
with no flags produces exactly what it produced before this module existed.

THREE MECHANISMS, CHECKED BEFORE THEY WERE BUILT

1. **`-af` accepts a branching graph.** `asplit` -> `atrim`/`asetpts` per kept
   range -> `concat` is a *simple* filtergraph (one input, one output), so the
   audio cuts stay in `-af`. MEASURED: a 12 s window with a 3 s cut came out at
   exactly 9.000 s. This matters because commit 26's loudness pass 1 also uses
   `-af` and must apply the identical chain -- if cuts had forced audio into
   `-filter_complex`, pass 1 would have measured uncut audio and pass 2
   normalised cut audio, wrong by however much was removed and undetectable.
2. **Video and audio stay in sync through cuts.** MEASURED on the same window:
   video 9.000 s / 270 frames (30 fps x 9), audio 9.000 s.
3. **One `volume` filter mutes several ranges.**
   `volume=enable='between(t,a,b)+between(t,c,d)':volume=0` MEASURED at
   -23.6 -> -50.8 dB inside a range and unchanged (-21.6 -> -21.6) outside it.
   The residual is leakage at the boundary, which is why ranges are padded.

WHAT §8.2's ONE LINE HIDES

> | Filler-word removal | Word timestamps + concat filter | Equal, far faster |

Cutting a word cuts the **video** too. On a talking head a jump cut is the
house style; on gameplay the character teleports. And every timestamp after a
cut moves, so captions have to be generated against the cut plan rather than
before it -- which is why `EditPlan` exists and why `render_stream` builds the
cuts first.

So the planner below is mostly refusals: whole words only, only from the tracks
named in config, never across another speaker's words, nothing shorter than a
glitch, and nothing at all if the plan would remove an implausible share of the
clip. That last one is the difference between "this clip had some ums in it"
and "the word list is wrong", and only the second is worth stopping for.

**Whether any of this improves a clip is not something the fixture can answer.**
The mechanism is deterministic and tested; the taste question needs footage.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from clipforge.extract.phrases import compile_patterns, load_phrases, normalise
from clipforge.render import RenderError
from clipforge.render.timeline import EPSILON, Cut, EditPlan
from clipforge.render.words import track_roles


class EditError(RenderError):
    pass


@dataclass(frozen=True)
class Span:
    """A word's place in SOURCE time, with who said it."""

    start: float
    end: float
    text: str
    role: str
    track: int | None

    @property
    def duration(self) -> float:
        return self.end - self.start


# --------------------------------------------------------------------------
# reading words out of the window
# --------------------------------------------------------------------------


def spans_in(
    conn: sqlite3.Connection, stream_id: str, plan: EditPlan,
    audio_track_map: str | None,
) -> list[Span]:
    """Every aligned word overlapping the clip, in source time.

    Unaligned words are skipped rather than interpolated -- the caption path
    interpolates because a caption missing a word is visibly broken, but
    *cutting* on an invented timestamp would remove audio at a time nobody
    measured. Silence is the safe failure here.
    """
    roles = track_roles(audio_track_map)
    found: list[Span] = []

    rows = conn.execute(
        "SELECT t_start, t_end, speaker, track, words FROM segments "
        "WHERE stream_id = ? AND t_end >= ? AND t_start <= ? ORDER BY t_start",
        (stream_id, plan.clip_start, plan.clip_end),
    ).fetchall()

    for row in rows:
        try:
            words = json.loads(row["words"] or "[]")
        except json.JSONDecodeError:
            continue
        role = roles.get(row["track"], "") if row["track"] is not None else ""
        for word in words:
            if word.get("start") is None or word.get("end") is None:
                continue
            start, end = float(word["start"]), float(word["end"])
            if end < plan.clip_start or start > plan.clip_end:
                continue
            found.append(Span(
                start=start, end=end, text=str(word.get("w") or ""),
                role=role, track=row["track"],
            ))

    return sorted(found, key=lambda s: (s.start, s.end))


# --------------------------------------------------------------------------
# §8.6 — profanity muting
# --------------------------------------------------------------------------


def matching(spans: list[Span], patterns) -> list[Span]:
    """Words matching a compiled word list.

    `fullmatch` against the normalised token, so "shit" matches and "shitake"
    does not -- the same word-boundary discipline §5.6 needed, for the same
    reason: a substring rule mutes syllables out of innocent words.
    """
    hits = []
    for span in spans:
        token = normalise(span.text)
        if token and any(pattern.fullmatch(token) for _p, pattern in patterns):
            hits.append(span)
    return hits


def mute_ranges(
    cfg, conn: sqlite3.Connection, stream_id: str, plan: EditPlan,
    audio_track_map: str | None = None, spans: list[Span] | None = None,
) -> list[tuple[float, float]]:
    """Profanity spans in OUTPUT time, padded and merged.

    Output time, because `-af` sees audio that starts at zero after `-ss` and,
    once cuts exist, has had ranges removed from the middle. `EditPlan.map_span`
    does both conversions, so a mute lands where the word actually ended up.

    Padded because of a measurement: the boundary leaks. A mute that starts
    50 ms late leaves the consonant audible, which defeats the entire point --
    C2, expand rather than contract.
    """
    if not bool(cfg.get("render.mute.enabled", False)):
        return []

    words = cfg.get("render.mute.words", None)
    terms = list(words) if words else load_phrases(cfg).profanity
    patterns = compile_patterns([str(t) for t in terms])
    if not patterns:
        return []

    if spans is None:
        spans = spans_in(conn, stream_id, plan, audio_track_map)

    pad = float(cfg.get("render.mute.pad_s", 0.08))
    ranges: list[tuple[float, float]] = []
    for span in matching(spans, patterns):
        mapped = plan.map_span(span.start - pad, span.end + pad)
        if mapped is None:
            continue
        ranges.append((max(mapped[0], 0.0), min(mapped[1], plan.duration)))

    return merge_ranges(ranges)


def merge_ranges(ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Sort and coalesce overlapping ranges, so `enable` stays short."""
    merged: list[tuple[float, float]] = []
    for start, end in sorted(ranges):
        if end - start <= EPSILON:
            continue
        if merged and start <= merged[-1][1] + EPSILON:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def mute_filter(ranges: list[tuple[float, float]]) -> str:
    """One `volume` filter covering every range.

    `+` is logical OR in ffmpeg's expression language (non-zero is true), so a
    single filter handles them all rather than one filter per swear word.
    """
    if not ranges:
        return ""
    conditions = "+".join(
        f"between(t,{start:.3f},{end:.3f})" for start, end in ranges)
    return f"volume=enable='{conditions}':volume=0"


# --------------------------------------------------------------------------
# §8.2 — filler removal
# --------------------------------------------------------------------------


@dataclass
class CutPlan:
    cuts: list[Cut]
    removed_s: float = 0.0
    #: Words that matched but were not cut, and why. Reported, because a
    #: planner that silently declines to do what it was asked is worse than one
    #: that explains itself.
    refused: list[str] = None          # type: ignore[assignment]

    def __post_init__(self):
        if self.refused is None:
            self.refused = []


def filler_cuts(
    cfg, conn: sqlite3.Connection, stream_id: str, plan: EditPlan,
    audio_track_map: str | None = None, spans: list[Span] | None = None,
) -> CutPlan:
    """Which filler words to remove, after every constraint has had a say.

    The constraints are the feature. Removing a word is easy; removing only the
    ones that can be removed without wrecking the clip is the part §8.2 does
    not mention.
    """
    if not bool(cfg.get("render.filler.enabled", False)):
        return CutPlan(cuts=[])

    terms = [str(w) for w in (cfg.get("render.filler.words", []) or [])]
    patterns = compile_patterns(terms)
    if not patterns:
        return CutPlan(cuts=[])

    if spans is None:
        spans = spans_in(conn, stream_id, plan, audio_track_map)

    roles = {str(r) for r in (cfg.get("render.filler.roles", ["mic"]) or [])}
    min_duration = float(cfg.get("render.filler.min_duration_s", 0.12))
    merge_gap = float(cfg.get("render.filler.merge_gap_s", 0.35))
    max_share = float(cfg.get("render.filler.max_share", 0.25))

    result = CutPlan(cuts=[])
    keep: list[Span] = []

    for span in matching(spans, patterns):
        # Only from the tracks named. Cutting the other person's speech because
        # the operator said "um" is not a thing anyone wants.
        if roles and span.role not in roles:
            result.refused.append(
                f"{span.text!r} at {span.start:.2f}s is on {span.role or 'no'} "
                f"track, not {sorted(roles)}")
            continue
        if span.duration < min_duration:
            result.refused.append(
                f"{span.text!r} at {span.start:.2f}s is {span.duration * 1000:.0f}ms, "
                f"below min_duration_s -- a cut that short is a glitch, not an edit")
            continue
        if overlaps_other_speech(span, spans):
            # Cutting here would take a syllable out of whoever else was
            # talking, and they did not say the filler word.
            result.refused.append(
                f"{span.text!r} at {span.start:.2f}s overlaps another track's "
                f"speech")
            continue
        keep.append(span)

    cuts = [Cut(s.start, s.end, "filler") for s in keep]
    cuts = _merge_cuts(cuts, merge_gap)

    removed = sum(cut.duration for cut in cuts)
    if removed > plan.source_duration * max_share:
        # Not "this clip was full of filler" -- almost certainly the word list
        # is matching something it should not. Refusing the whole plan is the
        # honest response; silently deleting a third of an approved moment is
        # not.
        result.refused.append(
            f"the plan would remove {removed:.1f}s of {plan.source_duration:.1f}s "
            f"({removed / plan.source_duration:.0%}), past "
            f"render.filler.max_share -- that is a word list problem, not a "
            f"clip full of filler. Nothing cut."
        )
        return result

    result.cuts = cuts
    result.removed_s = removed
    return result


def overlaps_other_speech(span: Span, spans: list[Span]) -> bool:
    """Is anyone on a different track talking across this word?"""
    return any(
        other.track is not None and span.track is not None
        and other.track != span.track
        and other.end > span.start + EPSILON
        and other.start < span.end - EPSILON
        for other in spans
    )


def _merge_cuts(cuts: list[Cut], gap: float) -> list[Cut]:
    """Coalesce cuts separated by less than `gap`.

    Two removals 200 ms apart leave a 200 ms island of audio between them,
    which reads as a stutter rather than as an edit. Merging takes the island
    with them.
    """
    merged: list[Cut] = []
    for cut in sorted(cuts, key=lambda c: c.t_start):
        if merged and cut.t_start - merged[-1].t_end <= gap:
            merged[-1] = Cut(merged[-1].t_start, max(merged[-1].t_end, cut.t_end),
                             "filler")
        else:
            merged.append(cut)
    return merged


# --------------------------------------------------------------------------
# the filter graphs
# --------------------------------------------------------------------------


def trim_chain(
    plan: EditPlan, *, audio: bool, entry: str = "", label: str = "",
) -> str:
    """`split`/`trim`/`concat` over the ranges that survive, in CLIP time.

    `entry` and `label` are the graph's input and output pads: empty for audio,
    because `-af` supplies an implicit single input and output, and `[0:v]` /
    `[vcut]` for the video side of a `-filter_complex`.

    Returns "" when nothing was cut. That is load bearing: a clip with no cuts
    must produce exactly the command it produced before this module existed,
    and an empty string is what makes that true rather than a `trim` spanning
    the whole window that would merely *probably* be equivalent.

    MEASURED: a 12 s window with a 3 s cut comes out at exactly 9.000 s on both
    streams -- video 270 frames at 30 fps, audio 9.000 s.
    """
    ranges = plan.keep_ranges()
    if len(ranges) <= 1:
        return ""

    kind = "a" if audio else ""
    setpts = "asetpts" if audio else "setpts"
    count = len(ranges)

    # `k` for "keep", so these pads cannot collide with the crop graph's.
    parts = [f"{entry}{kind}split={count}" + "".join(f"[k{i}]" for i in range(count))]
    for index, (start, end) in enumerate(ranges):
        # Clip-relative: `-ss` before `-i` means the decoder already starts at
        # clip_start, so a source-time trim would cut the wrong place.
        parts.append(
            f"[k{index}]{kind}trim={start - plan.clip_start:.3f}:"
            f"{end - plan.clip_start:.3f},{setpts}=PTS-STARTPTS[kt{index}]"
        )
    inputs = "".join(f"[kt{i}]" for i in range(count))
    video_out, audio_out = (0, 1) if audio else (1, 0)
    parts.append(f"{inputs}concat=n={count}:v={video_out}:a={audio_out}"
                 + (f"[{label}]" if label else ""))
    return ";".join(parts)


def audio_chain(plan: EditPlan, mute: list[tuple[float, float]]) -> str:
    """Cuts then muting, in that order.

    Order matters: the mute ranges are already in output time, which is the
    clock that exists *after* the cuts. Muting first would place them against a
    timeline that no longer exists by the time the audio is written.
    """
    parts = [p for p in (trim_chain(plan, audio=True), mute_filter(mute)) if p]
    # Comma, not semicolon: the mute chains onto the concat's output pad rather
    # than starting a new branch of the graph.
    return ",".join(parts)


def describe(cut_plan: CutPlan, mute: list[tuple[float, float]]) -> list[str]:
    """What the edit did, for the log and for `--dry-run`."""
    lines: list[str] = []
    if cut_plan.cuts:
        lines.append(
            f"filler    {len(cut_plan.cuts)} cut(s), {cut_plan.removed_s:.2f}s removed")
        for cut in cut_plan.cuts:
            lines.append(f"    {cut.t_start:.2f}-{cut.t_end:.2f}s")
    for note in cut_plan.refused:
        lines.append(f"    not cut: {note}")
    if mute:
        lines.append(f"mute      {len(mute)} range(s)")
        for start, end in mute:
            lines.append(f"    {start:.2f}-{end:.2f}s")
    return lines
