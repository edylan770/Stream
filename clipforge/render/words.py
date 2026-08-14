"""Words inside a clip window, in output time, tagged with who said them.

`segments.words` is a JSON array per segment and three modules already re-parse
it by hand — `extract/phrases.py`, `score/runner.py`, `extract/speakers.py` —
each with its own null handling. Nothing anywhere joins segments to a *window*.
Captions need exactly that, so it is written once here.

**Colour comes from `segments.track`, not `segments.speaker`.**

§8.3 says:

    operator -> color A
    party    -> color B
    both     -> alternate per word by source track

The third rule cannot be done from `speaker`. `speakers.classify` compares mic
and party energy across a segment's span and never looks at `segments.track`
(`extract/speakers.py:159`), so a segment labelled `both` still holds words from
exactly *one* track — there is nothing inside it to alternate between. What an
overlap actually produces is two segments, one per track, both labelled `both`.

And `speaker` can be wrong about a segment's own author. It is a statement about
the acoustic scene, not about provenance: a party-track segment during a
mic-dominant moment is labelled `operator`, and colouring by that paints the
party's words in the operator's colour.

`track` is provenance and cannot be wrong — whisperx transcribed that file. So
colouring by track implements all three of §8.3's cases with one rule: mic words
get colour A, party words colour B, and an overlap alternates by source track
*because the words came from different tracks*. `speaker` is the fallback only
when `track` is NULL, or when §4.2 found a single-track recording and there is
no provenance to read.

**Unaligned words keep their text.** §5.7's deviation is deliberate — a word
forced alignment could not place keeps `w` and loses `start`/`end`. Dropping it
here would put a caption on screen missing a word from the middle of a sentence,
so its times are interpolated across the gap between its aligned neighbours and
it is flagged. The flag is counted and reported: a transcript that is mostly
interpolated is a caption track that is mostly guesswork, and that should be
visible rather than inferred from captions looking slightly off.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from clipforge.render import RenderError
from clipforge.render.timeline import EPSILON, EditPlan

#: §4.2 role for the operator's own microphone, and for the other party. These
#: name config keys (`render.captions.styles.<role>`), so they are the vocabulary
#: the whole caption layer speaks.
MIC = "mic"
PARTY = "party"

#: What `segments.speaker` says, when there is no track to read. §5.8's
#: vocabulary, mapped onto the two roles that have a colour.
SPEAKER_ROLE = {"operator": MIC, "party": PARTY}


class WordError(RenderError):
    pass


@dataclass(frozen=True)
class RenderWord:
    """One word, placed on the *output* clock."""

    text: str
    #: Output-relative seconds. Never a stream time — see `render/timeline.py`.
    start: float
    end: float
    #: §4.2 role that produced it: 'mic', 'party', or '' when nothing said.
    role: str
    #: ffmpeg audio stream index, straight from `segments.track`.
    track: int | None
    #: §5.8's energy verdict, carried for the fallback and for reporting.
    speaker: str | None
    #: `segments.seq` — §12.1's LLM-facing identifier, so a caption can be
    #: traced back to the line it came from.
    segment_seq: int
    #: True when forced alignment gave no times and these were inferred.
    interpolated: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class WordLoad:
    """What `load` found, so a caller can report it rather than guess."""

    words: tuple[RenderWord, ...]
    interpolated: int = 0
    #: Words that fell entirely inside a cut, or outside the window.
    dropped: int = 0
    #: Words whose role could not be decided, and so take the default style.
    unroled: int = 0

    def __len__(self) -> int:
        return len(self.words)

    def __iter__(self):
        return iter(self.words)


# --------------------------------------------------------------------------
# track -> role
# --------------------------------------------------------------------------


def track_roles(audio_track_map: str | None) -> dict[int, str]:
    """Invert §4.2's `{"roles": {"mic": 1, "party": 3}}` into index -> role.

    Only `mic` and `party` carry a caption colour. `game` has no speech and
    `mixed` has both people in it, so neither can identify a speaker; a segment
    from either falls back to `speaker`.
    """
    if not audio_track_map:
        return {}
    try:
        roles = json.loads(audio_track_map).get("roles") or {}
    except (json.JSONDecodeError, AttributeError):
        return {}
    return {
        int(index): role
        for role, index in roles.items()
        if role in (MIC, PARTY) and index is not None
    }


def role_for(track: int | None, speaker: str | None, roles: dict[int, str]) -> str:
    """Provenance first, energy second. See the module docstring."""
    if track is not None and track in roles:
        return roles[track]
    return SPEAKER_ROLE.get(speaker or "", "")


# --------------------------------------------------------------------------
# alignment gaps
# --------------------------------------------------------------------------


def interpolate(raw: list[dict], t_start: float, t_end: float) -> list[dict]:
    """Give unaligned words times, evenly across the gap they sit in.

    Anchors are the previous aligned word's end and the next aligned word's
    start, falling back to the segment's own bounds at either edge. A run of
    unaligned words divides its gap equally — not accurate, and not pretending
    to be: `interpolated` marks every word this touched.

    A non-positive gap (two aligned words back to back with an unaligned one
    between them, which forced alignment does produce) yields zero-length
    words at the anchor. They still get a caption line, because the line runs
    to the *next* word's start rather than to the word's own end.
    """
    resolved: list[dict] = [dict(word) for word in raw]
    aligned = [
        index for index, word in enumerate(resolved)
        if word.get("start") is not None and word.get("end") is not None
    ]
    if len(aligned) == len(resolved):
        return resolved

    index = 0
    while index < len(resolved):
        word = resolved[index]
        if word.get("start") is not None and word.get("end") is not None:
            index += 1
            continue

        run_end = index
        while run_end < len(resolved) and (
            resolved[run_end].get("start") is None
            or resolved[run_end].get("end") is None
        ):
            run_end += 1

        low = float(resolved[index - 1]["end"]) if index else float(t_start)
        high = float(resolved[run_end]["start"]) if run_end < len(resolved) \
            else float(t_end)
        span = max(high - low, 0.0)
        count = run_end - index
        step = span / count if count else 0.0

        for offset in range(count):
            target = resolved[index + offset]
            target["start"] = low + step * offset
            target["end"] = low + step * (offset + 1)
            target["interpolated"] = True

        index = run_end

    return resolved


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def _segments(conn: sqlite3.Connection, stream_id: str, plan: EditPlan):
    """Segments overlapping the clip window, in time order.

    Ordered by `t_start` rather than `seq` because §5.8's merge assigns `seq`
    after interleaving two tracks, and a caption reads in time order.
    """
    return conn.execute(
        """
        SELECT seq, t_start, t_end, speaker, track, words
          FROM segments
         WHERE stream_id = ? AND t_end >= ? AND t_start <= ?
         ORDER BY t_start, seq
        """,
        (stream_id, plan.clip_start, plan.clip_end),
    ).fetchall()


def load(
    conn: sqlite3.Connection,
    stream_id: str,
    plan: EditPlan,
    audio_track_map: str | None = None,
) -> WordLoad:
    """Every word inside the clip, on the output clock, tagged with its role.

    Sorted by output start, then by track, so two tracks speaking at once come
    back interleaved by time but stay individually ordered — which is what the
    caption grouper needs to keep each speaker's sentence intact.
    """
    roles = track_roles(audio_track_map)
    found: list[RenderWord] = []
    interpolated = dropped = unroled = 0

    for row in _segments(conn, stream_id, plan):
        try:
            raw = json.loads(row["words"] or "[]")
        except json.JSONDecodeError as exc:
            raise WordError(
                f"segment {row['seq']} of {stream_id} has unreadable words JSON: {exc}"
            ) from exc
        if not raw:
            continue

        role = role_for(row["track"], row["speaker"], roles)
        if not role:
            unroled += 1

        for word in interpolate(raw, row["t_start"], row["t_end"]):
            text = str(word.get("w") or "").strip()
            if not text:
                continue
            span = plan.map_span(float(word["start"]), float(word["end"]))
            if span is None:
                dropped += 1
                continue
            was_guessed = bool(word.get("interpolated"))
            interpolated += int(was_guessed)
            found.append(RenderWord(
                text=text,
                start=span[0],
                end=span[1],
                role=role,
                track=row["track"],
                speaker=row["speaker"],
                segment_seq=int(row["seq"]),
                interpolated=was_guessed,
            ))

    found.sort(key=lambda w: (w.start, w.track if w.track is not None else -1,
                              w.segment_seq))
    return WordLoad(
        words=tuple(found),
        interpolated=interpolated,
        dropped=dropped,
        unroled=unroled,
    )


def by_role(words: tuple[RenderWord, ...] | list[RenderWord]) -> dict[str, list[RenderWord]]:
    """Split into one stream per role, preserving order within each.

    Captions group per role so that two people talking at once produce two
    stacked lines rather than one interleaved line of gibberish. Words with no
    role share a single stream under `''`.
    """
    streams: dict[str, list[RenderWord]] = {}
    for word in words:
        streams.setdefault(word.role, []).append(word)
    return streams


def span_of(words: tuple[RenderWord, ...] | list[RenderWord]) -> tuple[float, float]:
    """First start to last end. `(0.0, 0.0)` for nothing."""
    if not words:
        return (0.0, 0.0)
    return (min(w.start for w in words), max(w.end for w in words))


def coverage(load_result: WordLoad, plan: EditPlan) -> float:
    """Fraction of the clip that has a word on screen.

    Reported, not enforced. A clip with 5% coverage is a clip of gameplay with
    no talking, which is a legitimate thing to post — but it is also what a
    broken word loader looks like, and the number tells the two apart.
    """
    if plan.duration <= EPSILON or not load_result.words:
        return 0.0
    covered = sum(max(w.duration, 0.0) for w in load_result.words)
    return min(covered / plan.duration, 1.0)
