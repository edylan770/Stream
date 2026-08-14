"""Source time -> output time, for one clip.

Every §8 feature that touches time needs the same mapping and gets it wrong
independently if each one does its own arithmetic:

* **Captions** must be written in *output* time. A1 puts `-ss` before `-i`, and
  without `-copyts` ffmpeg restarts the output's timestamps at zero — so an ASS
  file carrying absolute stream times places every caption tens of minutes into
  a forty-second clip. ffmpeg exits 0 and the clip simply has no captions, which
  is the worst way for it to be wrong.
* **Filler removal** (§8.2) deletes ranges out of the middle. Every timestamp
  after a cut moves earlier by the cumulative removed duration, so captions have
  to be generated *against the cut plan*, not before it.
* **Profanity mutes** (§8.6) are ranges in the same output clock.

`EditPlan` is that one mapping. Phase 4's first commit only ever builds it with
no cuts — `map_time` is then a subtraction — but the API is here so that the
commit which adds filler removal supplies a different plan rather than reworking
the caption path.

**A cut is a half-open interval `[t_start, t_end)`.** An instant exactly on
`t_end` survives; one exactly on `t_start` does not. Without that rule a word
ending precisely where a cut begins would map into the cut and vanish.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from clipforge.render import RenderError

#: Times closer than this are the same instant. `candidates.t_start` is SQLite
#: REAL and word timestamps come back through JSON, so both carry a double's
#: worth of noise; `render/timebase.py` needed the same epsilon for exactly this
#: reason and it is not worth two different values.
EPSILON = 1e-9


class TimelineError(RenderError):
    pass


@dataclass(frozen=True)
class Cut:
    """A source range removed from the clip. Half-open: `[t_start, t_end)`."""

    t_start: float
    t_end: float
    #: Why it was removed — 'filler' today, and whatever §8 grows later. Carried
    #: so a report can say what it cut rather than just how much.
    reason: str = ""

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start

    def contains(self, t: float) -> bool:
        return self.t_start - EPSILON <= t < self.t_end - EPSILON


@dataclass(frozen=True)
class EditPlan:
    """What source range becomes the clip, and what is removed from inside it.

    `clip_start`/`clip_end` are source seconds. `cuts` are source ranges inside
    that window, sorted and disjoint — `build` enforces both, because an
    unsorted or overlapping list makes `map_time` monotonically wrong in a way
    no test on the output file could localise.
    """

    clip_start: float
    clip_end: float
    cuts: tuple[Cut, ...] = field(default=())

    @property
    def source_duration(self) -> float:
        """How much of the master is read. What `-t` gets."""
        return self.clip_end - self.clip_start

    @property
    def removed(self) -> float:
        return sum(cut.duration for cut in self.cuts)

    @property
    def duration(self) -> float:
        """How long the finished clip is."""
        return self.source_duration - self.removed

    def contains(self, t: float) -> bool:
        """Is this source instant inside the clip window at all?"""
        return self.clip_start - EPSILON <= t <= self.clip_end + EPSILON

    def is_cut(self, t: float) -> bool:
        return any(cut.contains(t) for cut in self.cuts)

    def output_of(self, t: float) -> float:
        """Source -> output for a time already known to survive.

        Unchecked on purpose: `map_time` and `map_span` do the checking, and
        this is the arithmetic they share. Calling it on a cut instant returns
        the output time of the cut's start, which is meaningless — hence
        private-by-convention use only.
        """
        shift = sum(cut.duration for cut in self.cuts if cut.t_end - EPSILON <= t)
        return max(t - self.clip_start - shift, 0.0)

    def map_time(self, t: float) -> float | None:
        """Source seconds -> output seconds, or None when the instant was cut.

        Returns None *only* for a time inside a cut or outside the window. A
        caller that wants a clamped answer for a word straddling an edge should
        clamp the word's bounds first — silently clamping here would let a word
        that is entirely outside the clip come back as 0.0.
        """
        if not self.contains(t) or self.is_cut(t):
            return None
        return self.output_of(t)

    def map_span(self, start: float, end: float) -> tuple[float, float] | None:
        """Map a span, keeping whatever part of it survives the cuts.

        This is what a word goes through. A word straddling a cut boundary is
        shortened rather than dropped — the syllable that survived is still on
        screen, and C2 says to expand rather than contract when in doubt.
        Returns None when nothing of the span is left.
        """
        # Overlap is checked BEFORE clamping. Clamping first would collapse a
        # word twenty seconds past the end onto the clip's last instant and
        # give it a caption line there.
        if end < self.clip_start - EPSILON or start > self.clip_end + EPSILON:
            return None
        start, end = self.clamp(start), self.clamp(end)
        if end - start <= EPSILON:
            # A zero-length word: an interpolated one that landed on top of its
            # own anchor. It still has text, so it still gets a caption line.
            if self.is_cut(start):
                return None
            mapped = self.output_of(start)
            return (mapped, mapped)

        pieces = [
            (max(keep_start, start), min(keep_end, end))
            for keep_start, keep_end in self.keep_ranges()
            if min(keep_end, end) - max(keep_start, start) > EPSILON
        ]
        if not pieces:
            return None
        return (self.output_of(pieces[0][0]), self.output_of(pieces[-1][1]))

    def clamp(self, t: float) -> float:
        """The nearest source instant inside the window."""
        return min(max(t, self.clip_start), self.clip_end)

    def keep_ranges(self) -> list[tuple[float, float]]:
        """The source ranges that survive, in order. What `trim`/`concat` gets.

        A plan with no cuts yields exactly one range, so the caller never needs
        a special case for "no editing".
        """
        ranges: list[tuple[float, float]] = []
        cursor = self.clip_start
        for cut in self.cuts:
            start = max(cut.t_start, self.clip_start)
            end = min(cut.t_end, self.clip_end)
            if end - cursor > EPSILON and start - cursor > EPSILON:
                ranges.append((cursor, start))
            cursor = max(cursor, end)
        if self.clip_end - cursor > EPSILON:
            ranges.append((cursor, self.clip_end))
        return ranges


def build(
    clip_start: float,
    clip_end: float,
    cuts: list[Cut] | tuple[Cut, ...] = (),
) -> EditPlan:
    """The only way to make an `EditPlan`, so the invariants hold everywhere.

    Cuts are clipped to the window, sorted, and merged where they touch or
    overlap. Merging rather than rejecting is deliberate: two independently
    derived cut sources (filler words and, later, anything else §8 grows) will
    legitimately produce adjacent ranges, and the caller should not have to
    reconcile them.
    """
    if clip_end - clip_start <= EPSILON:
        raise TimelineError(
            f"clip window is empty or inverted: {clip_start} -> {clip_end}"
        )

    inside: list[Cut] = []
    for cut in sorted(cuts, key=lambda c: (c.t_start, c.t_end)):
        start = max(cut.t_start, clip_start)
        end = min(cut.t_end, clip_end)
        if end - start <= EPSILON:
            continue
        if inside and start - inside[-1].t_end <= EPSILON:
            previous = inside[-1]
            reason = previous.reason if previous.reason == cut.reason else "merged"
            inside[-1] = Cut(previous.t_start, max(previous.t_end, end), reason)
        else:
            inside.append(Cut(start, end, cut.reason))

    if inside and inside[-1].t_end >= clip_end - EPSILON and len(inside) == 1 \
            and inside[0].t_start <= clip_start + EPSILON:
        # The whole clip was cut away. Better to say so than to hand ffmpeg a
        # zero-length concat and let it fail with a filter-graph error.
        raise TimelineError(
            f"every part of the window {clip_start:.3f}-{clip_end:.3f}s was cut"
        )

    return EditPlan(clip_start=clip_start, clip_end=clip_end, cuts=tuple(inside))
