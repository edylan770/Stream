"""One opinion per moment, read across scoring generations.

CLAUDE.md states the invariant this module exists to hold:

> Ratings are read across scoring generations, never `is_current`. A re-score
> must never drop a moment the operator approved.

`score.runner._inherit_ratings` carries a rating onto a new generation only when
the windows overlap by at least `score.rating_inherit_min_overlap` (0.5). A
re-score that shifts a window past that threshold — or splits one peak into two —
strands the approval on a superseded generation, where an `is_current` filter
cannot see it. §13.2 calls those judgment calls the one irreplaceable thing in
the system.

**But "any operator rating, any generation" introduces the opposite bug.** Rate a
moment *clip it*, re-score, watch it again and rate the overlapping candidate
*skip* — and a naive union still counts the approval, because the old one is
sitting there in generation 1.

So the rule is neither. Overlapping operator-rated windows are clustered into one
**moment** regardless of generation, and the most recently rated opinion in that
cluster is the verdict. An approval survives a re-score; a change of mind still
wins; and one judgment call is counted once.

WHY THIS IS NOT IN `render/selection.py` ANY MORE

It was, and it was the only implementation, written against what goes on a
timeline. §14's `signal_firing_rate_by_rating` needs the identical rule against
feature vectors, and §14 is explicit that the thing it must not do is count one
judgment twice. The cheapest moment to have one implementation is when the second
caller arrives, which is the argument commit 42 made for `clipforge/llm/`.

`render/selection.py` keeps `approved_moments` — deciding what goes on a timeline
IS a render concern — and re-exports the four names below, so every existing
caller is unchanged.

AND WHY `marker_anchoring` IS HERE TOO

Same reason. §7.4's fourth section and §14's `marker_precision` both ask "was
this moment marked?", and they want DIFFERENT answers to it — see
`MarkerAnchoring`. Two call sites computing that separately is how the two
readings drift into disagreeing about a moment with nothing to report it. The
hazard HANDOFF records for `gates.speech_activity`, one layer up.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field


@dataclass
class Moment:
    """One thing worth cutting, assembled from every generation that saw it."""

    t_start: float
    t_end: float
    rating: int
    rated_at: str
    #: Every candidate row that contributed, newest generation first.
    candidate_ids: list[int] = field(default_factory=list)
    #: Highest generation that contained this moment.
    generation: int = 0
    #: True when the deciding rating lives on a superseded generation — the
    #: case this module exists for.
    rescued: bool = False
    note: str | None = None

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start

    def overlaps(self, other_start: float) -> bool:
        return other_start <= self.t_end


def rated_candidates(conn: sqlite3.Connection, stream_id: str) -> list[dict]:
    """Every operator-rated candidate for a stream, across all generations.

    No `is_current` anywhere in this query, by design.
    """
    rows = conn.execute(
        """
        SELECT c.id, c.generation, c.is_current, c.t_start, c.t_end, c.t_peak,
               r.rating, r.rated_at, r.note, r.adjusted_start, r.adjusted_end
          FROM candidates c
          JOIN ratings r ON r.candidate_id = c.id
         WHERE c.stream_id = ?
           AND r.rating_source = 'operator'
         ORDER BY COALESCE(r.adjusted_start, c.t_start), c.generation
        """,
        (stream_id,),
    ).fetchall()

    return [
        {
            "id": row["id"],
            "generation": row["generation"],
            "is_current": bool(row["is_current"]),
            "t_start": float(row["adjusted_start"] if row["adjusted_start"] is not None
                             else row["t_start"]),
            "t_end": float(row["adjusted_end"] if row["adjusted_end"] is not None
                           else row["t_end"]),
            "rating": int(row["rating"]),
            # Ties are broken by id, so two ratings recorded in the same second
            # still resolve to the later one deterministically.
            "rated_at": row["rated_at"] or "",
            "note": row["note"],
            # Whether these boundaries are the operator's own (§7.3's nudge
            # keys) or the detector's. See `approved_moments` — it decides
            # which windows the union may draw from.
            "adjusted": row["adjusted_start"] is not None,
        }
        for row in rows
    ]


def cluster(rated: list[dict]) -> list[list[dict]]:
    """Group overlapping windows into moments, ignoring generation entirely."""
    groups: list[list[dict]] = []
    for entry in sorted(rated, key=lambda e: (e["t_start"], e["t_end"])):
        if groups and entry["t_start"] <= max(e["t_end"] for e in groups[-1]):
            groups[-1].append(entry)
        else:
            groups.append([entry])
    return groups


def verdict(group: list[dict]) -> dict:
    """The opinion that decides a moment: the most recent one recorded."""
    return max(group, key=lambda e: (e["rated_at"], e["id"]))


# --------------------------------------------------------------------------
# was this moment marked? — two readings, one derivation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MarkerAnchoring:
    """Whether a window was marked, and by which of two different tests.

    THE TWO READINGS ARE NOT INTERCHANGEABLE, which is the whole reason this is
    one object with two fields rather than one boolean.

    `press_inside` is the strict one: the operator pressed F1 or F2 and the
    press time, after §4.3's retro offset, landed inside this window. It reads
    `events` and nothing else, so it is a property of what happened during the
    stream.

    `contributed` is loose: a marker kernel was non-zero here and the profile
    weighted it. §4.3's plateau runs `pre_s` before and `post_s` after every
    press (25 s and 5 s), so a window merely NEAR a press contributes — and
    because it reads `contributing_signals`, which `score.features.breakdown`
    builds from weighted tracks only, **the answer changes when a weight
    changes**.

    §7.4's safety-net section wants the union: a moment anywhere near something
    the operator marked deliberately should not be buried by the weights.

    §14's `marker_precision` wants `press_inside` alone. It is an input to
    weight tuning, and a measurement that moves when the weights move cannot
    tune them. `spec/GUESSES.md` also records that the loose reading is
    suspiciously loose — on `fixture_long` every candidate is anchored under it,
    which empties §7.4's sections 2 and 3.
    """

    press_inside: tuple[float, ...] = ()
    contributed: bool = False

    @property
    def any(self) -> bool:
        """§7.4's reading: near a press, or scored by one."""
        return bool(self.press_inside) or self.contributed


#: The two §5.4.2 kinds a marker daemon writes. Selected by `source` rather than
#: by kind so a third marker key added later is counted without a code change —
#: which is the reason `events` has a `source` column at all (§3.2's rationale:
#: "new sensor = new `source` value").
MARKER_SOURCE = "marker"


def marker_times(conn: sqlite3.Connection, stream_id: str) -> list[float]:
    """Every marker press for a stream, in VOD seconds, ordered."""
    return [
        float(row["t"]) for row in conn.execute(
            "SELECT t FROM events WHERE stream_id = ? AND source = ? ORDER BY t",
            (stream_id, MARKER_SOURCE),
        )
    ]


def marker_anchoring(
    t_start: float, t_end: float, markers: list[float],
    contributions: dict | None = None,
) -> MarkerAnchoring:
    """Both readings for one window. `markers` comes from `marker_times`."""
    payload = contributions or {}
    # `> 0`, not `bool(...)`. A contribution is weight × kernel level, and both
    # are non-negative for a marker today — but `bool(-1.0)` is True and
    # `-1.0 > 0` is False, so the two readings would diverge the day anything
    # gives a marker a negative weight. This is the reading `review/queries.py`
    # has always used.
    return MarkerAnchoring(
        press_inside=tuple(t for t in markers if t_start <= t <= t_end),
        contributed=payload.get("marker_definite", 0) > 0 or payload.get("marker_maybe", 0) > 0,
    )
