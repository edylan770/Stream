"""What goes on the timeline: approved moments, read across generations.

HANDOFF states the requirement:

> Must read operator ratings **across generations**, not `is_current`: a
> re-score that drops a moment you already approved must not drop it from the
> export.

That is real. `score.runner._inherit_ratings` carries a rating onto the new
generation only when the windows overlap by at least
`score.rating_inherit_min_overlap` (0.5). A re-score that shifts a window past
that threshold — or splits one peak into two — strands the approval on a
superseded generation, where an `is_current` filter cannot see it. §13.2 calls
those judgment calls the one irreplaceable thing in the system.

**But "any operator rating >= 2, any generation" introduces the opposite bug.**
Rate a moment *clip it*, re-score, watch it again and rate the overlapping
candidate *skip* — and a naive union still exports it, because the old approval
is sitting there in generation 1.

So the rule is neither. Overlapping operator-rated windows are clustered into
one **moment** regardless of generation, and the most recently rated opinion in
that cluster is the verdict. An approval survives a re-score; a change of mind
still wins.

Two smaller rules fall out of the same reasoning:

* **`rating_source='inherited'` rows never count.** They are copies made by
  `_inherit_ratings`, and `inherited_from` points at the original — counting
  both would let one judgment call appear as two, which is exactly what §14's
  tuning is told to avoid.
* **`ratings.adjusted_start`/`adjusted_end` win over the candidate window.**
  §7.3's nudge keys write them, and the operator's trim outranks the detector's
  guess wherever it exists — including over the union rule above. A cluster
  containing an adjusted window is measured from the adjusted windows alone;
  otherwise an overlapping older generation would widen a hand-set boundary
  back out, silently undoing the edit that was just made.
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


def approved_moments(
    conn: sqlite3.Connection, stream_id: str, min_rating: int = 2
) -> list[Moment]:
    """Moments to put on the timeline, in stream order."""
    groups = cluster(rated_candidates(conn, stream_id))

    moments: list[Moment] = []
    for group in groups:
        decision = verdict(group)
        if decision["rating"] < min_rating:
            continue

        # The union of the window, not just the deciding row's: a re-score that
        # trimmed a window should not shorten a moment you approved at its
        # original length. C2 again — the operator can trim in Resolve, but
        # cannot recover a frame the export never referenced.
        keep = [e for e in group if e["rating"] >= min_rating]

        # **Unless the operator set a boundary by hand.** §7.3's nudge keys are
        # an explicit statement about where this moment starts and ends; a
        # candidate window that was merely rated is not. Taking the union across
        # both would let an overlapping older generation silently widen a trim
        # back out — the operator cuts three seconds of dead air off the front,
        # and the export puts it back with nothing reporting that it did.
        #
        # So an adjusted window suppresses the unadjusted ones. C2 still holds
        # among adjustments themselves: two nudged rows in one cluster union
        # normally, because both are the operator speaking.
        adjusted = [e for e in keep if e["adjusted"]]
        source = adjusted or keep

        moments.append(Moment(
            t_start=min(e["t_start"] for e in source),
            t_end=max(e["t_end"] for e in source),
            rating=decision["rating"],
            rated_at=decision["rated_at"],
            candidate_ids=[e["id"] for e in sorted(
                group, key=lambda e: -e["generation"])],
            generation=max(e["generation"] for e in group),
            # The safety net, made visible: nothing in the current generation
            # carries this judgment, so an is_current read would have lost it.
            rescued=not any(e["is_current"] for e in keep),
            note=decision["note"],
        ))

    return sorted(moments, key=lambda m: m.t_start)
