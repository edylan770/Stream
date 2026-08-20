"""What goes on the timeline: approved moments, read across generations.

THE RULE THIS RESTS ON MOVED, AND ONLY THE RULE MOVED. `Moment`,
`rated_candidates`, `cluster` and `verdict` now live in `clipforge/moments.py`
and are re-exported below, so every caller of this module is unchanged. They
moved because §14's `signal_firing_rate_by_rating` needs the identical
"one opinion per moment, across generations, latest wins" rule against feature
vectors rather than against a timeline, and §14's stated hazard is counting one
judgment twice — the same thing the rule exists to prevent. Commit 42's argument
for `clipforge/llm/`, applied to the second caller as it arrives.

What stayed here is `approved_moments`, because deciding what goes on a timeline
is a render concern and its two extra rules are about frames rather than about
ratings:

* **The union of the cluster's windows, not just the deciding row's.** A re-score
  that trimmed a window must not shorten a moment approved at its original
  length. C2 — the operator can trim in Resolve, but cannot recover a frame the
  export never referenced.
* **…unless the operator set a boundary by hand.** §7.3's nudge keys are an
  explicit statement about where a moment starts and ends; a candidate window
  that was merely rated is not. An adjusted window therefore suppresses the
  unadjusted ones in its cluster — otherwise an overlapping older generation
  silently widens a trim back out, and the three seconds of dead air the
  operator just cut off the front reappear in the export with nothing reporting
  that they did. Two nudged rows in one cluster still union: both are the
  operator speaking.
"""

from __future__ import annotations

import sqlite3

from clipforge.moments import Moment, cluster, rated_candidates, verdict

__all__ = ["Moment", "approved_moments", "cluster", "rated_candidates", "verdict"]


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

        keep = [e for e in group if e["rating"] >= min_rating]
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
