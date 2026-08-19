"""§14's three unbuilt metrics — the ones §17's tuning procedure runs on.

> | `signal_firing_rate_by_rating` | Which signals fired on approved vs.
> rejected — **the primary weight-tuning input** |
> | `marker_precision` | Fraction of marker-anchored candidates approved |
> | `marker_recall_proxy` | Fraction of approved candidates that had no marker
> (i.e. what the operator missed live) |

All three were named, referenced and protected by other code, and never once
calculated. Four places in this codebase carry a comment explaining why they take
care not to corrupt `signal_firing_rate_by_rating`; none of them computed it.

**Nothing new is collected here.** A9 already stores the full feature vector on
every candidate — "they cannot be reconstructed retroactively and they are the
input to all future tuning" — `ratings` holds the verdicts, and §4.3's presses
are in `events`. This is the aggregation that was missing.

**Computed on demand, never stored.** §14's table says "log to `tool_metrics`",
and these three are the exception: they are derivations over `candidates` and
`ratings`, so a stored copy is stale the moment one more candidate is rated.
`score/derived.py` draws the same line for the same reason — storing a derived
value freezes it into every stream ever processed. Recomputing is a single scan
over rows that are already indexed.

**This does not make any number true.** Zero streams exist. What it does is make
the guesses in `spec/GUESSES.md` falsifiable, which is the whole point of that
file — and `report` refuses to print a rate it does not have the sample size to
support, because a fraction over n=1 invites a decision it cannot carry.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
from dataclasses import dataclass, field

from clipforge.review.queries import is_marker_anchored, load_markers

#: §14's names, used verbatim so the report and the spec's table line up.
FIRING_METRIC = "signal_firing_rate_by_rating"
MARKER_PRECISION = "marker_precision"
MARKER_RECALL = "marker_recall_proxy"

#: §14 contrasts "approved vs. rejected". A `maybe` is neither: folding it into
#: either side would blur the one contrast being measured, so it is counted and
#: reported separately.
APPROVED, MAYBE, REJECTED = 2, 1, 0


@dataclass
class SignalStats:
    """One signal's behaviour across the operator's verdicts."""

    name: str
    group: str
    #: Values seen on approved / rejected candidates. NULLS ARE NOT HERE — see
    #: `collect`. A signal unobserved on a candidate says nothing about it.
    approved: list[float] = field(default_factory=list)
    rejected: list[float] = field(default_factory=list)
    maybes: list[float] = field(default_factory=list)

    @property
    def observed(self) -> int:
        return len(self.approved) + len(self.rejected) + len(self.maybes)

    @property
    def n_approved(self) -> int:
        return len(self.approved)

    @property
    def n_rejected(self) -> int:
        return len(self.rejected)

    def mean_approved(self) -> float | None:
        return statistics.fmean(self.approved) if self.approved else None

    def mean_rejected(self) -> float | None:
        return statistics.fmean(self.rejected) if self.rejected else None

    def separation(self) -> float | None:
        """THE headline: mean on approved minus mean on rejected.

        §14 asks for a firing *rate*, which needs a fired/not-fired line and
        therefore an invented constant. This needs none — it is a difference of
        means over the values already stored — and it answers the same question
        more directly: a signal that is systematically higher on the moments the
        operator kept is a signal worth more weight.

        The firing rate is reported beside it because §14 names it, not because
        it is the better number.
        """
        left, right = self.mean_approved(), self.mean_rejected()
        return None if left is None or right is None else left - right

    def fired_rate(self, values: list[float], threshold: float) -> float | None:
        return None if not values else sum(1 for v in values if v >= threshold) / len(values)

    def to_json(self, threshold: float) -> dict:
        approved_rate = self.fired_rate(self.approved, threshold)
        rejected_rate = self.fired_rate(self.rejected, threshold)
        return {
            "signal": self.name,
            "group": self.group,
            "observed": self.observed,
            "n_approved": self.n_approved,
            "n_rejected": self.n_rejected,
            "n_maybe": len(self.maybes),
            "mean_approved": _round(self.mean_approved()),
            "mean_rejected": _round(self.mean_rejected()),
            "separation": _round(self.separation()),
            "fired_rate_approved": _round(approved_rate),
            "fired_rate_rejected": _round(rejected_rate),
            "lift": (None if approved_rate is None or rejected_rate is None
                     else round(approved_rate - rejected_rate, 4)),
        }


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 4)


@dataclass
class MarkerStats:
    """§14's two marker metrics, and the counts they came from."""

    anchored: int = 0
    anchored_approved: int = 0
    approved: int = 0
    approved_without_marker: int = 0

    def precision(self) -> float | None:
        """Of the moments the operator marked live, how many were worth keeping.

        The direct test of `score.markers.retro_offset_s`: a low precision means
        the offset is putting windows somewhere the moment is not.
        """
        return None if not self.anchored else self.anchored_approved / self.anchored

    def recall_proxy(self) -> float | None:
        """Of the moments worth keeping, how many the operator did NOT mark.

        §14: "particularly valuable: it directly measures how many good clips the
        operator misses live, which is the exact worry that motivated automatic
        detection in the first place." A high number here is the whole project
        justifying itself; a number near zero means the markers were enough and
        the detector is not earning its keep.
        """
        return None if not self.approved else self.approved_without_marker / self.approved

    def to_json(self) -> dict:
        return {
            "anchored": self.anchored,
            "anchored_approved": self.anchored_approved,
            "approved": self.approved,
            "approved_without_marker": self.approved_without_marker,
            MARKER_PRECISION: _round(self.precision()),
            MARKER_RECALL: _round(self.recall_proxy()),
        }


# --------------------------------------------------------------------------
# collection
# --------------------------------------------------------------------------


def _rated_candidates(conn: sqlite3.Connection, stream_ids: list[str]):
    """Every current candidate an OPERATOR rated, with its vector.

    Three filters, each of which is silently wrong rather than loud if dropped:

    **`rating_source = 'operator'`.** Inherited ratings are copies carried onto a
    new generation by time overlap. `score/runner.py` says so where it writes
    them: the filter is what stops one verdict being counted twice as fresh
    evidence, and §14's tuning input is exactly the place that would happen.

    **`is_current = 1`.** An older generation was scored under different weights,
    so its vectors describe a scoring run that no longer exists.

    **`rating IS NOT NULL`** is implied by the join — an unrated candidate is not
    evidence either way, and counting it as a rejection would turn "not looked at
    yet" into "looked at and refused".
    """
    if not stream_ids:
        return []
    placeholders = ", ".join("?" for _ in stream_ids)
    return conn.execute(
        f"""
        SELECT c.id, c.stream_id, c.t_start, c.t_end,
               c.feature_vector, c.contributing_signals, r.rating
          FROM candidates c
          JOIN ratings r ON r.candidate_id = c.id
         WHERE c.stream_id IN ({placeholders})
           AND c.is_current = 1
           AND r.rating_source = 'operator'
         ORDER BY c.stream_id, c.t_peak
        """,
        stream_ids,
    ).fetchall()


def collect_signals(conn: sqlite3.Connection, stream_ids: list[str],
                    schema) -> dict[str, SignalStats]:
    """§14's `signal_firing_rate_by_rating`, gathered.

    **A null is not a zero, and this is the whole reason the function is not two
    lines of SQL.** `feature_vector` carries every declared signal on every
    candidate (A9), with null wherever the signal had no observation — an
    unvoiced frame for `mic_f0`, a stream with no input log for `input_rate`.
    Treating those as zero would say the pitch was average when the truth is
    that nobody was speaking, and `mic_f0` — unvoiced most of any stream — would
    come out looking like a signal that never fires.

    So each signal keeps its OWN denominator: the candidates where it was
    actually observed. `score/derived.py` states the same rule for the
    extraction side ("a derivation whose inputs are absent produces nothing; it
    does not produce zeros"); this is its analysis-side twin.
    """
    stats = {
        name: SignalStats(name=name, group=str(meta.get("group", "?")))
        for name, meta in schema.signals.items()
    }

    for row in _rated_candidates(conn, stream_ids):
        vector = json.loads(row["feature_vector"] or "{}")
        rating = int(row["rating"])
        for name, value in vector.items():
            entry = stats.get(name)
            # A key not in the schema is from an older `feature_schema_version`.
            # Skipped rather than invented: the schema is what declares the
            # comparable set, and its own header forbids reusing a key.
            if entry is None or value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if rating == APPROVED:
                entry.approved.append(number)
            elif rating == REJECTED:
                entry.rejected.append(number)
            elif rating == MAYBE:
                entry.maybes.append(number)

    return stats


def collect_markers(conn: sqlite3.Connection,
                    stream_ids: list[str]) -> MarkerStats:
    """§14's `marker_precision` and `marker_recall_proxy`.

    "Marker-anchored" is `queries.is_marker_anchored`, the same predicate §7.4's
    rail section uses — deliberately not a second definition here. A metric that
    scored it differently from the screen the operator actually rated on would
    be measuring something nobody saw.
    """
    stats = MarkerStats()
    markers_by_stream = {sid: load_markers(conn, sid) for sid in stream_ids}

    for row in _rated_candidates(conn, stream_ids):
        payload = json.loads(row["contributing_signals"] or "{}")
        contributions = {k: v for k, v in payload.items() if not k.startswith("_")}
        inside = [t for t in markers_by_stream.get(row["stream_id"], [])
                  if row["t_start"] <= t <= row["t_end"]]
        anchored = is_marker_anchored(inside, contributions)
        approved = int(row["rating"]) == APPROVED

        if anchored:
            stats.anchored += 1
            if approved:
                stats.anchored_approved += 1
        if approved:
            stats.approved += 1
            if not anchored:
                stats.approved_without_marker += 1

    return stats


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------


@dataclass
class Tuning:
    """Everything §17 needs, plus the sample sizes that say whether to trust it."""

    signals: list[dict] = field(default_factory=list)
    markers: dict = field(default_factory=dict)
    streams: int = 0
    rated: int = 0
    approved: int = 0
    rejected: int = 0
    threshold: float = 1.0
    min_samples: int = 30

    @property
    def enough(self) -> bool:
        """Whether a RATE may be printed at all.

        Gating on the smaller side of the contrast, not the total: 200 rejections
        and 2 approvals is not a sample that can say which signals predict
        approval, however large the total looks.
        """
        return min(self.approved, self.rejected) >= self.min_samples

    def to_json(self) -> dict:
        return {
            "streams": self.streams,
            "rated": self.rated,
            "approved": self.approved,
            "rejected": self.rejected,
            "enough_to_tune": self.enough,
            "firing_threshold_z": self.threshold,
            "min_samples_for_rate": self.min_samples,
            FIRING_METRIC: self.signals,
            **self.markers,
        }


def tuning_metrics(conn: sqlite3.Connection, cfg,
                   stream_ids: list[str] | None = None) -> Tuning:
    """§14's three, over the whole library by default.

    Library-wide because §17 tunes weights for the corpus, not per recording: a
    weight that is right for one stream and wrong for the next nine is not a
    weight worth setting. `clipforge metrics <id>` still narrows it.
    """
    if stream_ids is None:
        stream_ids = [r["id"] for r in conn.execute("SELECT id FROM streams ORDER BY id")]

    schema = cfg.feature_schema
    stats = collect_signals(conn, stream_ids, schema)
    markers = collect_markers(conn, stream_ids)
    threshold = float(cfg.get("metrics.firing_threshold_z"))

    rows = [entry.to_json(threshold) for entry in stats.values() if entry.observed]
    # Strongest separation first, and unranked within a group boundary — see
    # `report`. `None` sorts last: a signal seen on only one side of the
    # contrast has no separation to rank by.
    rows.sort(key=lambda r: (r["separation"] is None,
                             -abs(r["separation"] or 0.0)))

    rated = _rated_candidates(conn, stream_ids)
    by_rating = [int(r["rating"]) for r in rated]

    return Tuning(
        signals=rows,
        markers=markers.to_json(),
        streams=len(stream_ids),
        rated=len(rated),
        approved=sum(1 for r in by_rating if r == APPROVED),
        rejected=sum(1 for r in by_rating if r == REJECTED),
        threshold=threshold,
        min_samples=int(cfg.get("metrics.min_samples_for_rate")),
    )
