"""§14's weight-tuning metrics: which signals actually discriminate.

§14 calls `signal_firing_rate_by_rating` **the primary weight-tuning input** and
§17 builds its whole tuning procedure on it:

> after every ~5 streams, pull `signal_firing_rate_by_rating` from
> `tool_metrics`, compare firing rates on rating-2 vs rating-0 candidates, and
> adjust weights toward signals that discriminate.

Nothing computed it, and nothing had ever read `candidates.feature_vector` at
all — A9 has been filling that column since Phase 1 for exactly this moment.

TWO THINGS §14's ONE-LINE DESCRIPTION DOES NOT SURVIVE CONTACT WITH

**1. "Fired" is not one test.** `feature_vector` holds three different kinds of
number under one roof, and `feature_schema.yaml`'s `group` is what tells them
apart:

| group        | value at `t_peak`                    | range             |
|--------------|--------------------------------------|-------------------|
| `continuous` | rolling z-score                      | signed, unbounded |
| `events`     | decayed event kernel                 | 0..1              |
| `composite`  | ditto — §5.4.3 emits events          | 0..1              |
| `afk`, `menu_screen` | §6.4 gate ramp, NOT z-scored | 0..1              |

A single threshold across those cannot mean one thing. So the number this module
ranks by is **separation**, which needs no threshold at all: the probability
that a randomly chosen *clip it* moment scores higher on this signal than a
randomly chosen *skip* moment. It is a rank statistic, so it means the same
thing over a z-score, a kernel level and a gate ramp — which is the only reason
one column can cover all three.

0.5 is no discrimination. 1.0 is perfect. **Below 0.5 is the interesting case**:
a signal that discriminates the wrong way, which is a weight with the wrong
sign, and no firing rate would have made that visible as such.

§14's literal firing rate is reported beside it, because §14 names it and §17
says to pull that name out of `tool_metrics`. Events, composites and gates fire
at `> 0`, which is **grounded** — that is what a kernel level means. Only
continuous signals need a chosen cut, so there is exactly one arbitrary number
in this module: `tuning.firing_threshold_z`.

**2. Which ratings even count.** The obvious query is `review_metrics`' own —
`c.is_current = 1 AND r.rating_source = 'operator'` — and it returns NOTHING on
a re-scored stream, because the operator's row stays on the superseded
generation while the current one carries an `'inherited'` copy the filter
excludes. The primary tuning input would read zero on exactly the corpus it
exists to measure. `clipforge/moments.py` has the rule instead: one opinion per
moment, across generations, latest wins.

`verdict(group)["id"]` is the candidate the deciding opinion was actually formed
on, and therefore the candidate whose feature vector the operator was looking at
when they formed it. That is the pairing this module wants, and it needed no new
rule.

IT REFUSES RATHER THAN PRODUCING A TABLE THAT LOOKS LIKE EVIDENCE

Below `tuning.min_rated_moments` there is no ranking at all, and below
`tuning.min_moments_per_class` an individual signal reports why instead of a
number. §11.1's timing honesty applied one layer down: a discrimination table
built from three ratings is not a weak finding, it is a table that reads as
evidence and is not.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field

import numpy as np

from clipforge import moments
from clipforge.score.combined import ranks

#: §14's own name, verbatim. HANDOFF's rule: a metric renamed here is a metric
#: whatever reads it later cannot find.
FIRING_METRIC = "signal_firing_rate_by_rating"
PRECISION_METRIC = "marker_precision"
RECALL_METRIC = "marker_recall_proxy"

#: §7.3's "clip it" and "skip" — the two classes §17 says to compare. Rating 1
#: ("maybe") is counted and reported but is deliberately in neither: it is the
#: operator declining to decide, which is not evidence in either direction.
SKIP, CLIP = 0, 2

#: Kernel levels, composite kernels and §6.4 gate ramps are all zero when the
#: thing did not happen, so "fired" needs no chosen threshold for them. Only
#: `continuous` does. NOT config: it is what a kernel level means.
EVENT_FIRING_FLOOR = 0.0


@dataclass(frozen=True)
class SignalStat:
    """One row of the tuning table."""

    name: str
    group: str
    n_skip: int
    n_clip: int
    separation: float | None
    #: Why `separation` is None. Empty when it is not.
    reason: str = ""
    fired_skip: int = 0
    fired_clip: int = 0

    @property
    def rate_skip(self) -> float | None:
        return self.fired_skip / self.n_skip if self.n_skip else None

    @property
    def rate_clip(self) -> float | None:
        return self.fired_clip / self.n_clip if self.n_clip else None

    @property
    def observations(self) -> int:
        return self.n_skip + self.n_clip


@dataclass(frozen=True)
class MarkerStat:
    """§14's `marker_precision` and `marker_recall_proxy` for one stream.

    Computed on `press_inside` alone — see `moments.MarkerAnchoring`. The loose
    reading reads `contributing_signals`, which is built from weighted tracks,
    so it moves when a weight moves; a weight-tuning input cannot do that.
    """

    stream_id: str
    anchored: int = 0
    anchored_approved: int = 0
    approved: int = 0
    approved_unmarked: int = 0

    @property
    def precision(self) -> float | None:
        """§14: "fraction of marker-anchored candidates approved"."""
        return self.anchored_approved / self.anchored if self.anchored else None

    @property
    def recall_proxy(self) -> float | None:
        """§14: "fraction of approved candidates that had no marker" — what the
        operator missed live, which is the worry that motivated the detector."""
        return self.approved_unmarked / self.approved if self.approved else None


@dataclass
class Corpus:
    """What the numbers rest on. Printed first, always."""

    streams: int = 0
    rated_streams: int = 0
    moments: int = 0
    by_rating: dict[int, int] = field(default_factory=dict)
    #: Moments whose deciding candidate carried no usable vector at all.
    without_vector: int = 0
    #: `feature_schema_version` values seen. More than one is not an error —
    #: the metric reads declared keys, so a key present in both is comparable —
    #: but it is worth seeing.
    schema_versions: list[int] = field(default_factory=list)

    @property
    def comparable(self) -> int:
        """Moments in one of §17's two classes."""
        return self.by_rating.get(SKIP, 0) + self.by_rating.get(CLIP, 0)


# --------------------------------------------------------------------------
# the statistic
# --------------------------------------------------------------------------


def separation(skip: np.ndarray, clip: np.ndarray) -> float | None:
    """P(a random *clip it* value > a random *skip* value), ties counted half.

    The normalised Mann-Whitney U, i.e. the area under the ROC curve. Two
    properties earn it the primary column:

    * **No threshold.** A firing rate has to choose one, and the choice then
      decides the answer for every signal it is applied to.
    * **Rank-based.** It is invariant to the scale of the values, so a z-score
      that swings ±3, a kernel level in 0..1 and a §6.4 gate ramp all produce a
      number meaning the same thing.

    Computed from `score.combined.ranks` — the same hand-rolled, ties-averaged
    helper Spearman already uses here, rather than importing `scipy.stats` for
    six lines.

    None when either class is empty: undefined is not 0.5, and reporting 0.5
    would read as "measured, and it does not discriminate". The distinction
    `rank_agreement` already makes for an undefined correlation.
    """
    n_skip, n_clip = int(skip.size), int(clip.size)
    if not n_skip or not n_clip:
        return None

    combined = np.concatenate([skip, clip])
    # `ranks` is 0-based with ties averaged; U is defined over 1-based ranks.
    ordered = ranks(combined) + 1.0
    rank_sum_clip = float(ordered[n_skip:].sum())
    u = rank_sum_clip - n_clip * (n_clip + 1) / 2.0
    return round(u / (n_skip * n_clip), 4)


# --------------------------------------------------------------------------
# gathering
# --------------------------------------------------------------------------


def _decided(conn: sqlite3.Connection, stream_id: str) -> list[dict]:
    """One entry per moment: the deciding opinion and the row it was formed on.

    `moments.verdict` picks the most recent opinion in a cluster, so a moment
    the operator re-rated after a re-score contributes once, with the opinion
    they ended on. §14's stated hazard is counting one judgment twice.
    """
    return [moments.verdict(group)
            for group in moments.cluster(moments.rated_candidates(conn, stream_id))]


def _vectors(conn: sqlite3.Connection, ids: list[int]) -> dict[int, tuple[dict, dict, int]]:
    """`{candidate_id: (feature_vector, contributions, schema_version)}`.

    One query with an `IN (...)`, not one per moment — the rule `search._hydrate`
    and `queries.load_candidates` both already follow.
    """
    if not ids:
        return {}
    rows = conn.execute(
        "SELECT id, feature_vector, contributing_signals, feature_schema_version "
        "FROM candidates WHERE id IN (%s)" % ",".join("?" * len(ids)),
        ids,
    ).fetchall()

    out: dict[int, tuple[dict, dict, int]] = {}
    for row in rows:
        try:
            vector = json.loads(row["feature_vector"] or "{}")
        except json.JSONDecodeError:
            vector = {}
        payload = json.loads(row["contributing_signals"] or "{}")
        contributions = {k: v for k, v in payload.items() if not k.startswith("_")}
        out[int(row["id"])] = (
            vector if isinstance(vector, dict) else {},
            contributions,
            int(row["feature_schema_version"]),
        )
    return out


def _fires(group: str, value: float, threshold_z: float) -> bool:
    """§14's literal "did this signal fire", per value kind."""
    if group == "continuous":
        return value > threshold_z
    return value > EVENT_FIRING_FLOOR


def collect(
    conn: sqlite3.Connection, cfg, stream_ids: list[str],
) -> tuple[Corpus, list[SignalStat], list[MarkerStat]]:
    """Everything the report needs, in one pass over the corpus."""
    schema = cfg.feature_schema
    threshold_z = float(cfg.get("tuning.firing_threshold_z"))
    min_per_class = int(cfg.get("tuning.min_moments_per_class"))
    approved_at = int(cfg.get("tuning.approved_rating"))

    corpus = Corpus(streams=len(stream_ids))
    versions: set[int] = set()
    # {signal: {SKIP: [values], CLIP: [values]}}
    samples: dict[str, dict[int, list[float]]] = {
        name: {SKIP: [], CLIP: []} for name in schema.keys
    }
    markers: list[MarkerStat] = []

    for stream_id in stream_ids:
        decided = _decided(conn, stream_id)
        if decided:
            corpus.rated_streams += 1
        corpus.moments += len(decided)

        loaded = _vectors(conn, [int(d["id"]) for d in decided])
        presses = moments.marker_times(conn, stream_id)
        anchored = anchored_approved = approved = approved_unmarked = 0

        for entry in decided:
            rating = int(entry["rating"])
            corpus.by_rating[rating] = corpus.by_rating.get(rating, 0) + 1

            vector, contributions, version = loaded.get(int(entry["id"]), ({}, {}, 0))
            if not vector:
                corpus.without_vector += 1
            else:
                versions.add(version)

            # §14's marker pair. `press_inside` only — see MarkerStat.
            anchoring = moments.marker_anchoring(
                entry["t_start"], entry["t_end"], presses, contributions)
            is_anchored = bool(anchoring.press_inside)
            is_approved = rating >= approved_at
            anchored += is_anchored
            approved += is_approved
            anchored_approved += is_anchored and is_approved
            approved_unmarked += is_approved and not is_anchored

            if rating not in (SKIP, CLIP):
                continue
            # Declared keys only, never the JSON's own. A version-1 vector
            # carries context keys the current writer no longer emits, and
            # iterating the payload would score `mic_rms_db` — an absolute dB
            # level — as if it were a signal.
            for name in schema.keys:
                value = vector.get(name)
                if value is None:
                    continue
                samples[name][rating].append(float(value))

        markers.append(MarkerStat(
            stream_id=stream_id, anchored=anchored,
            anchored_approved=anchored_approved,
            approved=approved, approved_unmarked=approved_unmarked,
        ))

    corpus.schema_versions = sorted(versions)

    stats: list[SignalStat] = []
    for name in schema.keys:
        group = schema.signals[name].get("group", "")
        skip = np.asarray(samples[name][SKIP], dtype=np.float64)
        clip = np.asarray(samples[name][CLIP], dtype=np.float64)

        value = None
        reason = ""
        if skip.size < min_per_class or clip.size < min_per_class:
            reason = (f"needs {min_per_class} of each; "
                      f"have {skip.size} skip, {clip.size} clip it")
        else:
            value = separation(skip, clip)
            if value is None:
                reason = "undefined"

        stats.append(SignalStat(
            name=name, group=group,
            n_skip=int(skip.size), n_clip=int(clip.size),
            separation=value, reason=reason,
            fired_skip=int(sum(_fires(group, v, threshold_z) for v in skip)),
            fired_clip=int(sum(_fires(group, v, threshold_z) for v in clip)),
        ))

    # Most discriminating first; a signal with no number sorts to the bottom
    # rather than into the middle, where 0.5 would put it.
    stats.sort(key=lambda s: (s.separation is not None,
                              abs((s.separation or 0.5) - 0.5),
                              s.observations),
               reverse=True)
    return corpus, stats, markers


def rankable(corpus: Corpus, cfg) -> tuple[bool, str]:
    """Whether there is enough evidence to print a ranking at all."""
    minimum = int(cfg.get("tuning.min_rated_moments"))
    if corpus.comparable < minimum:
        return False, (
            f"{corpus.comparable} moment(s) rated skip or clip it, and "
            f"tuning.min_rated_moments is {minimum}. §17 tunes after every ~5 "
            f"streams; this is what that looks like before the first one."
        )
    return True, ""


def record(conn: sqlite3.Connection, corpus: Corpus,
           stats: list[SignalStat], markers: list[MarkerStat]) -> int:
    """§17: "pull `signal_firing_rate_by_rating` from `tool_metrics`"."""
    written = 0
    for stat in stats:
        if stat.separation is None and not stat.observations:
            continue
        conn.execute(
            "INSERT INTO tool_metrics (stream_id, metric, value, meta) VALUES (NULL, ?, ?, ?)",
            (FIRING_METRIC,
             # `value` is a bare REAL, so an undefined separation is stored as
             # 0.0 with a companion boolean — the shape `combined_rank_agreement`
             # already uses for an undefined Spearman.
             float(stat.separation) if stat.separation is not None else 0.0,
             json.dumps({
                 "signal": stat.name, "group": stat.group,
                 "separation_defined": stat.separation is not None,
                 "reason": stat.reason,
                 "n_skip": stat.n_skip, "n_clip": stat.n_clip,
                 "fired_skip": stat.fired_skip, "fired_clip": stat.fired_clip,
                 "rate_skip": stat.rate_skip, "rate_clip": stat.rate_clip,
                 "corpus_moments": corpus.moments,
                 "corpus_rated_streams": corpus.rated_streams,
             })),
        )
        written += 1

    for marker in markers:
        if not marker.anchored and not marker.approved:
            continue
        for metric, value in ((PRECISION_METRIC, marker.precision),
                              (RECALL_METRIC, marker.recall_proxy)):
            conn.execute(
                "INSERT INTO tool_metrics (stream_id, metric, value, meta) VALUES (?, ?, ?, ?)",
                (marker.stream_id, metric,
                 float(value) if value is not None else 0.0,
                 json.dumps({
                     "defined": value is not None,
                     "anchored": marker.anchored,
                     "anchored_approved": marker.anchored_approved,
                     "approved": marker.approved,
                     "approved_unmarked": marker.approved_unmarked,
                     "anchoring": "press_inside",
                 })),
            )
            written += 1
    return written
