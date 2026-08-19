"""§9.3 — chapter segmentation, deterministically.

> Segment boundaries are found **without** an LLM:
> 1. Transcript embedding shift — cosine distance between consecutive
>    rolling-window embeddings; peaks are topic boundaries
> 2. Game changes (from OCR or manual stream metadata)
> 3. Long silence gaps (> 60 s)
> 4. Scene changes (weak signal, tie-breaker only)
>
> Merge boundaries within 120 s. Target chapter length: 10–30 minutes.

**Source 2 is not implemented, because nothing can produce it.** `streams.games`
is a single untimed JSON array for the whole stream, and the OCR that §9.3's own
wording points at is Phase 7 — unbuilt, and the one thing §15 says to cut if it
proves difficult. A boundary source wired to a column that has no timestamps
would fire never and look implemented. It is recorded as a gap in
`spec/GUESSES.md` instead.

**Source 1 is unavailable on every stream that exists today**, which is the fact
this module is built around. Embedding shift needs `whisperx` → `embeddings`, and
Phase 2 ships off (`extract.whisperx.enabled: false`) against a corpus of zero
streams. So a real segmentation today runs on silence gaps and — if an OBS log
was attached — scene changes, which §9.3 itself ranks last of four.

That is not a reason to refuse. It IS a reason to say so: `Segmentation.sources`
records which signals actually contributed, it travels into the digest JSON, and
`digest_boundary_sources` goes to `tool_metrics`. A digest built from silence
alone is a different artifact from one built from embeddings, and nothing else in
the system would ever say which one you are holding.

**Two rules of different kinds.** "Merge within 120 s" is local: it dedupes a
cluster of boundary candidates describing one transition. "10–30 minutes" is
global: it is a statement about the shape of the whole stream. Local merging
cannot produce it — a 3-hour stream with eleven silence gaps in its first twenty
minutes merges to eleven boundaries there and none after. So there is a second,
explicitly global pass, and the resulting length distribution is logged, because
it is the only observation that would show either number is wrong.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import numpy as np

from clipforge import signals

#: Where a boundary came from. Ordered by §9.3's own ranking, strongest first --
#: `Boundary.strength` breaks ties within a source, this breaks them across.
SOURCE_EMBEDDING = "embedding_shift"
SOURCE_SILENCE = "silence_gap"
SOURCE_SCENE = "scene_change"

#: §9.3's ranking as a number, used when two candidates survive into the same
#: merge cluster. §9.3 calls scene changes a "tie-breaker only", which is exactly
#: this: it can position a boundary, never justify one on its own.
SOURCE_RANK = {SOURCE_EMBEDDING: 3, SOURCE_SILENCE: 2, SOURCE_SCENE: 1}


@dataclass(frozen=True)
class Boundary:
    """One candidate topic boundary."""

    t: float
    source: str
    #: In [0, 1] within its own source. NOT comparable across sources -- a
    #: cosine distance and a silence length are different quantities, and
    #: `SOURCE_RANK` is what orders them against each other.
    strength: float

    @property
    def rank(self) -> tuple[int, float]:
        return SOURCE_RANK.get(self.source, 0), self.strength


@dataclass
class Chapter:
    """One chapter. Times are VOD seconds; `seq_*` are `segments.seq` (§12.1)."""

    index: int
    t_start: float
    t_end: float
    #: The boundary that OPENED this chapter, or None for the first one.
    opened_by: str | None = None

    @property
    def duration_s(self) -> float:
        return self.t_end - self.t_start


@dataclass
class Segmentation:
    """The chapters, and an honest account of what produced them."""

    chapters: list[Chapter] = field(default_factory=list)
    #: Sources that contributed at least one SURVIVING boundary. Not the sources
    #: that were merely available -- an embedding series that produced no peak
    #: did not shape this segmentation and must not claim to have.
    sources: list[str] = field(default_factory=list)
    #: Why a source contributed nothing, keyed by source. Carried into the digest
    #: so "no embedding boundaries" can be told from "no transcript at all".
    absent: dict[str, str] = field(default_factory=dict)

    @property
    def lengths(self) -> list[float]:
        return [c.duration_s for c in self.chapters]


# --------------------------------------------------------------------------
# the three sources
# --------------------------------------------------------------------------


def embedding_boundaries(
    conn: sqlite3.Connection, stream_id: str, *, window_s: float,
    min_distance: float,
) -> tuple[list[Boundary], str]:
    """§9.3 source 1: cosine distance between consecutive rolling windows.

    Vectors are stored L2-normalised (see `extract/embeddings.py`), so cosine is
    a dot product and the distance is `1 - dot`. **Filtered to one model**: that
    module is explicit that a cosine between two geometries is "finite, ordered,
    and meaningless", and a library embedded with two models would otherwise
    produce a boundary at the point the model changed.

    Returns the boundaries and, when there are none, why -- the caller has to be
    able to distinguish "no transcript" from "one continuous topic".
    """
    rows = conn.execute(
        """
        SELECT s.seq, s.t_start, e.vec, e.model
          FROM segments s
          JOIN segment_embeddings e ON e.segment_id = s.id
         WHERE s.stream_id = ?
         ORDER BY s.t_start
        """,
        (stream_id,),
    ).fetchall()
    if not rows:
        return [], "no segment embeddings (needs whisperx + embeddings)"

    # The commonest model wins; a stream embedded twice is a real possibility
    # once a model is changed and only some streams are re-run.
    models: dict[str, int] = {}
    for row in rows:
        models[row["model"]] = models.get(row["model"], 0) + 1
    model = max(models, key=lambda name: models[name])
    rows = [row for row in rows if row["model"] == model]
    if len(rows) < 4:
        return [], f"only {len(rows)} embedded segment(s), too few to compare"

    times = np.array([float(row["t_start"]) for row in rows], dtype=np.float64)
    vectors = np.vstack([
        np.frombuffer(row["vec"], dtype=np.float32) for row in rows
    ]).astype(np.float64)

    # Rolling window MEANS, not individual segments: §9.3 says "rolling-window
    # embeddings", and one sentence is far too short a unit for a topic. The
    # mean of unit vectors is not itself unit, so it is re-normalised before the
    # dot product -- otherwise the "cosine" is scaled by how much the window
    # agreed with itself, and a coherent window would look FURTHER from its
    # neighbour than an incoherent one.
    out: list[Boundary] = []
    distances: list[tuple[float, float]] = []
    for index in range(1, len(rows)):
        split = times[index]
        before = vectors[(times >= split - window_s) & (times < split)]
        after = vectors[(times >= split) & (times < split + window_s)]
        if before.shape[0] < 2 or after.shape[0] < 2:
            continue
        left = _unit(before.mean(axis=0))
        right = _unit(after.mean(axis=0))
        if left is None or right is None:
            continue
        distances.append((split, float(1.0 - float(np.dot(left, right)))))

    if not distances:
        return [], "transcript too short for a rolling window"

    # Local maxima only. Every point in a topic change is "distant"; the
    # boundary is the peak, and taking all of them would put a boundary on every
    # sentence of a transition.
    values = np.array([d for _t, d in distances], dtype=np.float64)
    peak = float(values.max())
    if peak <= 0:
        return [], "no topic shift found in the transcript"

    for i, (t, value) in enumerate(distances):
        if value < min_distance:
            continue
        if i > 0 and distances[i - 1][1] > value:
            continue
        if i + 1 < len(distances) and distances[i + 1][1] > value:
            continue
        out.append(Boundary(t=t, source=SOURCE_EMBEDDING,
                            strength=min(1.0, value / peak)))
    if not out:
        return [], f"no shift reached min_distance={min_distance}"
    return out, ""


def _unit(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    return None if norm <= 0 else vector / norm


def silence_boundaries(
    conn: sqlite3.Connection, stream_id: str, *, gap_s: float,
    duration_s: float, floor_db: float,
) -> tuple[list[Boundary], str]:
    """§9.3 source 3: gaps longer than `gap_s`.

    Preferring `segments` when a transcript exists: a gap between transcribed
    speech is a gap in SPEECH, which is what §9.3 means. Falling back to
    `mic_rms` otherwise, because that is the only silence signal a Phase 1 stream
    has -- and a Phase 1 stream is every stream that exists.

    The boundary is placed at the gap's MIDPOINT rather than at either edge. A
    three-minute break belongs to neither the chapter before it nor the one
    after, and putting it at the start would open the new chapter with silence.
    """
    rows = conn.execute(
        "SELECT t_start, t_end FROM segments WHERE stream_id = ? "
        "AND TRIM(text) != '' ORDER BY t_start",
        (stream_id,),
    ).fetchall()

    spans: list[tuple[float, float]]
    if rows:
        spans = [(float(r["t_start"]), float(r["t_end"])) for r in rows]
        why_empty = "no speech gap longer than the threshold"
    else:
        series = signals.load(conn, stream_id, "mic_rms")
        if series is None or len(series) == 0:
            return [], "no transcript and no mic_rms to find silence in"
        spans = _loud_spans(series, floor_db)
        why_empty = "no quiet stretch longer than the threshold"
        if not spans:
            return [], "mic_rms never rises above the floor"

    gaps: list[tuple[float, float]] = []
    previous_end = spans[0][1]
    for start, end in spans[1:]:
        if start - previous_end >= gap_s:
            gaps.append((previous_end, start))
        previous_end = max(previous_end, end)
    # The tail: a stream that ends in twenty minutes of silence has a gap there
    # too, but it opens no chapter -- there is nothing after it.
    if not gaps:
        return [], why_empty

    longest = max(end - start for start, end in gaps)
    return [
        Boundary(t=(start + end) / 2.0, source=SOURCE_SILENCE,
                 strength=min(1.0, (end - start) / longest))
        for start, end in gaps
        if 0.0 < (start + end) / 2.0 < duration_s
    ], ""


def _loud_spans(series: signals.Series, floor_db: float) -> list[tuple[float, float]]:
    """Contiguous runs where the mic is above `floor_db`.

    Thresholding in dB directly, which is legitimate ONLY because this is a
    comparison and not an average: `x > floor` is the same question in dB as in
    linear power. Anything that combined these values would have to convert
    first -- see `energy_series` in build.py, which does.
    """
    values = series.values.astype(np.float64)
    above = np.isfinite(values) & (values > floor_db)
    if not above.any():
        return []
    edges = np.diff(above.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1) + 1)
    if above[0]:
        starts.insert(0, 0)
    if above[-1]:
        ends.append(len(above))
    return [(series.time_of(int(s)), series.time_of(int(e) - 1))
            for s, e in zip(starts, ends)]


def scene_boundaries(
    conn: sqlite3.Connection, stream_id: str, *, kinds: tuple[str, ...],
    duration_s: float,
) -> tuple[list[Boundary], str]:
    """§9.3 source 4: scene changes, "weak signal, tie-breaker only".

    Every scene change gets the same strength. There is no meaningful ordering
    between two of them, and inventing one would let this source do what §9.3
    says it must not: justify a boundary rather than position one. `SOURCE_RANK`
    keeps it below the other two in every merge.
    """
    if not kinds:
        return [], "no scene event kinds configured"
    placeholders = ", ".join("?" for _ in kinds)
    rows = conn.execute(
        f"SELECT t FROM events WHERE stream_id = ? AND kind IN ({placeholders}) "
        f"ORDER BY t",
        (stream_id, *kinds),
    ).fetchall()
    if not rows:
        return [], "no scene events (needs an OBS log attached at register time)"
    out = [Boundary(t=float(r["t"]), source=SOURCE_SCENE, strength=1.0)
           for r in rows if 0.0 < float(r["t"]) < duration_s]
    return out, "" if out else "scene events all fall outside the recording"


# --------------------------------------------------------------------------
# merging, and the shape of the whole stream
# --------------------------------------------------------------------------


def merge(boundaries: list[Boundary], within_s: float) -> list[Boundary]:
    """§9.3's "merge boundaries within 120 s", keeping the best of each cluster.

    Best by `Boundary.rank`: source first, then strength. A silence gap and a
    scene change describing the same break are one boundary, and it is the
    silence gap -- §9.3 ranks scene changes last precisely so that this comes
    out that way round.
    """
    if not boundaries:
        return []
    ordered = sorted(boundaries, key=lambda b: b.t)
    clusters: list[list[Boundary]] = [[ordered[0]]]
    for boundary in ordered[1:]:
        # Against the cluster's FIRST member, not its last: chaining off the
        # last lets a run of boundaries 119 s apart collapse the whole stream
        # into one cluster spanning hours.
        if boundary.t - clusters[-1][0].t <= within_s:
            clusters[-1].append(boundary)
        else:
            clusters.append([boundary])
    return [max(cluster, key=lambda b: b.rank) for cluster in clusters]


def enforce_lengths(
    boundaries: list[Boundary], *, duration_s: float, minimum_s: float,
    maximum_s: float, all_boundaries: list[Boundary],
) -> list[Boundary]:
    """The global pass: §9.3's "target chapter length: 10-30 minutes".

    Merging is local and cannot produce a target that is a statement about the
    whole stream. Two corrections, in this order:

    **Too long** -- reinstate the strongest boundary INSIDE the over-long
    chapter, taken from the pre-merge pool so that a candidate discarded by
    merging is still reachable. A chapter with no interior candidate at all is
    left alone: splitting it at its midpoint would be inventing a topic change,
    and a 40-minute chapter is a finding about the segmentation, not a defect to
    paper over.

    **Too short** -- drop the weaker of its two boundaries, absorbing it into
    that neighbour. Deliberately NOT applied when it would leave a single
    chapter: a 12-minute stream is one chapter, and that is correct rather than
    a failure to reach the minimum.
    """
    kept = sorted(boundaries, key=lambda b: b.t)
    pool = sorted(all_boundaries, key=lambda b: b.t)

    # Too long. Bounded by the pool: each pass consumes one candidate.
    for _ in range(len(pool) + 1):
        edges = [0.0] + [b.t for b in kept] + [duration_s]
        longest, at = 0.0, -1
        for index in range(len(edges) - 1):
            span = edges[index + 1] - edges[index]
            if span > longest:
                longest, at = span, index
        if longest <= maximum_s or at < 0:
            break
        lo, hi = edges[at], edges[at + 1]
        have = {b.t for b in kept}
        # Far enough inside that the split does not immediately create a chapter
        # below the minimum -- otherwise the two passes fight each other.
        inside = [b for b in pool
                  if lo + minimum_s <= b.t <= hi - minimum_s and b.t not in have]
        if not inside:
            break
        kept.append(max(inside, key=lambda b: b.rank))
        kept.sort(key=lambda b: b.t)

    # Too short.
    for _ in range(len(kept) + 1):
        if len(kept) <= 1:
            break
        edges = [0.0] + [b.t for b in kept] + [duration_s]
        shortest, at = duration_s, -1
        for index in range(len(edges) - 1):
            span = edges[index + 1] - edges[index]
            if span < shortest:
                shortest, at = span, index
        if shortest >= minimum_s or at < 0:
            break
        # The chapter at `at` is bounded by boundary at-1 and at (0-indexed into
        # `kept`); drop whichever is weaker, which is what absorbs it into the
        # neighbour on that side.
        options = [i for i in (at - 1, at) if 0 <= i < len(kept)]
        if not options:
            break
        kept.pop(min(options, key=lambda i: kept[i].rank))

    return kept


def to_chapters(boundaries: list[Boundary], duration_s: float) -> list[Chapter]:
    edges = sorted(boundaries, key=lambda b: b.t)
    chapters: list[Chapter] = []
    starts = [0.0] + [b.t for b in edges]
    ends = [b.t for b in edges] + [duration_s]
    for index, (start, end) in enumerate(zip(starts, ends)):
        chapters.append(Chapter(
            index=index,
            t_start=round(start, 3),
            t_end=round(end, 3),
            opened_by=edges[index - 1].source if index else None,
        ))
    return chapters


def segment(conn: sqlite3.Connection, stream_id: str, *, duration_s: float,
            settings: dict) -> Segmentation:
    """Everything above, in §9.3's order."""
    result = Segmentation()
    if duration_s <= 0:
        result.absent["stream"] = "the stream has no duration"
        return result

    found: list[Boundary] = []
    for source, (boundaries, why) in {
        SOURCE_EMBEDDING: embedding_boundaries(
            conn, stream_id,
            window_s=float(settings["embedding_window_s"]),
            min_distance=float(settings["embedding_min_distance"]),
        ),
        SOURCE_SILENCE: silence_boundaries(
            conn, stream_id,
            gap_s=float(settings["silence_gap_s"]),
            duration_s=duration_s,
            floor_db=float(settings["silence_floor_db"]),
        ),
        SOURCE_SCENE: scene_boundaries(
            conn, stream_id,
            kinds=tuple(settings["scene_event_kinds"]),
            duration_s=duration_s,
        ),
    }.items():
        if why:
            result.absent[source] = why
        found.extend(boundaries)

    merged = merge(found, float(settings["merge_within_s"]))
    kept = enforce_lengths(
        merged,
        duration_s=duration_s,
        minimum_s=float(settings["min_chapter_s"]),
        maximum_s=float(settings["max_chapter_s"]),
        all_boundaries=found,
    )

    result.chapters = to_chapters(kept, duration_s)
    # Sources that actually SHAPED this segmentation, not ones that were merely
    # readable. A source whose every candidate lost a merge did not.
    result.sources = sorted({b.source for b in kept}, key=lambda s: -SOURCE_RANK[s])
    return result
