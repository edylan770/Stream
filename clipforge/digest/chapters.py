"""§9.3 — chapter boundaries, found without an LLM.

> "Segment boundaries are found **without** an LLM:
>  1. Transcript embedding shift ...
>  2. Game changes ...
>  3. Long silence gaps (> 60 s)
>  4. Scene changes (weak signal, tie-breaker only)
>  Merge boundaries within 120 s. Target chapter length: 10-30 minutes."

Chapters are the unit §9.4's map-reduce chunks over, so this is what decides how
a three-hour stream is handed to a model. It runs **entirely on stored data** —
no Ollama call, no model of any kind — which is what makes §9.3's "deterministic
first" literally rather than approximately true. Window centroids are averaged
from the per-segment vectors `extract/embeddings.py` already stored.

WHICH OF §9.3's FOUR INPUTS ACTUALLY EXISTS

    long silence      LIVE on every stream (mic/party RMS -> derived.speech_gate)
    embedding shift   live only when `extract.whisperx.enabled` is on, which it
                      is not by default -- so usually absent
    scene changes     producer exists (commit 39) but ITS PARSER HAS NEVER SEEN
                      A REAL OBS LOG, so in practice it yields nothing yet
    game changes      NO PRODUCER AT ALL. `streams.games` is an untimed JSON
                      array; timed game data is §5.9's vision, Phase 7

**So on a stream processed with shipped defaults this is a long-silence
splitter**, and `Result.inputs` says so per input rather than letting four names
in the spec imply four sources agreed. Same discipline as commit 37's live-weight
table and `derived.missing_inputs`: a signal that could not be computed explains
itself instead of contributing a silent zero.

§9.3's EMBEDDING FORMULATION IS THE WEAKER OF TWO, MEASURED

§9.3 says "cosine distance between consecutive rolling-window embeddings". Tested
against a deliberately maximal topic change -- five Marvel Rivals lines followed
by five baking lines, embedded with the real model:

    formulation                       W=2      W=3      W=4     peak/mean
    §9.3's consecutive windows        MISS     hit      --       1.2-1.3x
    before/after centroids at a gap   hit      hit      hit      1.2-1.3x

The before/after form puts the peak on the seam at every window size; §9.3's
consecutive form misses it at one. So that is what this computes.

**The ratio is the more important number.** Even for two topics as unrelated as
a hero shooter and pastry, the boundary is only 1.2-1.3x the mean distance --
`nomic-embed-text` compresses similarity so hard that everything is roughly
half-similar to everything, the same property that killed a `min_similarity`
knob in `clipforge/search.py`. On the speech fixture, thirteen lines of ONE
continuous conversation with no topic change at all, consecutive distances still
run 0.000-0.466.

Two consequences:

**No absolute distance threshold.** The scale moves with window size -- max
0.466 at W=1 against 0.138 at W=2 -- so a fixed `min_distance` would mean
different things at different settings. Prominence comes from
`score/windows.py`'s existing peak finder, expressed in the distance array's
OWN standard deviations (`prominence_sd`), so it is scale-free.

**Every boundary carries its strength and the report prints it.** A 1.3x bump
must not present itself as a topic change. This is the `laughter` situation
exactly: the fixture can show the mechanism responds; it cannot show that these
are good chapters.

CHAPTERS TILE THE STREAM

No gaps, no overlaps, the first starts at 0 and the last ends at `duration_s`.
§9.2's structure is a partition and §9.4 chunks over it, so a hole would silently
drop transcript from the digest -- the one failure here that would be invisible
in the output.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field

import numpy as np

from clipforge import signals
from clipforge.extract import embeddings
from clipforge.score import derived, gates, grid, windows

#: §9.3's four, in the order it lists them. The value is what `Result.inputs`
#: reports when the input produced nothing.
INPUTS = ("embedding", "game", "silence", "scene")

#: `events.source` written by commit 39's `scene_events`.
SCENE_SOURCE = "scene"


class ChapterError(RuntimeError):
    pass


@dataclass
class Boundary:
    """One candidate chapter start, and what proposed it."""

    t: float
    source: str
    #: Comparable only WITHIN a source. Seconds of silence, prominence in
    #: standard deviations, or 1.0 for a scene switch, which has no magnitude.
    strength: float
    detail: str = ""
    #: Set when `merge` absorbs others into this one. Two inputs agreeing is
    #: worth seeing, and it is the only corroboration this stage can offer.
    corroborated_by: list[str] = field(default_factory=list)
    #: False for candidates kept only so an over-long chapter has somewhere
    #: honest to split.
    accepted: bool = True

    def to_json(self) -> dict:
        return {
            "t": round(self.t, 2),
            "source": self.source,
            "strength": round(self.strength, 4),
            "detail": self.detail,
            "corroborated_by": list(self.corroborated_by),
        }


@dataclass
class Chapter:
    index: int
    t_start: float
    t_end: float
    #: What opened this chapter. None for the first, which opens because the
    #: stream does.
    opened_by: Boundary | None = None

    @property
    def duration_s(self) -> float:
        return self.t_end - self.t_start

    def to_json(self) -> dict:
        return {
            "index": self.index,
            "t_start": round(self.t_start, 2),
            "t_end": round(self.t_end, 2),
            "duration_s": round(self.duration_s, 2),
            "opened_by": self.opened_by.to_json() if self.opened_by else None,
        }


@dataclass
class Result:
    chapters: list[Chapter] = field(default_factory=list)
    boundaries: list[Boundary] = field(default_factory=list)
    #: input name -> why it contributed nothing. Absent means it contributed.
    inputs: dict[str, str] = field(default_factory=dict)
    #: Chapters that could not be brought inside §9.3's target range, and why.
    unmet_targets: list[str] = field(default_factory=list)

    @property
    def contributing(self) -> list[str]:
        return [name for name in INPUTS if name not in self.inputs]


# --------------------------------------------------------------------------
# input 3: long silence — the only one live on every stream
# --------------------------------------------------------------------------


def silence_boundaries(
    conn: sqlite3.Connection, stream_id: str, cfg, duration_s: float
) -> tuple[list[Boundary], str]:
    """Runs of "no speech on either voice track" at least `min_silence_s` long.

    **The gate is §6.4's, not a third copy of it.** `gates.speech_activity`
    unions `derived.speech_gate` over `VOICE_KINDS`, and HANDOFF records that
    §6.4's "ANY speech detected" and §5.4.1's "VAD on both tracks" being asked
    twice was a real hazard — a test asserts the two agree sample for sample.
    Asking a third time here would reopen it.

    **The boundary is the END of the gap, not its middle.** The dead air belongs
    to the chapter that just finished, not to the one about to start: content
    stopped, then there was silence, then a new thing began. Splitting the gap
    down the middle would put half of a two-minute pause at the head of the next
    chapter, where §9.4 would hand it to a model as context.
    """
    grid_hz = float(cfg.get("score.score_grid_hz"))
    timeline = grid.build(duration_s, grid_hz)

    raw: dict[str, np.ndarray] = {}
    for name, series in signals.load_all(conn, stream_id).items():
        if name in gates.VOICE_KINDS:
            raw[name] = grid.resample(series, timeline)
    if not raw:
        return [], (f"no {' or '.join(gates.VOICE_KINDS)} for this stream — "
                    f"§9.3's silence input needs a voice track")

    inputs = derived.Inputs(timeline=timeline, grid_hz=grid_hz, raw=raw, cfg=cfg)
    baseline = grid.window_samples(cfg.get("score.rolling_baseline_window_s"), grid_hz)
    speech = gates.speech_activity(raw, inputs, baseline)
    if speech.size == 0:
        return [], "the speech gate produced nothing"

    minimum = float(cfg.get("digest.chapters.min_silence_s"))
    found: list[Boundary] = []
    for start, end in _runs(~speech):
        gap = (end - start) / grid_hz
        if gap < minimum:
            continue
        t = float(timeline[min(end, timeline.size - 1)])
        found.append(Boundary(
            t=t, source="silence", strength=gap,
            detail=f"{gap:.0f}s of silence, {timeline[start]:.0f}-{t:.0f}s",
        ))
    return found, ""


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """`[start, end)` index pairs for each run of True."""
    if flags.size == 0:
        return []
    padded = np.concatenate([[False], flags.astype(bool), [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist(), strict=True))


# --------------------------------------------------------------------------
# input 1: transcript embedding shift
# --------------------------------------------------------------------------


def embedding_boundaries(
    conn: sqlite3.Connection, stream_id: str, cfg
) -> tuple[list[Boundary], str]:
    """Topic shift, as the depth of the split between two neighbourhoods.

    For each gap between consecutive segments, the centroid of the `window`
    segments BEFORE it against the centroid of the `window` segments AFTER it.
    Deeper means the two sides talk about less similar things.

    Not §9.3's "consecutive rolling-window" form — MEASURED to miss a maximal
    topic seam at one of three window sizes where this form hits it at all
    three. See the module docstring.
    """
    rows = conn.execute(
        """
        SELECT s.t_start, s.t_end, e.vec, e.dim, e.model
          FROM segments s JOIN segment_embeddings e ON e.segment_id = s.id
         WHERE s.stream_id = ?
         ORDER BY s.t_start, s.seq
        """,
        (stream_id,),
    ).fetchall()
    if not rows:
        return [], ("no transcript embeddings — §5.7's transcription ships off "
                    "(`extract.whisperx.enabled: false`), so this input is "
                    "absent on a stream processed with defaults")

    # One geometry only: a cosine across two models is finite and meaningless.
    model = rows[0]["model"]
    rows = [r for r in rows if r["model"] == model]
    dim = int(rows[0]["dim"])
    rows = [r for r in rows if int(r["dim"]) == dim]

    window = int(cfg.get("digest.chapters.embedding.window"))
    if len(rows) < 2 * window:
        return [], (f"{len(rows)} embedded segment(s), and a window of {window} "
                    f"needs at least {2 * window} to compare two sides of a gap")

    vectors = np.stack([embeddings.from_blob(r["vec"]) for r in rows]).astype(np.float64)
    depths, times = [], []
    for i in range(window, len(rows) - window + 1):
        before = _centroid(vectors[i - window:i])
        after = _centroid(vectors[i:i + window])
        depths.append(1.0 - float(before @ after))
        # The gap itself: after the previous segment ends, before this one starts.
        times.append((float(rows[i - 1]["t_end"]) + float(rows[i]["t_start"])) / 2.0)

    depth = np.asarray(depths, dtype=np.float64)
    if depth.size < 3:
        return [], f"only {depth.size} comparable gap(s); too few to find a peak"

    # Prominence in the array's OWN standard deviations. An absolute threshold
    # would mean different things at different window sizes — measured, the
    # distance scale moves by 3x between W=1 and W=2.
    spread = float(depth.std())
    if spread <= 0:
        return [], "every gap has identical depth; nothing distinguishes a boundary"
    prominence = float(cfg.get("digest.chapters.embedding.prominence_sd")) * spread

    found = []
    for peak in windows.find(depth, prominence=prominence, min_value=float("-inf")):
        found.append(Boundary(
            t=times[peak.index], source="embedding",
            # In standard deviations, so it is comparable across streams even
            # though the raw distance is not.
            strength=(peak.value - peak.base) / spread,
            detail=f"depth {peak.value:.3f} vs mean {depth.mean():.3f}",
        ))
    return found, ""


def _centroid(block: np.ndarray) -> np.ndarray:
    """Mean of unit vectors, renormalised. Zero-length stays zero rather than NaN."""
    mean = block.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    return mean if norm == 0.0 else mean / norm


# --------------------------------------------------------------------------
# input 4: scene changes — §9.3's tie-breaker, §16's rejected scorer
# --------------------------------------------------------------------------


def scene_boundaries(
    conn: sqlite3.Connection, stream_id: str, cfg
) -> tuple[list[Boundary], str]:
    """The start of each scene span (commit 39 stores spans, so this is free).

    §9.3 calls this a "weak signal, tie-breaker only" and §16 rejects it as a
    scorer outright, so it never proposes a boundary alone — see `merge`.
    """
    if not bool(cfg.get("digest.chapters.scene_changes", True)):
        return [], "digest.chapters.scene_changes is false"

    rows = conn.execute(
        "SELECT t, meta FROM events WHERE stream_id = ? AND source = ? ORDER BY t",
        (stream_id, SCENE_SOURCE),
    ).fetchall()
    if not rows:
        return [], ("no scene events — `scene_events` needs an OBS log attached "
                    "with `register --obs-log`, and its parser has never been "
                    "checked against a real one")

    found = []
    for row in rows:
        if float(row["t"]) <= 0.0:
            continue        # the scene the recording opened in is not a change
        try:
            name = json.loads(row["meta"] or "{}").get("scene") or "?"
        except json.JSONDecodeError:
            name = "?"
        found.append(Boundary(t=float(row["t"]), source="scene", strength=1.0,
                              detail=f"switched to {name!r}"))
    return found, ""


# --------------------------------------------------------------------------
# merging and the target range
# --------------------------------------------------------------------------


#: Which input's timestamp wins when several land in one cluster. NOT a
#: preference: `silence` is the only one of §9.3's four validated on real data
#: and its time is the most precise (talking demonstrably resumed), `embedding`
#: is MEASURED weak at 1.2-1.3x the mean and localised only to within a window
#: of segments, and §9.3 itself calls `scene` a "weak signal, tie-breaker only".
PRIORITY = ("silence", "embedding", "scene")


def merge(boundaries: list[Boundary], within_s: float) -> list[Boundary]:
    """§9.3's "Merge boundaries within 120 s", keeping what corroborated what.

    **The survivor is the most trustworthy source in the cluster, not the
    earliest.** The first cut of this took the earliest, on the reasoning that
    starting a chapter at the first evidence loses no content — and the speech
    fixture immediately showed why that is wrong: a 1.21-sd embedding bump at
    29.4 s displaced a NINETEEN-SECOND silence at 52.0 s. The silence is the only
    input here validated against real data, and its timestamp means something
    exact (speech resumed); the embedding peak is a weak statistic localised to a
    window of segments. Deferring to it would have put the boundary 23 seconds
    early, in the middle of a sentence.

    Within one source the strongest wins, then the earliest, so the result never
    depends on row order.

    A cluster of nothing but scene changes is DROPPED. §9.3 calls scene changes
    a tie-breaker and §16 rejects them as a scorer; a tie-breaker that proposes
    boundaries on its own is just a weak detector, and this one is running on a
    parser that has never seen a real OBS log.
    """
    if not boundaries:
        return []

    ordered = sorted(boundaries, key=lambda b: (b.t, b.source))
    clusters: list[list[Boundary]] = [[ordered[0]]]
    for boundary in ordered[1:]:
        if boundary.t - clusters[-1][0].t <= within_s:
            clusters[-1].append(boundary)
        else:
            clusters.append([boundary])

    def rank(boundary: Boundary) -> tuple:
        order = (PRIORITY.index(boundary.source)
                 if boundary.source in PRIORITY else len(PRIORITY))
        return (order, -boundary.strength, boundary.t)

    survivors = []
    for cluster in clusters:
        sources = {b.source for b in cluster}
        if sources == {"scene"}:
            continue
        lead = min((b for b in cluster if b.source != "scene"), key=rank)
        lead.corroborated_by = sorted(sources - {lead.source})
        survivors.append(lead)
    return survivors


def build(duration_s: float, accepted: list[Boundary]) -> list[Chapter]:
    """Boundaries into a tiling of `[0, duration_s]`."""
    edges = [0.0] + [b.t for b in accepted] + [float(duration_s)]
    openers: list[Boundary | None] = [None] + list(accepted)
    chapters = []
    for index, (start, end) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        chapters.append(Chapter(index=index, t_start=start, t_end=end,
                                opened_by=openers[index]))
    return chapters


def apply_targets(
    chapters: list[Chapter], spare: list[Boundary], cfg
) -> tuple[list[Chapter], list[str]]:
    """§9.3's "target 10-30 minutes", as guidance rather than a constraint.

    Over-long chapters split at the strongest boundary that was found inside
    them but did not clear the merge — a real observation that lost a
    tie-break, never an invented midpoint. When there is no such candidate the
    chapter stays long and the shortfall is reported: a fabricated boundary in
    the middle of a forty-minute stretch would be a lie that §9.4 then hands to
    a model as a topic change.

    Under-short chapters merge into the previous one (the next, for the first),
    which cannot fabricate anything — it only removes a boundary.
    """
    notes: list[str] = []
    minimum = float(cfg.get("digest.chapters.target_min_s"))
    maximum = float(cfg.get("digest.chapters.target_max_s"))

    accepted = [c.opened_by for c in chapters if c.opened_by is not None]
    changed = True
    guard = 0
    while changed and guard < 64:
        changed, guard = False, guard + 1

        for chapter in build(chapters[-1].t_end, accepted):
            if chapter.duration_s <= maximum:
                continue
            inside = [b for b in spare
                      if chapter.t_start < b.t < chapter.t_end and b not in accepted]
            if not inside:
                continue
            best = max(inside, key=lambda b: b.strength)
            best.accepted = True
            accepted = sorted(accepted + [best], key=lambda b: b.t)
            changed = True
            break

    rebuilt = build(chapters[-1].t_end, accepted)
    kept = [b for b in accepted]
    for chapter in rebuilt:
        if chapter.duration_s < minimum and chapter.opened_by in kept:
            kept.remove(chapter.opened_by)
    rebuilt = build(rebuilt[-1].t_end, kept)

    for chapter in rebuilt:
        if chapter.duration_s > maximum:
            notes.append(
                f"chapter {chapter.index} is {chapter.duration_s / 60:.0f} min, over "
                f"the {maximum / 60:.0f} min target, and no boundary was found "
                f"inside it to split on")
    if len(rebuilt) == 1 and rebuilt[0].duration_s < minimum:
        notes.append(
            f"the whole stream is {rebuilt[0].duration_s / 60:.1f} min, under the "
            f"{minimum / 60:.0f} min target — one chapter is the honest answer")
    return rebuilt, notes


# --------------------------------------------------------------------------
# the whole thing
# --------------------------------------------------------------------------


def segment(conn: sqlite3.Connection, stream_id: str, cfg,
            duration_s: float | None = None) -> Result:
    """§9.3 end to end. Pure: reads the database, touches no model, writes nothing."""
    if duration_s is None:
        row = conn.execute("SELECT duration_s FROM streams WHERE id = ?",
                           (stream_id,)).fetchone()
        if row is None:
            raise ChapterError(f"no stream {stream_id!r}")
        duration_s = float(row["duration_s"] or 0.0)
    if duration_s <= 0:
        raise ChapterError(f"{stream_id} has no duration; it has not been probed")

    result = Result()
    proposed: list[Boundary] = []

    for name, produce in (
        ("silence", lambda: silence_boundaries(conn, stream_id, cfg, duration_s)),
        ("embedding", lambda: embedding_boundaries(conn, stream_id, cfg)),
        ("scene", lambda: scene_boundaries(conn, stream_id, cfg)),
    ):
        found, reason = produce()
        if reason:
            result.inputs[name] = reason
        elif not found:
            result.inputs[name] = "no boundaries met the threshold"
        proposed.extend(found)

    # §9.3's second input has no producer anywhere in the system, and inventing
    # one would be building ahead of the data (C5).
    result.inputs["game"] = (
        "no timed game data exists — `streams.games` is an untimed list, and "
        "per-moment game identification is §5.9's vision (Phase 7)")

    inside = [b for b in proposed if 0.0 < b.t < duration_s]
    survivors = merge(inside, float(cfg.get("digest.chapters.merge_within_s")))
    for boundary in inside:
        boundary.accepted = boundary in survivors

    chapters = build(duration_s, sorted(survivors, key=lambda b: b.t))
    chapters, notes = apply_targets(chapters, inside, cfg)

    result.chapters = chapters
    result.boundaries = sorted(inside, key=lambda b: b.t)
    result.unmet_targets = notes
    _assert_tiles(chapters, duration_s)
    return result


def _assert_tiles(chapters: list[Chapter], duration_s: float) -> None:
    """The one invariant that would be invisible downstream if it broke.

    §9.2's structure is a partition and §9.4 chunks over it, so a gap between
    two chapters silently drops that transcript from the digest — no error, just
    a stream the model was never shown part of.
    """
    if not chapters:
        raise ChapterError("segmentation produced no chapters")
    if abs(chapters[0].t_start) > 1e-6:
        raise ChapterError(f"first chapter starts at {chapters[0].t_start}, not 0")
    if abs(chapters[-1].t_end - duration_s) > 1e-6:
        raise ChapterError(
            f"last chapter ends at {chapters[-1].t_end}, not {duration_s}")
    for before, after in zip(chapters, chapters[1:], strict=False):
        if abs(before.t_end - after.t_start) > 1e-6:
            raise ChapterError(
                f"chapters {before.index} and {after.index} do not meet: "
                f"{before.t_end} then {after.t_start}")
        if after.duration_s <= 0:
            raise ChapterError(f"chapter {after.index} has no duration")
