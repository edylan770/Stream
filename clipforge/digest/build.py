"""§9.2's digest, minus the parts only a model can write.

Roughly sixty per cent of §9.2's structure is arithmetic over rows this database
already holds: the emotional arc, the recurring phrases, the top candidates and
their quotes, the chapter boundaries and each chapter's mean energy. Only the
chapter titles and summaries, `themes_observed`, `open_loops` and the label on a
top candidate need a model at all.

Building that half on its own, first, is what makes the digest layer produce
something on a machine with no API key and no Ollama — which is this machine, and
§9.1 says the corpus is the compounding asset, so the sooner it starts
accumulating the better. It is also what makes the layer testable: every number
below can be checked against a fixture, and none of it depends on what a model
happened to say.

**dB are logarithms.** `mic_rms` and friends are stored in dBFS, and the arc
averages them. Averaging decibels is averaging exponents — it under-weights the
loud moments, which are precisely the ones an "energy arc" exists to show. So
every mean here runs in linear power and converts back only at the end. This has
caused two real bugs in this project already.

**The digest is a row, not a cache.** §9.1: *"first-class rows, never regenerable
cache. Keep every version forever."* `write` therefore only ever INSERTs, at
`max(version) + 1`, and nothing in this module updates a digest in place.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field

import numpy as np

from clipforge import signals
from clipforge.digest import chapters as chapters_mod
from clipforge.extract import phrases as phrases_mod

#: §9.2's own key order, so a hand-read digest and the spec line up.
STRUCTURE_KEYS = (
    "stream_id", "date", "games", "duration_s", "chapters", "recurring_phrases",
    "emotional_arc", "open_loops", "top_candidates", "themes_observed",
)

#: What a model fills in later. Present and empty in v1 rather than absent, so
#: the shape of a digest does not depend on whether it has been enriched --
#: anything reading the corpus can treat every version the same way.
MODEL_AUTHORED = ("title", "summary", "notable_segment_ids")


@dataclass
class Digest:
    """One digest, in memory."""

    content: dict
    markdown: str
    #: Everything worth putting in `tool_metrics`. Not stored in the row.
    metrics: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# energy, in linear power
# --------------------------------------------------------------------------


def energy_series(conn: sqlite3.Connection, stream_id: str, *, roles: list[str],
                  floor_db: float) -> signals.Series | None:
    """Combined loudness across the configured roles, as a linear power series.

    Summed in LINEAR POWER, which is the only meaning "how loud was it" has
    across three tracks: two tracks at −20 dB are louder together than either
    alone, and adding the dB values would say −40.

    The floor is applied before conversion, not after. `extract/features.py`
    writes a floor value where a track was silent, and 10**(−90/10) is not zero
    but it is close enough that letting it through would make a silent stream's
    arc pure floating-point noise scaled to fill the range.
    """
    tracks = []
    for role in roles:
        series = signals.load(conn, stream_id, f"{role}_rms")
        if series is None or len(series) == 0:
            continue
        values = series.values.astype(np.float64)
        values = np.where(np.isfinite(values), values, floor_db)
        values = np.maximum(values, floor_db)
        tracks.append((series, 10.0 ** (values / 10.0)))

    if not tracks:
        return None

    # Resampled onto the longest track's grid rather than assuming they match.
    # They do today -- one hop for every role -- but a re-extraction at a
    # different hop would otherwise sum misaligned arrays silently.
    base = max(tracks, key=lambda pair: len(pair[0]))[0]
    grid = base.times()
    total = np.zeros(grid.size, dtype=np.float64)
    for series, power in tracks:
        total += np.interp(grid, series.times(), power,
                           left=power[0], right=power[-1])

    return signals.Series(kind="digest_energy", values=total,
                          sample_rate_hz=base.sample_rate_hz, t0=base.t0)


def normalise_energy(power: np.ndarray) -> np.ndarray:
    """Linear power to a 0-1 arc, via dB.

    §9.2's `energy` is unitless and unexplained. Rendering the SHAPE of the
    session means dB: doubling the power is one step, whether that is from quiet
    to less quiet or from loud to very loud, and a linear-power arc would be a
    flat line with four spikes on it. So this converts back to dB and scales
    that against the stream's own observed range -- which also makes the number
    comparable across streams recorded at different gains, and every stream so
    far was recorded at a different gain.
    """
    if power.size == 0:
        return power
    with np.errstate(divide="ignore"):
        db = 10.0 * np.log10(np.maximum(power, 1e-30))
    lo, hi = float(db.min()), float(db.max())
    if hi - lo < 1e-6:
        return np.zeros_like(db)
    return (db - lo) / (hi - lo)


def emotional_arc(conn: sqlite3.Connection, stream_id: str, *, duration_s: float,
                  bin_s: float, roles: list[str], laughter_kinds: list[str],
                  laughter_threshold: float, floor_db: float) -> list[dict]:
    """§9.2's `[{t_bin, energy, laughter_density}]`.

    `laughter_density` is the FRACTION of the bin above the laughter threshold,
    which is what "density" has to mean for a detector whose output is a
    continuous score: the share of the bin that was laughter. A mean of the
    score would be a different quantity wearing the same name.
    """
    energy = energy_series(conn, stream_id, roles=roles, floor_db=floor_db)
    laughter = [
        series for series in
        (signals.load(conn, stream_id, kind) for kind in laughter_kinds)
        if series is not None and len(series)
    ]
    if energy is None and not laughter:
        return []

    normalised = normalise_energy(energy.values.astype(np.float64)) if energy else None

    arc: list[dict] = []
    edges = np.arange(0.0, max(duration_s, bin_s), bin_s)
    for start in edges:
        end = start + bin_s
        entry: dict = {"t_bin": round(float(start), 1)}

        if energy is not None and normalised is not None:
            times = energy.times()
            mask = (times >= start) & (times < end)
            entry["energy"] = (round(float(normalised[mask].mean()), 4)
                               if mask.any() else None)
        else:
            entry["energy"] = None

        if laughter:
            shares = []
            for series in laughter:
                window = series.slice(start, end)
                finite = window[np.isfinite(window)]
                if finite.size:
                    shares.append(float((finite > laughter_threshold).mean()))
            entry["laughter_density"] = round(max(shares), 4) if shares else None
        else:
            entry["laughter_density"] = None

        arc.append(entry)
    return arc


def chapter_energy(arc: list[dict], chapter: chapters_mod.Chapter,
                   bin_s: float) -> float | None:
    """§9.2's `chapters[].energy_mean`, from the arc that is already built.

    From the arc rather than re-reading the series: two numbers claiming to be
    "the energy of this chapter" that were computed by different code will
    eventually disagree, and the one on screen would be the one nobody checked.
    """
    values = [entry["energy"] for entry in arc
              if entry["energy"] is not None
              and chapter.t_start <= entry["t_bin"] < chapter.t_end]
    if not values:
        return None
    return round(float(np.mean(values)), 4)


# --------------------------------------------------------------------------
# recurring phrases
# --------------------------------------------------------------------------


def recurring_phrases(conn: sqlite3.Connection, stream_id: str, cfg, *,
                      minimum: int, low: int, high: int,
                      limit: int) -> list[dict]:
    """§9.2's `[{phrase, count, segment_ids}]`.

    A third thing from the two phrase counters this project already has, and
    deliberately so: `phrases.find_repeats` fires on three occurrences inside a
    90 s window (§5.4.2's *bit in progress*), and §11.2's `ngrams` table counts
    across the whole library (a *catchphrase*). This is one stream, end to end --
    what got said repeatedly THIS session. The tokeniser, the normaliser and the
    filler list are shared with both, so the three cannot disagree about what a
    phrase is.

    `segment_ids` are `segments.seq`, per §12.1: those are the ids a model is
    ever shown, and a digest whose ids meant something else would be a trap for
    every later pass.
    """
    rows = conn.execute(
        "SELECT seq, text FROM segments WHERE stream_id = ? AND TRIM(text) != '' "
        "ORDER BY seq",
        (stream_id,),
    ).fetchall()
    if not rows:
        return []

    phrases = phrases_mod.load_phrases(cfg)
    seen: dict[str, list[int]] = {}
    for row in rows:
        # `set(...)` so a phrase said twice in one segment counts that segment
        # once. The unit of `count` is occurrences-in-distinct-segments, which
        # is what makes it a recurrence rather than a stutter.
        for phrase in set(phrases_mod.ngrams(row["text"] or "", low, high)):
            if phrases_mod.is_filler(phrase, phrases):
                continue
            seen.setdefault(phrase, []).append(int(row["seq"]))

    found = [
        {"phrase": phrase, "count": len(seqs), "segment_ids": seqs}
        for phrase, seqs in seen.items() if len(seqs) >= minimum
    ]
    found = _drop_subsumed(found)
    found.sort(key=lambda entry: (-entry["count"], entry["phrase"]))
    return found[:limit]


def _drop_subsumed(found: list[dict]) -> list[dict]:
    """Keep the longest phrase of any run that always occurs together.

    "the tier list" and "tier list" with identical segment ids are one bit, and
    listing both spends two of §9.2's slots saying it once. The same maximality
    rule `phrases.maximal` applies to §5.4.2's events, restated here because the
    unit differs -- segments rather than a sliding window.
    """
    by_phrase = {entry["phrase"]: entry for entry in found}
    out = []
    for entry in found:
        longer = [
            other for other in by_phrase.values()
            if other["phrase"] != entry["phrase"]
            and entry["phrase"] in other["phrase"]
            and other["segment_ids"] == entry["segment_ids"]
        ]
        if not longer:
            out.append(entry)
    return out


# --------------------------------------------------------------------------
# top candidates
# --------------------------------------------------------------------------


def top_candidates(conn: sqlite3.Connection, stream_id: str, *,
                   limit: int) -> list[dict]:
    """§9.2's `[{candidate_id, score, label, quote}]`.

    Rated "clip it" first, then everything else by score. The operator's verdict
    beats the detector's: a moment they approved is a better answer to "what was
    good about this stream" than a high composite nobody has looked at.

    **Ratings are read across scoring generations, never `is_current`** -- the
    standing invariant in CLAUDE.md, and the reason is the same here as
    everywhere: a re-score must never drop a moment the operator approved. But
    the CANDIDATES are current-generation only, since an old generation's windows
    were computed under different weights and would double up on the same moment.
    A rating that carried forward is on the current row already
    (`ratings.rating_source = 'inherited'`).

    `quote` is filled deterministically, from the segment overlapping the window
    most. §12.3 wants a verbatim quote on every selection so fabrication is
    machine-checkable; when the selection itself is deterministic there is
    nothing to fabricate, and taking the quote from the database rather than from
    a model removes the check by removing the risk. `label` is what is left for
    the model.
    """
    rows = conn.execute(
        """
        SELECT c.id, c.t_start, c.t_end, c.t_peak, c.score_combined, r.rating
          FROM candidates c
          LEFT JOIN ratings r ON r.candidate_id = c.id
         WHERE c.stream_id = ? AND c.is_current = 1
         ORDER BY (r.rating = 2) DESC, c.score_combined DESC
         LIMIT ?
        """,
        (stream_id, limit),
    ).fetchall()

    out = []
    for row in rows:
        out.append({
            "candidate_id": int(row["id"]),
            "score": round(float(row["score_combined"]), 4),
            "t_start": round(float(row["t_start"]), 3),
            "t_end": round(float(row["t_end"]), 3),
            "rating": None if row["rating"] is None else int(row["rating"]),
            "label": None,
            "quote": _best_quote(conn, stream_id, float(row["t_start"]),
                                 float(row["t_end"])),
        })
    return out


def _best_quote(conn: sqlite3.Connection, stream_id: str, t_start: float,
                t_end: float) -> str | None:
    """The segment overlapping this window most, verbatim."""
    rows = conn.execute(
        "SELECT text, t_start, t_end FROM segments WHERE stream_id = ? "
        "AND t_end >= ? AND t_start <= ? AND TRIM(text) != ''",
        (stream_id, t_start, t_end),
    ).fetchall()
    if not rows:
        return None
    best = max(rows, key=lambda r: (min(t_end, float(r["t_end"]))
                                    - max(t_start, float(r["t_start"]))))
    return str(best["text"]).strip() or None


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------


def build(conn: sqlite3.Connection, stream_id: str, cfg, *,
          settings: dict) -> Digest:
    """The deterministic digest, in §9.2's shape."""
    stream = conn.execute(
        "SELECT * FROM streams WHERE id = ?", (stream_id,)
    ).fetchone()
    if stream is None:
        raise LookupError(f"no stream {stream_id!r}")
    duration_s = float(stream["duration_s"] or 0.0)

    segmentation = chapters_mod.segment(
        conn, stream_id, duration_s=duration_s, settings=settings)

    bin_s = float(settings["arc_bin_s"])
    arc = emotional_arc(
        conn, stream_id,
        duration_s=duration_s,
        bin_s=bin_s,
        roles=list(settings["energy_roles"]),
        laughter_kinds=list(settings["laughter_kinds"]),
        laughter_threshold=float(settings["laughter_threshold"]),
        floor_db=float(settings["silence_floor_db"]),
    )

    chapter_rows = []
    for chapter in segmentation.chapters:
        seqs = _seq_range(conn, stream_id, chapter.t_start, chapter.t_end)
        chapter_rows.append({
            "index": chapter.index,
            "t_start": chapter.t_start,
            "t_end": chapter.t_end,
            "title": None,
            "summary": None,
            # §9.2 has a per-chapter `game`. There is no timed game source (see
            # chapters.py), so this carries the stream's games rather than
            # claiming to know which one was on at this point.
            "game": None,
            "energy_mean": chapter_energy(arc, chapter, bin_s),
            "notable_segment_ids": [],
            # NOT IN §9.2. The segment range is what scopes a model's ids to this
            # chapter, and §12.2's validation cannot be done without it.
            "seq_start": seqs[0],
            "seq_end": seqs[1],
            "opened_by": chapter.opened_by,
        })

    content = {
        "stream_id": stream_id,
        "date": stream["date"],
        "games": json.loads(stream["games"]) if stream["games"] else [],
        "duration_s": round(duration_s, 3),
        "chapters": chapter_rows,
        "recurring_phrases": recurring_phrases(
            conn, stream_id, cfg,
            minimum=int(settings["phrase_min_count"]),
            low=int(settings["phrase_min_words"]),
            high=int(settings["phrase_max_words"]),
            limit=int(settings["phrase_limit"]),
        ),
        "emotional_arc": arc,
        "open_loops": [],
        "top_candidates": top_candidates(
            conn, stream_id, limit=int(settings["top_candidates"])),
        "themes_observed": [],
        # NOT IN §9.2, and the most important field on a digest built today.
        # Which boundary signals actually shaped the chapters, and why each
        # absent one was absent. A digest segmented on silence alone is a
        # different artifact from one segmented on embedding shift, and at month
        # twelve nothing else would say which is in your hand.
        "segmentation": {
            "sources": segmentation.sources,
            "absent": segmentation.absent,
        },
    }

    markdown = render_markdown(content)
    return Digest(
        content=content,
        markdown=markdown,
        metrics={
            "digest_chapter_count": float(len(chapter_rows)),
            "digest_word_count": float(len(markdown.split())),
            "digest_boundary_sources": float(len(segmentation.sources)),
            "chapter_lengths_s": [round(v, 1) for v in segmentation.lengths],
            "sources": segmentation.sources,
            "absent": segmentation.absent,
        },
    )


def _seq_range(conn: sqlite3.Connection, stream_id: str, t_start: float,
               t_end: float) -> tuple[int | None, int | None]:
    row = conn.execute(
        "SELECT MIN(seq) AS lo, MAX(seq) AS hi FROM segments "
        "WHERE stream_id = ? AND t_end >= ? AND t_start < ? AND TRIM(text) != ''",
        (stream_id, t_start, t_end),
    ).fetchone()
    if row is None or row["lo"] is None:
        return None, None
    return int(row["lo"]), int(row["hi"])


# --------------------------------------------------------------------------
# the markdown mirror
# --------------------------------------------------------------------------


def _clock(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def render_markdown(content: dict) -> str:
    """`digests.markdown` — the human-readable mirror §3.2 asks for.

    Rendered from the JSON, never assembled alongside it, so the two cannot
    disagree about what the digest says.
    """
    games = ", ".join(content.get("games") or []) or "—"
    lines = [
        f"# {content['stream_id']}",
        "",
        f"{content.get('date') or '—'} · {games} · "
        f"{_clock(content.get('duration_s') or 0)}",
        "",
    ]

    segmentation = content.get("segmentation") or {}
    sources = segmentation.get("sources") or []
    lines += [
        "## Chapters",
        "",
        f"*{len(content.get('chapters') or [])} chapter(s), from: "
        f"{', '.join(sources) if sources else 'no boundary signal at all'}*",
        "",
    ]
    for chapter in content.get("chapters") or []:
        title = chapter.get("title") or "(untitled)"
        lines.append(
            f"### {chapter['index'] + 1}. {title}  "
            f"`{_clock(chapter['t_start'])}–{_clock(chapter['t_end'])}`"
        )
        if chapter.get("energy_mean") is not None:
            lines.append(f"energy {chapter['energy_mean']:.2f}")
        if chapter.get("summary"):
            lines += ["", chapter["summary"]]
        lines.append("")

    phrases = content.get("recurring_phrases") or []
    if phrases:
        lines += ["## Recurring phrases", ""]
        lines += [f"- **{p['phrase']}** ×{p['count']}" for p in phrases]
        lines.append("")

    loops = content.get("open_loops") or []
    if loops:
        lines += ["## Open loops", ""]
        lines += [f"- *({loop.get('kind') or 'loop'})* {loop['text']}"
                  for loop in loops]
        lines.append("")

    top = content.get("top_candidates") or []
    if top:
        lines += ["## Top candidates", ""]
        for entry in top:
            label = entry.get("label") or "(unlabelled)"
            lines.append(f"- `{_clock(entry['t_start'])}` **{label}** "
                         f"· {entry['score']:.2f}")
            if entry.get("quote"):
                lines.append(f"  > {entry['quote']}")
        lines.append("")

    themes = content.get("themes_observed") or []
    if themes:
        lines += ["## Themes", "", ", ".join(themes), ""]

    absent = segmentation.get("absent") or {}
    if absent:
        lines += ["## Segmentation notes", ""]
        lines += [f"- **{source}**: {why}" for source, why in sorted(absent.items())]
        lines.append("")

    return "\n".join(lines)


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------


def next_version(conn: sqlite3.Connection, stream_id: str) -> int:
    row = conn.execute(
        "SELECT MAX(version) AS v FROM digests WHERE stream_id = ?", (stream_id,)
    ).fetchone()
    return int(row["v"] or 0) + 1


def latest(conn: sqlite3.Connection, stream_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM digests WHERE stream_id = ? ORDER BY version DESC LIMIT 1",
        (stream_id,),
    ).fetchone()


def write(conn: sqlite3.Connection, stream_id: str, digest: Digest, *,
          model_used: str | None) -> int:
    """INSERT a new version. Never an UPDATE — §9.1 keeps every version forever.

    Caller owns the transaction.
    """
    version = next_version(conn, stream_id)
    conn.execute(
        "INSERT INTO digests (stream_id, version, content, markdown, model_used) "
        "VALUES (?, ?, ?, ?, ?)",
        (stream_id, version, json.dumps(digest.content, indent=2, sort_keys=False),
         digest.markdown, model_used),
    )
    return version
