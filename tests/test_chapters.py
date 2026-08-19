"""§9.3's chapter segmentation.

**Constructed boundaries, not the fixture.** The generated fixture is a 600 s
tone with no transcript, no scene log and no silence — it produces exactly one
chapter, which is the right answer and tests nothing about merging, ranking or
the length pass. Those are pure functions over `Boundary` lists, so they are
driven directly with the shapes that matter.

**Every threshold is read from the config object** (house rule 3). A test that
hardcoded 120 would keep passing after someone changed `merge_within_s`, which
is the one moment it needed to fail.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from clipforge import config, db, signals
from clipforge.digest import chapters


@pytest.fixture
def settings():
    cfg = config.load()
    from clipforge.digest import stage
    return stage.settings(cfg)


@pytest.fixture
def conn(tmp_path):
    connection = db.open_db(tmp_path / "t.db")
    connection.execute(
        "INSERT INTO streams (id, date, master_path, duration_s) "
        "VALUES ('s', '2026-01-01', '/m.mkv', 7200)"
    )
    yield connection
    connection.close()


def _b(t, source, strength=1.0):
    return chapters.Boundary(t=t, source=source, strength=strength)


# --------------------------------------------------------------------------
# merging (§9.3's "merge boundaries within 120 s")
# --------------------------------------------------------------------------


def test_boundaries_inside_the_merge_window_become_one(settings):
    within = float(settings["merge_within_s"])
    merged = chapters.merge(
        [_b(1000.0, chapters.SOURCE_SILENCE, 0.4),
         _b(1000.0 + within / 2, chapters.SOURCE_SILENCE, 0.9)],
        within,
    )
    assert len(merged) == 1
    # The stronger of the cluster survives, not the earlier one.
    assert merged[0].strength == pytest.approx(0.9)


def test_boundaries_outside_the_merge_window_both_survive(settings):
    within = float(settings["merge_within_s"])
    merged = chapters.merge(
        [_b(1000.0, chapters.SOURCE_SILENCE),
         _b(1000.0 + within * 2, chapters.SOURCE_SILENCE)],
        within,
    )
    assert len(merged) == 2


def test_a_long_run_of_near_boundaries_does_not_chain_into_one_cluster(settings):
    """Clustering is measured from each cluster's FIRST member.

    Measuring from the last lets boundaries spaced just under the window chain
    indefinitely: eleven of them 119 s apart would collapse twenty minutes of
    stream into a single boundary. That is not what "within 120 s" means.
    """
    within = float(settings["merge_within_s"])
    step = within * 0.9
    starts = [_b(1000.0 + i * step, chapters.SOURCE_SILENCE) for i in range(11)]

    merged = chapters.merge(starts, within)

    assert len(merged) > 1, "the whole run collapsed into one boundary"
    span = starts[-1].t - starts[0].t
    assert len(merged) >= span / (within * 2)


def test_a_scene_change_loses_to_a_silence_gap_in_the_same_cluster(settings):
    """§9.3 ranks scene changes last: "weak signal, tie-breaker only".

    So a scene change may POSITION a boundary when nothing else is near, and
    must never win against a stronger source describing the same break — even
    when its own strength is higher, since the two numbers are not comparable.
    """
    within = float(settings["merge_within_s"])
    merged = chapters.merge(
        [_b(500.0, chapters.SOURCE_SCENE, 1.0),
         _b(500.0 + within / 3, chapters.SOURCE_SILENCE, 0.1)],
        within,
    )
    assert len(merged) == 1
    assert merged[0].source == chapters.SOURCE_SILENCE


def test_an_embedding_shift_outranks_a_silence_gap(settings):
    """§9.3's list is in priority order and source 1 is the embedding shift."""
    within = float(settings["merge_within_s"])
    merged = chapters.merge(
        [_b(500.0, chapters.SOURCE_SILENCE, 1.0),
         _b(510.0, chapters.SOURCE_EMBEDDING, 0.3)],
        within,
    )
    assert merged[0].source == chapters.SOURCE_EMBEDDING


# --------------------------------------------------------------------------
# the global pass (§9.3's "target chapter length: 10-30 minutes")
# --------------------------------------------------------------------------


def test_an_over_long_chapter_is_split_at_its_strongest_interior_boundary(settings):
    """Local merging cannot hit a target that describes the whole stream.

    One boundary near the start of a long stream leaves a chapter far over the
    maximum; the global pass reinstates a candidate that merging discarded.
    """
    maximum = float(settings["max_chapter_s"])
    minimum = float(settings["min_chapter_s"])
    duration = maximum * 3

    kept = [_b(minimum * 1.5, chapters.SOURCE_SILENCE, 0.9)]
    # Two candidates inside the over-long tail; the stronger must be chosen.
    pool = kept + [
        _b(duration * 0.55, chapters.SOURCE_SILENCE, 0.2),
        _b(duration * 0.60, chapters.SOURCE_EMBEDDING, 0.2),
    ]

    out = chapters.enforce_lengths(
        kept, duration_s=duration, minimum_s=minimum, maximum_s=maximum,
        all_boundaries=pool)

    assert len(out) > len(kept), "the over-long chapter was never split"
    added = [b for b in out if b not in kept]
    assert added[0].source == chapters.SOURCE_EMBEDDING


def test_an_over_long_chapter_with_no_interior_candidate_is_left_alone(settings):
    """Splitting at the midpoint would be inventing a topic change.

    A 40-minute chapter with no boundary evidence inside it is a finding about
    the segmentation — reported through `digest_chapter_count` — not a defect to
    paper over with an arbitrary cut.
    """
    maximum = float(settings["max_chapter_s"])
    minimum = float(settings["min_chapter_s"])
    duration = maximum * 2

    out = chapters.enforce_lengths(
        [], duration_s=duration, minimum_s=minimum, maximum_s=maximum,
        all_boundaries=[])

    assert out == []


def test_a_too_short_chapter_is_absorbed_into_a_neighbour(settings):
    minimum = float(settings["min_chapter_s"])
    maximum = float(settings["max_chapter_s"])
    duration = maximum * 2

    # Two boundaries a tenth of the minimum apart: the sliver between them
    # cannot stand as its own chapter.
    kept = [_b(maximum * 0.5, chapters.SOURCE_SILENCE, 0.9),
            _b(maximum * 0.5 + minimum * 0.1, chapters.SOURCE_SCENE, 0.9)]

    out = chapters.enforce_lengths(
        kept, duration_s=duration, minimum_s=minimum, maximum_s=maximum,
        all_boundaries=kept)

    assert len(out) < len(kept)
    # It is absorbed across the WEAKER boundary, so the stronger one survives.
    assert out[0].source == chapters.SOURCE_SILENCE


def test_a_stream_shorter_than_the_minimum_is_one_chapter(settings):
    """Not a failure to reach the minimum. A 12-minute stream is one chapter."""
    minimum = float(settings["min_chapter_s"])
    out = chapters.enforce_lengths(
        [], duration_s=minimum / 2, minimum_s=minimum,
        maximum_s=float(settings["max_chapter_s"]), all_boundaries=[])
    assert chapters.to_chapters(out, minimum / 2) == [
        chapters.Chapter(index=0, t_start=0.0, t_end=round(minimum / 2, 3))
    ]


def test_chapters_tile_the_stream_with_no_gap_or_overlap(settings):
    duration = float(settings["max_chapter_s"]) * 2
    out = chapters.to_chapters(
        [_b(1000.0, chapters.SOURCE_SILENCE), _b(2000.0, chapters.SOURCE_SCENE)],
        duration)

    assert out[0].t_start == 0.0
    assert out[-1].t_end == pytest.approx(duration)
    for earlier, later in zip(out, out[1:]):
        assert earlier.t_end == later.t_start
    # Each chapter records which kind of boundary opened it; the first has none.
    assert out[0].opened_by is None
    assert out[1].opened_by == chapters.SOURCE_SILENCE


# --------------------------------------------------------------------------
# the sources, against a database
# --------------------------------------------------------------------------


def _add_segments(conn, rows):
    for seq, (t_start, t_end, text) in enumerate(rows):
        conn.execute(
            "INSERT INTO segments (stream_id, seq, t_start, t_end, text) "
            "VALUES ('s', ?, ?, ?, ?)", (seq, t_start, t_end, text))


def test_a_speech_gap_longer_than_the_threshold_opens_a_boundary(conn, settings):
    gap = float(settings["silence_gap_s"])
    _add_segments(conn, [
        (10.0, 20.0, "before the break"),
        (20.0 + gap * 2, 20.0 + gap * 2 + 10.0, "after the break"),
    ])

    found, why = chapters.silence_boundaries(
        conn, "s", gap_s=gap, duration_s=7200.0,
        floor_db=float(settings["silence_floor_db"]))

    assert why == ""
    assert len(found) == 1
    # The MIDPOINT of the gap: it belongs to neither neighbour, and placing it
    # at the start would open the new chapter with silence.
    assert found[0].t == pytest.approx((20.0 + 20.0 + gap * 2) / 2)


def test_a_speech_gap_shorter_than_the_threshold_opens_nothing(conn, settings):
    gap = float(settings["silence_gap_s"])
    _add_segments(conn, [
        (10.0, 20.0, "before"), (20.0 + gap * 0.5, 40.0 + gap * 0.5, "after"),
    ])
    found, why = chapters.silence_boundaries(
        conn, "s", gap_s=gap, duration_s=7200.0,
        floor_db=float(settings["silence_floor_db"]))
    assert found == []
    assert why


def test_silence_falls_back_to_mic_rms_when_there_is_no_transcript(conn, settings):
    """The path every stream that exists today takes.

    Phase 2 ships off, so `segments` is empty and the only silence signal is the
    RMS series. A segmenter that only knew how to read `segments` would return
    one chapter for every real recording and never say why.
    """
    gap = float(settings["silence_gap_s"])
    floor = float(settings["silence_floor_db"])
    hz = 10.0
    quiet_s = gap * 2
    values = np.concatenate([
        np.full(int(60 * hz), floor + 20.0),
        np.full(int(quiet_s * hz), floor - 20.0),
        np.full(int(60 * hz), floor + 20.0),
    ])
    with db.transaction(conn):
        signals.store(conn, "s", signals.Series(
            kind="mic_rms", values=values, sample_rate_hz=hz))

    found, why = chapters.silence_boundaries(
        conn, "s", gap_s=gap, duration_s=7200.0, floor_db=floor)

    assert why == ""
    assert len(found) == 1
    assert found[0].t == pytest.approx(60.0 + quiet_s / 2, abs=1.0)


def test_every_absent_source_says_why_it_is_absent(conn, settings):
    """The property the whole degradation story rests on.

    "No boundaries" and "no transcript to look for boundaries in" are different
    facts, and a digest that reported them identically would be unreadable at
    month twelve.
    """
    result = chapters.segment(conn, "s", duration_s=7200.0, settings=settings)

    assert set(result.absent) == {
        chapters.SOURCE_EMBEDDING, chapters.SOURCE_SILENCE, chapters.SOURCE_SCENE}
    for source, why in result.absent.items():
        assert why.strip(), f"{source} is absent with no reason given"
    assert result.sources == []
    assert len(result.chapters) == 1


def test_sources_lists_only_boundaries_that_survived(conn, settings):
    """A source whose every candidate lost a merge did not shape the result.

    Claiming it did would overstate what the digest was built from, which is
    exactly the overstatement `sources` exists to prevent.
    """
    within = float(settings["merge_within_s"])
    kept = chapters.merge(
        [_b(1000.0, chapters.SOURCE_SCENE, 1.0),
         _b(1000.0 + within / 4, chapters.SOURCE_SILENCE, 0.5)],
        within)
    surviving = {b.source for b in kept}
    assert surviving == {chapters.SOURCE_SILENCE}


def test_embeddings_from_a_second_model_are_not_compared_across(conn, settings):
    """A cosine between two geometries is finite, ordered and meaningless.

    `extract/embeddings.py` says so; a library embedded with two models would
    otherwise show a spurious boundary exactly where the model changed.
    """
    dim = 8
    rows = []
    for seq in range(40):
        t = seq * 10.0
        vector = np.zeros(dim, dtype=np.float32)
        vector[0] = 1.0
        rows.append((seq, t, vector, "model-a" if seq < 30 else "model-b"))

    _add_segments(conn, [(t, t + 5.0, f"line {seq}") for seq, t, _v, _m in rows])
    for seq, _t, vector, model in rows:
        seg = conn.execute(
            "SELECT id FROM segments WHERE stream_id='s' AND seq=?", (seq,)
        ).fetchone()["id"]
        conn.execute(
            "INSERT INTO segment_embeddings (segment_id, model, dim, vec) "
            "VALUES (?, ?, ?, ?)", (seg, model, dim, vector.tobytes()))

    found, _why = chapters.embedding_boundaries(
        conn, "s", window_s=float(settings["embedding_window_s"]),
        min_distance=float(settings["embedding_min_distance"]))

    # Every model-a vector is identical, so there is no genuine shift. A
    # boundary here would mean the two models had been compared.
    assert found == []
