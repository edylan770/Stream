"""§9.2's digest — the deterministic half.

Every threshold comes from the config object (house rule 3). The one number
written literally below is `10 * log10`, which is the definition of a decibel
rather than a tunable.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from clipforge import config, db, signals
from clipforge.digest import build
from clipforge.digest import stage as digest_stage
from clipforge.pipeline.context import StageContext


@pytest.fixture
def cfg():
    return config.load()


@pytest.fixture
def settings(cfg):
    return digest_stage.settings(cfg)


@pytest.fixture
def conn(tmp_path):
    connection = db.open_db(tmp_path / "t.db")
    connection.execute(
        "INSERT INTO streams (id, date, title, games, master_path, duration_s) "
        "VALUES ('s', '2026-01-01', 'A stream', '[\"Elden Ring\"]', '/m.mkv', 1200)"
    )
    yield connection
    connection.close()


def _segments(conn, rows):
    for seq, (t_start, t_end, text) in enumerate(rows):
        conn.execute(
            "INSERT INTO segments (stream_id, seq, t_start, t_end, text) "
            "VALUES ('s', ?, ?, ?, ?)", (seq, t_start, t_end, text))


def _candidate(conn, t_start, t_end, score, rating=None):
    cur = conn.execute(
        "INSERT INTO candidates (stream_id, generation, is_current, profile, "
        "t_start, t_end, t_peak, score_entertainment, score_gameplay, "
        "score_combined, feature_vector, feature_schema_version, config_version) "
        "VALUES ('s', 1, 1, 'entertainment', ?, ?, ?, 0, 0, ?, '{}', 1, 'x@y')",
        (t_start, t_end, (t_start + t_end) / 2, score))
    if rating is not None:
        conn.execute("INSERT INTO ratings (candidate_id, rating) VALUES (?, ?)",
                     (cur.lastrowid, rating))
    return cur.lastrowid


# --------------------------------------------------------------------------
# dB are logarithms — the invariant that has caused two real bugs here
# --------------------------------------------------------------------------


def test_energy_is_averaged_in_linear_power_not_in_decibels(conn, settings):
    """CLAUDE.md: "Convert to linear power before averaging or ratioing them."

    Half a stream at -10 dB and half at -50 dB. The mean POWER is
    (10^-1 + 10^-5)/2, which is -13.0 dB — dominated by the loud half, because
    that is what loudness means. Averaging the dB values instead gives -30 dB,
    which is quieter than anything that actually happened and would flatten
    precisely the peaks an energy arc exists to show.
    """
    hz = 10.0
    loud_db, quiet_db = -10.0, -50.0
    values = np.concatenate([np.full(int(300 * hz), loud_db),
                             np.full(int(300 * hz), quiet_db)])
    with db.transaction(conn):
        signals.store(conn, "s", signals.Series(
            kind="mic_rms", values=values, sample_rate_hz=hz))

    series = build.energy_series(
        conn, "s", roles=["mic"], floor_db=float(settings["silence_floor_db"]))

    mean_power = float(series.values.astype(np.float64).mean())
    in_db = 10.0 * np.log10(mean_power)

    honest = 10.0 * np.log10((10.0 ** (loud_db / 10) + 10.0 ** (quiet_db / 10)) / 2)
    naive = (loud_db + quiet_db) / 2

    assert in_db == pytest.approx(honest, abs=0.1)
    assert abs(in_db - naive) > 10.0, "the mean looks like a mean of decibels"


def test_two_tracks_are_summed_in_power_so_together_is_louder_than_either(
        conn, settings):
    """Adding dB would say two tracks at -20 are -40: quieter than silence."""
    hz = 10.0
    level = -20.0
    with db.transaction(conn):
        for kind in ("mic_rms", "party_rms"):
            signals.store(conn, "s", signals.Series(
                kind=kind, values=np.full(int(60 * hz), level), sample_rate_hz=hz))

    both = build.energy_series(conn, "s", roles=["mic", "party"],
                               floor_db=float(settings["silence_floor_db"]))
    one = build.energy_series(conn, "s", roles=["mic"],
                              floor_db=float(settings["silence_floor_db"]))

    assert float(both.values.mean()) > float(one.values.mean())
    assert 10.0 * np.log10(float(both.values.mean())) == pytest.approx(
        level + 10.0 * np.log10(2), abs=0.1)


def test_the_arc_is_binned_at_the_configured_width(conn, settings):
    hz = 10.0
    duration = 1200.0
    bin_s = float(settings["arc_bin_s"])
    with db.transaction(conn):
        signals.store(conn, "s", signals.Series(
            kind="mic_rms", values=np.full(int(duration * hz), -20.0),
            sample_rate_hz=hz))

    arc = build.emotional_arc(
        conn, "s", duration_s=duration, bin_s=bin_s,
        roles=list(settings["energy_roles"]),
        laughter_kinds=list(settings["laughter_kinds"]),
        laughter_threshold=float(settings["laughter_threshold"]),
        floor_db=float(settings["silence_floor_db"]))

    assert len(arc) == int(duration / bin_s)
    assert [entry["t_bin"] for entry in arc[:3]] == [0.0, bin_s, bin_s * 2]


def test_laughter_density_is_the_share_of_the_bin_above_the_threshold(
        conn, settings):
    """"Density", not "mean score" — they are different quantities."""
    hz = 10.0
    bin_s = float(settings["arc_bin_s"])
    threshold = float(settings["laughter_threshold"])
    per_bin = int(bin_s * hz)
    # First bin: a quarter of it is laughter. Second bin: none.
    values = np.concatenate([
        np.full(per_bin // 4, threshold + 0.2),
        np.full(per_bin - per_bin // 4, threshold - 0.2),
        np.full(per_bin, threshold - 0.2),
    ])
    with db.transaction(conn):
        signals.store(conn, "s", signals.Series(
            kind="mic_laughter", values=values, sample_rate_hz=hz))

    arc = build.emotional_arc(
        conn, "s", duration_s=bin_s * 2, bin_s=bin_s,
        roles=list(settings["energy_roles"]),
        laughter_kinds=list(settings["laughter_kinds"]),
        laughter_threshold=threshold,
        floor_db=float(settings["silence_floor_db"]))

    assert arc[0]["laughter_density"] == pytest.approx(0.25, abs=0.02)
    assert arc[1]["laughter_density"] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# recurring phrases
# --------------------------------------------------------------------------


def test_a_phrase_repeated_enough_times_is_recorded_with_its_segment_ids(
        conn, cfg, settings):
    minimum = int(settings["phrase_min_count"])
    _segments(conn, [
        (i * 30.0, i * 30.0 + 5.0, "the tier list again")
        for i in range(minimum)
    ] + [(900.0, 905.0, "something else entirely")])

    found = build.recurring_phrases(
        conn, "s", cfg, minimum=minimum,
        low=int(settings["phrase_min_words"]),
        high=int(settings["phrase_max_words"]),
        limit=int(settings["phrase_limit"]))

    # The MAXIMAL phrase survives, matching `phrases.maximal` for §5.4.2: when
    # "tier list" and "the tier list again" occur on exactly the same segments
    # they are one bit, and the longer one is what was actually said.
    phrases = {entry["phrase"] for entry in found}
    assert "the tier list again" in phrases
    assert "tier list" not in phrases

    entry = next(e for e in found if e["phrase"] == "the tier list again")
    assert entry["count"] == minimum
    # §12.1: the ids a model is ever shown are `segments.seq`.
    assert entry["segment_ids"] == list(range(minimum))


def test_a_phrase_below_the_minimum_is_not_recorded(conn, cfg, settings):
    minimum = int(settings["phrase_min_count"])
    _segments(conn, [(i * 30.0, i * 30.0 + 5.0, "the tier list again")
                     for i in range(minimum - 1)])
    found = build.recurring_phrases(
        conn, "s", cfg, minimum=minimum,
        low=int(settings["phrase_min_words"]),
        high=int(settings["phrase_max_words"]),
        limit=int(settings["phrase_limit"]))
    assert found == []


def test_filler_is_dropped_so_it_cannot_top_the_list(conn, cfg, settings):
    """Otherwise §9.2's recurring_phrases is a list of the operator's stopwords.

    Shares `phrases.is_filler` with §5.4.2 and §11.2, so the three counters
    cannot disagree about what a phrase is.
    """
    minimum = int(settings["phrase_min_count"])
    _segments(conn, [(i * 30.0, i * 30.0 + 5.0, "you know what i mean")
                     for i in range(minimum * 3)])

    found = build.recurring_phrases(
        conn, "s", cfg, minimum=minimum,
        low=int(settings["phrase_min_words"]),
        high=int(settings["phrase_max_words"]),
        limit=int(settings["phrase_limit"]))

    from clipforge.extract import phrases as phrases_mod
    loaded = phrases_mod.load_phrases(cfg)
    for entry in found:
        assert not phrases_mod.is_filler(entry["phrase"], loaded)


def test_a_shorter_phrase_inside_a_longer_one_is_not_listed_twice(
        conn, cfg, settings):
    """"the tier list" and "tier list" on identical segments are one bit."""
    minimum = int(settings["phrase_min_count"])
    _segments(conn, [(i * 30.0, i * 30.0 + 5.0, "ranking the tier list now")
                     for i in range(minimum)])

    found = build.recurring_phrases(
        conn, "s", cfg, minimum=minimum,
        low=int(settings["phrase_min_words"]),
        high=int(settings["phrase_max_words"]),
        limit=int(settings["phrase_limit"]))

    phrases = [entry["phrase"] for entry in found]
    for shorter in phrases:
        longer = [p for p in phrases if p != shorter and shorter in p]
        assert not longer, f"{shorter!r} is subsumed by {longer!r}"


# --------------------------------------------------------------------------
# top candidates
# --------------------------------------------------------------------------


def test_an_approved_moment_outranks_a_higher_scoring_unrated_one(conn, settings):
    """The operator's verdict beats the detector's.

    A moment they rated "clip it" is a better answer to "what was good about
    this stream" than a high composite nobody has looked at.
    """
    _candidate(conn, 100.0, 110.0, score=9.9)              # unrated, highest
    approved = _candidate(conn, 200.0, 210.0, score=0.1, rating=2)

    top = build.top_candidates(conn, "s", limit=int(settings["top_candidates"]))

    assert top[0]["candidate_id"] == approved
    assert top[0]["rating"] == 2


def test_the_quote_is_verbatim_from_the_segment_overlapping_the_window(
        conn, settings):
    """§12.3 wants a machine-checkable quote on every selection.

    Filled from the database rather than by a model: when the selection is
    deterministic there is nothing to fabricate, so the check is removed by
    removing the risk.
    """
    said = "I cannot believe that actually worked"
    _segments(conn, [(0.0, 50.0, "unrelated chatter"),
                     (100.0, 108.0, said)])
    _candidate(conn, 99.0, 110.0, score=1.0)

    top = build.top_candidates(conn, "s", limit=int(settings["top_candidates"]))

    assert top[0]["quote"] == said


def test_an_older_generation_does_not_double_up_the_same_moment(conn, settings):
    """Candidates are current-generation only.

    An old generation's windows were computed under different weights; including
    them would list the same moment twice with two scores.
    """
    _candidate(conn, 100.0, 110.0, score=5.0)
    conn.execute(
        "INSERT INTO candidates (stream_id, generation, is_current, profile, "
        "t_start, t_end, t_peak, score_entertainment, score_gameplay, "
        "score_combined, feature_vector, feature_schema_version, config_version) "
        "VALUES ('s', 0, 0, 'entertainment', 100, 110, 105, 0, 0, 9.9, '{}', 1, 'x@y')")

    top = build.top_candidates(conn, "s", limit=int(settings["top_candidates"]))

    assert len(top) == 1
    assert top[0]["score"] == pytest.approx(5.0)


# --------------------------------------------------------------------------
# the artifact, and how it is stored
# --------------------------------------------------------------------------


def test_the_digest_has_every_key_9_2_asks_for(conn, cfg, settings):
    _candidate(conn, 100.0, 110.0, score=1.0)
    digest = build.build(conn, "s", cfg, settings=settings)
    for key in build.STRUCTURE_KEYS:
        assert key in digest.content, f"§9.2 asks for {key!r}"


def test_the_model_authored_fields_are_present_and_empty(conn, cfg, settings):
    """Present rather than absent, so every version has the same shape.

    Anything reading the corpus can then treat a v1 and a v2 identically.
    """
    _candidate(conn, 100.0, 110.0, score=1.0)
    digest = build.build(conn, "s", cfg, settings=settings)

    assert digest.content["themes_observed"] == []
    assert digest.content["open_loops"] == []
    for chapter in digest.content["chapters"]:
        assert chapter["title"] is None
        assert chapter["summary"] is None
        assert chapter["notable_segment_ids"] == []
    for entry in digest.content["top_candidates"]:
        assert entry["label"] is None


def test_the_digest_records_which_boundary_signals_it_actually_had(
        conn, cfg, settings):
    """The most important field on a digest built today.

    Segmented on silence alone is a different artifact from segmented on
    embedding shift, and nothing else in the system would say which.
    """
    _candidate(conn, 100.0, 110.0, score=1.0)
    digest = build.build(conn, "s", cfg, settings=settings)

    segmentation = digest.content["segmentation"]
    assert segmentation["sources"] == []
    assert set(segmentation["absent"]) == {
        "embedding_shift", "silence_gap", "scene_change"}


def test_writing_a_digest_never_overwrites_an_earlier_version(conn, cfg, settings):
    """§9.1: "first-class rows, never regenerable cache. Keep every version."

    The obvious implementation UPDATEs, and would destroy the corpus one
    re-run at a time.
    """
    _candidate(conn, 100.0, 110.0, score=1.0)
    digest = build.build(conn, "s", cfg, settings=settings)

    with db.transaction(conn):
        first = build.write(conn, "s", digest, model_used=None)
    with db.transaction(conn):
        second = build.write(conn, "s", digest, model_used="manual:test")

    assert (first, second) == (1, 2)
    rows = conn.execute(
        "SELECT version, model_used FROM digests WHERE stream_id='s' "
        "ORDER BY version").fetchall()
    assert [r["version"] for r in rows] == [1, 2]
    assert rows[0]["model_used"] is None
    assert rows[1]["model_used"] == "manual:test"


def test_the_markdown_mirror_is_rendered_from_the_json(conn, cfg, settings):
    """Rendered from the content, never assembled alongside it.

    Two renderings of "what the digest says" that were built separately will
    eventually disagree, and the one on screen is the one nobody checked.
    """
    _segments(conn, [(100.0, 108.0, "a memorable line")])
    _candidate(conn, 99.0, 110.0, score=1.0)
    digest = build.build(conn, "s", cfg, settings=settings)

    assert digest.markdown == build.render_markdown(digest.content)
    assert "a memorable line" in digest.markdown
    assert digest.content["stream_id"] in digest.markdown


# --------------------------------------------------------------------------
# the stage
# --------------------------------------------------------------------------


def test_the_stage_never_defers_not_even_before_anything_has_scored(conn, cfg):
    """`available` answers "can this machine run it", never "are inputs ready".

    Two things this pins at once. There is no API key anywhere in this project,
    so a key check here would mean the stage never ran — that is the whole
    reason the deterministic half exists.

    And the runner evaluates `available` for EVERY stage up front, in `plan()`,
    before any of them has run. A candidates check here therefore deferred
    `digest` on a fresh stream and ran it on the next invocation, so
    `clipforge run` could never say "everything up to date" after one pass.
    Input readiness is `requires=("score",)`, which the registry declares and
    the runner already honours.
    """
    ctx = StageContext(cfg=cfg, conn=conn, stream_id="s", log=lambda *_: None)

    # No candidates, no transcript, nothing scored.
    assert digest_stage.available(ctx) == (True, "")

    _candidate(conn, 100.0, 110.0, score=1.0)
    assert digest_stage.available(ctx) == (True, "")


def test_a_stream_that_scored_nothing_still_gets_a_digest(conn, cfg, settings):
    """An empty `top_candidates` is the honest answer, not a reason to defer.

    The arc, the chapters and the recurring phrases do not depend on a candidate
    existing, and a stream with none is exactly the stream worth having a digest
    of — it is the one you want to ask what happened during.
    """
    digest = build.build(conn, "s", cfg, settings=settings)
    assert digest.content["top_candidates"] == []
    assert digest.content["chapters"], "no chapters for an unscored stream"


def test_the_stage_verifies_against_a_row_not_a_file(conn, cfg):
    """The product is a row, so deleting it must un-do the stage."""
    _candidate(conn, 100.0, 110.0, score=1.0)
    ctx = StageContext(cfg=cfg, conn=conn, stream_id="s", log=lambda *_: None)

    assert digest_stage.verify(ctx)[0] is False
    digest_stage.run(ctx)
    assert digest_stage.verify(ctx)[0] is True

    conn.execute("DELETE FROM digests WHERE stream_id='s'")
    assert digest_stage.verify(ctx)[0] is False


def test_the_stage_logs_the_metrics_that_would_falsify_its_own_guesses(conn, cfg):
    """GUESSWORK DISCIPLINE rule 4.

    Every number in the `digest:` config block is unvalidated. Without these
    three metrics none of them could ever be shown wrong.
    """
    _candidate(conn, 100.0, 110.0, score=1.0)
    ctx = StageContext(cfg=cfg, conn=conn, stream_id="s", log=lambda *_: None)
    digest_stage.run(ctx)

    rows = {r["metric"]: r for r in conn.execute(
        "SELECT metric, value, meta FROM tool_metrics WHERE stream_id='s'")}

    assert {"digest_chapter_count", "digest_word_count",
            "digest_boundary_sources"} <= set(rows)
    # The chapter-length distribution is the only observation that would show
    # min_chapter_s/max_chapter_s or merge_within_s to be wrong.
    assert "lengths_s" in json.loads(rows["digest_chapter_count"]["meta"])
    # WHICH sources, not just how many.
    assert "sources" in json.loads(rows["digest_boundary_sources"]["meta"])


def test_params_covers_every_digest_setting_the_stage_reads(cfg):
    """A key that changed behaviour without invalidating the stage would leave
    the previous digest looking current."""
    used = digest_stage.settings(cfg)
    for key in digest_stage.SETTING_KEYS:
        assert key in used
        assert used[key] is not None, f"digest.{key} is missing from config"
