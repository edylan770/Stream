"""§14's weight-tuning metrics.

Two kinds of assertion here, kept apart on purpose.

`separation` is asserted against **constructed ground truth** — perfectly
separated data must give 1.0, identical data 0.5, inverted data 0.0. Those are
arithmetic facts about a rank statistic, not tolerances chosen to make a fixture
pass, so there is nothing for a wrong number to hide behind.

Everything else is asserted against the config object. The two refusal
thresholds are read from `cfg`, never written down here — the rule every numeric
test in this project follows.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from clipforge import config, db, tuning
from tests.conftest import hand_rows


@pytest.fixture
def bare(tmp_path):
    cfg = config.load(overrides=[f"paths.data_root={(tmp_path / 'data').as_posix()}"])
    conn = db.open_db(cfg.db_path)
    conn.execute(
        "INSERT INTO streams (id, date, master_path, marker_time_base) "
        "VALUES ('s', '2026-08-14', 'D:/m.mkv', 'vod')"
    )
    yield cfg, conn
    conn.close()


def _permissive(cfg, **over):
    """The same config with the two refusals lowered out of the way.

    Named rather than inlined so a test that is ABOUT a refusal cannot silently
    be reading a config that disabled it.
    """
    overrides = [f"paths.data_root={cfg.data_root.as_posix()}",
                 "tuning.min_rated_moments=1",
                 "tuning.min_moments_per_class=1"]
    overrides += [f"tuning.{k}={v}" for k, v in over.items()]
    return config.load(overrides=overrides)


def _rows(vectors_by_rating, *, t0=0.0, gap=100.0):
    """`[(gen, current, t_start, t_end, rating, rated_at, source)]` plus vectors.

    Windows are spaced by `gap` so `moments.cluster` keeps them apart: two
    overlapping windows are ONE moment by design, which is right for the metric
    and wrong for a fixture that means to describe several.
    """
    entries, vectors = [], []
    for index, (rating, vector) in enumerate(vectors_by_rating):
        start = t0 + index * gap
        entries.append((1, 1, start, start + 10.0, rating,
                        f"2026-08-14 10:{index:02d}:00", "operator"))
        vectors.append(vector)
    return entries, vectors


# --------------------------------------------------------------------------
# the statistic, as arithmetic
# --------------------------------------------------------------------------


def test_perfect_separation_is_one():
    assert tuning.separation(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])) == 1.0


def test_identical_distributions_are_a_half():
    assert tuning.separation(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])) == 0.5


def test_a_signal_that_discriminates_backwards_is_below_a_half():
    """The case a firing rate cannot show as such, and the reason this is the
    ranking column: below 0.5 is a weight with the wrong sign."""
    assert tuning.separation(np.array([4.0, 5.0, 6.0]), np.array([1.0, 2.0, 3.0])) == 0.0


def test_every_value_tied_is_a_half_not_undefined():
    """`ranks` averages ties, so a signal that fired identically everywhere
    scores exactly 0.5 — measured, and genuinely 'discriminates nothing'."""
    assert tuning.separation(np.array([2.0, 2.0]), np.array([2.0, 2.0])) == 0.5


def test_an_empty_class_is_undefined_and_not_a_half():
    """`rank_agreement` makes the same distinction for an undefined Spearman:
    reporting 0.5 would read as "measured, and it does not discriminate"."""
    assert tuning.separation(np.array([]), np.array([1.0, 2.0])) is None
    assert tuning.separation(np.array([1.0, 2.0]), np.array([])) is None


def test_the_statistic_is_scale_free():
    """Why one column can cover z-scores, kernel levels and §6.4 gate ramps at
    once: only the ORDER of the values is read."""
    skip, clip = np.array([0.1, 0.2]), np.array([0.3, 0.4])
    assert tuning.separation(skip, clip) == tuning.separation(skip * 1000, clip * 1000)


# --------------------------------------------------------------------------
# reading the corpus
# --------------------------------------------------------------------------


def test_a_signal_authored_to_separate_comes_out_at_one(bare):
    cfg, conn = bare
    entries, vectors = _rows([(0, {"mic_rms": 0.1}), (0, {"mic_rms": 0.2}),
                              (2, {"mic_rms": 5.0}), (2, {"mic_rms": 6.0})])
    hand_rows(conn, "s", entries, feature_vector=vectors)

    _, stats, _ = tuning.collect(conn, _permissive(cfg), ["s"])
    mic = next(s for s in stats if s.name == "mic_rms")
    assert (mic.n_skip, mic.n_clip) == (2, 2)
    assert mic.separation == 1.0
    assert stats[0].name == "mic_rms"          # sorts to the top


def test_maybe_is_counted_but_is_in_neither_class(bare):
    """§17 compares rating-2 against rating-0. Rating 1 is the operator
    declining to decide, which is not evidence in either direction."""
    cfg, conn = bare
    entries, vectors = _rows([(0, {"mic_rms": 0.0}), (1, {"mic_rms": 99.0}),
                              (2, {"mic_rms": 1.0})])
    hand_rows(conn, "s", entries, feature_vector=vectors)

    corpus, stats, _ = tuning.collect(conn, _permissive(cfg), ["s"])
    mic = next(s for s in stats if s.name == "mic_rms")
    assert corpus.by_rating == {0: 1, 1: 1, 2: 1}
    assert (mic.n_skip, mic.n_clip) == (1, 1)
    assert mic.separation == 1.0               # the 99.0 played no part


def test_a_rating_stranded_on_a_superseded_generation_is_still_counted(bare):
    """THE test that says reading through `moments` was necessary rather than
    tidy. After commit 43 a rated stream ALWAYS mints a new generation on
    re-score, so the operator's own row lives on the superseded one — and the
    `is_current = 1 AND rating_source = 'operator'` query every other §14 metric
    used would see nothing at all."""
    cfg, conn = bare
    hand_rows(
        conn, "s",
        [(1, 0, 0.0, 10.0, 0, "2026-08-14 10:00:00", "operator"),
         (1, 0, 100.0, 110.0, 2, "2026-08-14 10:01:00", "operator")],
        feature_vector=[{"mic_rms": 0.0}, {"mic_rms": 9.0}],
    )
    # Generation 2 is current and carries nothing an `is_current` read would use.
    hand_rows(conn, "s", [(2, 1, 500.0, 510.0, 1, "2026-08-14 10:02:00", "inherited")])

    blind = conn.execute(
        "SELECT COUNT(*) FROM candidates c JOIN ratings r ON r.candidate_id = c.id "
        "WHERE c.stream_id = 's' AND c.is_current = 1 AND r.rating_source = 'operator'"
    ).fetchone()[0]
    assert blind == 0

    corpus, stats, _ = tuning.collect(conn, _permissive(cfg), ["s"])
    mic = next(s for s in stats if s.name == "mic_rms")
    assert (mic.n_skip, mic.n_clip) == (1, 1)
    assert corpus.moments == 2


def test_a_moment_rated_twice_across_generations_counts_once(bare):
    """§14's stated hazard. The later opinion is the one that counts, and the
    earlier one does not also contribute a sample."""
    cfg, conn = bare
    hand_rows(
        conn, "s",
        [(1, 0, 0.0, 10.0, 2, "2026-08-14 10:00:00", "operator"),
         (2, 1, 1.0, 11.0, 0, "2026-08-14 11:00:00", "operator")],
        feature_vector=[{"mic_rms": 9.0}, {"mic_rms": 0.5}],
    )

    corpus, stats, _ = tuning.collect(conn, _permissive(cfg), ["s"])
    mic = next(s for s in stats if s.name == "mic_rms")
    assert corpus.moments == 1
    assert (mic.n_skip, mic.n_clip) == (1, 0)   # the change of mind won
    assert mic.separation is None               # and one class is now empty


def test_only_declared_keys_are_scored(bare):
    """A version-1 vector carries context keys the current writer no longer
    emits — `mic_rms_db` is an absolute dB level, not a signal. Iterating the
    stored JSON rather than the schema would rank it. Found by reading the real
    database, where every candidate predates feature_schema version 2."""
    cfg, conn = bare
    entries, vectors = _rows([(0, {"mic_rms": 0.0, "mic_rms_db": -60.0}),
                              (2, {"mic_rms": 1.0, "mic_rms_db": -6.0})])
    hand_rows(conn, "s", entries, feature_vector=vectors)

    _, stats, _ = tuning.collect(conn, _permissive(cfg), ["s"])
    names = {s.name for s in stats}
    assert "mic_rms" in names
    assert "mic_rms_db" not in names
    assert names == set(cfg.feature_schema.keys)


def test_a_null_value_is_not_an_observation(bare):
    """§5.4.1's pitch gaps and every unproduced signal are stored as `null`.
    Reading one as 0.0 would invent a measurement nobody made."""
    cfg, conn = bare
    entries, vectors = _rows([(0, {"mic_rms": 1.0, "laugh_party": None}),
                              (2, {"mic_rms": 2.0, "laugh_party": None})])
    hand_rows(conn, "s", entries, feature_vector=vectors)

    _, stats, _ = tuning.collect(conn, _permissive(cfg), ["s"])
    laugh = next(s for s in stats if s.name == "laugh_party")
    assert (laugh.n_skip, laugh.n_clip) == (0, 0)
    assert laugh.separation is None


# --------------------------------------------------------------------------
# "fired" branches on the value's kind
# --------------------------------------------------------------------------


def test_a_continuous_signal_fires_against_the_configured_threshold(bare):
    cfg, conn = bare
    tuned = _permissive(cfg, firing_threshold_z=2.0)
    threshold = float(tuned.get("tuning.firing_threshold_z"))
    entries, vectors = _rows([(0, {"mic_rms": threshold - 0.5}),
                              (2, {"mic_rms": threshold + 0.5})])
    hand_rows(conn, "s", entries, feature_vector=vectors)

    _, stats, _ = tuning.collect(conn, tuned, ["s"])
    mic = next(s for s in stats if s.name == "mic_rms")
    assert (mic.fired_skip, mic.fired_clip) == (0, 1)


def test_an_event_fires_above_zero_whatever_the_threshold_is(bare):
    """A kernel level is zero exactly when the event did not happen, so `> 0`
    is what "fired" MEANS for it — not a chosen cut. The threshold is set
    absurdly high here and must not affect the answer."""
    cfg, conn = bare
    tuned = _permissive(cfg, firing_threshold_z=99.0)
    entries, vectors = _rows([(0, {"laugh_party": 0.0}), (2, {"laugh_party": 0.4})])
    hand_rows(conn, "s", entries, feature_vector=vectors)

    _, stats, _ = tuning.collect(conn, tuned, ["s"])
    laugh = next(s for s in stats if s.name == "laugh_party")
    assert (laugh.fired_skip, laugh.fired_clip) == (0, 1)


def test_the_threshold_moves_the_rates_and_never_the_ranking(bare):
    """The reason `sep` is the ranking column: it depends on no threshold, so a
    guess in `tuning.firing_threshold_z` cannot decide which signal looks
    best."""
    cfg, conn = bare
    entries, vectors = _rows([(0, {"mic_rms": 0.4}), (0, {"mic_rms": 0.6}),
                              (2, {"mic_rms": 2.4}), (2, {"mic_rms": 2.6})])
    hand_rows(conn, "s", entries, feature_vector=vectors)

    low = tuning.collect(conn, _permissive(cfg, firing_threshold_z=0.5), ["s"])[1]
    high = tuning.collect(conn, _permissive(cfg, firing_threshold_z=9.0), ["s"])[1]

    pick = lambda stats: next(s for s in stats if s.name == "mic_rms")   # noqa: E731
    assert pick(low).separation == pick(high).separation
    assert pick(low).fired_clip != pick(high).fired_clip


# --------------------------------------------------------------------------
# it refuses rather than producing a table that looks like evidence
# --------------------------------------------------------------------------


def test_the_whole_ranking_is_refused_below_the_configured_corpus_size(bare):
    cfg, conn = bare
    minimum = int(cfg.get("tuning.min_rated_moments"))
    entries, vectors = _rows([(0, {"mic_rms": 0.0}), (2, {"mic_rms": 1.0})])
    hand_rows(conn, "s", entries, feature_vector=vectors)

    corpus, _, _ = tuning.collect(conn, cfg, ["s"])
    ok, why = tuning.rankable(corpus, cfg)
    assert corpus.comparable < minimum
    assert not ok
    assert str(minimum) in why


def test_one_signal_is_refused_below_the_configured_observation_count(bare):
    cfg, conn = bare
    per_class = int(cfg.get("tuning.min_moments_per_class"))
    entries, vectors = _rows([(0, {"mic_rms": 0.0}), (2, {"mic_rms": 1.0})])
    hand_rows(conn, "s", entries, feature_vector=vectors)

    _, stats, _ = tuning.collect(conn, cfg, ["s"])
    mic = next(s for s in stats if s.name == "mic_rms")
    assert mic.separation is None
    assert str(per_class) in mic.reason


def test_a_signal_with_no_number_sorts_below_one_that_has_one(bare):
    """0.5 is "measured, and it discriminates nothing"; None is "not measured".
    Sorting the second into the middle would present absence as a finding."""
    cfg, conn = bare
    entries, vectors = _rows([(0, {"mic_rms": 1.0}), (2, {"mic_rms": 1.0})])
    hand_rows(conn, "s", entries, feature_vector=vectors)

    _, stats, _ = tuning.collect(conn, _permissive(cfg), ["s"])
    measured = [s for s in stats if s.separation is not None]
    unmeasured = [s for s in stats if s.separation is None]
    assert measured and unmeasured
    assert stats.index(measured[-1]) < stats.index(unmeasured[0])


# --------------------------------------------------------------------------
# §14's marker pair
# --------------------------------------------------------------------------


def _press(conn, stream_id, t):
    with db.transaction(conn):
        conn.execute(
            "INSERT INTO events (stream_id, t, source, kind) VALUES (?, ?, 'marker', ?)",
            (stream_id, t, "marker_definite"),
        )


def test_marker_precision_counts_approved_over_anchored(bare):
    cfg, conn = bare
    entries, vectors = _rows([(2, {"mic_rms": 1.0}), (0, {"mic_rms": 0.0})])
    hand_rows(conn, "s", entries, feature_vector=vectors)
    _press(conn, "s", 5.0)      # inside the first window (0..10)
    _press(conn, "s", 105.0)    # inside the second (100..110)

    _, _, markers = tuning.collect(conn, _permissive(cfg), ["s"])
    stat = markers[0]
    assert (stat.anchored, stat.anchored_approved) == (2, 1)
    assert stat.precision == 0.5


def test_marker_recall_proxy_is_the_approved_moments_nobody_marked(bare):
    """§14 singles this one out: "how many good clips the operator misses
    live", which is the worry that motivated automatic detection."""
    cfg, conn = bare
    entries, vectors = _rows([(2, {"mic_rms": 1.0}), (2, {"mic_rms": 1.0})])
    hand_rows(conn, "s", entries, feature_vector=vectors)
    _press(conn, "s", 5.0)      # only the first was marked

    _, _, markers = tuning.collect(conn, _permissive(cfg), ["s"])
    stat = markers[0]
    assert (stat.approved, stat.approved_unmarked) == (2, 1)
    assert stat.recall_proxy == 0.5


def test_marker_precision_ignores_a_contribution_with_no_press(bare):
    """§4.3's plateau runs 25 s before a press, so a window merely NEAR one
    carries a marker contribution. §7.4 wants that; §14 must not have it, or the
    number moves when a weight moves. See `moments.MarkerAnchoring`."""
    cfg, conn = bare
    entries, vectors = _rows([(2, {"mic_rms": 1.0})])
    hand_rows(conn, "s", entries, feature_vector=vectors,
              contributions=[{"marker_definite": 3.0}])
    # No events row at all — nothing was actually pressed inside the window.

    _, _, markers = tuning.collect(conn, _permissive(cfg), ["s"])
    assert markers[0].anchored == 0
    assert markers[0].precision is None
    assert markers[0].approved_unmarked == 1


def test_zeroing_a_marker_weight_leaves_the_precision_unmoved(bare):
    """The asymmetry stated as a property rather than as prose: `press_inside`
    reads `events`, so no weight can touch it."""
    cfg, conn = bare
    entries, vectors = _rows([(2, {"mic_rms": 1.0}), (0, {"mic_rms": 0.0})])
    ids = hand_rows(conn, "s", entries, feature_vector=vectors,
                    contributions=[{"marker_definite": 3.0}, {"marker_definite": 3.0}])
    _press(conn, "s", 5.0)
    before = tuning.collect(conn, _permissive(cfg), ["s"])[2][0].precision

    # A re-score under a zero marker weight drops it from contributing_signals.
    with db.transaction(conn):
        for candidate_id in ids:
            conn.execute("UPDATE candidates SET contributing_signals = '{}' WHERE id = ?",
                         (candidate_id,))

    assert tuning.collect(conn, _permissive(cfg), ["s"])[2][0].precision == before


# --------------------------------------------------------------------------
# recording, and the subtree it is configured from
# --------------------------------------------------------------------------


def test_recording_writes_section_14s_own_metric_name(bare):
    """HANDOFF's rule: a metric renamed here is one whatever reads it later
    cannot find. §17 says to pull `signal_firing_rate_by_rating` by name."""
    cfg, conn = bare
    entries, vectors = _rows([(0, {"mic_rms": 0.0}), (2, {"mic_rms": 1.0})])
    hand_rows(conn, "s", entries, feature_vector=vectors)
    _press(conn, "s", 105.0)

    study = tuning.collect(conn, _permissive(cfg), ["s"])
    with db.transaction(conn):
        tuning.record(conn, *study)

    names = {r[0] for r in conn.execute("SELECT DISTINCT metric FROM tool_metrics")}
    assert {"signal_firing_rate_by_rating", "marker_precision",
            "marker_recall_proxy"} <= names


def test_an_undefined_separation_is_stored_with_a_companion_flag(bare):
    """`tool_metrics.value` is a bare REAL, so None has to be 0.0 plus a
    boolean — otherwise "undefined" and "no discrimination at all" become the
    same row. The shape `combined_rank_agreement` already uses."""
    cfg, conn = bare
    entries, vectors = _rows([(0, {"mic_rms": 0.0}), (0, {"mic_rms": 1.0})])
    hand_rows(conn, "s", entries, feature_vector=vectors)

    study = tuning.collect(conn, _permissive(cfg), ["s"])
    with db.transaction(conn):
        tuning.record(conn, *study)

    row = conn.execute(
        "SELECT value, meta FROM tool_metrics WHERE metric = 'signal_firing_rate_by_rating' "
        "AND json_extract(meta, '$.signal') = 'mic_rms'"
    ).fetchone()
    assert row["value"] == 0.0
    assert json.loads(row["meta"])["separation_defined"] is False


def test_the_tuning_subtree_cannot_invalidate_a_candidate():
    """Reading the instrument must not move the thing it measures. `tuning` is
    outside VERSIONED_SUBTREES for the reason `llm:` and `render:` are."""
    from clipforge.config import VERSIONED_SUBTREES

    assert "tuning" not in VERSIONED_SUBTREES
    base = config.load()
    moved = config.load(overrides=["tuning.firing_threshold_z=9.0",
                                   "tuning.min_rated_moments=1"])
    assert base.version == moved.version
