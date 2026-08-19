"""§14's three weight-tuning metrics.

Every threshold here comes from the config object (house rule 3). The signal
names come from `config.load_feature_schema()` rather than being restated, for
the reason the schema file gives itself: five months of vectors depend on those
names being stable, and a test that hardcoded them would keep passing after one
changed.

**The four filters get a test each.** Each is silently wrong rather than loud —
drop any of them and the metric still produces a plausible-looking number, which
is the worst way for a tuning input to fail.
"""

from __future__ import annotations

import json

import pytest

from clipforge import config, db
from clipforge.review import queries, tuning


@pytest.fixture
def cfg():
    return config.load()


@pytest.fixture
def conn(tmp_path):
    _NEXT_WINDOW[0] = 100.0
    connection = db.open_db(tmp_path / "t.db")
    connection.execute(
        "INSERT INTO streams (id, date, master_path, duration_s) "
        "VALUES ('s', '2026-01-01', '/m.mkv', 3600)"
    )
    yield connection
    connection.close()


def _signal_names(cfg, group="continuous", count=2):
    """Real schema keys, so the test cannot drift from the declared set."""
    names = [n for n, m in cfg.feature_schema.signals.items()
             if m.get("group") == group]
    assert len(names) >= count, "the schema no longer has enough signals to test"
    return names[:count]


#: `candidates` is UNIQUE on (stream_id, generation, profile, t_peak), so every
#: candidate needs its own window. Allocated here rather than spelled out at each
#: call site: none of these tests is about where in the stream a moment sits.
_NEXT_WINDOW = [100.0]


def _candidate(conn, cfg, *, rating=None, vector=None, t_start=None,
               t_end=None, generation=1, is_current=1,
               rating_source="operator", contributions=None):
    if t_start is None:
        t_start = _NEXT_WINDOW[0]
        _NEXT_WINDOW[0] += 100.0
    if t_end is None:
        t_end = t_start + 40.0
    full = cfg.feature_schema.empty_vector()
    full.update(vector or {})
    cur = conn.execute(
        "INSERT INTO candidates (stream_id, generation, is_current, profile, "
        "t_start, t_end, t_peak, score_entertainment, score_gameplay, "
        "score_combined, contributing_signals, feature_vector, "
        "feature_schema_version, config_version) "
        "VALUES ('s', ?, ?, 'entertainment', ?, ?, ?, 0, 0, 1, ?, ?, ?, 'x@y')",
        (generation, is_current, t_start, t_end, (t_start + t_end) / 2,
         json.dumps(contributions or {}), json.dumps(full),
         cfg.feature_schema.version))
    if rating is not None:
        conn.execute(
            "INSERT INTO ratings (candidate_id, rating, rating_source) "
            "VALUES (?, ?, ?)", (cur.lastrowid, rating, rating_source))
    return cur.lastrowid


def _marker(conn, t):
    conn.execute(
        "INSERT INTO events (stream_id, t, kind, source) "
        "VALUES ('s', ?, 'marker_definite', 'marker')", (t,))


# --------------------------------------------------------------------------
# the four filters
# --------------------------------------------------------------------------


def test_a_null_signal_does_not_count_against_its_own_firing_rate(conn, cfg):
    """A9 stores every declared signal with null where it was not observed.

    Treating those as zero would say "the pitch was average" when the truth is
    that nobody was speaking — and `mic_f0`, unvoiced through most of any
    stream, would come out looking like a signal that never fires. Each signal
    keeps its own denominator: the candidates where it was actually observed.
    """
    loud, quiet = _signal_names(cfg)
    # `loud` observed on both; `quiet` observed on neither.
    _candidate(conn, cfg, rating=2, vector={loud: 3.0})
    _candidate(conn, cfg, rating=0, vector={loud: -1.0})
    # Two more candidates where BOTH are null.
    _candidate(conn, cfg, rating=2)
    _candidate(conn, cfg, rating=0)

    stats = tuning.collect_signals(conn, ["s"], cfg.feature_schema)

    assert stats[loud].observed == 2
    assert stats[loud].n_approved == 1 and stats[loud].n_rejected == 1
    # Not 4 observations with two zeros in them.
    assert stats[loud].approved == [3.0]
    assert stats[loud].rejected == [-1.0]
    # A signal nobody observed has no stats at all, rather than a 0.0 mean.
    assert stats[quiet].observed == 0
    assert stats[quiet].separation() is None


def test_an_inherited_rating_is_not_counted_as_fresh_evidence(conn, cfg):
    """`score/runner.py` says outright that this filter is what stops a verdict
    being counted twice, and §14's tuning input is where it would happen."""
    name, _ = _signal_names(cfg)
    _candidate(conn, cfg, rating=2, vector={name: 3.0})
    _candidate(conn, cfg, rating=2, vector={name: 3.0},
               rating_source="inherited")

    stats = tuning.collect_signals(conn, ["s"], cfg.feature_schema)

    assert stats[name].n_approved == 1


def test_a_maybe_lands_on_neither_side_of_the_contrast(conn, cfg):
    """§14 contrasts "approved vs. rejected". A maybe is neither.

    Folding it into either side would blur the one contrast being measured, so
    it is counted separately and kept out of both means.
    """
    name, _ = _signal_names(cfg)
    _candidate(conn, cfg, rating=2, vector={name: 3.0})
    _candidate(conn, cfg, rating=1, vector={name: 99.0})
    _candidate(conn, cfg, rating=0, vector={name: 1.0})

    stats = tuning.collect_signals(conn, ["s"], cfg.feature_schema)

    assert stats[name].approved == [3.0]
    assert stats[name].rejected == [1.0]
    assert stats[name].maybes == [99.0]
    # The maybe's extreme value must not have moved the separation.
    assert stats[name].separation() == pytest.approx(2.0)


def test_an_older_generation_is_not_counted(conn, cfg):
    """Its vectors describe a scoring run under weights that no longer exist."""
    name, _ = _signal_names(cfg)
    _candidate(conn, cfg, rating=2, vector={name: 1.0})
    _candidate(conn, cfg, rating=2, vector={name: 99.0},
               generation=0, is_current=0)

    stats = tuning.collect_signals(conn, ["s"], cfg.feature_schema)

    assert stats[name].approved == [1.0]


def test_an_unrated_candidate_is_not_a_rejection(conn, cfg):
    """"Not looked at yet" is not "looked at and refused"."""
    name, _ = _signal_names(cfg)
    _candidate(conn, cfg, rating=2, vector={name: 1.0})
    _candidate(conn, cfg, rating=None, vector={name: -5.0})

    stats = tuning.collect_signals(conn, ["s"], cfg.feature_schema)

    assert stats[name].rejected == []


# --------------------------------------------------------------------------
# what the numbers say
# --------------------------------------------------------------------------


def test_a_signal_that_predicts_approval_shows_a_positive_separation(conn, cfg):
    """The headline number, and it needs no threshold to compute."""
    good, noise = _signal_names(cfg)
    for value in (2.0, 2.5, 3.0):
        _candidate(conn, cfg, rating=2, vector={good: value, noise: 1.0})
    for value in (-1.0, -0.5, 0.0):
        _candidate(conn, cfg, rating=0, vector={good: value, noise: 1.0})

    stats = tuning.collect_signals(conn, ["s"], cfg.feature_schema)

    assert stats[good].separation() > 2.0
    # A signal at the same level on both sides carries no information, and says
    # so with a separation of zero rather than by being absent.
    assert stats[noise].separation() == pytest.approx(0.0)


def test_a_signal_lower_on_approved_moments_is_reported_not_hidden(conn, cfg):
    """A reliably negative separation is just as informative — it wants a
    negative weight, or it is a §6.4 gate."""
    name, _ = _signal_names(cfg)
    _candidate(conn, cfg, rating=2, vector={name: -2.0})
    _candidate(conn, cfg, rating=0, vector={name: 2.0})

    stats = tuning.collect_signals(conn, ["s"], cfg.feature_schema)

    assert stats[name].separation() == pytest.approx(-4.0)


def test_the_firing_threshold_comes_from_config(conn, cfg):
    """House rule 3, and the reason `separation` is the headline: this number
    is arbitrary, so nothing may hardcode it."""
    name, _ = _signal_names(cfg)
    threshold = float(cfg.get("metrics.firing_threshold_z"))
    _candidate(conn, cfg, rating=2, vector={name: threshold + 0.5})
    _candidate(conn, cfg, rating=2, vector={name: threshold - 0.5})

    stats = tuning.collect_signals(conn, ["s"], cfg.feature_schema)
    entry = stats[name]

    assert entry.fired_rate(entry.approved, threshold) == pytest.approx(0.5)
    # Raising the bar above both values takes the rate to zero — proving the
    # threshold is actually being applied rather than baked in.
    assert entry.fired_rate(entry.approved, threshold + 10) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# markers (§14, GUESSES gap 2)
# --------------------------------------------------------------------------


def test_marker_precision_is_the_share_of_marked_moments_that_survived(
        conn, cfg):
    _marker(conn, 120.0)                                   # inside the window
    _candidate(conn, cfg, rating=2, t_start=100.0, t_end=140.0)
    _marker(conn, 320.0)
    _candidate(conn, cfg, rating=0, t_start=300.0, t_end=340.0)

    stats = tuning.collect_markers(conn, ["s"])

    assert stats.anchored == 2
    assert stats.anchored_approved == 1
    assert stats.precision() == pytest.approx(0.5)


def test_marker_recall_proxy_counts_approved_moments_with_no_marker(conn, cfg):
    """§14 calls this the valuable one: how much the operator missed live.

    It is the number that says whether automatic detection is earning its keep
    at all — near zero would mean the markers were already enough.
    """
    _marker(conn, 120.0)
    _candidate(conn, cfg, rating=2, t_start=100.0, t_end=140.0)
    _candidate(conn, cfg, rating=2, t_start=500.0, t_end=540.0)   # unmarked
    _candidate(conn, cfg, rating=2, t_start=700.0, t_end=740.0)   # unmarked

    stats = tuning.collect_markers(conn, ["s"])

    assert stats.approved == 3
    assert stats.approved_without_marker == 2
    assert stats.recall_proxy() == pytest.approx(2 / 3)


def test_a_retro_marker_outside_the_window_still_counts_as_anchored(conn, cfg):
    """§4.3's whole point: the press lands ~20 s AFTER the moment.

    So a window covering the moment frequently does not contain the press that
    found it, and the `marker_definite` contribution is what records the link.
    Counting only presses inside the window would make `marker_precision` — the
    direct test of `retro_offset_s` — measure the offset it is meant to check.
    """
    _candidate(conn, cfg, rating=2, t_start=100.0, t_end=140.0,
               contributions={"marker_definite": 3.0})
    _marker(conn, 200.0)     # well outside the window

    stats = tuning.collect_markers(conn, ["s"])

    assert stats.anchored == 1
    assert stats.approved_without_marker == 0


def test_the_metric_and_the_review_rail_agree_on_marker_anchored(conn, cfg):
    """One predicate, two callers.

    §7.4's fourth rail section and §14's marker metrics must not disagree about
    what a marker-anchored candidate is — a metric scoring it differently from
    the screen the operator rated on would be measuring something nobody saw.
    """
    contributions = {"marker_maybe": 1.5}
    _candidate(conn, cfg, rating=2, t_start=100.0, t_end=140.0,
               contributions=contributions)

    from_metric = tuning.collect_markers(conn, ["s"]).anchored
    from_rail = queries.is_marker_anchored([], contributions)

    assert from_rail is True
    assert from_metric == 1


# --------------------------------------------------------------------------
# sample-size honesty
# --------------------------------------------------------------------------


def test_a_rate_is_withheld_below_the_configured_sample_size(conn, cfg):
    """The part that matters most.

    A fraction over n=1 reads exactly like a fraction over n=1000, and §17's
    procedure is someone changing a weight because of a number on this screen.
    """
    name, _ = _signal_names(cfg)
    _candidate(conn, cfg, rating=2, vector={name: 3.0})
    _candidate(conn, cfg, rating=0, vector={name: 0.0})

    result = tuning.tuning_metrics(conn, cfg, ["s"])

    assert result.min_samples == int(cfg.get("metrics.min_samples_for_rate"))
    assert not result.enough
    # The counts are real and still reported; only the fractions are withheld.
    assert result.approved == 1 and result.rejected == 1


def test_enough_is_gated_on_the_smaller_side_not_the_total(conn, cfg):
    """200 rejections and 2 approvals cannot say which signals predict approval,
    however large the total looks."""
    minimum = int(cfg.get("metrics.min_samples_for_rate"))
    name, _ = _signal_names(cfg)
    for i in range(minimum + 5):
        _candidate(conn, cfg, rating=0, vector={name: 0.0},
                   t_start=100.0 + i * 50, t_end=140.0 + i * 50)
    _candidate(conn, cfg, rating=2, vector={name: 3.0},
               t_start=9000.0, t_end=9040.0)

    result = tuning.tuning_metrics(conn, cfg, ["s"])

    assert result.rated > minimum
    assert not result.enough


def test_enough_flips_once_both_sides_reach_the_threshold(conn, cfg):
    minimum = int(cfg.get("metrics.min_samples_for_rate"))
    name, _ = _signal_names(cfg)
    for i in range(minimum):
        _candidate(conn, cfg, rating=2, vector={name: 3.0},
                   t_start=100.0 + i * 50, t_end=140.0 + i * 50)
        _candidate(conn, cfg, rating=0, vector={name: 0.0},
                   t_start=20000.0 + i * 50, t_end=20040.0 + i * 50)

    result = tuning.tuning_metrics(conn, cfg, ["s"])

    assert result.enough
    row = next(r for r in result.signals if r["signal"] == name)
    assert row["fired_rate_approved"] == pytest.approx(1.0)
    assert row["fired_rate_rejected"] == pytest.approx(0.0)
    assert row["lift"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# the report shape
# --------------------------------------------------------------------------


def test_signals_are_ranked_by_absolute_separation(conn, cfg):
    """Ranked by |separation|: a signal reliably LOWER on approved moments is
    just as informative as one that is higher."""
    strong, weak = _signal_names(cfg)
    _candidate(conn, cfg, rating=2, vector={strong: -5.0, weak: 0.1})
    _candidate(conn, cfg, rating=0, vector={strong: 5.0, weak: 0.0})

    result = tuning.tuning_metrics(conn, cfg, ["s"])

    assert result.signals[0]["signal"] == strong
    assert result.signals[0]["separation"] < 0


def test_every_reported_signal_carries_its_sample_size(conn, cfg):
    """So a rate can never be read without the n it came from."""
    name, _ = _signal_names(cfg)
    _candidate(conn, cfg, rating=2, vector={name: 1.0})
    _candidate(conn, cfg, rating=0, vector={name: 0.0})

    for row in tuning.tuning_metrics(conn, cfg, ["s"]).signals:
        assert {"n_approved", "n_rejected", "observed"} <= set(row)


def test_a_signal_nobody_observed_is_absent_rather_than_zero(conn, cfg):
    name, _ = _signal_names(cfg)
    _candidate(conn, cfg, rating=2, vector={name: 1.0})

    reported = {r["signal"] for r in tuning.tuning_metrics(conn, cfg, ["s"]).signals}

    assert name in reported
    assert len(reported) == 1, "unobserved signals are being reported as 0.0"


def test_the_report_covers_the_whole_library_by_default(conn, cfg):
    """§17 tunes weights for the corpus: a weight right for one stream and wrong
    for the next nine is not a weight worth setting."""
    conn.execute("INSERT INTO streams (id, date, master_path, duration_s) "
                 "VALUES ('s2', '2026-01-02', '/m2.mkv', 3600)")
    name, _ = _signal_names(cfg)
    _candidate(conn, cfg, rating=2, vector={name: 1.0})

    everything = tuning.tuning_metrics(conn, cfg)

    assert everything.streams == 2


def test_the_metric_names_match_section_14s_table():
    """A different name here means §14's table and this report are two
    documents about different things."""
    assert tuning.FIRING_METRIC == "signal_firing_rate_by_rating"
    assert tuning.MARKER_PRECISION == "marker_precision"
    assert tuning.MARKER_RECALL == "marker_recall_proxy"
