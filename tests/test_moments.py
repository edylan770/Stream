"""`clipforge.moments` — one opinion per moment, and "was this marked?".

The rule itself is covered by `tests/test_selection.py`, which passes unmodified
across the move and is therefore the evidence that this was a move rather than a
rewrite. What is tested here is what the move ADDED: that the re-export is an
alias rather than a copy, and the two marker readings §7.4 and §14 need.
"""

from __future__ import annotations

import pytest

from clipforge import config, db, moments
from clipforge.render import selection


# --------------------------------------------------------------------------
# the move: aliases, not copies
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["Moment", "rated_candidates", "cluster", "verdict"])
def test_selection_re_exports_the_same_object(name):
    """IDENTITY, not equality.

    A copied function compares equal to nothing and would pass any behavioural
    test today, then drift the first time one of the two is edited. `is` is the
    only assertion that cannot be satisfied by a second implementation — the
    same check commit 42 used when §12's validator left `render/hooks.py`.
    """
    assert getattr(selection, name) is getattr(moments, name)


def test_the_rule_lives_in_exactly_one_module():
    """§14's hazard is counting one judgment twice, and two implementations of
    "which opinion wins" is how that happens without anyone deciding to."""
    source = (moments.__file__, selection.__file__)
    assert source[0] != source[1]
    for name in ("rated_candidates", "cluster", "verdict"):
        assert getattr(moments, name).__module__ == "clipforge.moments"
        assert getattr(selection, name).__module__ == "clipforge.moments"


# --------------------------------------------------------------------------
# §7.4's reading and §14's reading are different tests
# --------------------------------------------------------------------------


def test_a_press_inside_the_window_anchors_it_under_both_readings():
    anchoring = moments.marker_anchoring(10.0, 20.0, [15.0], {})
    assert anchoring.press_inside == (15.0,)
    assert anchoring.any


def test_a_press_outside_the_window_is_not_press_inside():
    anchoring = moments.marker_anchoring(10.0, 20.0, [25.0], {})
    assert anchoring.press_inside == ()
    assert not anchoring.any


def test_a_contribution_with_no_press_anchors_only_the_loose_reading():
    """THE distinction the split exists for.

    §4.3's plateau runs 25 s before and 5 s after each press, so a window merely
    NEAR a marker carries a contribution without containing the press. §7.4's
    safety net wants that window; §14's `marker_precision` must not count it,
    because it is measuring whether marking predicts approval.
    """
    anchoring = moments.marker_anchoring(10.0, 20.0, [40.0], {"marker_definite": 3.0})
    assert anchoring.press_inside == ()
    assert anchoring.contributed
    assert anchoring.any


def test_the_loose_reading_is_weight_dependent_and_the_strict_one_is_not():
    """`contributing_signals` holds weight × value and `features.breakdown`
    drops weight-0 signals, so zeroing a marker weight empties `contributed`
    while `press_inside` is unmoved. That asymmetry is exactly why §14 cannot
    use the loose reading: a weight-tuning input that moves when a weight moves
    cannot tune anything."""
    weighted = moments.marker_anchoring(10.0, 20.0, [15.0], {"marker_maybe": 1.5})
    unweighted = moments.marker_anchoring(10.0, 20.0, [15.0], {})

    assert weighted.contributed and not unweighted.contributed
    assert weighted.press_inside == unweighted.press_inside


def test_a_negative_contribution_does_not_anchor():
    """`> 0`, not `bool(...)`. Nothing gives a marker a negative weight today,
    so this asserts the reading rather than a behaviour anyone has seen."""
    assert not moments.marker_anchoring(10.0, 20.0, [], {"marker_definite": -1.0}).contributed


def test_missing_contributions_are_the_same_as_none():
    assert (moments.marker_anchoring(0.0, 1.0, [], None)
            == moments.marker_anchoring(0.0, 1.0, [], {}))


# --------------------------------------------------------------------------
# marker_times reads by source, not by kind
# --------------------------------------------------------------------------


@pytest.fixture
def bare(tmp_path):
    cfg = config.load(overrides=[f"paths.data_root={(tmp_path / 'data').as_posix()}"])
    conn = db.open_db(cfg.db_path)
    conn.execute(
        "INSERT INTO streams (id, date, master_path, marker_time_base) "
        "VALUES ('s', '2026-08-14', 'D:/m.mkv', 'vod')"
    )
    yield conn
    conn.close()


def test_marker_times_selects_by_source_so_a_new_key_needs_no_code_change(bare):
    """§3.2's rationale for the `events` shape is "new sensor = new `source`
    value". A third marker hotkey would arrive as a new `kind` under the same
    source, and filtering by kind here would silently ignore it."""
    with db.transaction(bare):
        bare.executemany(
            "INSERT INTO events (stream_id, t, source, kind) VALUES ('s', ?, ?, ?)",
            [
                (5.0, moments.MARKER_SOURCE, "marker_maybe"),
                (9.0, moments.MARKER_SOURCE, "marker_definite"),
                (7.0, moments.MARKER_SOURCE, "marker_invented_later"),
                (8.0, "laugh", "laugh_party"),
            ],
        )

    assert moments.marker_times(bare, "s") == [5.0, 7.0, 9.0]
