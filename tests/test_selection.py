"""Reading ratings across generations.

The headline case is built from a REAL re-score — register, run, rate, score
again — rather than from hand-inserted rows, because the thing under test is
whether a genuine generation change strands a judgment call.
"""

from __future__ import annotations

import pytest

from clipforge import config, db
from clipforge.ingest import register
from clipforge.pipeline import runner
from clipforge.render import selection
from clipforge.score import runner as score_runner


def make_stream(root, fixture_dir, manifest, overrides=()):
    cfg = config.load(overrides=[
        f"paths.data_root={(root / 'data').as_posix()}", *overrides,
    ])
    db.open_db(cfg.db_path).close()
    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    result = register.register_stream(
        cfg, conn, register.RegisterSpec(master=fixture_dir / manifest["video"])
    )
    engine = runner.Runner(cfg=cfg, conn=conn, stream_id=result.stream_id,
                           log=lambda *_: None)
    engine.execute(engine.plan())
    return cfg, result.stream_id, conn


@pytest.fixture(scope="module")
def scored(tmp_path_factory, fixture_dir, manifest):
    """A processed stream whose candidates have been rated by 'the operator'."""
    cfg, stream_id, conn = make_stream(
        tmp_path_factory.mktemp("selection"), fixture_dir, manifest
    )
    rows = conn.execute(
        "SELECT id FROM candidates WHERE stream_id = ? AND is_current = 1 "
        "ORDER BY t_start", (stream_id,)
    ).fetchall()
    with db.transaction(conn):
        for row, rating in zip(rows, [2, 0, 2, 1], strict=False):
            conn.execute(
                "INSERT INTO ratings (candidate_id, rating, review_ms, rating_source) "
                "VALUES (?, ?, 1200, 'operator')",
                (row["id"], rating),
            )
    yield cfg, stream_id, conn
    conn.close()


def rescore(cfg, conn, stream_id, overrides):
    """Re-score into a new generation. Config must differ or it replaces."""
    fresh = config.load(overrides=[
        f"paths.data_root={cfg.data_root.as_posix()}", *overrides,
    ])
    from clipforge.pipeline.context import StageContext
    ctx = StageContext(cfg=fresh, conn=conn, stream_id=stream_id, log=lambda *_: None)
    return score_runner.run(ctx)


# --------------------------------------------------------------------------
# the requirement
# --------------------------------------------------------------------------


def test_an_approval_survives_a_rescore_that_strands_it(scored):
    """HANDOFF: "a re-score that drops a moment you already approved must not
    drop it from the export."

    Inheritance is disabled outright (an overlap threshold nothing can satisfy),
    which is exactly the state a re-score reaches when no new candidate matches
    the old window closely enough."""
    cfg, stream_id, conn = scored
    before = selection.approved_moments(conn, stream_id, min_rating=2)
    assert before, "fixture should have approved moments to begin with"

    rescore(cfg, conn, stream_id, [
        "score.smoothing_sigma_s=3.5",
        "score.rating_inherit_min_overlap=1.01",   # nothing can inherit
    ])

    stranded = conn.execute(
        "SELECT COUNT(*) FROM candidates c JOIN ratings r ON r.candidate_id = c.id "
        "WHERE c.stream_id = ? AND c.is_current = 0 AND r.rating_source = 'operator'",
        (stream_id,),
    ).fetchone()[0]
    assert stranded, "the re-score should have left ratings on a superseded generation"

    after = selection.approved_moments(conn, stream_id, min_rating=2)
    assert [round(m.t_start, 3) for m in after] == [round(m.t_start, 3) for m in before]
    assert any(m.rescued for m in after), "the safety net should be visible"


def test_an_is_current_read_would_have_lost_them(scored):
    """The bug being prevented, demonstrated rather than asserted in prose."""
    _cfg, stream_id, conn = scored
    naive = conn.execute(
        "SELECT COUNT(*) FROM candidates c JOIN ratings r ON r.candidate_id = c.id "
        "WHERE c.stream_id = ? AND c.is_current = 1 AND r.rating_source = 'operator' "
        "AND r.rating >= 2",
        (stream_id,),
    ).fetchone()[0]
    assert len(selection.approved_moments(conn, stream_id, min_rating=2)) > naive


# --------------------------------------------------------------------------
# the opposite bug: a change of mind must still win
# --------------------------------------------------------------------------


def hand_rows(conn, stream_id, entries):
    """Insert candidate/rating pairs directly, for the edge cases."""
    made = []
    with db.transaction(conn):
        for gen, current, t0, t1, rating, when, source in entries:
            cursor = conn.execute(
                "INSERT INTO candidates (stream_id, generation, is_current, profile, "
                "t_start, t_end, t_peak, score_entertainment, score_gameplay, "
                "score_combined, feature_vector, feature_schema_version, config_version) "
                "VALUES (?, ?, ?, 'naive', ?, ?, ?, 1, 0, 1, '{}', 1, 'test@0')",
                (stream_id, gen, current, t0, t1, (t0 + t1) / 2),
            )
            conn.execute(
                "INSERT INTO ratings (candidate_id, rating, rated_at, rating_source) "
                "VALUES (?, ?, ?, ?)",
                (cursor.lastrowid, rating, when, source),
            )
            made.append(cursor.lastrowid)
    return made


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


def test_a_later_skip_beats_an_earlier_approval(bare):
    """Rate it 'clip it', re-score, watch it again, rate it 'skip'. A naive
    union across generations would still export it."""
    hand_rows(bare, "s", [
        (1, 0, 100.0, 140.0, 2, "2026-08-10 10:00:00", "operator"),
        (2, 1, 105.0, 145.0, 0, "2026-08-12 11:00:00", "operator"),
    ])
    assert selection.approved_moments(bare, "s", min_rating=2) == []


def test_a_later_approval_beats_an_earlier_skip(bare):
    """And the reverse, so the rule is recency and not pessimism."""
    hand_rows(bare, "s", [
        (1, 0, 100.0, 140.0, 0, "2026-08-10 10:00:00", "operator"),
        (2, 1, 105.0, 145.0, 2, "2026-08-12 11:00:00", "operator"),
    ])
    moments = selection.approved_moments(bare, "s", min_rating=2)
    assert len(moments) == 1
    assert moments[0].t_start == 105.0


def test_non_overlapping_moments_stay_separate(bare):
    hand_rows(bare, "s", [
        (1, 1, 100.0, 140.0, 2, "2026-08-10 10:00:00", "operator"),
        (1, 1, 300.0, 340.0, 2, "2026-08-10 10:01:00", "operator"),
    ])
    assert len(selection.approved_moments(bare, "s", min_rating=2)) == 2


def test_an_approved_window_is_never_shortened_by_a_rescore(bare):
    """The union of the approved windows: the operator can trim in Resolve but
    cannot recover a frame the export never referenced (C2)."""
    hand_rows(bare, "s", [
        (1, 0, 100.0, 140.0, 2, "2026-08-10 10:00:00", "operator"),
        (2, 1, 110.0, 145.0, 2, "2026-08-12 11:00:00", "operator"),
    ])
    moment = selection.approved_moments(bare, "s", min_rating=2)[0]
    assert (moment.t_start, moment.t_end) == (100.0, 145.0)


# --------------------------------------------------------------------------
# inherited copies, and operator trims
# --------------------------------------------------------------------------


def test_inherited_ratings_are_never_counted(bare):
    """They are copies made by _inherit_ratings; counting both would let one
    judgment call appear as two, which §14's tuning is told to avoid."""
    hand_rows(bare, "s", [
        (1, 0, 100.0, 140.0, 2, "2026-08-10 10:00:00", "operator"),
        (2, 1, 100.0, 140.0, 2, "2026-08-12 11:00:00", "inherited"),
    ])
    moments = selection.approved_moments(bare, "s", min_rating=2)
    assert len(moments) == 1
    assert len(moments[0].candidate_ids) == 1


def test_an_inherited_row_alone_still_exports_its_moment(bare):
    """The original operator row is what carries it, and it is still there."""
    hand_rows(bare, "s", [
        (1, 0, 10.0, 20.0, 2, "2026-08-10 10:00:00", "operator"),
        (2, 1, 10.5, 20.5, 2, "2026-08-12 11:00:00", "inherited"),
    ])
    assert len(selection.approved_moments(bare, "s", min_rating=2)) == 1


def test_an_operator_trim_outranks_the_detector(bare):
    """§7.3's nudge keys are not built yet, so adjusted_* is NULL in practice —
    but the operator's boundary must win the moment it exists."""
    made = hand_rows(bare, "s", [
        (1, 1, 100.0, 140.0, 2, "2026-08-10 10:00:00", "operator"),
    ])
    bare.execute(
        "UPDATE ratings SET adjusted_start = 112.5, adjusted_end = 133.25 "
        "WHERE candidate_id = ?", (made[0],)
    )
    moment = selection.approved_moments(bare, "s", min_rating=2)[0]
    assert (moment.t_start, moment.t_end) == (112.5, 133.25)


# --------------------------------------------------------------------------
# thresholds and ordering
# --------------------------------------------------------------------------


def test_min_rating_selects_what_it_says(bare):
    hand_rows(bare, "s", [
        (1, 1, 10.0, 20.0, 2, "2026-08-10 10:00:00", "operator"),
        (1, 1, 40.0, 50.0, 1, "2026-08-10 10:01:00", "operator"),
        (1, 1, 70.0, 80.0, 0, "2026-08-10 10:02:00", "operator"),
    ])
    assert len(selection.approved_moments(bare, "s", min_rating=2)) == 1
    assert len(selection.approved_moments(bare, "s", min_rating=1)) == 2
    assert len(selection.approved_moments(bare, "s", min_rating=0)) == 3


def test_moments_come_back_in_stream_order(bare):
    hand_rows(bare, "s", [
        (1, 1, 300.0, 340.0, 2, "2026-08-10 10:00:00", "operator"),
        (1, 1, 100.0, 140.0, 2, "2026-08-10 10:01:00", "operator"),
        (1, 1, 200.0, 240.0, 2, "2026-08-10 10:02:00", "operator"),
    ])
    starts = [m.t_start for m in selection.approved_moments(bare, "s", min_rating=2)]
    assert starts == sorted(starts)


def test_a_stream_with_no_ratings_yields_nothing(bare):
    assert selection.approved_moments(bare, "s", min_rating=2) == []


def test_same_second_ratings_resolve_deterministically(bare):
    """`rated_at` is whole seconds, so two ratings can tie. The later row id
    wins, which is the later keypress."""
    hand_rows(bare, "s", [
        (1, 0, 100.0, 140.0, 2, "2026-08-10 10:00:00", "operator"),
        (2, 1, 100.0, 140.0, 0, "2026-08-10 10:00:00", "operator"),
    ])
    assert selection.approved_moments(bare, "s", min_rating=2) == []
