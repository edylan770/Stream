"""The score stage against the fixture, whose regions and markers are known.

The long fixture is used throughout: at 600 s it is twice the 300 s rolling
baseline window, so §6.2's z-score actually rolls. On the 60 s fixture every
sample is baselined against the whole signal, which is the global normalization
§6.2 exists to avoid.
"""

from __future__ import annotations

import json
import shutil

import pytest

from clipforge import cli, config, db, paths
from clipforge.pipeline import runner
from clipforge.score import runner as score_runner
from tests.fixtures.make_fixture import FixtureSpec, generate, load_manifest


@pytest.fixture(scope="session")
def long_fixture():
    """600 s: long enough for the rolling baseline to mean something."""
    directory = generate(FixtureSpec(name="long", duration_s=600.0))
    return directory, load_manifest(directory)


@pytest.fixture(scope="session")
def scored(tmp_path_factory, long_fixture):
    directory, manifest = long_fixture
    root = tmp_path_factory.mktemp("score")
    cfg = config.load(overrides=[f"paths.data_root={(root / 'data').as_posix()}"])
    conn = db.open_db(cfg.db_path)

    anchor = json.loads((directory / manifest["anchor"]).read_text())["record_start_epoch_ms"]
    conn.execute(
        "INSERT INTO streams (id, date, master_path, marker_time_base, record_start_epoch_ms) "
        "VALUES ('fx', '2026-08-14', ?, 'epoch', ?)",
        (str(directory / manifest["video"]), anchor),
    )
    sp = paths.StreamPaths(cfg.data_root, "fx").ensure()
    shutil.copy2(directory / manifest["markers_file"], sp.markers_jsonl)

    engine = runner.Runner(cfg=cfg, conn=conn, stream_id="fx", log=lambda *_: None)
    engine.mark_external_done("register_stream")
    results = engine.execute(engine.plan())
    yield cfg, conn, engine, manifest, results
    conn.close()


def candidates(conn, current_only=True):
    where = "AND is_current = 1" if current_only else ""
    return conn.execute(
        f"SELECT * FROM candidates WHERE stream_id='fx' {where} ORDER BY score_combined DESC"
    ).fetchall()


def _rescore_with(cfg, conn, stream_id="fx"):
    """Force a score pass under `cfg`.

    Used to restore the session fixture's state after a test scores under a
    different configuration — the `scored` fixture is session-scoped, so
    leaving a foreign config_version current would break the next test to run.
    """
    engine = runner.Runner(cfg=cfg, conn=conn, stream_id=stream_id,
                           force={"score"}, log=lambda *_: None)
    return engine.execute([d for d in engine.plan() if d.stage == "score"])


# --------------------------------------------------------------------------
# the stage runs and produces sane candidates
# --------------------------------------------------------------------------


def test_pipeline_completes(scored):
    cfg, conn, engine, manifest, results = scored
    assert all(r.status == "done" for r in results), [r.error for r in results]


def test_candidates_exist(scored):
    cfg, conn, engine, manifest, _ = scored
    assert len(candidates(conn)) > 0


def test_candidate_rate_lands_in_the_spec_range(scored, manifest=None):
    """§6.7: 80-150 per 3 hours, i.e. 27-50 per hour."""
    cfg, conn, engine, manifest, _ = scored
    rows = candidates(conn)
    per_hour = len(rows) / (manifest["duration_s"] / 3600.0)
    low, high = cfg.get("score.peak.target_candidates_per_hour")
    assert low <= per_hour <= high, f"{per_hour:.1f}/hour outside {low}-{high}"


def test_windows_respect_the_configured_bounds(scored):
    cfg, conn, engine, manifest, _ = scored
    lo = cfg.get("score.window.min_window_s")
    hi = cfg.get("score.window.max_window_s")
    for row in candidates(conn):
        length = row["t_end"] - row["t_start"]
        assert lo - 0.2 <= length <= hi + 0.2, f"{length:.1f}s window"


def test_windows_are_not_all_clamped_to_the_maximum(scored):
    """The pedestal bug: with base-relative thresholds every window came out at
    max_window_s because expansion never terminated on the marker plateau."""
    cfg, conn, engine, manifest, _ = scored
    hi = cfg.get("score.window.max_window_s")
    lengths = [row["t_end"] - row["t_start"] for row in candidates(conn)]
    assert not all(abs(length - hi) < 0.5 for length in lengths)


def test_every_window_contains_its_peak(scored):
    cfg, conn, engine, manifest, _ = scored
    for row in candidates(conn):
        assert row["t_start"] <= row["t_peak"] <= row["t_end"]


def test_windows_stay_inside_the_stream(scored):
    cfg, conn, engine, manifest, _ = scored
    for row in candidates(conn):
        assert row["t_start"] >= 0.0
        assert row["t_end"] <= manifest["duration_s"] + 0.2


# --------------------------------------------------------------------------
# candidates land on real content
# --------------------------------------------------------------------------


def test_peaks_land_inside_known_loud_regions(scored):
    """The whole chain, end to end: authored audio at a known level becomes a
    candidate at that timestamp."""
    cfg, conn, engine, manifest, _ = scored
    regions = [(r["t_start"], r["t_end"]) for r in manifest["regions"]]

    for row in candidates(conn):
        inside = any(
            start - 2.0 <= row["t_peak"] <= end + 2.0 for start, end in regions
        )
        assert inside, f"peak at {row['t_peak']:.1f}s is not near any authored region"


def test_marker_moments_fall_inside_their_candidate_windows(scored):
    """§17's stated tuning metric for marker_retro_offset_s: 'fraction of
    marker-anchored windows where the moment is inside the window'.

    Every candidate that a marker plateau contributed to should contain the
    moment the operator was reacting to — that is what the asymmetric plateau
    exists to guarantee, where a point estimate at t-20 would not.
    """
    cfg, conn, engine, manifest, _ = scored
    rows = candidates(conn)

    covered = 0
    for marker in manifest["markers"]:
        moment = marker["intended_event_t"]
        if any(row["t_start"] <= moment <= row["t_end"] for row in rows):
            covered += 1

    # Not every marker produces a candidate — §6.7's rate deliberately keeps
    # only the strongest moments. But of the candidates that exist, each should
    # be anchored on a real marked moment.
    assert covered >= len(rows), (
        f"only {covered} marker moments covered by {len(rows)} candidates"
    )


def test_marker_contribution_reaches_full_weight(scored):
    """Unit-peak kernels: marker_definite's weight is 3.0, so its contribution
    at a marked moment is exactly 3.0 — not 0.01 as a unit-area Gaussian would
    give."""
    cfg, conn, engine, manifest, _ = scored
    weight = cfg.profile.weights["marker_definite"]
    top = json.loads(candidates(conn)[0]["contributing_signals"])
    assert top["marker_definite"] == pytest.approx(weight, abs=0.01)


# --------------------------------------------------------------------------
# explainability (§14's tuning input)
# --------------------------------------------------------------------------


def test_contributions_sum_to_the_recorded_raw_total(scored):
    """A breakdown that does not add up cannot be tuned against."""
    from clipforge.score.features import TOTAL_RAW, TOTAL_SMOOTHED

    cfg, conn, engine, manifest, _ = scored
    for row in candidates(conn):
        payload = json.loads(row["contributing_signals"])
        signals_only = {
            k: v for k, v in payload.items() if not k.startswith("_")
        }
        assert sum(signals_only.values()) == pytest.approx(payload[TOTAL_RAW], abs=1e-4)
        assert TOTAL_SMOOTHED in payload


def test_the_stored_score_is_the_smoothed_value(scored):
    """It has to be: that is the value peak-finding ranked on, so anything else
    would make the ordering disagree with the scores."""
    from clipforge.score.features import TOTAL_SMOOTHED

    cfg, conn, engine, manifest, _ = scored
    for row in candidates(conn):
        payload = json.loads(row["contributing_signals"])
        assert row["score_entertainment"] == pytest.approx(payload[TOTAL_SMOOTHED], abs=1e-4)


def test_breakdown_carries_the_decibels_behind_the_z_score(scored):
    """'mic_rms z = 2.8' only means something next to 'which is -18 dB against
    a -34 dB baseline'."""
    cfg, conn, engine, manifest, _ = scored
    payload = json.loads(candidates(conn)[0]["contributing_signals"])
    assert "_mic_rms_db" in payload
    assert "_mic_rms_baseline_db" in payload
    assert payload["_mic_rms_db"] > payload["_mic_rms_baseline_db"]


def test_feature_vector_holds_exactly_the_declared_schema(scored):
    """A9's value is that a Phase 1 vector and a Phase 3 vector have identical
    shape, so this field carries the schema's keys and nothing else."""
    cfg, conn, engine, manifest, _ = scored
    expected = set(cfg.feature_schema.keys)
    for row in candidates(conn):
        assert set(json.loads(row["feature_vector"])) == expected


def test_the_vector_holds_signals_the_profile_does_not_weight(scored):
    """A9: "Feature vectors are logged in full, always, EVEN FOR SIGNALS NOT
    CURRENTLY WEIGHTED. They cannot be reconstructed retroactively and they are
    the input to all future tuning."

    This test used to assert the opposite — that the vector held exactly the
    three signals the naive profile weights — because `build_tracks` only
    loaded what the profile named. That was invisible while Phase 1 had three
    signals and the profile weighted all three; `game_rms` and `party_rms` are
    extracted for any multi-track recording and weighted by nothing, so they
    were being dropped on the floor.
    """
    from clipforge import signals

    cfg, conn, engine, manifest, _ = scored
    vector = json.loads(candidates(conn)[0]["feature_vector"])
    computed = {k for k, v in vector.items() if v is not None}

    stored = set(signals.kinds(conn, "fx"))
    weighted = set(cfg.profile.weights)

    unweighted = stored - weighted
    assert unweighted, "the fixture should extract a signal the profile ignores"
    assert unweighted <= computed, f"A9 requires {sorted(unweighted)} in the vector"
    assert stored <= computed

    # Everything else — the ~25 signals no phase computes yet — is still null,
    # which is what makes a Phase 1 vector and a Phase 3 vector comparable.
    assert not computed - stored - weighted
    assert len(vector) == len(cfg.feature_schema.keys)


def test_unweighted_signals_do_not_appear_in_the_breakdown(scored):
    """The vector answers "what was observed"; contributing_signals answers
    "what moved the score". A row of 0.000 in the review UI's `?` panel is noise
    in the one place that explains the number beside it."""
    cfg, conn, engine, manifest, _ = scored
    contributions = json.loads(candidates(conn)[0]["contributing_signals"])
    named = {k for k in contributions if not k.startswith("_")}
    assert named <= set(cfg.profile.weights)


def test_unweighted_signals_do_not_move_the_score(scored):
    """Weight 0 contributes weight * value = 0, so loading them for A9 cannot
    change what scored."""
    cfg, conn, engine, manifest, _ = scored
    for row in candidates(conn):
        contributions = json.loads(row["contributing_signals"])
        total = sum(v for k, v in contributions.items() if not k.startswith("_"))
        assert total == pytest.approx(contributions["_total_raw"], abs=1e-6)


def test_feature_schema_version_is_recorded(scored):
    cfg, conn, engine, manifest, _ = scored
    for row in candidates(conn):
        assert row["feature_schema_version"] == cfg.feature_schema.version


def test_phase_1_scores_are_recorded_honestly(scored):
    """One profile exists, so gameplay is 0.0 rather than a faked §6.5 product
    and combined mirrors entertainment."""
    cfg, conn, engine, manifest, _ = scored
    for row in candidates(conn):
        assert row["score_gameplay"] == 0.0
        assert row["score_combined"] == pytest.approx(row["score_entertainment"])
        assert row["profile"] == "naive"


# --------------------------------------------------------------------------
# generations and re-scoring (§6.1)
# --------------------------------------------------------------------------


def test_rescoring_with_the_same_config_replaces_in_place(scored):
    """One generation per distinct scoring configuration, so the numbers stay
    meaningful when comparing weight experiments."""
    cfg, conn, engine, manifest, _ = scored
    before = {row["id"] for row in candidates(conn)}
    generation = candidates(conn)[0]["generation"]

    forced = runner.Runner(cfg=cfg, conn=conn, stream_id="fx",
                           force={"score"}, log=lambda *_: None)
    forced.execute([d for d in forced.plan() if d.stage == "score"])

    after = candidates(conn)
    assert after[0]["generation"] == generation
    assert {row["id"] for row in after} != before      # rows were rewritten
    assert len({row["generation"] for row in candidates(conn, current_only=False)}) == 1


def test_changing_a_weight_creates_a_new_generation_and_keeps_the_old(scored):
    cfg, conn, engine, manifest, _ = scored
    original = candidates(conn)[0]["generation"]

    retuned = config.load(overrides=[
        f"paths.data_root={cfg.data_root.as_posix()}",
        "score.markers.pre_s=20.0",
    ])
    assert retuned.version != cfg.version

    engine2 = runner.Runner(cfg=retuned, conn=conn, stream_id="fx",
                            force={"score"}, log=lambda *_: None)
    engine2.execute([d for d in engine2.plan() if d.stage == "score"])

    current = candidates(conn)
    assert current[0]["generation"] == original + 1
    assert current[0]["config_version"] == retuned.version
    # The superseded generation is still there, untouched.
    old = conn.execute(
        "SELECT count(*) FROM candidates WHERE stream_id='fx' AND generation = ?", (original,)
    ).fetchone()[0]
    assert old > 0

    _rescore_with(cfg, conn)          # restore the session fixture's config


def test_operator_ratings_survive_a_rescore_and_are_carried_forward(scored):
    """§13.2 calls ratings irreplaceable and §6.1 expects re-scoring to be
    routine. Those only coexist if a re-score does not cost a review session."""
    cfg, conn, engine, manifest, _ = scored
    target = candidates(conn)[0]
    conn.execute(
        "INSERT OR REPLACE INTO ratings (candidate_id, rating, note) VALUES (?, 2, 'keep this')",
        (target["id"],),
    )

    retuned = config.load(overrides=[
        f"paths.data_root={cfg.data_root.as_posix()}",
        "score.smoothing_sigma_s=2.5",
    ])
    engine2 = runner.Runner(cfg=retuned, conn=conn, stream_id="fx",
                            force={"score"}, log=lambda *_: None)
    engine2.execute([d for d in engine2.plan() if d.stage == "score"])

    # The operator's own row is untouched on the old candidate...
    original = conn.execute(
        "SELECT rating, rating_source FROM ratings WHERE candidate_id = ?", (target["id"],)
    ).fetchone()
    assert (original["rating"], original["rating_source"]) == (2, "operator")

    # ...and a copy landed on the new generation, tagged so §14 cannot
    # double-count it as fresh evidence.
    inherited = conn.execute(
        """
        SELECT r.rating, r.rating_source, r.inherited_from, r.note
          FROM ratings r JOIN candidates c ON c.id = r.candidate_id
         WHERE c.stream_id = 'fx' AND c.is_current = 1 AND r.rating_source = 'inherited'
        """
    ).fetchall()
    assert len(inherited) == 1
    assert inherited[0]["rating"] == 2
    assert inherited[0]["note"] == "keep this"
    assert inherited[0]["inherited_from"] == target["id"]

    # Restore the session fixture's config: `scored` is session-scoped, so a
    # test that leaves a different config_version current would break whichever
    # test happens to run next.
    _rescore_with(cfg, conn)


def test_only_one_generation_is_current(scored):
    cfg, conn, engine, manifest, _ = scored
    generations = conn.execute(
        "SELECT DISTINCT generation FROM candidates WHERE stream_id='fx' AND is_current=1"
    ).fetchall()
    assert len(generations) == 1


def test_verify_notices_candidates_from_a_different_config(scored):
    from clipforge.pipeline import stages

    cfg, conn, engine, manifest, _ = scored
    _rescore_with(cfg, conn)          # establish the precondition explicitly

    ok, _ = stages.get("score").verify(engine.ctx)
    assert ok

    retuned = config.load(overrides=[
        f"paths.data_root={cfg.data_root.as_posix()}", "score.spacing.factor=0.4",
    ])
    engine2 = runner.Runner(cfg=retuned, conn=conn, stream_id="fx", log=lambda *_: None)
    ok, why = stages.get("score").verify(engine2.ctx)
    assert not ok
    assert "scored by" in why


# --------------------------------------------------------------------------
# the command
# --------------------------------------------------------------------------


def test_score_command_forces_rather_than_skipping(scored, capsys):
    """§6.1 calls scoring infinitely rerunnable, so `clipforge score` must not
    report 'up to date' the way `run` correctly does."""
    cfg, conn, engine, manifest, _ = scored
    conn.commit()

    assert cli.main(
        ["score", "fx", "--set", f"paths.data_root={cfg.data_root.as_posix()}"]
    ) == 0
    out = capsys.readouterr().out
    assert "candidates" in out
    assert "up to date" not in out


def test_score_command_lists_contributors(scored, capsys):
    cfg, conn, engine, manifest, _ = scored
    conn.commit()

    assert cli.main(
        ["score", "fx", "--list", "--set", f"paths.data_root={cfg.data_root.as_posix()}"]
    ) == 0
    out = capsys.readouterr().out
    assert "top contributors" in out
    assert "marker_definite" in out or "mic_rms" in out


# --------------------------------------------------------------------------
# the degraded case: no markers at all
# --------------------------------------------------------------------------


def test_a_stream_with_no_markers_still_scores_on_audio(tmp_path, long_fixture):
    """Testvid's situation: mic_rms alone. The marker weights contribute zero
    rather than failing."""
    directory, manifest = long_fixture
    cfg = config.load(overrides=[f"paths.data_root={(tmp_path / 'data').as_posix()}"])
    conn = db.open_db(cfg.db_path)
    conn.execute(
        "INSERT INTO streams (id, date, master_path, marker_time_base) "
        "VALUES ('bare', '2026-08-14', ?, 'vod')",
        (str(directory / manifest["video"]),),
    )
    engine = runner.Runner(cfg=cfg, conn=conn, stream_id="bare", log=lambda *_: None)
    engine.mark_external_done("register_stream")
    results = engine.execute(engine.plan())
    assert all(r.status == "done" for r in results), [r.error for r in results]

    rows = conn.execute(
        "SELECT contributing_signals FROM candidates WHERE stream_id='bare' AND is_current=1"
    ).fetchall()
    assert rows
    payload = json.loads(rows[0]["contributing_signals"])
    assert payload["marker_definite"] == 0.0
    assert payload["mic_rms"] != 0.0
    conn.close()
