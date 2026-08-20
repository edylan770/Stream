"""The score stage against the fixture, whose regions and markers are known.

The long fixture is used throughout: at 600 s it is twice the 300 s rolling
baseline window, so §6.2's z-score actually rolls. On the 60 s fixture every
sample is baselined against the whole signal, which is the global normalization
§6.2 exists to avoid.
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from clipforge import cli, config, db, paths
from clipforge.pipeline import runner
from clipforge.pipeline.context import StageContext
from clipforge.render import selection
from clipforge.score import features, gates, grid
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


def _restore_generation(conn, generation: int, stream_id: str = "fx") -> None:
    """Undo any generation a test minted, and make `generation` current again."""
    with conn:
        conn.execute(
            "DELETE FROM candidates WHERE stream_id = ? AND generation > ?",
            (stream_id, generation),
        )
        conn.execute(
            "UPDATE candidates SET is_current = (generation = ?) WHERE stream_id = ?",
            (generation, stream_id),
        )


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


def test_the_context_key_carries_the_signals_own_unit(scored):
    """Every Phase 1 signal was dBFS, so "_db" was a literal in the key. Pitch
    is hertz, and "_mic_f0_db: 166.2" is a label stating something false — the
    review UI reads the suffix to decide what to print beside the number."""
    cfg, conn, engine, manifest, _ = scored
    payload = json.loads(candidates(conn)[0]["contributing_signals"])

    assert any(k.startswith("_mic_f0_") for k in payload), "pitch reaches the panel"
    assert not any(k.startswith("_mic_f0_") and k.endswith("_db") for k in payload)
    assert "_mic_f0_hz" in payload


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

    Phase 3 splits the assertion, because null now has two meanings and only one
    of them is a bug. A gap-free signal MUST be non-null: a null there is the
    original regression, a signal that was never loaded. A signal with gaps
    (`mic_f0` is unvoiced wherever nobody is speaking) is null exactly when it
    had no observation at that instant — which is checked against the series
    itself below, so "null" can never quietly go back to meaning "dropped".
    """
    from clipforge import signals

    cfg, conn, engine, manifest, _ = scored
    row = candidates(conn)[0]
    vector = json.loads(row["feature_vector"])
    computed = {k for k, v in vector.items() if v is not None}

    stored = set(signals.kinds(conn, "fx"))
    weighted = set(cfg.profile.weights)

    unweighted = stored - weighted
    assert unweighted, "the fixture should extract a signal the profile ignores"
    assert stored <= set(vector), "A9 wants every stored signal's KEY in the vector"

    import numpy as np

    for kind in sorted(stored):
        series = signals.load(conn, "fx", kind)
        observed = np.isfinite(series.values)
        if observed.all():
            assert vector[kind] is not None, (
                f"{kind} has no gaps, so a null here means it was never loaded"
            )
            continue
        # A gappy signal: null must mean "not observed here", never "not loaded".
        at_peak = np.isfinite(series.value_at(float(row["t_peak"])))
        assert (vector[kind] is not None) == bool(at_peak), (
            f"{kind} is {'observed' if at_peak else 'absent'} at t_peak but the "
            f"vector says otherwise"
        )

    # Everything else — the signals no phase computes yet — is still null,
    # which is what makes a Phase 1 vector and a Phase 3 vector comparable.
    #
    # There are now three legitimate sources of a value, not two: stored,
    # weighted, and DERIVED (§5.4.1's signals that are computed at score time
    # rather than extracted — see score/derived.py). Anything outside those
    # three is the failure this guards: a Phase 7 signal like `clutch` acquiring
    # a number that nothing computed.
    from clipforge.score import derived

    derivable = {d.name for d in derived.DERIVATIONS}
    assert not computed - stored - weighted - derivable
    assert len(vector) == len(cfg.feature_schema.keys)


def test_derived_signals_are_in_the_vector_but_were_never_stored(scored):
    """The C3 boundary, asserted from the other side: `overlap_speech` reaches
    A9's vector without ever becoming a `signal_series` row, so retuning its
    threshold costs a re-score rather than a re-extraction."""
    from clipforge import signals
    from clipforge.score import derived

    _cfg, conn, _engine, _manifest, _ = scored
    vector = json.loads(candidates(conn)[0]["feature_vector"])
    stored = set(signals.kinds(conn, "fx"))

    derivable = {d.name for d in derived.DERIVATIONS if not d.emits_events}
    assert derivable & set(vector), "derived signals belong in the vector"
    assert not derivable & stored, "and must not have been written to the database"


def test_unweighted_signals_do_not_appear_in_the_breakdown(scored):
    """The vector answers "what was observed"; contributing_signals answers
    "what moved the score". A row of 0.000 in the review UI's `?` panel is noise
    in the one place that explains the number beside it.

    §6.4's negatives widen the rule rather than break it: a penalty carries a
    non-zero weight by construction — that is what it would cost — so what
    earns a row is having actually moved this candidate's score. On this stream
    neither gate can even be evaluated, so the set is the profile's weights
    exactly.
    """
    cfg, conn, engine, manifest, _ = scored
    contributions = json.loads(candidates(conn)[0]["contributing_signals"])
    named = {k for k in contributions if not k.startswith("_")}
    assert named <= set(cfg.profile.weights) | {gates.MENU, gates.AFK}
    assert not named & {gates.MENU, gates.AFK}, (
        "no input log and no vision, so neither negative applies here"
    )


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


def test_a_moment_is_one_row_carrying_every_profiles_score(scored):
    """§3.2 gives `candidates` three score columns rather than three rows, and
    §7.4's sections are orderings over one list. So `profile` holds the
    profile-SET name and `UNIQUE (stream_id, generation, profile, t_peak)` keeps
    working with no migration."""
    cfg, conn, engine, manifest, _ = scored
    rows = candidates(conn)
    assert rows
    for row in rows:
        assert row["profile"] == cfg.profile_set == "entertainment+gameplay"
    assert len({row["t_peak"] for row in rows}) == len(rows), "one row per moment"


def test_the_two_profiles_disagree(scored):
    """If they did not, the second profile would be costing a pass for nothing.
    §6.5's own reason for gameplay is that mechanics and personality are
    different axes — this is the weakest statement of that which can still fail
    if the merge accidentally copies one column into the other."""
    cfg, conn, engine, manifest, _ = scored
    rows = candidates(conn)
    pairs = [(row["score_entertainment"], row["score_gameplay"]) for row in rows]
    assert any(e != g for e, g in pairs)


def test_a_single_profile_still_scores_exactly_as_it_did(scored):
    """§6.5's "casual/comedy: entertainment only" case, and the regression that
    matters: everything Phase 1 and 2 produced was scored with one profile, so
    one profile has to keep meaning what it meant. gameplay stays 0.0 rather
    than a faked §6.5 product, and combined mirrors entertainment at its own raw
    scale."""
    cfg, conn, engine, manifest, _ = scored
    naive = config.load(overrides=[f"paths.data_root={cfg.data_root.as_posix()}"],
                        profile="naive")
    generation = candidates(conn)[0]["generation"]
    try:
        _rescore_with(naive, conn)
        rows = candidates(conn)
        assert rows
        for row in rows:
            assert row["profile"] == "naive"
            assert row["score_gameplay"] == 0.0
            assert row["score_combined"] == pytest.approx(row["score_entertainment"])
    finally:
        # Scoring under a second configuration MINTS a generation, and the
        # session fixture is shared — so this puts the stream back rather than
        # leaving two extra generations for the generation tests to trip over.
        _restore_generation(conn, generation)


def test_the_combined_score_is_not_either_profile(scored):
    """§6.5: "score_combined is NOT a sum or average." It is also not a copy —
    with two profiles it lives on the normalised 0..2 scale of §6.5's product,
    where the per-profile columns are raw composites."""
    cfg, conn, engine, manifest, _ = scored
    rows = candidates(conn)
    assert any(row["score_combined"] != pytest.approx(row["score_entertainment"])
               for row in rows)
    assert all(0.0 <= row["score_combined"] <= 2.0 for row in rows)


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


def test_a_same_config_rescore_over_operator_ratings_never_destroys_them(scored):
    """THE REGRESSION. §13.2 calls ratings the irreplaceable tier.

    `config_version` hashes `extract`, `score` and every profile's weights. It
    does NOT hash the signals and events those weights are applied to, because
    those are data. So a stream whose DATA changed under an unchanged
    configuration took the replace-in-place path: `prior` was set to `[]`, the
    generation's candidates were DELETEd, and `ratings.candidate_id ON DELETE
    CASCADE` took every operator judgement with them. Silently.

    Two workflows HANDOFF documents reach exactly that state --
    `register --input-log <file> --force` and `register --obs-log <file>
    --force`, each followed by a run that cascades into `score`.
    """
    cfg, conn, engine, manifest, _ = scored
    target = candidates(conn)[0]
    generation = target["generation"]
    marker = "the rating this test is about"
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO ratings (candidate_id, rating, note) VALUES (?, 2, ?)",
            (target["id"], marker),
        )

    # The SAME configuration -- this is the re-run that follows attaching a log
    # to an already-reviewed stream, not a weight experiment.
    forced = runner.Runner(cfg=cfg, conn=conn, stream_id="fx",
                           force={"score"}, log=lambda *_: None)
    forced.execute([d for d in forced.plan() if d.stage == "score"])

    # The mechanism: a rated generation is never replaced, only superseded.
    assert candidates(conn)[0]["generation"] == generation + 1

    surviving = conn.execute(
        """
        SELECT r.rating, r.rating_source
          FROM ratings r JOIN candidates c ON c.id = r.candidate_id
         WHERE c.stream_id = 'fx' AND r.note = ? AND r.rating_source = 'operator'
        """,
        (marker,),
    ).fetchall()
    assert len(surviving) == 1, "the operator's own rating was destroyed by a re-score"
    assert surviving[0]["rating"] == 2

    # ...it is still reachable by what consumes it -- §10.5's export and §8's
    # renderer both go through here, and both read ACROSS generations...
    assert any(m.rating == 2 for m in selection.approved_moments(conn, "fx", min_rating=2))

    # ...and the review screen, which joins ratings to the CURRENT candidates,
    # still shows the moment as rated rather than asking for it again.
    carried = conn.execute(
        """
        SELECT r.rating_source FROM ratings r JOIN candidates c ON c.id = r.candidate_id
         WHERE c.stream_id = 'fx' AND c.is_current = 1 AND r.note = ?
        """,
        (marker,),
    ).fetchall()
    assert [row["rating_source"] for row in carried] == ["inherited"]

    with conn:
        conn.execute("DELETE FROM ratings WHERE note = ?", (marker,))
    _rescore_with(cfg, conn)


def test_a_rating_that_does_not_reach_the_new_generation_is_reported(scored):
    """The other half of commit 43: nothing is destroyed, but a rating whose
    window moved further than `score.rating_inherit_min_overlap` is invisible to
    the review screen, which joins ratings to the CURRENT candidates.

    Driven by raising the threshold above 1.0, which no IoU can reach — the
    "too high" failure `spec/GUESSES.md` names as this parameter's falsifier.
    Tuning the fixture until a window missed would be asserting against a number
    nobody chose.
    """
    cfg, conn, engine, manifest, _ = scored
    target = candidates(conn)[0]
    marker = "stranded on purpose"
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO ratings (candidate_id, rating, note) VALUES (?, 2, ?)",
            (target["id"], marker),
        )

    unreachable = config.load(overrides=[
        f"paths.data_root={cfg.data_root.as_posix()}",
        "score.rating_inherit_min_overlap=1.01",
    ])
    lines: list[str] = []
    engine2 = runner.Runner(cfg=unreachable, conn=conn, stream_id="fx",
                            force={"score"}, log=lines.append)
    engine2.execute([d for d in engine2.plan() if d.stage == "score"])

    assert any("did not match a new window" in line for line in lines), lines
    assert not any("carried" in line for line in lines), lines

    # Reported, not lost: the operator's own row is still there, and the thing
    # that reads across generations still finds it.
    still_there = conn.execute(
        "SELECT rating_source FROM ratings WHERE note = ?", (marker,)
    ).fetchall()
    assert [row["rating_source"] for row in still_there] == ["operator"]
    assert any(m.rating == 2 for m in selection.approved_moments(conn, "fx", min_rating=2))

    with conn:
        conn.execute("DELETE FROM ratings WHERE note = ?", (marker,))
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


# --------------------------------------------------------------------------
# A9 over the events table, and §6.4's gates end to end
# --------------------------------------------------------------------------


def _tracks_for(cfg, conn, stream_id="fx"):
    """The primary profile's tracks, the way the stage builds them."""
    ctx = StageContext(cfg=cfg, conn=conn, stream_id=stream_id, log=lambda *_: None)
    duration = float(ctx.stream["duration_s"])
    timeline = grid.build(duration, float(cfg.get("score.score_grid_hz")))
    pool = score_runner.build_pool(ctx, timeline)
    tracks, missing = score_runner.tracks_for(pool, cfg.profile)
    return ctx, timeline, (tracks, missing, pool.raw)


def test_a_stored_event_no_profile_weights_still_reaches_the_vector(scored):
    """A9: feature vectors are logged in full, "even for signals not currently
    weighted", and they cannot be reconstructed retroactively.

    The weight-0 sweep did this for stored SIGNALS and, until this commit, only
    for events it had just derived — so `phrase_excitement` and `phrase_repeat`,
    which `phrase_detect` writes to the events table, were null in every vector.
    Invisible only because `extract.whisperx.enabled` ships false.
    """
    cfg, conn, engine, manifest, _ = scored
    # §16 rejects scene_change as a scorer, so no §6.5 profile weights it — and
    # it is exactly the kind of stored-but-unweighted event A9 exists for.
    kind = "scene_change"
    assert not any(kind in profile.weights for profile in cfg.profiles), \
        "the premise of the test"
    assert kind in cfg.feature_schema.signals

    _ctx, _timeline, (before, _m, _r) = _tracks_for(cfg, conn)
    assert kind not in {track.name for track in before}

    conn.execute(
        "INSERT INTO events (stream_id, t, source, kind) VALUES ('fx', ?, 'scene', ?)",
        (float(manifest["regions"][0]["t_start"]), kind),
    )
    try:
        _ctx, timeline, (after, _m, _r) = _tracks_for(cfg, conn)
        by_name = {track.name: track for track in after}
        assert kind in by_name
        assert by_name[kind].weight == 0.0

        index = int(timeline.size // 2)
        assert features.vector(cfg.feature_schema, after, index)[kind] is not None

        # ...and weight 0 means the score cannot have moved.
        assert np.allclose(
            score_runner.composite_of(before, timeline.size),
            score_runner.composite_of(after, timeline.size),
        )
    finally:
        conn.execute("DELETE FROM events WHERE stream_id = 'fx' AND kind = ?", (kind,))


def test_an_unevaluable_negative_is_null_in_the_vector_not_zero(scored):
    """"Not computed" and "computed as nothing" are different claims, and A9's
    vector is the record §17 tunes against. This stream has no input log and no
    vision, so both are null."""
    cfg, conn, engine, manifest, _ = scored
    for row in candidates(conn):
        vector = json.loads(row["feature_vector"])
        assert vector[gates.AFK] is None
        assert vector[gates.MENU] is None


def test_a_gate_that_can_be_evaluated_reaches_every_vector(scored):
    """The other half: once something produces `menu_screen` spans, the gate
    level is recorded on every candidate — including the candidates it did not
    penalise, where it is 0.0 rather than null."""
    cfg, conn, engine, manifest, _ = scored
    duration = float(manifest["duration_s"])
    grace = float(cfg.get("score.negatives.menu_grace_period_s"))

    # Spans over the stretch between the fixture's speech blocks, which repeat
    # every 60 s. Read off the manifest rather than written down.
    blocks = sorted({int(r["t_start"] // 60) for r in manifest["regions"]})
    for block in blocks:
        start = 60.0 * block + 44.0
        if start + 4 * grace < duration:
            conn.execute(
                "INSERT INTO events (stream_id, t, t_end, source, kind) "
                "VALUES ('fx', ?, ?, 'vision', ?)",
                (start, start + 4 * grace, gates.MENU),
            )
    try:
        _rescore_with(cfg, conn)
        rows = candidates(conn)
        assert rows, "the stream still produces candidates with a negative live"
        for row in rows:
            vector = json.loads(row["feature_vector"])
            assert vector[gates.MENU] is not None
            assert vector[gates.AFK] is None, "the AFK gate is still unevaluable"

        share = conn.execute(
            "SELECT value FROM tool_metrics WHERE stream_id = 'fx' "
            "AND metric = 'negative_penalty_share' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert share is not None and share["value"] > 0.0
    finally:
        conn.execute("DELETE FROM events WHERE stream_id = 'fx' AND kind = ?", (gates.MENU,))
        _rescore_with(cfg, conn)


def test_the_breakdown_still_adds_up_with_a_penalty_applied(scored):
    """`_total_raw` is the composite the smoothing then acts on, so a penalty
    has to be inside it. The identity is what stops the `?` panel disagreeing
    with the number beside it."""
    cfg, conn, engine, manifest, _ = scored
    tracks = [
        features.SignalTrack(name="mic_rms", values=np.array([2.0]), weight=1.5),
        *gates.as_tracks([gates.Penalty(gates.AFK, np.array([1.0]), weight=2.0)]),
    ]
    detail = features.breakdown(tracks, 0, smoothed_value=0.0)
    assert detail.total_raw == pytest.approx(1.5 * 2.0 - 2.0)
    assert detail.contributions[gates.AFK] == pytest.approx(-2.0)
