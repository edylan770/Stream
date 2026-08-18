"""§5.4.1's derived signals, computed at score time.

The interesting tests are measurements, not assertions about intent: the speech
fixture's manifest records every utterance per track and the one window where
both people talk, so `overlap_speech` is scored against ground truth rather than
trusted. Where a tolerance is needed it is read from config — the predicted
overlap may overrun the authored one by `vad.hangover_s` and not a millisecond
more, because that is what the hangover is.
"""

from __future__ import annotations

import numpy as np
import pytest

from clipforge import config, db, signals
from clipforge.extract import features
from clipforge.ingest import register
from clipforge.pipeline import runner
from clipforge.pipeline.context import StageContext
from clipforge.score import derived, grid
from clipforge.score import runner as score_runner


@pytest.fixture(scope="module")
def cfg():
    return config.load()


#: Every synthetic case below is built on this timeline, and its arrays are
#: sized from it rather than from a literal — `grid.build` spans the stream
#: inclusively, so a 90 s stream at 10 Hz is 901 samples and not 900.
DURATION_S = 90.0


def timeline_for(cfg, duration_s: float = DURATION_S) -> np.ndarray:
    return grid.build(duration_s, float(cfg.get("score.score_grid_hz")))


# --------------------------------------------------------------------------
# the log-domain conversion (finding 9)
# --------------------------------------------------------------------------


def test_an_octave_is_twelve_semitones():
    out = derived.to_semitones(np.array([100.0, 200.0, 400.0, 50.0]))
    assert out[1] - out[0] == pytest.approx(12.0)
    assert out[2] - out[0] == pytest.approx(24.0)
    assert out[3] - out[0] == pytest.approx(-12.0)


def test_the_semitone_reference_cancels_out():
    """Only differences are ever used, so the reference is arbitrary — and a
    test that pinned it would be testing the constant rather than the maths."""
    hz = np.array([120.0, 180.0])
    a = derived.to_semitones(hz)
    b = 12 * np.log2(hz / 440.0)
    assert (a[1] - a[0]) == pytest.approx(b[1] - b[0])


def test_unvoiced_stays_unvoiced_through_the_conversion():
    out = derived.to_semitones(np.array([np.nan, 150.0, 0.0, -1.0]))
    assert np.isnan(out[0]) and np.isnan(out[2]) and np.isnan(out[3])
    assert np.isfinite(out[1])


def test_a_linear_deviation_understates_what_a_log_one_finds():
    """Why this conversion exists. Two speakers each swinging an octave have the
    same prosodic range; in hertz the low one looks half as expressive."""
    low = np.array([80.0, 160.0] * 20)
    high = np.array([240.0, 480.0] * 20)

    assert np.std(high) > 2 * np.std(low)                       # hertz: wrong
    assert np.std(derived.to_semitones(high)) == pytest.approx(
        np.std(derived.to_semitones(low))                       # semitones: right
    )


# --------------------------------------------------------------------------
# the primitives
# --------------------------------------------------------------------------


def test_rolling_std_needs_enough_observations():
    """A "prosodic range" over two voiced frames is noise wearing the name of a
    signal, and zero would claim the speaker was monotone."""
    values = np.full(50, np.nan)
    values[:3] = [1.0, 2.0, 3.0]
    out = derived.rolling_std(values, window=11, min_observations=8)
    assert np.isnan(out).all()

    out = derived.rolling_std(values, window=11, min_observations=2)
    assert np.isfinite(out[:3]).any()


def test_rolling_max_looks_backwards_only():
    """§5.4.1's silence is "within 2 s OF high activity" — activity that has not
    happened yet cannot explain a silence now, so the window trails."""
    values = np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0])
    out = derived.rolling_max(values, window=3)
    assert out[1] == 0.0, "the spike at index 2 must not reach backwards"
    assert out[2] == 5.0
    assert out[4] == 5.0
    assert out[5] == 0.0, "and it must expire"


def test_hold_bridges_the_gaps_between_words():
    active = np.array([True, False, False, True, False, False, False])
    out = derived.hold(active, samples=2)
    assert list(out) == [True, True, True, True, True, True, False]


def test_short_runs_are_dropped():
    active = np.array([True, False, True, True, True, False])
    out = derived.drop_short_runs(active, samples=3)
    assert list(out) == [False, False, True, True, True, False]


def test_a_run_at_the_very_end_is_still_measured():
    assert not derived.drop_short_runs(np.array([False, True, True]), samples=3).any()


def test_ramp_reaches_one_and_resets():
    out = derived.ramp_up(np.array([True] * 4 + [False] + [True]), samples=2)
    assert out[0] == pytest.approx(0.5)
    assert out[1] == pytest.approx(1.0)
    assert out[3] == pytest.approx(1.0)
    assert out[4] == 0.0
    assert out[5] == pytest.approx(0.5), "and starts again from the bottom"


# --------------------------------------------------------------------------
# sudden_silence
# --------------------------------------------------------------------------


def loud_then_quiet(cfg, floor=-50.0, peak=-15.0):
    """Speech from 20 s to 30 s, then nothing. Returns (timeline, mic_rms)."""
    timeline = timeline_for(cfg)
    mic = np.full(timeline.size, floor)
    mic[(timeline >= 20.0) & (timeline < 30.0)] = peak
    return timeline, mic


def test_sudden_silence_fires_after_the_activity_stops(cfg):
    timeline, mic = loud_then_quiet(cfg)
    values, _u, _e = derived.compute({"mic_rms": mic}, timeline, cfg)
    times = timeline[values["sudden_silence"] > 0.5]

    assert times.size > 0
    # Just after the mic goes quiet at t=30, and bounded by the lookback.
    assert times.min() == pytest.approx(30.0, abs=0.5)
    assert times.max() <= 30.0 + float(cfg.get("score.derived.silence.lookback_s")) + 0.5


def test_silence_that_follows_nothing_does_not_fire(cfg):
    """§5.4.1 is specific: silence WITHIN 2 s OF HIGH ACTIVITY. An hour of quiet
    is not a moment, and a detector that fires on it would nominate the dullest
    stretch of every stream."""
    timeline = timeline_for(cfg)
    values, _u, _e = derived.compute(
        {"mic_rms": np.full(timeline.size, -50.0)}, timeline, cfg)
    assert not (values["sudden_silence"] > 0.5).any()


def test_the_drop_is_what_detects_not_the_level(cfg):
    """REGRESSION. The first cut tested `z <= quiet_z` with a negative threshold
    and MEASURED zero firings on every input, because on a real stream SILENCE
    IS THE BASELINE — the operator is quiet most of the time, so z during
    silence is about zero and never meaningfully negative.

    This fixture makes that explicit: speech occupies a ninth of the timeline,
    so the mean sits near the quiet level, and any detector keyed on "below the
    baseline" finds nothing.
    """
    timeline, mic = loud_then_quiet(cfg)
    baseline_samples = grid.window_samples(
        cfg.get("score.rolling_baseline_window_s"),
        float(cfg.get("score.score_grid_hz")))
    z, _b, _s = grid.rolling_zscore(
        mic, baseline_samples, float(cfg.get("score.zscore_std_floor")))

    quiet_z = float(z[timeline.searchsorted(40.0)])   # deep in the later silence
    assert quiet_z > -0.5, "silence sits at the baseline, not below it"

    values, _u, _e = derived.compute({"mic_rms": mic}, timeline, cfg)
    assert (values["sudden_silence"] > 0.5).any(), "and it must fire anyway"


def test_sudden_silence_is_a_ramp_not_a_step(cfg):
    timeline, mic = loud_then_quiet(cfg)
    values, _u, _e = derived.compute({"mic_rms": mic}, timeline, cfg)
    levels = np.unique(values["sudden_silence"])
    assert levels.size > 2, "a square wave would take only two values"
    assert levels.max() == pytest.approx(1.0)
    assert levels.min() == 0.0


# --------------------------------------------------------------------------
# overlap_speech, against the fixture's own ground truth
# --------------------------------------------------------------------------


def stage_ctx(tmp_path, fixture_dir, manifest, full_plan=False):
    cfg = config.load(overrides=[f"paths.data_root={(tmp_path / 'data').as_posix()}"])
    db.open_db(cfg.db_path).close()
    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    result = register.register_stream(
        cfg, conn, register.RegisterSpec(master=fixture_dir / manifest["video"]))
    engine = runner.Runner(cfg=cfg, conn=conn, stream_id=result.stream_id,
                           log=lambda *_: None)
    engine.execute(engine.plan() if full_plan else engine.plan(
        ["register_stream", "probe", "audio_split"]))
    return cfg, conn, StageContext(cfg=cfg, conn=conn, stream_id=result.stream_id,
                                   log=lambda *_: None)


@pytest.fixture(scope="module")
def speech_signals(tmp_path_factory, speech_fixture_dir, speech_manifest):
    """mic_rms and party_rms of the speech fixture, on the scoring grid."""
    cfg, conn, ctx = stage_ctx(
        tmp_path_factory.mktemp("derived"), speech_fixture_dir, speech_manifest)
    try:
        signal_hz = float(cfg.get("extract.signal_hz"))
        series = {
            f"{role}_rms": features.rms_series(
                ctx.paths.audio(role), f"{role}_rms", signal_hz=signal_hz,
                frame_s=float(cfg.get("extract.rms.frame_s")),
                db_floor=float(cfg.get("extract.rms.db_floor")), role=role)
            for role in ("mic", "party")
        }
    finally:
        conn.close()

    timeline = grid.build(float(speech_manifest["duration_s"]),
                          float(cfg.get("score.score_grid_hz")))
    raw = {k: grid.resample(v, timeline) for k, v in series.items()}
    return cfg, timeline, raw


def mask_from(spans, timeline):
    out = np.zeros(timeline.size, dtype=bool)
    for start, end in spans:
        out |= (timeline >= start) & (timeline <= end)
    return out


def mic_utterances(manifest):
    return [u for u in manifest["utterances"] if u["track"] == "mic"]


def test_sudden_silence_fires_after_every_authored_mic_utterance(speech_signals,
                                                                 speech_manifest):
    """MEASURED: 9 of 9. Every line the operator speaks is followed by the
    signal, within a couple of tenths of the line's authored end."""
    cfg, timeline, raw = speech_signals
    values, _u, _e = derived.compute(raw, timeline, cfg)
    fired = values["sudden_silence"] > 0.5
    reach = (float(cfg.get("score.derived.silence.lookback_s"))
             + float(cfg.get("score.derived.silence.ramp_s")))

    for utterance in mic_utterances(speech_manifest):
        after = ((timeline >= utterance["t_end"])
                 & (timeline <= utterance["t_end"] + reach))
        assert fired[after].any(), (
            f"nothing fired after the mic line ending at {utterance['t_end']}s"
        )


def test_sudden_silence_ignores_the_party_track(speech_signals, speech_manifest):
    """§5.4.1 defines this on the MIC. The fixture's party-only lines end at
    24.9, 28.8 and 32.6 s with the operator already silent throughout — nothing
    changed on the mic, so nothing may fire.

    This is what caught a wrong assertion in the first draft of these tests: the
    manifest's longest silence begins where the PARTY stops, and expecting a mic
    signal there was reading the fixture rather than the spec.
    """
    cfg, timeline, raw = speech_signals
    values, _u, _e = derived.compute(raw, timeline, cfg)
    fired = values["sudden_silence"] > 0.5

    mic_ends = [u["t_end"] for u in mic_utterances(speech_manifest)]
    reach = (float(cfg.get("score.derived.silence.lookback_s"))
             + float(cfg.get("score.derived.silence.ramp_s")))

    for utterance in speech_manifest["utterances"]:
        if utterance["track"] != "party":
            continue
        # Only the party ends that are not shadowed by a mic end nearby.
        if any(abs(utterance["t_end"] - end) <= reach for end in mic_ends):
            continue
        after = ((timeline >= utterance["t_end"])
                 & (timeline <= utterance["t_end"] + reach))
        assert not fired[after].any(), (
            f"fired after a PARTY line ending at {utterance['t_end']}s"
        )


def test_overlap_finds_every_authored_overlapping_sample(speech_signals,
                                                         speech_manifest):
    """Recall against the manifest's own `overlap_windows`. C2 is explicit that
    a false negative costs a clip, so this one is not allowed to miss."""
    cfg, timeline, raw = speech_signals
    values, _u, _e = derived.compute(raw, timeline, cfg)
    predicted = values["overlap_speech"] > 0.5
    truth = mask_from(speech_manifest["overlap_windows"], timeline)

    assert truth.any(), "the fixture authors exactly one overlap"
    assert predicted[truth].all(), "every authored overlapping sample must fire"


def test_overlap_overruns_by_the_hangover_and_no_further(speech_signals,
                                                         speech_manifest):
    """The tolerance is read from config rather than written down: the gate is
    ALLOWED to run past the end of speech by `vad.hangover_s`, because that is
    what bridges the gaps between words. Anything beyond that is a real error.
    """
    cfg, timeline, raw = speech_signals
    values, _u, _e = derived.compute(raw, timeline, cfg)
    predicted = timeline[values["overlap_speech"] > 0.5]

    start, end = speech_manifest["overlap_windows"][0]
    hangover = float(cfg.get("score.derived.vad.hangover_s"))
    step = 1.0 / float(cfg.get("score.score_grid_hz"))

    assert predicted.min() == pytest.approx(start, abs=step)
    assert end <= predicted.max() <= end + hangover + step


def test_overlap_never_fires_in_the_authored_silence(speech_signals,
                                                     speech_manifest):
    """A4's twenty seconds of nothing. Two people talking over each other is the
    signal; the fixture's longest silence is the strongest negative it has."""
    cfg, timeline, raw = speech_signals
    values, _u, _e = derived.compute(raw, timeline, cfg)
    longest = max(speech_manifest["silence_windows"], key=lambda w: w[1] - w[0])

    quiet = mask_from([longest], timeline)
    assert not (values["overlap_speech"] > 0.5)[quiet].any()


def test_overlap_is_the_and_of_two_tracks_not_either(speech_signals,
                                                     speech_manifest):
    """§5.4.1: "boolean AND". The fixture has long stretches where exactly one
    person talks, and none of them is an overlap."""
    cfg, timeline, raw = speech_signals
    values, _u, _e = derived.compute(raw, timeline, cfg)
    predicted = values["overlap_speech"] > 0.5

    solo = mask_from(
        [(u["t_start"], u["t_end"]) for u in speech_manifest["utterances"]
         if u["track"] == "party" and u["t_end"] < 40.0],   # before the overlap
        timeline)
    assert solo.any()
    assert not predicted[solo].any()


def test_each_track_gate_finds_all_of_its_own_speech(speech_signals,
                                                     speech_manifest):
    cfg, timeline, raw = speech_signals
    inputs = derived.Inputs(timeline=timeline, grid_hz=10.0, raw=raw, cfg=cfg)
    baseline_samples = grid.window_samples(
        cfg.get("score.rolling_baseline_window_s"), 10.0)

    for role in ("mic", "party"):
        gate = derived.speech_gate(raw[f"{role}_rms"], inputs, baseline_samples)
        truth = mask_from(
            [(u["t_start"], u["t_end"]) for u in speech_manifest["utterances"]
             if u["track"] == role], timeline)
        assert gate[truth].all(), f"{role} gate missed authored speech"


# --------------------------------------------------------------------------
# absent inputs produce nothing, never zeros
# --------------------------------------------------------------------------


def test_a_derivation_without_its_inputs_is_absent(cfg):
    """Not zero. `input_stillness` on a stream with no input log is UNKNOWN, and
    a zero there reads as "perfectly still" — which §6.4's AFK penalty acts on."""
    timeline = timeline_for(cfg)
    values, _u, events = derived.compute(
        {"mic_rms": np.full(timeline.size, -50.0)}, timeline, cfg)

    assert "sudden_silence" in values
    assert "overlap_speech" not in values, "needs party_rms"
    assert "input_stillness" not in values, "needs an input log"
    assert "mic_f0_variance" not in values, "needs pitch"
    assert not events.get("reaction_onset")


def test_missing_inputs_are_reportable(cfg):
    missing = derived.missing_inputs({"mic_rms": np.zeros(10)})
    assert missing["overlap_speech"] == ["party_rms"]
    assert "input_rate" in missing["input_stillness"]
    # A derivation short of an upstream DERIVED signal says so too, rather than
    # naming only the stored arrays it could not reach.
    assert "overlap_speech" in missing["reaction_onset"]
    assert "sudden_silence" not in missing, "it had everything it needed"


def test_the_report_agrees_with_what_actually_ran(cfg):
    """The two walks must not drift: anything `compute` skipped has to appear in
    `missing_inputs`, and anything it produced must not."""
    timeline = timeline_for(cfg)
    raw = {"mic_rms": np.full(timeline.size, -50.0)}

    values, _units, events = derived.compute(dict(raw), timeline, cfg)
    missing = derived.missing_inputs(raw)
    produced = set(values) | {k for k, v in events.items() if v is not None}

    for derivation in derived.DERIVATIONS:
        ran = derivation.name in produced
        assert ran != (derivation.name in missing), derivation.name


def test_stillness_treats_a_coverage_gap_as_unknown_not_still(cfg):
    """THE trap of Phase 3, stated as a test. A stretch where the logger was not
    running has no input rate. Reading that as stillness is how a stream with no
    input log at all would score as motionless from end to end."""
    timeline = timeline_for(cfg)
    rate = np.zeros(timeline.size)
    busy = (timeline >= 10.0) & (timeline < 20.0)
    rate[busy] = 10.0                                  # activity, then nothing
    coverage = np.ones(timeline.size)
    down = (timeline >= 40.0) & (timeline < 60.0)      # the logger was down here
    coverage[down] = 0.0

    values, _u, _e = derived.compute(
        {"input_rate": rate, "input_coverage": coverage}, timeline, cfg)
    stillness = values["input_stillness"]

    assert stillness[down].max() == 0.0, "no coverage, no claim"
    after = (timeline >= 20.0) & (timeline < 24.0)
    assert stillness[after].max() > 0.0, "but real stillness after real input fires"


# --------------------------------------------------------------------------
# reaction_onset (§5.4.3)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def noise_signals(tmp_path_factory, fixture_dir, manifest):
    """mic_rms and party_rms of the noise fixture, on the scoring grid."""
    cfg, conn, ctx = stage_ctx(
        tmp_path_factory.mktemp("derived_noise"), fixture_dir, manifest)
    try:
        signal_hz = float(cfg.get("extract.signal_hz"))
        series = {
            f"{role}_rms": features.rms_series(
                ctx.paths.audio(role), f"{role}_rms", signal_hz=signal_hz,
                frame_s=float(cfg.get("extract.rms.frame_s")),
                db_floor=float(cfg.get("extract.rms.db_floor")), role=role)
            for role in ("mic", "party")
        }
    finally:
        conn.close()

    timeline = grid.build(float(manifest["duration_s"]),
                          float(cfg.get("score.score_grid_hz")))
    return cfg, timeline, {k: grid.resample(v, timeline) for k, v in series.items()}


def region(manifest, prefix, track):
    """The first authored region with this label prefix, on this track."""
    return next(r for r in manifest["regions"]
                if r["label"].startswith(prefix) and r["source_track"] == track)


def test_a_mic_dropping_out_mid_overlap_is_not_a_reaction(noise_signals, manifest):
    """MEASURED, and the reason `reaction_onset` requires a rising edge.

    The noise fixture's block pattern puts the mic's loud region inside the
    party's, so the operator "stops dead" while the party carries on. An earlier
    cut of this signal tested for a CONCURRENT overlap and fired here — a real
    moment, found through the wrong mechanism, because `sudden_silence` (the mic
    went quiet) and `overlap_speech` (both mics live) can only hold at the same
    instant inside the speech gate's `hangover_s`. The output therefore depended
    on a constant chosen to stop one sentence being chopped into nine.

    §5.4.3 says the overlap FOLLOWS the silence. Both timings come from the
    manifest, so if the block pattern moves the test moves with it.
    """
    cfg, timeline, raw = noise_signals
    mic_region = region(manifest, "speech_peak", "mic")
    party_region = region(manifest, "party_laugh", "party")

    # The shape this test depends on, asserted rather than assumed.
    assert party_region["t_start"] < mic_region["t_end"] < party_region["t_end"]

    _v, _u, events = derived.compute(raw, timeline, cfg)
    window = float(cfg.get("score.derived.reaction.window_s"))
    assert not any(mic_region["t_end"] <= t <= mic_region["t_end"] + window
                   for t in events["reaction_onset"]), (
        "firing here would be the hangover doing the detecting"
    )


def test_neither_fixture_contains_a_real_reaction_onset(speech_signals, noise_signals):
    """Said out loud so it is not mistaken for coverage.

    §5.4.3's shape is: everyone goes quiet for a beat, and THEN everyone talks at
    once. Neither fixture authors it — the noise fixture's overlap starts before
    its silence, and the speech fixture's long silence has decayed to zero by the
    time its overlap begins 19 seconds later. So the only tests of this signal
    are synthetic, and the real one is a hand-labelled clip.
    """
    for cfg, timeline, raw in (speech_signals, noise_signals):
        _v, _u, events = derived.compute(raw, timeline, cfg)
        assert not events["reaction_onset"]


def test_the_speech_fixtures_near_miss(speech_signals, speech_manifest):
    """The speech fixture's overlap starts 0.6 s after its long silence ENDS,
    which is inside §5.4.3's 2 s window — so it looks like it should fire.

    It does not, and for the right reason: `sudden_silence` marks the *onset* of
    quiet, not the whole of it, and after nineteen seconds it has long since
    decayed. §5.4.3 describes a reaction to something that just stopped.

    (An earlier draft of this test claimed the gap was 19 s. That misread the
    manifest — 19.4 s is the silence's DURATION and the gap is 0.6 s. Right
    conclusion, wrong reason, so the reason is now what gets asserted.)
    """
    cfg, timeline, raw = speech_signals
    values, _u, _e = derived.compute(raw, timeline, cfg)

    longest = max(speech_manifest["silence_windows"], key=lambda w: w[1] - w[0])
    overlap_start = speech_manifest["overlap_windows"][0][0]
    window = float(cfg.get("score.derived.reaction.window_s"))

    assert overlap_start - longest[1] < window, "the gap really is inside the window"

    lead_in = (timeline >= overlap_start - window) & (timeline <= overlap_start)
    assert values["sudden_silence"][lead_in].max() < float(
        cfg.get("score.derived.reaction.silence_level"))


def test_reaction_onset_is_events_not_a_level(cfg):
    """§5.4.3's sibling flick_signature says "emit composite event", and this
    names a moment rather than a level — so it goes through the same Gaussian
    kernel as every other event instead of inventing a second mechanism."""
    timeline, mic, party = classic_reaction(cfg)
    values, _u, events = derived.compute(
        {"mic_rms": mic, "party_rms": party}, timeline, cfg)

    assert "reaction_onset" not in values, "a moment, not a level"
    assert events["reaction_onset"], "the classic shape must be detected"
    # The beat of silence runs 20-21 s and both then talk from 21.5 s.
    assert all(21.0 <= t <= 24.0 for t in events["reaction_onset"])


def classic_reaction(cfg):
    """§5.4.3's own shape: the operator talks, stops dead, and then BOTH of them
    burst into speech. Synthetic because no fixture authors it."""
    timeline = timeline_for(cfg)
    mic = np.full(timeline.size, -50.0)
    party = np.full(timeline.size, -55.0)

    mic[(timeline >= 10.0) & (timeline < 20.0)] = -15.0    # operator talking
    #  20.0 - 21.5: the beat of silence
    both = (timeline >= 21.5) & (timeline < 26.0)
    mic[both] = -14.0                                       # and then both at once
    party[both] = -12.0
    return timeline, mic, party


def test_one_event_per_reaction_not_one_per_sample(cfg):
    """A four-second reaction would otherwise produce forty overlapping kernels,
    and score.events.combine is `sum`."""
    timeline, mic, party = classic_reaction(cfg)
    _v, _u, events = derived.compute(
        {"mic_rms": mic, "party_rms": party}, timeline, cfg)
    assert len(events["reaction_onset"]) == 1


# --------------------------------------------------------------------------
# the per-signal z-score floor
# --------------------------------------------------------------------------


def test_the_floor_falls_back_to_the_shared_default(cfg):
    assert score_runner.std_floor_for(cfg, "mic_rms") == pytest.approx(
        float(cfg.get("score.zscore_std_floor")))


def test_a_zero_to_one_signal_gets_its_own_floor(cfg):
    """`score.zscore_std_floor` is in DECIBELS. Applied to a 0..1 gate it binds
    for anything firing under 25% of the time, pinning z at ~2.0 however rare
    the firing is — the dB number rescuing a signal it knows nothing about."""
    shared = float(cfg.get("score.zscore_std_floor"))
    for name in ("overlap_speech", "sudden_silence", "input_stillness"):
        assert score_runner.std_floor_for(cfg, name) < shared


def test_the_override_actually_changes_the_z_score(cfg):
    """Not just config plumbing: the whole point is that a rare 0/1 signal can
    exceed z = 2, which the dB floor forbids."""
    gate = np.zeros(3000)
    gate[1500:1520] = 1.0                        # fires under 1% of the time

    with_db_floor, _b, _s = grid.rolling_zscore(gate, 3000, 0.5)
    with_own_floor, _b, _s = grid.rolling_zscore(
        gate, 3000, score_runner.std_floor_for(cfg, "overlap_speech"))

    assert with_db_floor.max() == pytest.approx(2.0, abs=0.1)
    assert with_own_floor.max() > with_db_floor.max() * 2


# --------------------------------------------------------------------------
# reaching the scorer
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scored_speech(tmp_path_factory, speech_fixture_dir, speech_manifest):
    """The speech fixture taken all the way through `score`, once.

    The whole default plan in one call rather than a partial one: `Runner.plan`
    marks a stage blocked when a dependency is absent from the SAME plan, even
    if it already ran, so a two-step plan silently produces nothing.
    """
    tmp_path = tmp_path_factory.mktemp("scored_speech")
    cfg, conn, ctx = stage_ctx(tmp_path, speech_fixture_dir, speech_manifest,
                               full_plan=True)
    yield cfg, conn, ctx
    conn.close()


def test_derived_signals_reach_the_feature_vector(scored_speech):
    """A9 wants the full vector, and a derived signal is exactly as
    unreconstructable later as a stored one."""
    import json

    _cfg, conn, ctx = scored_speech
    row = conn.execute(
        "SELECT feature_vector, contributing_signals FROM candidates "
        "WHERE stream_id = ? AND is_current = 1 LIMIT 1", (ctx.stream_id,)
    ).fetchone()
    assert row is not None, "the stream must still produce candidates"

    vector = json.loads(row["feature_vector"])
    for name in ("sudden_silence", "overlap_speech", "mic_f0_variance"):
        assert name in vector, f"{name} is declared in feature_schema.yaml"

    # At least one of them has to carry a real observation, or "derived signals
    # reach the vector" would pass on a vector of nulls.
    assert any(vector[name] is not None
               for name in ("sudden_silence", "overlap_speech"))


def test_a_derived_signal_can_be_weighted_like_any_other(scored_speech):
    """The point of computing them before the profile is applied: §6.5 weights
    `overlap_speech` at 1.5 and must not need a special case to do it."""
    cfg, conn, ctx = scored_speech
    weighted = config.load(overrides=[
        f"paths.data_root={cfg.data_root.as_posix()}",
        "score.profile=naive",
    ])
    timeline = grid.build(float(ctx.stream["duration_s"]),
                          float(weighted.get("score.score_grid_hz")))
    tracks, _missing, _raw = score_runner.build_tracks(
        StageContext(cfg=weighted, conn=conn, stream_id=ctx.stream_id,
                     log=lambda *_: None),
        timeline,
    )
    by_name = {t.name: t for t in tracks}
    assert "overlap_speech" in by_name
    assert "sudden_silence" in by_name
    assert by_name["overlap_speech"].weight == 0.0, "naive weights it at nothing"
    assert np.isfinite(by_name["overlap_speech"].values).all()
