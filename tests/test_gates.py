"""§6.4's gated negatives.

§6.4 is unusually prescriptive — "this is a specific requirement and must be
implemented exactly as described" — so most of what follows is its own sentences
turned into assertions: a joke in the lobby is never penalised, the grace period
and the speech reset are both mandatory, both AFK conditions are required and
either alone is insufficient, and the penalty ramps rather than steps.

The rest is the trap this phase keeps meeting. A stream with no input log has
UNKNOWN input, not idle input, and every test below that could confuse the two
is written from both sides.

No number here is written down: durations come from the config object and are
converted with the same `score.score_grid_hz` the scorer uses.
"""

from __future__ import annotations

import numpy as np
import pytest

from clipforge import config
from clipforge.score import derived, features, gates, grid

#: Long enough that `afk_threshold_s` (60 s) plus its ramp fits inside one
#: silent stretch with room either side, and that the rolling baseline window
#: is not the whole stream.
DURATION_S = 400.0

#: dBFS levels for the synthetic tracks. Only their DIFFERENCE matters — every
#: threshold in the gate is relative to the track's own rolling baseline, which
#: is the property that keeps a microphone's gain out of the signal.
FLOOR_DB = -60.0
SPEECH_DB = -20.0


@pytest.fixture(scope="module")
def cfg():
    return config.load()


def timeline_for(cfg, duration_s: float = DURATION_S) -> np.ndarray:
    return grid.build(duration_s, float(cfg.get("score.score_grid_hz")))


def samples_for(cfg, seconds: float) -> int:
    return max(int(round(float(seconds) * float(cfg.get("score.score_grid_hz")))), 1)


def track(timeline: np.ndarray, spans, level: float = SPEECH_DB,
          floor: float = FLOOR_DB) -> np.ndarray:
    """A flat track at `floor` with `spans` raised to `level`."""
    values = np.full(timeline.size, float(floor), dtype=np.float64)
    for start, end in spans:
        values[(timeline >= start) & (timeline < end)] = float(level)
    return values


def audio(timeline: np.ndarray, mic_spans, party_spans=()) -> dict:
    return {
        "mic_rms": track(timeline, mic_spans),
        "party_rms": track(timeline, party_spans),
    }


def input_log(timeline: np.ndarray, *, idle_spans, uncovered_spans=(),
              rate: float = 8.0, velocity: float = 1500.0) -> dict:
    """§4.4's three arrays, with the same conventions `input_signals` writes.

    Outside `idle_spans` the operator is playing; inside them the logger is
    running and recording nothing. Inside `uncovered_spans` the logger was NOT
    running: the rates are NaN and the coverage is 0, which is the distinction
    the whole AFK gate turns on.
    """
    rates = np.full(timeline.size, float(rate))
    velocities = np.full(timeline.size, float(velocity))
    coverage = np.ones(timeline.size)

    for start, end in idle_spans:
        inside = (timeline >= start) & (timeline < end)
        rates[inside] = 0.0
        velocities[inside] = 0.0
    for start, end in uncovered_spans:
        inside = (timeline >= start) & (timeline < end)
        rates[inside] = np.nan
        velocities[inside] = np.nan
        coverage[inside] = 0.0

    return {"input_rate": rates, "mouse_velocity": velocities,
            "input_coverage": coverage}


def by_name(penalties) -> dict:
    return {penalty.name: penalty for penalty in penalties}


def whole_stream(timeline: np.ndarray):
    return [(float(timeline[0]), float(timeline[-1]))]


# --------------------------------------------------------------------------
# THE shared helper — §6.4's "not per-signal ad hoc logic"
# --------------------------------------------------------------------------


def test_run_length_counts_and_resets():
    out = gates.run_length(np.array([True, True, False, True, True, True]))
    assert list(out) == [1, 2, 0, 1, 2, 3]


def test_run_length_of_nothing():
    assert gates.run_length(np.zeros(0, dtype=bool)).size == 0


def test_sustained_waits_out_the_full_period_before_anything_happens():
    active = np.ones(20, dtype=bool)
    out = gates.sustained(active, sustain_samples=5, ramp_samples=2)
    assert not out[:4].any(), "nothing may happen inside the sustain period"
    assert out[4] > 0.0


def test_sustained_ramps_rather_than_steps():
    active = np.ones(20, dtype=bool)
    out = gates.sustained(active, sustain_samples=3, ramp_samples=4)
    rising = out[out > 0]
    assert rising[0] < 1.0, "§6.4: 'ramping in gradually (not a step function)'"
    assert np.all(np.diff(rising[:4]) > 0)
    assert rising.max() == pytest.approx(1.0)


def test_sustained_is_removed_not_decayed():
    """§6.4's words are 'reset the timer to zero, remove penalty'. A penalty
    that faded out over a second would keep punishing a moment that had already
    started."""
    active = np.ones(30, dtype=bool)
    active[20] = False
    out = gates.sustained(active, sustain_samples=3, ramp_samples=4)
    assert out[19] > 0.0
    assert out[20] == 0.0
    assert out[21] == 0.0, "and the timer restarts from zero, it does not resume"


# --------------------------------------------------------------------------
# §6.4's menu / lobby requirements
# --------------------------------------------------------------------------


def test_a_joke_in_the_lobby_is_never_penalised(cfg):
    """§6.4: "Requirement: jokes in the lobby must never be penalized. The
    8-second grace period and the speech-reset are both mandatory." """
    timeline = timeline_for(cfg)
    grace = float(cfg.get("score.negatives.menu_grace_period_s"))

    # An utterance every `grace - 2` seconds: nobody is quiet for long enough.
    spans = [(t, t + 1.0) for t in np.arange(2.0, DURATION_S, grace - 2.0)]
    raw = audio(timeline, spans)

    penalties, reasons = gates.compute(
        raw, timeline, cfg, menu_intervals=whole_stream(timeline)
    )
    menu = by_name(penalties)[gates.MENU]
    assert gates.MENU not in reasons
    assert not menu.fired, "the operator was in a lobby and talking throughout"


def test_speech_removes_a_penalty_that_had_already_ramped(cfg):
    """"ANY speech detected -> reset the 8-second timer to zero, remove
    penalty." Not "ramp it back down"."""
    timeline = timeline_for(cfg)
    grace = float(cfg.get("score.negatives.menu_grace_period_s"))
    ramp = float(cfg.get("score.negatives.ramp_s"))
    joke_at = 40.0 + grace + ramp

    raw = audio(timeline, [(10.0, 12.0), (joke_at, joke_at + 2.0),
                           (DURATION_S - 20.0, DURATION_S - 18.0)])
    penalties, _ = gates.compute(
        raw, timeline, cfg, menu_intervals=whole_stream(timeline)
    )
    menu = by_name(penalties)[gates.MENU].values

    before = menu[(timeline > joke_at - 1.0) & (timeline < joke_at)]
    during = menu[(timeline >= joke_at) & (timeline < joke_at + 2.0)]
    assert before.max() == pytest.approx(1.0), "it had ramped all the way in"
    assert not during.any(), "and one joke removes it outright"


def test_the_grace_period_is_waited_out_in_full(cfg):
    timeline = timeline_for(cfg)
    grace = float(cfg.get("score.negatives.menu_grace_period_s"))
    grid_hz = float(cfg.get("score.score_grid_hz"))

    raw = audio(timeline, [(DURATION_S - 30.0, DURATION_S - 28.0)])
    penalties, _ = gates.compute(
        raw, timeline, cfg, menu_intervals=whole_stream(timeline)
    )
    menu = by_name(penalties)[gates.MENU].values

    first = int(np.flatnonzero(menu > 0)[0])
    assert timeline[first] >= grace - 1.0 / grid_hz
    assert timeline[first] < grace + 1.0 / grid_hz, (
        "and not appreciably later either — the grace period is the whole delay"
    )


def test_the_menu_gate_needs_the_audio_to_be_flat(cfg):
    """§6.4's third condition, and it is not implied by the second: the speech
    gate fires above `baseline + vad.margin_db`, this fires above the bare
    baseline. Audio elevated but under the speech threshold is still audio."""
    timeline = timeline_for(cfg)
    margin = float(cfg.get("score.derived.vad.margin_db"))
    grace = float(cfg.get("score.negatives.menu_grace_period_s"))

    raw = audio(timeline, [(20.0, 22.0), (DURATION_S - 20.0, DURATION_S - 18.0)])
    # Half the margin above the floor: too quiet to be speech, too loud to be
    # "audio is flat".
    murmur = (150.0, 150.0 + 4 * grace)
    inside = (timeline >= murmur[0]) & (timeline < murmur[1])
    raw["mic_rms"][inside] = FLOOR_DB + margin / 2.0

    inputs = derived.Inputs(timeline=timeline, grid_hz=float(cfg.get("score.score_grid_hz")),
                            raw=raw, cfg=cfg)
    baseline_samples = grid.window_samples(
        cfg.get("score.rolling_baseline_window_s"), inputs.grid_hz
    )
    speech, quiet, _used = gates._speech_and_quiet(raw, inputs, baseline_samples)
    assert not speech[inside].any(), "the murmur is below the speech threshold"
    assert not quiet[inside].any(), "but it is above the rolling baseline"

    penalties, _ = gates.compute(
        raw, timeline, cfg, menu_intervals=whole_stream(timeline)
    )
    menu = by_name(penalties)[gates.MENU].values
    assert not menu[inside].any()
    assert menu.max() > 0.0, "while the genuinely flat stretches are penalised"


def test_digital_silence_satisfies_the_energy_condition(cfg):
    """The reason the comparison is `<=` where §6.4 writes `<`.

    A stretch pinned at `extract.rms.db_floor` equals its own rolling mean when
    the whole baseline window is pinned too, so the strict reading fails on the
    most unambiguous "nothing is happening" audio there is.
    """
    timeline = timeline_for(cfg)
    raw = audio(timeline, [])                    # flat at the floor throughout
    penalties, _ = gates.compute(
        raw, timeline, cfg, menu_intervals=whole_stream(timeline)
    )
    assert by_name(penalties)[gates.MENU].fired


def test_the_menu_gate_is_declared_and_inert_without_a_producer(cfg):
    """§5.9's screen classification is Phase 7. The gate is built; nothing
    writes `menu_screen` rows, and that reads as "not evaluated" rather than as
    "never active"."""
    timeline = timeline_for(cfg)
    penalties, reasons = gates.compute(
        audio(timeline, [(10.0, 12.0)]), timeline, cfg, menu_intervals=None
    )
    assert gates.MENU not in by_name(penalties)
    assert "Phase 7" in reasons[gates.MENU]


def test_a_menu_span_comes_from_events_t_end(cfg):
    """§6.4 needs a STATE and `feature_schema.yaml` declares an EVENT —
    `events.t_end` is what reconciles them, and vision must set it."""
    timeline = timeline_for(cfg)
    active = gates.menu_active([(10.0, 20.0)], timeline)
    assert not active[timeline < 10.0].any()
    assert active[(timeline >= 10.0) & (timeline <= 20.0)].all()
    assert not active[timeline > 20.0].any()

    instantaneous = gates.menu_active([(10.0, 10.0)], timeline)
    assert instantaneous.sum() == 1, (
        "a row with no t_end covers one sample and can never satisfy a grace "
        "period — which is why Phase 7 has to set it"
    )


# --------------------------------------------------------------------------
# §6.4's AFK requirements
# --------------------------------------------------------------------------


def afk_case(cfg, *, mic_spans, idle_spans, uncovered_spans=(), drop=()):
    timeline = timeline_for(cfg)
    raw = audio(timeline, mic_spans)
    raw.update(input_log(timeline, idle_spans=idle_spans,
                         uncovered_spans=uncovered_spans))
    for key in drop:
        raw.pop(key)
    penalties, reasons = gates.compute(raw, timeline, cfg, menu_intervals=None)
    return timeline, by_name(penalties), reasons


def quiet_and_idle(cfg):
    """A stretch long enough to satisfy `afk_threshold_s` and ramp fully."""
    threshold = float(cfg.get("score.negatives.afk_threshold_s"))
    ramp = float(cfg.get("score.negatives.ramp_s"))
    return (100.0, 100.0 + threshold + ramp + 20.0)


def test_afk_needs_both_conditions_and_fires_when_it_has_them(cfg):
    """§6.4: "Both conditions required. Either one alone is insufficient." """
    span = quiet_and_idle(cfg)
    talk = [(10.0, 12.0), (DURATION_S - 30.0, DURATION_S - 28.0)]

    _t, both, reasons = afk_case(cfg, mic_spans=talk, idle_spans=[span])
    assert gates.AFK not in reasons
    assert both[gates.AFK].fired, "no input and no speech, for long enough"

    # ...idle input, but the operator is talking the whole way through it
    chatter = talk + [(t, t + 1.0) for t in np.arange(span[0], span[1], 5.0)]
    _t, talking, _ = afk_case(cfg, mic_spans=chatter, idle_spans=[span])
    assert not talking[gates.AFK].fired, "speech alone is enough to refuse it"

    # ...silent, but playing
    _t, playing, _ = afk_case(cfg, mic_spans=talk, idle_spans=[])
    assert not playing[gates.AFK].fired, "input alone is enough to refuse it"


def test_a_moving_mouse_is_input_activity(cfg):
    """§6.4 says "no input activity" and never defines it; §4.4's logger records
    keys+clicks and mouse velocity as separate fields. A mouse moving with no
    clicks is somebody playing."""
    span = quiet_and_idle(cfg)
    timeline = timeline_for(cfg)
    raw = audio(timeline, [(10.0, 12.0)])
    raw.update(input_log(timeline, idle_spans=[span]))

    inside = (timeline >= span[0]) & (timeline < span[1])
    raw["mouse_velocity"][inside] = 400.0        # aiming, not typing

    penalties, _ = gates.compute(raw, timeline, cfg, menu_intervals=None)
    assert not by_name(penalties)[gates.AFK].fired


def test_a_stream_with_no_input_log_gets_no_afk_penalty_anywhere(cfg):
    """The one that matters. Every stream that exists today is this stream."""
    timeline = timeline_for(cfg)
    raw = audio(timeline, [(10.0, 12.0)])        # silent for 388 of 400 seconds
    penalties, reasons = gates.compute(raw, timeline, cfg, menu_intervals=None)

    assert gates.AFK not in by_name(penalties), (
        "absent, not a zero array — §6.4's penalty must not be evaluable at all "
        "without §4.4's log"
    )
    assert "UNKNOWN rather than idle" in reasons[gates.AFK]


@pytest.mark.parametrize("missing", gates.AFK_KINDS)
def test_every_input_array_is_required_including_the_coverage_mask(cfg, missing):
    span = quiet_and_idle(cfg)
    _t, penalties, reasons = afk_case(
        cfg, mic_spans=[(10.0, 12.0)], idle_spans=[span], drop=[missing]
    )
    assert gates.AFK not in penalties
    assert missing in reasons[gates.AFK]


def test_an_unlogged_stretch_is_unknown_input_and_not_idle_input(cfg):
    """THE distinction commit 35 built `input_coverage` for, tested where it is
    finally consumed. The same audio and the same silence, twice: once with a
    logger that was running and recorded nothing, once with a logger that was
    not running at all."""
    span = quiet_and_idle(cfg)
    talk = [(10.0, 12.0), (DURATION_S - 30.0, DURATION_S - 28.0)]

    _t, logged, _ = afk_case(cfg, mic_spans=talk, idle_spans=[span])
    assert logged[gates.AFK].fired, "a real zero rate is a real observation"

    _t, unlogged, _ = afk_case(cfg, mic_spans=talk, idle_spans=[],
                               uncovered_spans=[span])
    assert not unlogged[gates.AFK].fired, (
        "and an unrecorded stretch is not an observation of anything"
    )


def test_a_gap_in_the_log_restarts_the_timer_rather_than_extending_it(cfg):
    """An uncovered sample is not idle, so it breaks the run — the conservative
    direction. A logger that drops out halfway through an AFK stretch means the
    penalty starts again, not that it carries on through the blind spot."""
    threshold = float(cfg.get("score.negatives.afk_threshold_s"))
    timeline = timeline_for(cfg)
    raw = audio(timeline, [(10.0, 12.0)])
    # Idle for well over the threshold, with a two-second blind spot in the
    # middle and not enough room after it to re-qualify.
    raw.update(input_log(timeline, idle_spans=[(100.0, 100.0 + threshold + 10.0)],
                         uncovered_spans=[(100.0 + threshold - 5.0,
                                           100.0 + threshold - 3.0)]))
    penalties, _ = gates.compute(raw, timeline, cfg, menu_intervals=None)
    assert not by_name(penalties)[gates.AFK].fired


# --------------------------------------------------------------------------
# consistency with the rest of the scorer
# --------------------------------------------------------------------------


def test_the_gates_speech_is_the_same_speech_as_overlap_speech(cfg):
    """§6.4's "ANY speech detected" and §5.4.1's "VAD on both tracks" are the
    same question asked twice. A second energy test here could disagree with the
    first and nothing would report it."""
    timeline = timeline_for(cfg)
    raw = audio(timeline, [(10.0, 14.0), (100.0, 103.0)],
                party_spans=[(12.0, 16.0), (200.0, 204.0)])

    grid_hz = float(cfg.get("score.score_grid_hz"))
    inputs = derived.Inputs(timeline=timeline, grid_hz=grid_hz, raw=raw, cfg=cfg)
    baseline_samples = grid.window_samples(
        cfg.get("score.rolling_baseline_window_s"), grid_hz
    )
    speech, _quiet, used = gates._speech_and_quiet(raw, inputs, baseline_samples)

    expected = np.zeros(timeline.size, dtype=bool)
    for kind in used:
        expected |= derived.speech_gate(raw[kind], inputs, baseline_samples)
    assert np.array_equal(speech, expected)
    assert speech.any(), "and it is not vacuously equal to nothing"


def test_without_a_voice_track_nothing_can_be_gated(cfg):
    """§6.4 conditions every negative on audio being flat. With no voice track
    there is nothing to be flat, so neither penalty is evaluated — rather than
    both firing across a stream nobody listened to."""
    timeline = timeline_for(cfg)
    raw = {"game_rms": track(timeline, [])}
    raw.update(input_log(timeline, idle_spans=[(0.0, DURATION_S)]))

    penalties, reasons = gates.compute(
        raw, timeline, cfg, menu_intervals=whole_stream(timeline)
    )
    assert penalties == []
    assert set(reasons) == {gates.MENU, gates.AFK}


def test_the_penalty_is_subtracted_and_never_scaled(cfg):
    """A composite of weighted z-scores is SIGNED. Scaling `-1.8` by 0.5 gives
    `-0.9`, which RAISES the composite everywhere it is already negative — which
    is exactly where a menu or an AFK stretch sits."""
    composite = np.array([-1.8, 0.0, 3.0])
    penalty = gates.Penalty(name=gates.AFK, values=np.ones(3), weight=2.0)
    out = gates.apply(composite, [penalty])
    assert list(out) == [-3.8, -2.0, 1.0]
    assert (out < composite).all()


def test_applying_nothing_changes_nothing(cfg):
    composite = np.array([1.0, -2.0, 3.0])
    assert np.array_equal(gates.apply(composite, []), composite)


def test_a_penalty_that_did_not_fire_here_is_not_in_the_breakdown(cfg):
    """A row of `-0.000` is noise in the one panel that exists to explain the
    number beside it — the reason weight-0 signals are excluded too. The weight
    test alone cannot do it: a penalty's weight is non-zero by construction."""
    values = np.array([0.0, 0.0, 1.0])
    tracks = gates.as_tracks([gates.Penalty(gates.AFK, values, weight=2.0)])

    quiet_moment = features.breakdown(tracks, 0, smoothed_value=0.0)
    assert gates.AFK not in quiet_moment.contributions

    penalised_moment = features.breakdown(tracks, 2, smoothed_value=0.0)
    assert penalised_moment.contributions[gates.AFK] == pytest.approx(-2.0)


def test_a_penalty_track_carries_no_unit_and_no_baseline(cfg):
    """It is neither a z-score nor an event kernel, so the review UI's context
    line has nothing to say about it — "0.7 against a baseline of 0.02" tells a
    reader nothing the number beside it did not."""
    tracks = gates.as_tracks(
        [gates.Penalty(gates.AFK, np.ones(3), weight=2.0)]
    )
    assert features.unweighted_context(tracks, 0) == {}
    assert tracks[0].is_penalty


# --------------------------------------------------------------------------
# the configuration itself
# --------------------------------------------------------------------------


def test_the_ramp_outlasts_the_smoothing_that_follows_it(cfg):
    """The one thing about `ramp_s` that is not arbitrary. §6.2 smooths the
    composite AFTER the penalty is applied, so a ramp much shorter than that
    sigma is smoothed into something indistinguishable from a step and §6.4's
    "not a step function" stops meaning anything."""
    assert (float(cfg.get("score.negatives.ramp_s"))
            > float(cfg.get("score.smoothing_sigma_s")))


def test_17s_two_negative_parameters_are_no_longer_deferred(cfg):
    """They sat in `deferred:` from commit 2 because nothing read them. Moving
    one out of that block is what building the phase that consumes it means."""
    assert cfg.get("deferred.negatives", None) is None
    assert float(cfg.get("score.negatives.menu_grace_period_s")) > 0
    assert float(cfg.get("score.negatives.afk_threshold_s")) > 0


def test_the_negatives_are_versioned_into_config_version():
    """`score.*` is inside VERSIONED_SUBTREES, so retuning a penalty mints a new
    generation rather than silently rewriting candidates that were reviewed
    under different numbers."""
    base = config.load()
    louder = config.load(overrides=["score.negatives.afk_penalty=99.0"])
    assert base.version != louder.version


def test_both_negatives_are_declared_in_the_feature_schema(cfg):
    """A9's vector records the gate level, so the names have to be the schema's
    own — otherwise `features.vector` drops them on the floor."""
    for name in (gates.MENU, gates.AFK):
        assert name in cfg.feature_schema.signals
