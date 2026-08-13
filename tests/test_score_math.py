"""The scoring primitives: grid, z-score, kernels, windows, spacing.

These are pure functions over arrays, so they get tested against values that can
be worked out by hand rather than against the fixture.
"""

from __future__ import annotations

import numpy as np
import pytest

from clipforge.score import features, grid, kernels, windows
from clipforge.signals import Series


# --------------------------------------------------------------------------
# grid and resampling
# --------------------------------------------------------------------------


def test_grid_spans_the_stream_at_the_requested_rate():
    g = grid.build(10.0, 10.0)
    assert g[0] == 0.0
    assert g[-1] == pytest.approx(10.0)
    assert len(g) == 101


def test_grid_rejects_a_zero_duration():
    with pytest.raises(ValueError, match="duration"):
        grid.build(0.0, 10.0)


def test_resample_respects_t0():
    """Series carry t0 = frame_s/2 because a frame measures the centre of its
    window; ignoring it would shift every signal by half a frame."""
    series = Series(kind="x", values=np.array([0.0, 10.0]), sample_rate_hz=1.0, t0=1.0)
    g = np.array([1.0, 1.5, 2.0])
    np.testing.assert_allclose(grid.resample(series, g), [0.0, 5.0, 10.0])


def test_resample_holds_the_endpoints_outside_the_series():
    """The alternative is a cliff to zero at both ends of every stream."""
    series = Series(kind="x", values=np.array([5.0, 7.0]), sample_rate_hz=1.0, t0=1.0)
    g = np.array([0.0, 1.0, 2.0, 9.0])
    np.testing.assert_allclose(grid.resample(series, g), [5.0, 5.0, 7.0, 7.0])


def test_resample_of_an_empty_series_is_zeros():
    series = Series(kind="x", values=np.array([]), sample_rate_hz=10.0)
    assert not grid.resample(series, np.arange(5.0)).any()


# --------------------------------------------------------------------------
# rolling z-score
# --------------------------------------------------------------------------


def test_constant_input_gives_zero_not_infinity():
    """A digitally silent stretch is exactly constant, so sigma is exactly
    zero. Without the floor, every dither becomes a z-score of fifty."""
    z, mean, std = grid.rolling_zscore(np.full(500, -80.0), 100, std_floor=0.5)
    assert np.all(np.isfinite(z))
    assert np.allclose(z, 0.0)
    assert np.allclose(std, 0.0)
    assert np.allclose(mean, -80.0)


def test_a_spike_scores_high_against_a_quiet_baseline():
    values = np.full(1000, -50.0)
    values[500] = -20.0
    z, _, _ = grid.rolling_zscore(values, 300, std_floor=0.01)
    assert z[500] > 10
    assert abs(z[0]) < 1


def test_the_baseline_rolls_rather_than_being_global():
    """§6.2's whole point: a global constant finds 'the loud game', not 'the
    loud moment'. Two halves at different levels must both read as normal."""
    values = np.concatenate([np.full(1000, -60.0), np.full(1000, -30.0)])
    rng = np.random.default_rng(0)
    values = values + rng.standard_normal(2000) * 2.0

    z, mean, _ = grid.rolling_zscore(values, 200, std_floor=0.5)
    assert mean[100] < -55 and mean[1900] > -35        # the baseline followed
    assert abs(np.mean(z[100:900])) < 0.5              # neither half is "the loud one"
    assert abs(np.mean(z[1100:1900])) < 0.5


def test_centering_keeps_the_variance_accurate():
    """E[x^2] - E[x]^2 on dB values around -50 subtracts 2500 from 2525.
    Centering first is what keeps three digits of that difference."""
    rng = np.random.default_rng(1)
    values = -50.0 + rng.standard_normal(20000) * 3.0
    _, _, std = grid.rolling_zscore(values, 4001, std_floor=0.0)
    assert std[10000] == pytest.approx(3.0, rel=0.05)


def test_window_truncates_at_the_edges():
    z, mean, _ = grid.rolling_zscore(np.arange(100.0), 51, std_floor=0.1)
    assert np.all(np.isfinite(z))
    assert mean[0] < mean[50] < mean[-1]


def test_empty_input():
    z, mean, std = grid.rolling_zscore(np.zeros(0), 10, 0.5)
    assert z.size == mean.size == std.size == 0


# --------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------


def test_plateau_peaks_at_exactly_one():
    """Unit PEAK, not unit area. A unit-area Gaussian at sigma=10 s peaks near
    0.004, so §6.5's marker_definite: 3.0 would contribute 0.01 against
    z-scores that swing plus or minus three."""
    g = np.linspace(0, 100, 1001)
    k = kernels.plateau(g, [50.0], pre_s=25.0, post_s=5.0, shoulder_s=3.0)
    assert k.max() == pytest.approx(1.0)


def test_plateau_is_asymmetric_around_the_press():
    """§4.3: the moment happened BEFORE the press, never after.

    pre_s=25 / post_s=5 spans [t-25, t+5], so the kernel reaches five times
    further back than forward.
    """
    g = np.linspace(0, 100, 10001)
    k = kernels.plateau(g, [50.0], pre_s=25.0, post_s=5.0, shoulder_s=0.0)

    def at(t):
        return k[np.argmin(abs(g - t))]

    assert at(30.0) == pytest.approx(1.0)   # 20 s before: well inside
    assert at(54.0) == pytest.approx(1.0)   # 4 s after: inside post_s
    assert at(56.0) == pytest.approx(0.0)   # 6 s after: outside
    assert at(24.0) == pytest.approx(0.0)   # 26 s before: outside
    # The asymmetry itself: 20 s back is covered, 20 s forward is not.
    assert at(50.0 - 20.0) == pytest.approx(1.0)
    assert at(50.0 + 20.0) == pytest.approx(0.0)


def test_plateau_covers_the_whole_reaction_range():
    """§4.3 says the delay is 5-15 s, a range. A point estimate at t-20 misses
    by 14 s when the true delay was 6."""
    g = np.linspace(0, 100, 10001)
    k = kernels.plateau(g, [50.0], pre_s=25.0, post_s=5.0, shoulder_s=3.0)
    for delay in (5.0, 10.0, 15.0):
        moment = 50.0 - delay
        assert k[np.argmin(abs(g - moment))] == pytest.approx(1.0)


def test_shoulders_are_smooth_and_reach_zero():
    g = np.linspace(0, 100, 10001)
    k = kernels.plateau(g, [50.0], pre_s=10.0, post_s=5.0, shoulder_s=4.0)
    assert k[np.argmin(abs(g - 36.0))] == pytest.approx(0.0, abs=1e-9)
    assert 0.0 < k[np.argmin(abs(g - 38.0))] < 1.0
    assert k[np.argmin(abs(g - 40.0))] == pytest.approx(1.0)


def test_overlapping_markers_combine_with_max():
    """Two presses eight seconds apart describe ONE moment. Summing would give
    marker_definite a contribution of 6.0 and let a flurry of presses dominate
    the whole stream."""
    g = np.linspace(0, 100, 1001)
    k = kernels.plateau(g, [50.0, 58.0], pre_s=25.0, post_s=5.0, shoulder_s=3.0, combine="max")
    assert k.max() == pytest.approx(1.0)


def test_sum_mode_is_available_for_other_event_kinds():
    """Three kills in eight seconds genuinely IS stronger than one."""
    g = np.linspace(0, 100, 1001)
    k = kernels.gaussian(g, [50.0, 50.0], sigma_s=2.0, combine="sum")
    assert k.max() == pytest.approx(2.0)


def test_gaussian_peaks_at_one():
    g = np.linspace(0, 100, 1001)
    assert kernels.gaussian(g, [50.0], sigma_s=5.0).max() == pytest.approx(1.0, abs=1e-3)


def test_no_events_gives_a_zero_kernel():
    g = np.linspace(0, 10, 101)
    assert not kernels.plateau(g, [], pre_s=25.0, post_s=5.0, shoulder_s=3.0).any()


def test_retro_offset_is_the_plateaus_centre_of_mass():
    """§17 tunes marker_retro_offset_s against 'the fraction of marker-anchored
    windows where the moment is inside'. The plateau replaced that number, so
    the parameter has to still correspond to something."""
    assert kernels.centre_of_mass(25.0, 5.0) == pytest.approx(10.0)
    assert kernels.centre_of_mass(45.0, 5.0) == pytest.approx(20.0)


# --------------------------------------------------------------------------
# peaks and windows
# --------------------------------------------------------------------------


def bump(length=1000, centre=500, height=5.0, width=40.0, pedestal=0.0):
    x = np.arange(length, dtype=float)
    return pedestal + height * np.exp(-0.5 * ((x - centre) / width) ** 2)


def test_peaks_below_the_minimum_value_are_rejected():
    """A local maximum in negative territory is a local maximum of quietness.

    §6.3's exit = 0.35 * v_peak is also *above* the peak when v_peak < 0, so
    the expansion loop stops immediately and the window comes out empty.
    """
    # A genuine local maximum whose value is still negative: a slightly less
    # quiet moment inside a quiet stretch.
    composite = bump(height=3.0) - 10.0
    assert composite.max() < 0

    assert windows.find(composite, 0.1, min_value=0.0) == []
    assert len(windows.find(composite, 0.1, min_value=-100.0)) == 1


def test_expansion_terminates_on_a_pedestal():
    """The bug the fixture exposed: a marker plateau holds the composite at 3.0
    for thirty seconds, so 0.35 * 4.5 = 1.6 is BELOW the pedestal and expansion
    ran to the end of the stream every time."""
    composite = bump(height=1.5, pedestal=3.0)
    g = np.arange(composite.size) / 10.0
    peaks = windows.find(composite, 0.1, min_value=0.0)
    assert len(peaks) == 1

    t_start, t_end = windows.expand(
        composite, g, peaks[0], exit_ratio=0.35,
        min_window_s=8.0, max_window_s=60.0, duration_s=g[-1],
    )
    assert t_end - t_start < 60.0     # not clamped to the maximum
    assert t_start <= g[peaks[0].index] <= t_end


def test_window_always_contains_its_peak():
    """The schema enforces t_start <= t_peak <= t_end, so clamping must not be
    able to push the peak out of its own window."""
    for peak_t in (0.0, 1.0, 30.0, 59.0, 60.0):
        t_start, t_end = windows.clamp(
            peak_t - 0.1, peak_t + 0.1, peak_t,
            min_window_s=8.0, max_window_s=60.0, duration_s=60.0,
        )
        assert t_start <= peak_t <= t_end


@pytest.mark.parametrize("length", [0.5, 8.0, 30.0, 120.0])
def test_clamped_windows_respect_the_bounds(length):
    t_start, t_end = windows.clamp(
        100.0, 100.0 + length, 100.0 + length / 2,
        min_window_s=8.0, max_window_s=60.0, duration_s=600.0,
    )
    assert 8.0 - 1e-9 <= (t_end - t_start) <= 60.0 + 1e-9


def test_windows_never_run_past_the_stream():
    t_start, t_end = windows.clamp(
        595.0, 605.0, 599.0, min_window_s=8.0, max_window_s=60.0, duration_s=600.0
    )
    assert t_start >= 0.0 and t_end <= 600.0


def test_a_short_window_at_the_start_slides_rather_than_truncating():
    t_start, t_end = windows.clamp(
        0.0, 1.0, 0.5, min_window_s=8.0, max_window_s=60.0, duration_s=600.0
    )
    assert t_start == 0.0
    assert t_end == pytest.approx(8.0)


# --------------------------------------------------------------------------
# peak merging — hysteresis_enter's job
# --------------------------------------------------------------------------


def test_a_wobble_on_one_bump_is_one_moment():
    """§6.3 defines hysteresis_enter and never reads it; this is the job."""
    x = np.arange(1000, dtype=float)
    composite = 5 * np.exp(-0.5 * ((x - 500) / 100) ** 2)
    composite[480:520] += 0.05 * np.sin(np.linspace(0, 6 * np.pi, 40))

    peaks = windows.find(composite, 0.0, min_value=0.0)
    assert len(peaks) > 1
    assert len(windows.merge_peaks(composite, peaks, enter=0.6)) == 1


def test_genuinely_separate_moments_are_kept():
    composite = bump(centre=200, width=20) + bump(centre=800, width=20)
    peaks = windows.find(composite, 0.1, min_value=0.0)
    assert len(windows.merge_peaks(composite, peaks, enter=0.6)) == 2


def test_merging_keeps_the_higher_peak():
    x = np.arange(600, dtype=float)
    composite = (3 * np.exp(-0.5 * ((x - 200) / 60) ** 2)
                 + 5 * np.exp(-0.5 * ((x - 300) / 60) ** 2))
    peaks = windows.find(composite, 0.0, min_value=0.0)
    merged = windows.merge_peaks(composite, peaks, enter=0.6)
    assert len(merged) == 1
    assert 250 < merged[0].index < 350


# --------------------------------------------------------------------------
# auto-calibration
# --------------------------------------------------------------------------


def test_target_scales_with_duration():
    assert windows.target_range(10800.0, [27, 50], 3) == (81, 150)   # §6.7's numbers
    assert windows.target_range(3600.0, [27, 50], 3) == (27, 50)


def test_target_is_floored_for_short_streams():
    """A 60 s fixture would otherwise target 0.45 candidates."""
    assert windows.target_range(60.0, [27, 50], 3) == (3, 3)


def test_calibration_finds_a_prominence_in_range():
    rng = np.random.default_rng(3)
    x = np.arange(6000, dtype=float)
    composite = rng.standard_normal(6000) * 0.2
    for centre in range(300, 6000, 300):
        composite += 4.0 * np.exp(-0.5 * ((x - centre) / 25) ** 2)

    result = windows.calibrate(
        composite, target_low=8, target_high=12, min_value=0.0, enter=0.6, max_iterations=24
    )
    assert result.reached_target
    assert 8 <= result.count <= 12


def test_calibration_reports_a_shortfall_rather_than_pretending():
    composite = bump(height=5.0)
    result = windows.calibrate(
        composite, target_low=20, target_high=30, min_value=0.0, enter=0.6, max_iterations=24
    )
    assert not result.reached_target
    assert result.count < 20
    assert "only" in result.reason


def test_calibration_on_a_flat_composite():
    result = windows.calibrate(
        np.full(100, 2.0), target_low=3, target_high=5,
        min_value=0.0, enter=0.6, max_iterations=24,
    )
    assert result.count == 0
    assert "flat" in result.reason


def test_calibration_counts_candidates_not_raw_peaks():
    """Measured on the fixture: three peaks became two candidates after
    merging. Calibrating on peaks would target a number nobody ever sees."""
    x = np.arange(1000, dtype=float)
    composite = 5 * np.exp(-0.5 * ((x - 500) / 100) ** 2)
    composite[480:520] += 0.05 * np.sin(np.linspace(0, 6 * np.pi, 40))

    raw = len(windows.find(composite, 0.0, 0.0))
    merged = windows.candidate_count(composite, 0.0, 0.0, enter=0.6)
    assert raw > merged == 1


# --------------------------------------------------------------------------
# spacing (§6.6)
# --------------------------------------------------------------------------


def make_windows(*specs):
    return [
        windows.Window(index=i, t_peak=t, t_start=t - 4, t_end=t + 4, value=s, score=s)
        for i, (t, s) in enumerate(specs)
    ]


def test_spacing_penalises_a_neighbour_once():
    result = windows.apply_spacing(
        make_windows((100.0, 5.0), (110.0, 4.0)), window_s=30.0, factor=0.5
    )
    by_peak = {w.t_peak: w for w in result}
    assert by_peak[100.0].score == 5.0
    assert by_peak[110.0].score == pytest.approx(2.0)
    assert by_peak[110.0].suppressed_by == 100.0


def test_accepted_only_never_penalises_twice():
    """§6.6 as written gives the third candidate 0.5 * 0.5 = 0.25, because a
    suppressed candidate goes on to suppress others."""
    result = windows.apply_spacing(
        make_windows((100.0, 5.0), (110.0, 4.0), (120.0, 3.0)),
        window_s=30.0, factor=0.5, mode="accepted_only",
    )
    by_peak = {w.t_peak: w for w in result}
    assert by_peak[120.0].score == pytest.approx(1.5)   # 3.0 * 0.5, not * 0.25


def test_spec_literal_mode_compounds_as_the_spec_describes():
    result = windows.apply_spacing(
        make_windows((100.0, 5.0), (110.0, 4.0), (120.0, 3.0)),
        window_s=30.0, factor=0.5, mode="spec_literal",
    )
    by_peak = {w.t_peak: w for w in result}
    assert by_peak[120.0].score == pytest.approx(1.5)


def test_distant_candidates_are_untouched():
    result = windows.apply_spacing(
        make_windows((100.0, 5.0), (200.0, 4.0)), window_s=30.0, factor=0.5
    )
    assert all(w.suppressed_by is None for w in result)


def test_suppressed_candidates_are_kept_not_dropped():
    """C2: recall over precision, and the operator can always lengthen a clip."""
    result = windows.apply_spacing(
        make_windows((100.0, 5.0), (105.0, 4.0)), window_s=30.0, factor=0.5
    )
    assert len(result) == 2


def test_result_is_ordered_by_final_score():
    result = windows.apply_spacing(
        make_windows((100.0, 5.0), (105.0, 4.9), (300.0, 3.0)), window_s=30.0, factor=0.5
    )
    assert [round(w.score, 2) for w in result] == [5.0, 3.0, 2.45]


# --------------------------------------------------------------------------
# breakdown and feature vector
# --------------------------------------------------------------------------


def tracks_for(values: dict[str, tuple[float, float]]):
    return [
        features.SignalTrack(name=name, values=np.array([value]), weight=weight)
        for name, (value, weight) in values.items()
    ]


def test_contributions_sum_to_the_raw_total():
    """A breakdown that does not add up to the score is worse than none."""
    tracks = tracks_for({"mic_rms": (2.0, 1.0), "marker_definite": (1.0, 3.0)})
    detail = features.breakdown(tracks, 0, smoothed_value=4.1)
    assert detail.contributions == {"marker_definite": 3.0, "mic_rms": 2.0}
    assert detail.total_raw == pytest.approx(5.0)


def test_the_smoothing_gap_is_recorded_not_hidden():
    """§6.2 smooths AFTER summing, so the smoothed value at t_peak is not the
    sum of weight x signal."""
    tracks = tracks_for({"mic_rms": (2.0, 1.0)})
    payload = features.breakdown(tracks, 0, smoothed_value=1.6).to_json_dict()
    assert payload[features.TOTAL_RAW] == 2.0
    assert payload[features.TOTAL_SMOOTHED] == 1.6


def test_contributions_are_ordered_by_magnitude():
    tracks = tracks_for({"a": (0.1, 1.0), "b": (5.0, 1.0), "c": (-2.0, 1.0)})
    detail = features.breakdown(tracks, 0, smoothed_value=3.1)
    assert list(detail.contributions) == ["b", "c", "a"]


# --------------------------------------------------------------------------
# word-boundary snapping (§6.3) — real since Phase 2 supplies word timestamps
# --------------------------------------------------------------------------


def snap(t_start, t_end, starts, ends, **kwargs):
    options = {
        "max_distance_s": 0.5, "min_window_s": 8.0, "max_window_s": 60.0,
        "duration_s": 600.0,
    }
    options.update(kwargs)
    return windows.snap_to_words(
        t_start, t_end, starts=np.array(starts), ends=np.array(ends), **options
    )


def test_boundaries_move_onto_nearby_words():
    """§6.3: "snap t_start / t_end to nearest word boundary from WhisperX"."""
    assert snap(10.2, 20.2, [10.0, 15.0], [12.0, 20.5]) == (10.0, 20.5)


def test_a_boundary_in_silence_stays_put():
    """A window ending in silence has no word near it, and nearest-boundary
    snapping would drag the edge to a word twenty seconds away."""
    assert snap(100.0, 110.0, [10.0], [12.0]) == (100.0, 110.0)


def test_the_leash_is_the_only_thing_stopping_that():
    """Same geometry, a leash wide enough to reach: it does move."""
    moved = snap(100.0, 110.0, [99.0], [111.0], max_distance_s=2.0)
    assert moved == (99.0, 111.0)


def test_the_start_prefers_a_word_start_at_or_before_it():
    """§8.2 says the point of snapping is "no clipped syllables", and §6.3's
    "nearest" can clip: a start that moves forward cuts the word it lands in."""
    start, _end = snap(10.3, 30.0, [10.0, 10.4], [30.0])
    assert start == 10.0


def test_the_end_prefers_a_word_end_at_or_after_it():
    _start, end = snap(10.0, 29.7, [10.0], [29.6, 30.0])
    assert end == 30.0


def test_a_snap_past_max_window_is_skipped():
    """Re-clamping afterwards would silently undo the snap and leave the window
    neither snapped nor honestly unsnapped.

    max_window_s is the bound that snapping actually threatens: preferring a
    start before and an end after means it GROWS the window, never shrinks it."""
    # 60.0 s exactly; snapping outward on both edges would make it 60.6.
    assert snap(10.0, 70.0, [9.7], [70.3], max_window_s=60.0) == (10.0, 70.0)


def test_a_snap_under_min_window_is_skipped():
    """The rarer direction: nothing in the preferred direction is within reach,
    so both edges fall back inward."""
    assert snap(10.0, 18.2, [10.4], [17.9], min_window_s=8.0) == (10.0, 18.2)


def test_a_snap_past_the_end_of_the_stream_is_skipped():
    assert snap(10.0, 99.8, [10.0], [100.2], duration_s=100.0) == (10.0, 99.8)


def test_no_words_means_no_movement():
    assert snap(10.0, 20.0, [], []) == (10.0, 20.0)


# --------------------------------------------------------------------------
# A9 — unweighted signals reach the vector, not the breakdown
# --------------------------------------------------------------------------


def test_unweighted_signals_are_excluded_from_the_breakdown():
    """A9 requires them in the feature vector, so build_tracks loads them at
    weight 0 — but a row of 0.000 in the review UI's `?` panel is noise in the
    one place that explains the number beside it."""
    tracks = tracks_for({"mic_rms": (2.0, 1.0), "speech_rate": (3.0, 0.0)})
    detail = features.breakdown(tracks, 0, smoothed_value=2.0)
    assert "speech_rate" not in detail.contributions
    assert detail.total_raw == pytest.approx(2.0)


def test_unweighted_signals_still_reach_the_feature_vector():
    """A9: "logged in full, always, even for signals not currently weighted"."""
    from clipforge import config

    schema = config.load().feature_schema
    tracks = tracks_for({"mic_rms": (2.0, 1.0), "speech_rate": (3.0, 0.0)})
    vector = features.vector(schema, tracks, 0)
    assert vector["speech_rate"] == pytest.approx(3.0)
    assert vector["mic_rms"] == pytest.approx(2.0)
