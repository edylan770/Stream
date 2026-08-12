"""The frame grid — HANDOFF's hard part of §10.5.

The rate comes from the NTSC fixture's manifest, so nothing here is a literal:
if the fixture is ever regenerated at a different rate these tests follow it.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from clipforge.render.timebase import (
    DOWN,
    NEAREST,
    UP,
    TimeBase,
    TimebaseError,
    parse_value,
)


@pytest.fixture
def ntsc(ntsc_manifest) -> TimeBase:
    """29.97, straight from the fixture's declared rate."""
    rate = Fraction(ntsc_manifest["fps"])
    return TimeBase(rate.numerator, rate.denominator)


# --------------------------------------------------------------------------
# the rational itself
# --------------------------------------------------------------------------


def test_the_rate_is_exact_not_a_float(ntsc, ntsc_manifest):
    """29.97 is 30000/1001. A float fps cannot express that, which is the whole
    reason streams.fps_num/fps_den exist."""
    assert ntsc.rate == Fraction(ntsc_manifest["fps"])
    assert ntsc.frame_duration == 1 / Fraction(ntsc_manifest["fps"])
    assert ntsc.frame_duration != Fraction(1 / float(Fraction(ntsc_manifest["fps"])))


def test_frame_duration_is_the_reciprocal_not_the_rate(ntsc):
    """The rate is num/den; the frame duration is den/num. Inverting this is
    the single easiest way to produce a file that is wrong by 0.1%."""
    assert ntsc.frame_duration == Fraction(ntsc.den, ntsc.num)
    assert ntsc.frame_duration < 1


def test_a_stream_without_an_exact_rate_is_refused():
    row = {"id": "x", "fps_num": None, "fps_den": None}
    with pytest.raises(TimebaseError, match="probe"):
        TimeBase.from_stream(row)


@pytest.mark.parametrize("num,den", [(0, 1), (30, 0), (-30, 1)])
def test_impossible_rates_are_refused(num, den):
    with pytest.raises(TimebaseError):
        TimeBase(num, den)


# --------------------------------------------------------------------------
# every emitted value lands on the grid
# --------------------------------------------------------------------------


def test_emitted_values_are_exact_multiples_of_the_frame_duration(ntsc):
    """The requirement, stated directly."""
    for frames in (0, 1, 2, 29, 30, 1000, 53947):
        parsed = parse_value(ntsc.value(frames))
        assert parsed == frames * ntsc.frame_duration
        assert parsed % ntsc.frame_duration == 0


def test_values_keep_the_timescale_as_the_denominator(ntsc):
    """Fraction would reduce 30 frames at 29.97 to 1001/1000s — the same instant
    and exactly what a naive parser mishandles. Final Cut keeps the timescale."""
    text = ntsc.value(30)
    assert text.endswith("s")
    numerator, _, denominator = text[:-1].partition("/")
    assert int(denominator) == ntsc.num
    assert int(numerator) == 30 * ntsc.den


def test_zero_is_the_literal_zero_form(ntsc):
    assert ntsc.value(0) == "0s"
    assert parse_value("0s") == 0


def test_a_long_timeline_does_not_drift(ntsc, ntsc_manifest):
    """A float accumulation over an hour of 29.97 drifts by whole frames. The
    exact form cannot: the last frame is precisely n * 1001/30000."""
    frames = ntsc.frame_at(3600.0, DOWN)
    assert parse_value(ntsc.value(frames)) == Fraction(frames * ntsc.den, ntsc.num)
    assert float(ntsc.seconds(frames)) == pytest.approx(3600.0, abs=float(ntsc.frame_duration))


def test_integer_rates_still_produce_valid_values(manifest):
    """The 30 fps fixture — the easy case must not regress while fixing 29.97."""
    rate = Fraction(manifest["fps"])
    timebase = TimeBase(rate.numerator, rate.denominator)
    assert parse_value(timebase.value(90)) == Fraction(90, rate)


# --------------------------------------------------------------------------
# rounding direction (C2: a false negative costs a clip)
# --------------------------------------------------------------------------


def test_starts_floor_and_ends_ceil(ntsc):
    """Rounding to nearest can eat the first or last frame of a moment."""
    mid = float(ntsc.seconds(100) + ntsc.frame_duration / 2)
    assert ntsc.frame_at(mid, DOWN) == 100
    assert ntsc.frame_at(mid, UP) == 101


def test_a_window_never_shrinks_onto_the_grid(ntsc):
    """Whatever the boundaries, the exported clip covers at least the moment."""
    for offset in (0.001, 0.017, 0.033, 0.5):
        start_s, end_s = 10.0 + offset, 14.0 + offset
        start = ntsc.frame_at(start_s, DOWN)
        end = ntsc.frame_at(end_s, UP)
        assert float(ntsc.seconds(start)) <= start_s
        assert float(ntsc.seconds(end)) >= end_s


@pytest.mark.parametrize("frame", [1, 250, 899, 53947])
def test_a_time_exactly_on_a_boundary_does_not_expand(ntsc, frame):
    """MEASURED BUG. Frame 250 at 29.97 is 250250/30000 s exactly, but the
    nearest double is a hair larger, so ceil returned 251 and every
    frame-aligned window grew by a frame at each end. Boundaries are matched
    within BOUNDARY_EPSILON for exactly this reason."""
    exact = float(ntsc.seconds(frame))
    assert ntsc.frame_at(exact, DOWN) == frame
    assert ntsc.frame_at(exact, UP) == frame
    assert ntsc.frame_at(exact, NEAREST) == frame


def test_the_epsilon_does_not_swallow_a_real_sub_frame_offset(ntsc):
    """It has to separate float noise from a genuine boundary, not flatten
    everything onto the nearest frame."""
    a_third_of_a_frame = float(ntsc.seconds(100) + ntsc.frame_duration / 3)
    assert ntsc.frame_at(a_third_of_a_frame, DOWN) == 100
    assert ntsc.frame_at(a_third_of_a_frame, UP) == 101


def test_nearest_is_available_but_not_the_default(ntsc):
    below = float(ntsc.seconds(10) + ntsc.frame_duration * Fraction(1, 4))
    assert ntsc.frame_at(below, NEAREST) == 10
    above = float(ntsc.seconds(10) + ntsc.frame_duration * Fraction(3, 4))
    assert ntsc.frame_at(above, NEAREST) == 11


def test_an_unknown_rounding_mode_is_refused(ntsc):
    with pytest.raises(TimebaseError):
        ntsc.frame_at(1.0, "sideways")


# --------------------------------------------------------------------------
# parsing back
# --------------------------------------------------------------------------


def test_both_spelled_forms_parse():
    assert parse_value("1001/30000s") == Fraction(1001, 30000)
    assert parse_value("5s") == 5


def test_a_value_without_the_suffix_is_refused():
    with pytest.raises(TimebaseError):
        parse_value("1001/30000")
