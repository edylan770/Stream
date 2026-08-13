"""Source time -> output time.

Every §8 feature that touches time goes through `EditPlan`. The cases that
matter are the ones where an off-by-one is invisible in the output file: a
caption written in stream time produces a clip with no captions and an ffmpeg
exit code of 0.
"""

from __future__ import annotations

import pytest

from clipforge.render import timeline
from clipforge.render.timeline import Cut


def test_a_plan_with_no_cuts_is_a_subtraction():
    plan = timeline.build(100.0, 160.0)

    assert plan.duration == pytest.approx(60.0)
    assert plan.map_time(100.0) == pytest.approx(0.0)
    assert plan.map_time(130.0) == pytest.approx(30.0)
    assert plan.map_time(160.0) == pytest.approx(60.0)


def test_times_outside_the_window_do_not_map():
    plan = timeline.build(100.0, 160.0)

    assert plan.map_time(99.0) is None
    assert plan.map_time(161.0) is None


def test_a_cut_shifts_everything_after_it():
    """§8.2's filler removal. Removing 2 s at t=110 moves every later caption
    2 s earlier, which is why captions are generated against the plan."""
    plan = timeline.build(100.0, 160.0, [Cut(110.0, 112.0, "filler")])

    assert plan.duration == pytest.approx(58.0)
    assert plan.map_time(105.0) == pytest.approx(5.0)
    assert plan.map_time(111.0) is None
    assert plan.map_time(120.0) == pytest.approx(18.0)


def test_a_cut_is_half_open_so_a_word_ending_on_it_survives():
    """Without this rule a word ending exactly where a cut begins vanishes."""
    plan = timeline.build(0.0, 10.0, [Cut(4.0, 5.0)])

    assert plan.map_time(4.0) is None      # the cut owns its start
    assert plan.map_time(5.0) == pytest.approx(4.0)


def test_cuts_are_sorted_clipped_and_merged():
    plan = timeline.build(
        0.0, 10.0,
        [Cut(6.0, 7.0), Cut(-5.0, 1.0), Cut(2.0, 3.0), Cut(3.0, 4.0), Cut(9.0, 20.0)],
    )

    assert [(c.t_start, c.t_end) for c in plan.cuts] == [
        (0.0, 1.0), (2.0, 4.0), (6.0, 7.0), (9.0, 10.0)
    ]
    assert plan.removed == pytest.approx(5.0)
    assert plan.duration == pytest.approx(5.0)


def test_keep_ranges_covers_the_window_exactly():
    plan = timeline.build(0.0, 10.0, [Cut(2.0, 3.0), Cut(6.0, 6.5)])

    ranges = plan.keep_ranges()

    assert ranges == [(0.0, 2.0), (3.0, 6.0), (6.5, 10.0)]
    assert sum(end - start for start, end in ranges) == pytest.approx(plan.duration)


def test_keep_ranges_is_a_single_range_when_nothing_is_cut():
    """So `trim`/`concat` never needs a special case for 'no editing'."""
    assert timeline.build(3.0, 8.0).keep_ranges() == [(3.0, 8.0)]


def test_a_span_straddling_a_cut_is_shortened_not_dropped():
    """C2: the syllable that survived is still worth showing."""
    plan = timeline.build(0.0, 10.0, [Cut(4.0, 5.0)])

    assert plan.map_span(3.5, 4.5) == pytest.approx((3.5, 4.0))
    assert plan.map_span(4.5, 5.5) == pytest.approx((4.0, 4.5))


def test_a_span_entirely_inside_a_cut_is_dropped():
    plan = timeline.build(0.0, 10.0, [Cut(4.0, 5.0)])

    assert plan.map_span(4.2, 4.8) is None


def test_a_span_entirely_outside_the_window_is_dropped():
    """Clamping before checking would collapse it onto the clip's last instant
    and give it a caption there."""
    plan = timeline.build(10.0, 20.0)

    assert plan.map_span(30.0, 30.4) is None
    assert plan.map_span(1.0, 2.0) is None


def test_a_span_overlapping_an_edge_is_clamped_inwards():
    plan = timeline.build(10.0, 20.0)

    assert plan.map_span(9.5, 10.5) == pytest.approx((0.0, 0.5))
    assert plan.map_span(19.5, 21.0) == pytest.approx((9.5, 10.0))


def test_a_zero_length_span_still_maps():
    """An interpolated word can land on top of its own anchor."""
    plan = timeline.build(0.0, 10.0)

    assert plan.map_span(4.0, 4.0) == pytest.approx((4.0, 4.0))


def test_an_empty_or_inverted_window_is_refused():
    with pytest.raises(timeline.TimelineError):
        timeline.build(10.0, 10.0)
    with pytest.raises(timeline.TimelineError):
        timeline.build(10.0, 5.0)


def test_cutting_the_whole_window_is_refused():
    """Better than handing ffmpeg a zero-length concat and reading the filter
    graph error that comes back."""
    with pytest.raises(timeline.TimelineError, match="was cut"):
        timeline.build(0.0, 10.0, [Cut(0.0, 10.0)])
