"""§6.5's combined score and §7.4's four sections.

§6.5 states one requirement in testable terms — "any formulation is acceptable
provided it satisfies: **high+high >> high+low**" — and one formula that does
not run. Both are asserted here: the requirement as a property, and the failure
mode of the literal formula as the reason `normalise` exists at all.

§7.4's sections are tested for the property it does not state and that decides
whether the screen is usable: they are DISJOINT and they cover everything.
"""

from __future__ import annotations

import numpy as np
import pytest

from clipforge import config
from clipforge.review import queries
from clipforge.score import combined as combined_score
from clipforge.score import runner as score_runner
from clipforge.score import windows


@pytest.fixture(scope="module")
def cfg():
    return config.load()


ALPHA = 0.5


# --------------------------------------------------------------------------
# §6.5's formula, and the two things wrong with it as written
# --------------------------------------------------------------------------


def test_high_and_high_beats_high_and_low():
    """§6.5's own acceptance criterion, in its own terms. The whole reason
    combined is not a sum: "that intersection is the operator's actual
    differentiator"."""
    both = combined_score.combined(np.array([1.0]), np.array([1.0]), ALPHA)[0]
    lopsided = combined_score.combined(np.array([1.0]), np.array([0.1]), ALPHA)[0]
    assert both > 2 * lopsided, "high+high >> high+low"


def test_a_balanced_pair_beats_a_lopsided_pair_of_the_same_total():
    """The property that makes it "product-like" rather than an average: 0.6+0.6
    and 1.0+0.2 sum the same, and only one of them is a moment where both
    profiles fired."""
    balanced = combined_score.combined(np.array([0.6]), np.array([0.6]), ALPHA)[0]
    lopsided = combined_score.combined(np.array([1.0]), np.array([0.2]), ALPHA)[0]
    assert balanced > lopsided


def test_the_spec_formula_is_nan_on_a_signed_composite():
    """WHY `normalise` is not optional. A profile composite is a sum of SIGNED
    weighted z-scores, and §6.5's `e_n ** alpha` is a fractional power — which
    is NaN for a negative base. A NaN score sorts unpredictably, and before
    commit 31's json fix it would have taken the whole review payload down."""
    with np.errstate(invalid="ignore"):
        literal = np.array([-0.4]) ** ALPHA
    assert np.isnan(literal).all()

    survived = combined_score.combined(
        combined_score.normalise(np.array([-0.4, 2.0])),
        combined_score.normalise(np.array([1.0, 3.0])),
        ALPHA,
    )
    assert np.isfinite(survived).all()


def test_normalise_is_share_of_the_best_not_min_max():
    """The difference is not cosmetic. Min-max sends the stream's WEAKEST
    candidate to exactly 0, and the product form then returns 0 for it whatever
    the other profile said — so on a stream where gameplay is mostly a marker
    detector, the funniest unmarked moment would rank last in the section §7.4
    puts first."""
    scores = np.array([1.0, 2.0, 4.0])
    out = combined_score.normalise(scores)
    assert out[-1] == pytest.approx(1.0)
    assert out[0] > 0.0, "the weakest candidate is not annihilated"
    assert out[0] == pytest.approx(0.25), "ratios are preserved"

    weakest_in_gameplay = combined_score.combined(
        combined_score.normalise(np.array([9.0, 1.0])),   # brilliant, then dull
        combined_score.normalise(np.array([1.0, 5.0])),   # dull, then brilliant
        ALPHA,
    )
    assert weakest_in_gameplay[0] > 0.0


def test_a_non_positive_composite_normalises_to_zero():
    """Below the rolling baseline is not a weaker kind of notable, it is the
    absence of it — and zero is what the product form needs as an annihilator."""
    out = combined_score.normalise(np.array([-3.0, 0.0, 2.0]))
    assert out[0] == 0.0 and out[1] == 0.0
    assert out[2] == pytest.approx(1.0)


def test_a_profile_that_scored_nothing_does_not_divide_by_zero():
    assert list(combined_score.normalise(np.array([-1.0, -2.0]))) == [0.0, 0.0]
    assert list(combined_score.normalise(np.zeros(0))) == []


def test_alpha_leans_the_geometric_part(cfg):
    """§17 lists `combined_score_alpha`; §6.5 never says which way it leans.
    Above 0.5 it weights the primary profile, which is the direction the
    operator would move it if entertainment turned out to matter more."""
    e, g = np.array([0.9]), np.array([0.3])
    even = combined_score.combined(e, g, 0.5)[0]
    leaning = combined_score.combined(e, g, 0.9)[0]
    assert leaning > even
    assert float(cfg.get("score.combined.alpha")) == 0.5


# --------------------------------------------------------------------------
# the agreement metric — the number this phase cannot answer for itself
# --------------------------------------------------------------------------


def test_identical_rankings_agree_completely():
    scores = np.array([5.0, 3.0, 1.0, 4.0])
    out = combined_score.rank_agreement(scores, scores.copy(), top_n=2)
    assert out["spearman"] == pytest.approx(1.0)
    assert out["top_n_overlap"] == pytest.approx(1.0)


def test_a_reversed_ranking_disagrees_completely():
    out = combined_score.rank_agreement(
        np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0]), top_n=1
    )
    assert out["spearman"] == pytest.approx(-1.0)
    assert out["top_n_overlap"] == pytest.approx(0.0)


def test_ties_make_the_correlation_undefined_rather_than_zero():
    """Reporting 0.0 would read as "the two rankings disagree", which is a
    different claim from "one of them has no ordering at all"."""
    out = combined_score.rank_agreement(np.array([1.0, 2.0]), np.array([7.0, 7.0]))
    assert out["spearman"] is None


def test_one_candidate_has_no_ordering_to_agree_about():
    out = combined_score.rank_agreement(np.array([1.0]), np.array([1.0]))
    assert out["spearman"] is None and out["n"] == 1


def test_rankings_must_cover_the_same_candidates():
    with pytest.raises(ValueError):
        combined_score.rank_agreement(np.array([1.0, 2.0]), np.array([1.0]))


def test_ranks_average_their_ties():
    """Spearman is Pearson over ranks, and ties have to share a rank or the
    correlation is computed against an ordering nobody chose."""
    ranks = combined_score._ranks(np.array([5.0, 1.0, 5.0, 3.0]))
    assert ranks[0] == ranks[2] == pytest.approx(2.5)
    assert ranks[1] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# §7.4's four sections
# --------------------------------------------------------------------------


def view(index, *, combined, entertainment, gameplay, marker=False,
         profile="entertainment+gameplay"):
    return queries.CandidateView(
        id=index, index=index, t_start=float(index), t_end=float(index) + 5.0,
        t_peak=float(index) + 1.0, score=combined,
        score_entertainment=entertainment, score_gameplay=gameplay,
        profile=profile, marker_anchored=marker,
    )


def a_spread(n=30, markers=(3, 7, 25)):
    """A plausible candidate set: descending combined, mixed profile leanings."""
    return [
        view(i,
             combined=float(n - i) / n,
             entertainment=float(n - i),
             gameplay=float(i if i % 3 else n - i),
             marker=i in markers)
        for i in range(n)
    ]


def test_the_sections_are_disjoint_and_cover_everything(cfg):
    """The property §7.4 does not state and that decides whether the screen is
    usable: one moment in three sections would spend §7.1's four-second budget
    rating the same window three times."""
    top_n = int(cfg.get("review.sections.combined_top_n"))
    out = queries.assign_sections(a_spread(), top_n)

    assert len(out) == 30
    assert len({v.id for v in out}) == 30
    assert all(v.section in dict(queries.SECTIONS) for v in out)


def test_the_sections_come_back_in_7_4s_order(cfg):
    """"Combined-score winners — the best content, reviewed first"."""
    out = queries.assign_sections(a_spread(), int(cfg.get("review.sections.combined_top_n")))
    seen = [v.section for v in out]
    order = [key for key, _ in queries.SECTIONS]
    assert seen == sorted(seen, key=order.index)
    assert seen[0] == "combined"


def test_the_combined_section_is_the_top_n_by_combined(cfg):
    top_n = int(cfg.get("review.sections.combined_top_n"))
    spread = a_spread()
    out = queries.assign_sections(spread, top_n)

    winners = [v for v in out if v.section == "combined"]
    assert len(winners) == top_n
    best = sorted(spread, key=lambda v: -v.score)[:top_n]
    assert {v.id for v in winners} == {v.id for v in best}


def test_a_marker_candidate_is_in_the_combined_section_or_the_safety_net(cfg):
    """§7.4's fourth section is "marker-anchored candidates that did NOT rank
    highly". One that did rank highly belongs at the top, not in the safety
    net."""
    top_n = int(cfg.get("review.sections.combined_top_n"))
    out = queries.assign_sections(a_spread(), top_n)
    for v in out:
        if v.marker_anchored:
            assert v.section in {"combined", "markers"}
        else:
            assert v.section != "markers"


def test_the_middle_two_sections_split_on_which_profile_is_more_confident(cfg):
    """Compared as a SHARE of each profile's own best — raw composites are not
    comparable across profiles, since §6.5's two sum 20.7 and 21.0 of weight of
    which very different amounts have a producer."""
    views = [
        view(0, combined=0.9, entertainment=10.0, gameplay=1.0),
        view(1, combined=0.8, entertainment=1.0, gameplay=10.0),
    ]
    out = {v.id: v.section for v in queries.assign_sections(views, top_n=0)}
    assert out[0] == "entertainment"
    assert out[1] == "gameplay"


def test_within_a_section_the_ranking_is_that_sections_own_score():
    views = [
        view(0, combined=0.9, entertainment=1.0, gameplay=0.0),
        view(1, combined=0.8, entertainment=9.0, gameplay=0.0),
    ]
    out = queries.assign_sections(views, top_n=0)
    entertainment = [v for v in out if v.section == "entertainment"]
    assert [v.id for v in entertainment] == [1, 0], "ranked by entertainment, not combined"


def test_a_single_profile_stream_gets_no_sections():
    """With one profile there is no gameplay ranking and combined mirrors the
    primary, so four headers would be four names for one ordering. The rail
    says so in prose instead."""
    views = [view(i, combined=1.0, entertainment=1.0, gameplay=0.0, profile="naive")
             for i in range(3)]
    out = queries.assign_sections(views, top_n=2)
    assert all(v.section == "" for v in out)
    assert not queries.is_sectioned("naive")
    assert queries.is_sectioned("entertainment+gameplay")


def test_index_is_renumbered_into_review_order(cfg):
    out = queries.assign_sections(a_spread(), int(cfg.get("review.sections.combined_top_n")))
    assert [v.index for v in out] == list(range(len(out)))


def test_no_candidates_is_not_an_error():
    assert queries.assign_sections([], top_n=20) == []


# --------------------------------------------------------------------------
# §6.2 step 8's cross-profile merge
# --------------------------------------------------------------------------


class _Profile:
    """Just enough of a §6.5 profile for the merge, which only reads the name."""

    def __init__(self, name):
        self.name = name


def a_run(name, spans):
    """One profile's windows: `(t_start, t_end, t_peak, score)` each."""
    built = [
        windows.Window(index=int(peak * 10), t_peak=peak, t_start=start, t_end=end,
                       value=score, score=score)
        for start, end, peak, score in spans
    ]
    return score_runner.ProfileRun(
        profile=_Profile(name), tracks=[], smoothed=np.zeros(1),
        calibration=None, built=built, missing=[],
        normalised=combined_score.normalise(np.array([w.score for w in built])),
    )


def test_overlapping_windows_from_two_profiles_become_one_moment():
    """§6.2 step 8. C2 decides the boundaries: earliest start, latest end,
    because expanding is the direction in which a false positive is cheap."""
    merged = score_runner.merge_across_profiles([
        a_run("entertainment", [(10.0, 20.0, 15.0, 5.0)]),
        a_run("gameplay", [(18.0, 30.0, 25.0, 4.0)]),
    ])
    assert len(merged) == 1
    assert (merged[0].t_start, merged[0].t_end) == (10.0, 30.0)
    assert merged[0].found_by == ("entertainment", "gameplay")


def test_windows_that_do_not_overlap_stay_separate():
    merged = score_runner.merge_across_profiles([
        a_run("entertainment", [(10.0, 20.0, 15.0, 5.0)]),
        a_run("gameplay", [(40.0, 50.0, 45.0, 4.0)]),
    ])
    assert len(merged) == 2
    assert [m.found_by for m in merged] == [("entertainment",), ("gameplay",)]


def test_the_peak_comes_from_the_more_confident_profile():
    """Compared as a SHARE of each profile's own best. Raw composites are not
    comparable across profiles — §6.5's two sum 20.7 and 21.0 of weight, of
    which very different amounts have a producer today."""
    # gameplay's window is its ONLY one, so it is that profile's best (1.0);
    # entertainment's is a third of its own best, even though it scores higher.
    merged = score_runner.merge_across_profiles([
        a_run("entertainment", [(10.0, 20.0, 15.0, 3.0), (100.0, 110.0, 105.0, 9.0)]),
        a_run("gameplay", [(18.0, 30.0, 25.0, 2.0)]),
    ])
    moment = next(m for m in merged if m.t_start == 10.0)
    assert moment.t_peak == 25.0, "the profile that is more sure of it wins the peak"


def test_two_windows_from_one_profile_are_never_merged():
    """That has been §6.6's business since Phase 1, and merging them here would
    silently change what a single-profile run produces."""
    merged = score_runner.merge_across_profiles([
        a_run("entertainment", [(10.0, 25.0, 15.0, 5.0), (20.0, 35.0, 30.0, 4.0)]),
    ])
    assert len(merged) == 2
    assert [(m.t_start, m.t_end) for m in merged] == [(10.0, 25.0), (20.0, 35.0)]


def test_a_second_window_from_the_same_profile_becomes_its_own_moment():
    """A moment absorbs at most one window per profile. The overlapping leftover
    is a near-duplicate for §6.6 to suppress, not a second merge."""
    merged = score_runner.merge_across_profiles([
        a_run("entertainment", [(10.0, 20.0, 15.0, 5.0)]),
        a_run("gameplay", [(12.0, 22.0, 16.0, 4.0), (14.0, 24.0, 18.0, 3.0)]),
    ])
    assert len(merged) == 2
    assert [len(m.found_by) for m in merged] == [2, 1]


def test_a_window_joins_the_moment_it_overlaps_most():
    merged = score_runner.merge_across_profiles([
        a_run("entertainment", [(0.0, 12.0, 6.0, 5.0), (11.0, 40.0, 30.0, 4.0)]),
        a_run("gameplay", [(10.0, 35.0, 20.0, 3.0)]),
    ])
    joined = [m for m in merged if len(m.found_by) == 2]
    assert len(joined) == 1
    # It shares 2 s with 0..12 and 24 s with 11..40, so it joins the second —
    # visible in the end, which only the second moment could have supplied.
    assert joined[0].t_end == 40.0
    # ...and the union then reaches back to the gameplay window's own start.
    assert joined[0].t_start == 10.0
    assert [m.t_end for m in merged if len(m.found_by) == 1] == [12.0]


class _Ctx:
    """A StageContext stand-in for the two functions that only read config."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.logged = []

    def log(self, message):
        self.logged.append(message)


def test_a_merged_window_is_clamped_back_into_6_3s_range(cfg):
    """Each profile clamped its own windows, but two that overlap by a second
    and extend in opposite directions union to almost twice `max_window_s` —
    and a 110-second candidate is not a moment, it is two."""
    max_window = float(cfg.get("score.window.max_window_s"))
    merged = score_runner.merge_across_profiles([
        a_run("entertainment", [(0.0, max_window, max_window / 2, 5.0)]),
        a_run("gameplay", [(max_window - 1.0, 2 * max_window - 1.0, max_window, 4.0)]),
    ])
    assert merged[0].duration_s > max_window, "the union really does breach it"

    ctx = _Ctx(cfg)
    score_runner._clamp_merged(ctx, merged, duration_s=1000.0)
    assert merged[0].duration_s == pytest.approx(max_window)
    assert merged[0].t_start <= merged[0].t_peak <= merged[0].t_end, (
        "§3.2's CHECK (t_peak BETWEEN t_start AND t_end) refuses anything else"
    )
    assert ctx.logged, "and it says so, because it is the merge undoing its own work"


def test_a_window_inside_the_range_is_left_exactly_where_it_was(cfg):
    merged = score_runner.merge_across_profiles([
        a_run("entertainment", [(10.0, 20.0, 15.0, 5.0)]),
    ])
    before = (merged[0].t_start, merged[0].t_end)
    score_runner._clamp_merged(_Ctx(cfg), merged, duration_s=1000.0)
    assert (merged[0].t_start, merged[0].t_end) == before
