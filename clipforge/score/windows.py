"""Peaks into candidate windows (§6.2 steps 6-7, §6.3, §6.6, §6.7).

§6.3 opens with the reason any of this exists:

> "Rank **windows**, not seconds. Without hysteresis you get hundreds of
> one-second fragments."

Three places where the spec's pseudocode does not survive contact with a signed
composite, and what is done instead:

**Peaks must clear a minimum value.** `exit_threshold = 0.35 * v_peak` assumes
`v_peak > 0`. A weighted sum of z-scores is signed, so for `v_peak = −2` the
exit threshold is `−0.7`, which is *above* the peak: the expansion loop stops
immediately and the window is empty. A local maximum in negative territory is a
local maximum of quietness and should never have been a candidate.

**`hysteresis_enter` is given a job.** §6.3 defines it and then never reads it —
the expansion loop only tests `exit_threshold`. Here it decides whether two
nearby peaks are two moments or one: a later peak opens a new window only if
the composite dipped below `enter × peak` between them.

**Thresholds are measured from each peak's own base, not from zero.** This is
the one that bites in practice. §6.3's `exit_threshold = 0.35 * v_peak` assumes
the composite returns to zero between moments. It does not: a marker plateau
holds `marker_definite` at 3.0 for thirty seconds, so a peak of 4.5 sits on a
pedestal of 3.0, and `0.35 * 4.5 = 1.6` is *below the pedestal*. Expansion never
terminates and every window clamps to `max_window_s` — measured on the fixture,
where every candidate came out 54 seconds long.

Using the peak's base — the level scipy already computes to measure prominence —
makes the threshold `base + 0.35 * (peak - base)`, i.e. "descend 65% of the way
back down". That is what §6.3 means by "falls back down", it is invariant to
whatever pedestal the peak is sitting on, and it costs nothing because
`find_peaks` returns the bases anyway.

**Auto-calibration floors its target.** §6.7 wants 80-150 candidates per three
hours. That rate applied to a ten-minute stream asks for four, and to a
one-minute fixture asks for half a candidate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks


@dataclass
class Peak:
    """A local maximum and the level it rises from.

    `base` is what makes the thresholds pedestal-invariant: it is the level the
    composite must descend to before reaching higher ground, which is exactly
    scipy's definition of a prominence base.
    """

    index: int
    value: float
    base: float

    @property
    def prominence(self) -> float:
        return self.value - self.base

    def level_at(self, ratio: float) -> float:
        """The absolute value `ratio` of the way up from base to peak."""
        return self.base + float(ratio) * self.prominence


@dataclass
class Window:
    """One candidate, before it becomes a database row."""

    index: int
    t_peak: float
    t_start: float
    t_end: float
    value: float
    score: float
    prominence: float = 0.0
    suppressed_by: float | None = None

    @property
    def duration_s(self) -> float:
        return self.t_end - self.t_start


@dataclass
class Calibration:
    """What auto-calibration settled on, and whether it worked."""

    prominence: float
    count: int
    target_low: int
    target_high: int
    iterations: int
    reached_target: bool
    reason: str = ""


# --------------------------------------------------------------------------
# peak finding
# --------------------------------------------------------------------------


def find(composite: np.ndarray, prominence: float, min_value: float) -> list[Peak]:
    """Peaks above `min_value`, with the base each rises from.

    scipy returns the midpoint of a flat maximum, which matters here: a marker
    plateau over a silent stretch is genuinely flat on top, and that is exactly
    the case where the marker is the only evidence there is.
    """
    if composite.size == 0:
        return []
    indices, properties = find_peaks(composite, prominence=max(float(prominence), 0.0))
    if indices.size == 0:
        return []

    prominences = properties["prominences"]
    return [
        Peak(index=int(i), value=float(composite[i]), base=float(composite[i] - p))
        for i, p in zip(indices, prominences, strict=True)
        if composite[i] > float(min_value)
    ]


def target_range(duration_s: float, per_hour: list[float], minimum: int) -> tuple[int, int]:
    """§6.7's rate turned into an absolute count for this stream."""
    hours = max(duration_s, 0.0) / 3600.0
    low, high = float(per_hour[0]), float(per_hour[1])
    return (
        max(int(round(low * hours)), int(minimum)),
        max(int(round(high * hours)), int(minimum)),
    )


def candidate_count(
    composite: np.ndarray, prominence: float, min_value: float, enter: float
) -> int:
    """How many candidates a given prominence would actually yield.

    Counts *after* peak merging, because that is what ends up in the database.
    Calibrating on raw peaks would target a number the operator never sees:
    measured on the fixture, three peaks became two candidates.
    """
    return len(merge_peaks(composite, find(composite, prominence, min_value), enter))


def calibrate(
    composite: np.ndarray, *, target_low: int, target_high: int,
    min_value: float, enter: float, max_iterations: int,
) -> Calibration:
    """Bisect the prominence threshold to land inside §6.7's range.

    Candidate count is non-increasing in prominence, so bisection is well
    defined. When no threshold reaches the range — a stream with fewer distinct
    moments than the target — the closest achievable is used and the shortfall
    is reported rather than silently accepted.
    """
    if composite.size == 0:
        return Calibration(0.0, 0, target_low, target_high, 0, False, "empty composite")

    span = float(np.max(composite) - np.min(composite))
    if span <= 0:
        return Calibration(0.0, 0, target_low, target_high, 0, False, "composite is flat")

    def count_at(prominence: float) -> int:
        return candidate_count(composite, prominence, min_value, enter)

    lo, hi = 0.0, span
    midpoint = (target_low + target_high) / 2.0
    best = Calibration(0.0, count_at(0.0), target_low, target_high, 0, False)

    for iteration in range(1, int(max_iterations) + 1):
        prominence = (lo + hi) / 2.0
        count = count_at(prominence)

        if abs(count - midpoint) < abs(best.count - midpoint):
            best = Calibration(prominence, count, target_low, target_high, iteration, False)

        if target_low <= count <= target_high:
            return Calibration(prominence, count, target_low, target_high, iteration, True)
        if count > target_high:
            lo = prominence   # too many candidates: raise the bar
        else:
            hi = prominence   # too few: lower it

    achievable = count_at(0.0)
    if achievable < target_low:
        return Calibration(
            0.0, achievable, target_low, target_high, int(max_iterations), False,
            f"only {achievable} candidate(s) exist above the minimum value at any threshold",
        )
    return Calibration(
        best.prominence, best.count, target_low, target_high, int(max_iterations), False,
        f"bisection settled at {best.count}, outside {target_low}-{target_high}",
    )


# --------------------------------------------------------------------------
# merging and expansion
# --------------------------------------------------------------------------


def merge_peaks(composite: np.ndarray, peaks: list[Peak], enter: float) -> list[Peak]:
    """Collapse peaks that belong to the same moment.

    This is `hysteresis_enter`'s job — §6.3 defines the parameter and §17 lists
    it as tunable, but the pseudocode never reads it. Two peaks separated by a
    trough that never descends far enough are one bump with a wobble on top; the
    higher is kept.

    Measured from the taller peak's base, for the same reason expansion is: on a
    marker plateau every trough is high in absolute terms and the comparison
    would be meaningless.
    """
    if len(peaks) <= 1:
        return list(peaks)

    kept = [peaks[0]]
    for peak in peaks[1:]:
        previous = kept[-1]
        trough = float(np.min(composite[previous.index:peak.index + 1]))
        taller = previous if previous.value >= peak.value else peak
        if trough < taller.level_at(enter):
            kept.append(peak)                        # a genuine separate moment
        elif peak.value > previous.value:
            kept[-1] = peak                          # same moment, better peak
    return kept


def expand(
    composite: np.ndarray, grid: np.ndarray, peak: Peak, *,
    exit_ratio: float, min_window_s: float, max_window_s: float, duration_s: float,
) -> tuple[float, float]:
    """§6.3: grow outward while the composite stays above the exit threshold.

    The threshold is `base + ratio * prominence`, not `ratio * value` — see the
    module docstring. On a pedestal the latter never terminates.
    """
    threshold = peak.level_at(exit_ratio)

    left = peak.index
    while left > 0 and composite[left - 1] > threshold:
        left -= 1
    right = peak.index
    last = composite.size - 1
    while right < last and composite[right + 1] > threshold:
        right += 1

    return clamp(
        float(grid[left]), float(grid[right]), float(grid[peak.index]),
        min_window_s=min_window_s, max_window_s=max_window_s, duration_s=duration_s,
    )


def clamp(
    t_start: float, t_end: float, t_peak: float, *,
    min_window_s: float, max_window_s: float, duration_s: float,
) -> tuple[float, float]:
    """Force the window into [min, max] while keeping the peak inside it.

    §6.3 says "clamp to [min_window_s=8, max_window_s=60]" without saying how.
    Growing and shrinking around `t_peak` is the only variant that cannot move
    the peak out of its own window — which the schema forbids outright.
    """
    duration_s = max(duration_s, 0.0)
    t_peak = min(max(t_peak, 0.0), duration_s)

    length = t_end - t_start
    if length < min_window_s:
        deficit = (min_window_s - length) / 2.0
        t_start, t_end = t_start - deficit, t_end + deficit
    elif length > max_window_s:
        half = max_window_s / 2.0
        t_start, t_end = t_peak - half, t_peak + half

    # Slide rather than truncate at the stream edges, so a moment near the start
    # still gets a full-length window.
    if t_start < 0.0:
        t_end += -t_start
        t_start = 0.0
    if t_end > duration_s:
        t_start = max(0.0, t_start - (t_end - duration_s))
        t_end = duration_s

    return min(t_start, t_peak), max(t_end, t_peak)


# --------------------------------------------------------------------------
# spacing (§6.6)
# --------------------------------------------------------------------------


def apply_spacing(
    windows: list[Window], *, window_s: float, factor: float, mode: str = "accepted_only"
) -> list[Window]:
    """§6.6, made deterministic.

    As written, §6.6 penalises a third candidate twice (0.5 x 0.5 = 0.25) and
    freezes its iteration order against pre-penalty scores while mutating those
    same scores. `accepted_only` is a single non-max-suppression pass in which
    only un-suppressed candidates suppress others, so each is penalised at most
    once and the outcome does not depend on how the ties fell.

    Suppressed candidates are kept at a reduced score rather than dropped: C2 is
    recall over precision, and the operator can always lengthen a clip.
    Distance is measured peak-to-peak.

    The final ranking differs from the suppression order — that is inherent to
    §6.6 penalising scores after sorting by them, not a defect here.
    """
    ordered = sorted(windows, key=lambda w: w.score, reverse=True)
    accepted: list[Window] = []

    for window in ordered:
        suppressor = next(
            (a for a in accepted if abs(a.t_peak - window.t_peak) <= window_s), None
        )
        if suppressor is None:
            accepted.append(window)
            continue

        window.score *= float(factor)
        window.suppressed_by = suppressor.t_peak
        if mode != "accepted_only":
            # §6.6 literally: a suppressed candidate can still suppress others,
            # which is what allows the compounding penalty.
            accepted.append(window)

    return sorted(windows, key=lambda w: w.score, reverse=True)


def build(
    composite: np.ndarray, grid: np.ndarray, peaks: list[Peak], *,
    exit_ratio: float, min_window_s: float, max_window_s: float, duration_s: float,
) -> list[Window]:
    out = []
    for peak in peaks:
        t_start, t_end = expand(
            composite, grid, peak, exit_ratio=exit_ratio,
            min_window_s=min_window_s, max_window_s=max_window_s, duration_s=duration_s,
        )
        out.append(Window(
            index=peak.index, t_peak=float(grid[peak.index]),
            t_start=t_start, t_end=t_end,
            value=peak.value, score=peak.value, prominence=peak.prominence,
        ))
    return out
