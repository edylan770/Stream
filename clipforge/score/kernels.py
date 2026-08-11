"""Discrete events into continuous signals (§6.2 step 4).

**Every kernel peaks at exactly 1.0.** §6.2 says "apply a decay kernel
(Gaussian, sigma configurable per event kind)" and stops there, which leaves the
most consequential detail unstated. A unit-*area* Gaussian at sigma=10 s peaks
near 0.004; multiplied by §6.5's `marker_definite: 3.0` the highest-weighted
signal in the entire system would contribute about 0.01 against z-scores that
swing plus or minus three. The weights would be uninterpretable and the markers
silently ignored. Unit-peak normalization is what makes 1.0 mean "one standard
deviation of notability" and the §6.5 table mean anything at all.

**The marker kernel is a plateau, not a point.** §4.3 anchors a marker at
`t − 20s`, but it also says realization plus reaction is 5-15 seconds — a
*range*. A symmetric bump at `t − 20` is a point estimate that misses by
fourteen seconds when the real delay was six. The plateau covers the whole
plausible span at full height and falls off over a shoulder, which serves C2:
recall over precision, because a false positive costs three seconds of review
and a false negative costs the clip.
"""

from __future__ import annotations

import numpy as np

#: Marker kinds get the asymmetric plateau; anything else (Phase 3+ kills,
#: laughs) gets a symmetric Gaussian, which is what §6.2 describes.
MARKER_PREFIX = "marker_"


def plateau(
    grid: np.ndarray,
    times: np.ndarray | list[float],
    *,
    pre_s: float,
    post_s: float,
    shoulder_s: float,
    combine: str = "max",
) -> np.ndarray:
    """Unit-peak asymmetric plateau at each event time.

    Full height across `[t - pre_s, t + post_s]`, falling to zero over
    `shoulder_s` on each side with a raised cosine. The asymmetry is the point:
    the moment being marked happened *before* the press, never after.
    """
    out = np.zeros_like(grid, dtype=np.float64)
    if len(times) == 0:
        return out

    shoulder = max(float(shoulder_s), 0.0)
    for t in times:
        start, end = t - float(pre_s), t + float(post_s)
        contribution = np.zeros_like(out)

        inside = (grid >= start) & (grid <= end)
        contribution[inside] = 1.0

        if shoulder > 0:
            rising = (grid >= start - shoulder) & (grid < start)
            contribution[rising] = _raised_cosine((grid[rising] - (start - shoulder)) / shoulder)
            falling = (grid > end) & (grid <= end + shoulder)
            contribution[falling] = _raised_cosine((end + shoulder - grid[falling]) / shoulder)

        out = np.maximum(out, contribution) if combine == "max" else out + contribution

    return np.clip(out, 0.0, None)


def gaussian(
    grid: np.ndarray,
    times: np.ndarray | list[float],
    *,
    sigma_s: float,
    combine: str = "sum",
) -> np.ndarray:
    """Unit-peak Gaussian at each event time — §6.2's decay kernel.

    Summing is right here and wrong for markers: three kills in eight seconds
    genuinely is stronger evidence than one, whereas two marker presses eight
    seconds apart describe the same moment.
    """
    out = np.zeros_like(grid, dtype=np.float64)
    if len(times) == 0 or sigma_s <= 0:
        return out

    for t in times:
        # Unit PEAK, not unit area: no 1/(sigma*sqrt(2*pi)) factor.
        contribution = np.exp(-0.5 * np.square((grid - t) / float(sigma_s)))
        out = np.maximum(out, contribution) if combine == "max" else out + contribution

    return out


def _raised_cosine(x: np.ndarray) -> np.ndarray:
    """Smooth 0 to 1 over x in [0, 1]. Continuous in value and slope."""
    return 0.5 * (1.0 - np.cos(np.pi * np.clip(x, 0.0, 1.0)))


def for_kind(kind: str, grid: np.ndarray, times: list[float], cfg) -> np.ndarray:
    """Build the kernel a given event kind should contribute."""
    if kind.startswith(MARKER_PREFIX):
        return plateau(
            grid, times,
            pre_s=cfg.get("score.markers.pre_s"),
            post_s=cfg.get("score.markers.post_s"),
            shoulder_s=cfg.get("score.markers.shoulder_s"),
            combine=str(cfg.get("score.markers.combine")),
        )
    return gaussian(
        grid, times,
        sigma_s=cfg.get("score.events.default_sigma_s"),
        combine=str(cfg.get("score.events.combine")),
    )


def centre_of_mass(pre_s: float, post_s: float) -> float:
    """Where the plateau's mass sits relative to the press.

    §17 tunes `marker_retro_offset_s` against "the fraction of marker-anchored
    windows where the moment is inside the window". The plateau replaced that
    single number, so this is what that parameter now corresponds to — keeping
    §17's tuning procedure meaningful rather than orphaning it.
    """
    return (float(pre_s) - float(post_s)) / 2.0
