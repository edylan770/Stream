"""The common time grid, and the rolling z-score (§6.2 steps 2-3).

§6.2 is emphatic about why the z-score has to roll:

> "Mic gain drifts. Games have different loudness. Energy changes over three
> hours. A global normalization constant means the detector finds 'the loud
> game' rather than 'the loud moment'."

Two details the spec does not mention but that decide whether the numbers are
right:

**The variance is computed on centered data.** `E[x²] − E[x]²` over dB values
around −50 subtracts 2500 from 2525 to get 25, and the accumulated error in a
cumulative sum over 144k samples is large enough to matter at that ratio.
Subtracting the global mean first costs one line and two orders of magnitude of
error.

**Sigma gets a floor.** §6.2 divides by the rolling standard deviation and says
nothing about it approaching zero. A digitally silent stretch is exactly
constant, so its sigma is exactly zero, and without the floor every dither in
the noise becomes a z-score of fifty — making the top candidate of every stream
a quiet moment.
"""

from __future__ import annotations

import numpy as np

from clipforge.signals import Series


def build(duration_s: float, grid_hz: float) -> np.ndarray:
    """Uniform timestamps `t_k = k / grid_hz` spanning the stream."""
    if duration_s <= 0:
        raise ValueError("stream duration must be positive to build a scoring grid")
    count = int(np.floor(duration_s * grid_hz)) + 1
    return np.arange(count, dtype=np.float64) / grid_hz


def resample(series: Series, grid: np.ndarray) -> np.ndarray:
    """Put a series onto the grid.

    Not a cast: signals carry `t0 = frame_s/2` because a frame-based
    measurement describes the centre of its window, so their timestamps do not
    start at zero and need not share a rate. Outside the series' own span
    `np.interp` holds the endpoint value, which is the right behaviour for an
    envelope — the alternative is a cliff to zero at both ends of every stream.
    """
    if len(series) == 0:
        return np.zeros_like(grid)
    if len(series) == 1:
        return np.full_like(grid, float(series.values[0]))
    return np.interp(grid, series.times(), series.values.astype(np.float64))


def rolling_zscore(
    values: np.ndarray, window_samples: int, std_floor: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """§6.2 step 3: z against a centred rolling window.

    Returns `(z, mean, std)` — the baseline is worth keeping because "why did
    this score highly" is usually answered by "because the five minutes around
    it were quiet", and that is invisible from z alone.

    Windows truncate at the stream edges rather than padding: the first and
    last few minutes are baselined against fewer samples, which is honest, where
    reflecting or zero-padding would invent content.
    """
    n = values.size
    if n == 0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty, empty

    half = max(int(window_samples) // 2, 0)
    data = values.astype(np.float64)

    # Center before accumulating. See the module docstring.
    offset = float(np.mean(data))
    centered = data - offset

    cumsum = np.concatenate(([0.0], np.cumsum(centered)))
    cumsum2 = np.concatenate(([0.0], np.cumsum(np.square(centered))))

    index = np.arange(n)
    lo = np.maximum(index - half, 0)
    hi = np.minimum(index + half + 1, n)
    count = (hi - lo).astype(np.float64)

    total = cumsum[hi] - cumsum[lo]
    total2 = cumsum2[hi] - cumsum2[lo]

    mean_centered = total / count
    # Float error can push a near-zero variance just below zero.
    variance = np.maximum(total2 / count - np.square(mean_centered), 0.0)
    std = np.sqrt(variance)

    effective_std = np.maximum(std, float(std_floor))
    z = (centered - mean_centered) / effective_std
    return z, mean_centered + offset, std


def window_samples(window_s: float, grid_hz: float) -> int:
    return max(int(round(window_s * grid_hz)), 1)
