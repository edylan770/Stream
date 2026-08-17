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

**Some signals have gaps, and a gap is not a zero.** Every Phase 1 signal was
defined at every instant: RMS always has a value, even over silence. Pitch does
not — roughly a fifth of speech frames are unvoiced and silence is unvoiced
entirely, so `mic_f0` arrives full of NaN, and Phase 3's input signals will
arrive with holes wherever the logger was not running. A cumulative sum poisons
every sample *after* the first NaN, so the naive path does not degrade near a
gap, it destroys the rest of the stream. Both functions below take a masked
path when the input has gaps and the original path exactly when it does not, so
no existing signal's numbers move.
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

    times = series.times()
    values = series.values.astype(np.float64)
    finite = np.isfinite(values)
    if finite.all():
        return np.interp(grid, times, values)

    if not finite.any():
        return np.full_like(grid, np.nan)

    # A gap must not be bridged. Interpolating across four unvoiced seconds
    # would draw a smooth pitch glide between two words and hand it to
    # `mic_f0_variance` as prosody. So the observed samples are interpolated
    # normally and then every grid point whose NEAREST observation was missing
    # is put back to NaN — no invention, and no widening of the gap either.
    out = np.interp(grid, times[finite], values[finite])
    nearest = np.clip(
        np.round((grid - series.t0) * series.sample_rate_hz).astype(np.int64),
        0, values.size - 1,
    )
    out[~finite[nearest]] = np.nan
    return out


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
    observed = np.isfinite(data)

    # Center before accumulating. See the module docstring.
    offset = float(np.mean(data[observed])) if observed.any() else 0.0
    centered = data - offset

    # A missing sample must contribute to neither the sums nor the count. Zero
    # in the accumulator plus zero in the count is exactly "this sample was not
    # observed"; leaving the NaN in would make every later sample NaN too,
    # because a cumulative sum never recovers.
    filled = np.where(observed, centered, 0.0)
    cumsum = np.concatenate(([0.0], np.cumsum(filled)))
    cumsum2 = np.concatenate(([0.0], np.cumsum(np.square(filled))))
    cumcount = np.concatenate(([0.0], np.cumsum(observed.astype(np.float64))))

    index = np.arange(n)
    lo = np.maximum(index - half, 0)
    hi = np.minimum(index + half + 1, n)
    count = cumcount[hi] - cumcount[lo]

    total = cumsum[hi] - cumsum[lo]
    total2 = cumsum2[hi] - cumsum2[lo]

    # A window with no observation at all has no baseline to offer. That is a
    # NaN rather than a zero: "the pitch here was average" is a claim, and there
    # was nothing here to average.
    safe = np.maximum(count, 1.0)
    mean_centered = np.where(count > 0, total / safe, np.nan)
    # Float error can push a near-zero variance just below zero.
    variance = np.maximum(total2 / safe - np.square(np.nan_to_num(mean_centered)), 0.0)
    std = np.where(count > 0, np.sqrt(variance), np.nan)

    effective_std = np.maximum(np.nan_to_num(std, nan=0.0), float(std_floor))
    z = (centered - mean_centered) / effective_std
    # An unobserved sample has no z-score of its own, whatever its neighbours
    # said. `composite_of` is what decides that a missing signal contributes
    # nothing to the sum; this function's job is only to refuse to invent one.
    z = np.where(observed, z, np.nan)
    return z, mean_centered + offset, std


def window_samples(window_s: float, grid_hz: float) -> int:
    return max(int(round(window_s * grid_hz)), 1)
