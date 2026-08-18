"""`score` — signals and events into candidates (§5.1 stage 14, §6).

§6.1 is the load-bearing claim:

> "Scoring is a **pure function** over stored signals. It must be re-runnable
> across the entire back catalog in seconds. **Consequence:** the first 20
> streams are not wasted on bad weights."

Everything here follows from that. No file is read except the database, nothing
is written except `candidates` rows, and a re-score costs nothing but CPU. The
one thing that must survive a re-score is the operator's ratings, which §13.2
calls the irreplaceable tier — so candidates are append-only generations and
ratings are carried onto the new generation by time overlap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.ndimage import gaussian_filter1d

from clipforge import db, signals
from clipforge.pipeline.context import StageContext
from clipforge.score import combined as combined_score
from clipforge.score import derived, features, gates, grid, kernels, windows

#: §5.4.1 continuous signals live in `signal_series`; §5.4.2 events live in
#: `events`. A profile weight naming neither is reported rather than ignored.
EVENT_SOURCE_KINDS = ("marker_maybe", "marker_definite")


class ScoreError(RuntimeError):
    pass


@dataclass
class ScoreResult:
    generation: int
    profile: str
    config_version: str
    candidates: int
    calibration: windows.Calibration
    inherited_ratings: int = 0
    replaced_generation: bool = False
    missing_weights: list[str] = field(default_factory=list)
    #: §6.5 runs a set. `calibration` and `missing_weights` stay the primary's,
    #: because everything that reads one profile still legitimately wants one.
    calibrations: dict = field(default_factory=dict)
    missing_by_profile: dict = field(default_factory=dict)
    #: §6.5's combined ranking against the primary's — how much the combined
    #: section actually differs from the list under it. Empty when one profile
    #: runs, because a ranking cannot disagree with itself.
    rank_agreement: dict = field(default_factory=dict)
    #: How many moments each combination of profiles found. The complement to
    #: `rank_agreement`: two profiles can order the same moments identically and
    #: still disagree about which moments there ARE, and §6.2 step 8's merge is
    #: what turns the second answer into candidates.
    found_by: dict = field(default_factory=dict)
    #: §6.4. `penalty_shares` is the fraction of the stream each negative held
    #: down; `gate_reasons` says why a negative was not evaluated at all, which
    #: is a different thing from "it never fired" and must not read as one.
    penalty_shares: dict[str, float] = field(default_factory=dict)
    gate_reasons: dict[str, str] = field(default_factory=dict)
    #: What the stream looks like with §6.4 switched off, independently
    #: calibrated. None when nothing fired, so there was nothing to counterfact.
    counterfactual: dict | None = None


# --------------------------------------------------------------------------
# building the composite
# --------------------------------------------------------------------------


def std_floor_for(cfg, name: str) -> float:
    """The z-score's sigma floor for one signal.

    `score.zscore_std_floor` is in DECIBELS and coupled to `extract.rms.db_floor`
    — it exists so a digitally silent stretch, whose sigma is exactly zero, does
    not turn dither into z = 50. Phase 3 added signals on other scales, and one
    number cannot serve them: for a 0..1 gate firing under 25% of the time sigma
    is below 0.5, so the dB floor binds and z lands at exactly ~2.0 however rare
    the firing is. That was the dB number rescuing a signal it knew nothing
    about, by luck.
    """
    overrides = cfg.get("score.zscore_std_floor_by_signal", {}) or {}
    if name in overrides:
        return float(overrides[name])
    return float(cfg.get("score.zscore_std_floor"))


def gather_raw(
    ctx: StageContext, timeline: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Every stored signal, resampled onto the grid, before any normalization.

    Split out from `build_tracks` because §5.4.1's derived signals are functions
    of these *raw* arrays — `sudden_silence` needs the RMS, not its z-score —
    and because it puts "what does this stream have" in one place.
    """
    raw: dict[str, np.ndarray] = {}
    units: dict[str, str] = {}
    for name, series in signals.load_all(ctx.conn, ctx.stream_id).items():
        raw[name] = grid.resample(series, timeline)
        units[name] = str(series.params.get("unit", "dB"))
    return raw, units


@dataclass
class Pool:
    """Everything a scoring run computes ONCE, before any profile is applied.

    §6.5 runs two profiles over one stream, and the only thing that differs
    between them is the weights. The z-scores, the derived signals, the event
    kernels and §6.4's gates are all profile-independent — so computing them per
    profile would re-read every BLOB and recompute every rolling z-score for a
    stage §6.1 promises is free.

    The raw arrays are kept alongside the normalised ones because §6.4's gates
    read them: "audio energy below its rolling baseline" is a statement about
    dB, not about a z-score.
    """

    ctx: StageContext
    timeline: np.ndarray
    raw: dict[str, np.ndarray]
    units: dict[str, str]
    events: dict[str, list[float]]
    baseline_samples: int
    _z: dict[str, features.SignalTrack] = field(default_factory=dict)
    _kernels: dict[str, np.ndarray] = field(default_factory=dict)

    def continuous(self, name: str, weight: float) -> features.SignalTrack:
        """A z-scored track, computed once and re-weighted per profile."""
        if name not in self._z:
            raw = self.raw[name]
            z, baseline, _ = grid.rolling_zscore(
                raw, self.baseline_samples, std_floor_for(self.ctx.cfg, name)
            )
            self._z[name] = features.SignalTrack(
                name=name, values=z, weight=0.0, is_event=False,
                baseline=baseline, raw=raw, unit=self.units.get(name, "dB"),
            )
        return replace(self._z[name], weight=float(weight))

    def event(self, name: str, weight: float) -> features.SignalTrack:
        if name not in self._kernels:
            self._kernels[name] = kernels.for_kind(
                name, self.timeline, list(self.events.get(name, [])), self.ctx.cfg
            )
        return features.SignalTrack(
            name=name, values=self._kernels[name], weight=float(weight), is_event=True,
        )

    def has(self, name: str) -> bool:
        return name in self.raw or name in self.events or name in EVENT_SOURCE_KINDS


def build_pool(ctx: StageContext, timeline: np.ndarray) -> Pool:
    """Every signal and event this stream has, on the grid, unweighted.

    Continuous signals are z-scored (§6.2 step 3); event kernels are not. That
    asymmetry is deliberate and is what makes §6.5's weights comparable: both
    arrive in units where 1.0 is roughly one standard deviation of notability,
    z-scores by construction and kernels by unit-peak normalization.

    §5.4.1's derived signals (`score/derived.py`) join the same pool as the
    stored ones before anything is weighted, so the profile lookup and A9's
    weight-0 sweep both see them with no special case. §6.2 puts step 5's
    composites before step 6's weighting for the same reason.
    """
    raw_values, units = gather_raw(ctx, timeline)
    events = load_events(ctx)

    # §6.2 step 5: "Compute composite/correlation signals" — before the profiles
    # are applied, so a derived signal is weightable exactly like a stored one.
    derived_values, derived_units, derived_events = derived.compute(
        raw_values, timeline, ctx.cfg
    )
    raw_values.update(derived_values)
    units.update(derived_units)
    for kind, times in derived_events.items():
        events.setdefault(kind, list(times))

    return Pool(
        ctx=ctx, timeline=timeline, raw=raw_values, units=units, events=events,
        baseline_samples=grid.window_samples(
            ctx.cfg.get("score.rolling_baseline_window_s"),
            ctx.cfg.get("score.score_grid_hz"),
        ),
    )


def tracks_for(pool: Pool, profile) -> tuple[list[features.SignalTrack], list[str]]:
    """One profile's weighted tracks, plus the weights nothing produced.

    Called once per §6.5 profile over the same pool, so a second profile costs
    a dictionary lookup and a multiply rather than a second pass over the
    database.
    """
    tracks: list[features.SignalTrack] = []
    missing: list[str] = []

    for name, weight in profile.weights.items():
        if name in pool.raw:
            tracks.append(pool.continuous(name, float(weight)))
        elif name in pool.events or name in EVENT_SOURCE_KINDS:
            tracks.append(pool.event(name, float(weight)))
        else:
            missing.append(name)

    # A9: "Feature vectors are logged in full, always, EVEN FOR SIGNALS NOT
    # CURRENTLY WEIGHTED. They cannot be reconstructed retroactively and they
    # are the input to all future tuning."
    #
    # Loading only what the profile weights satisfied that by accident while
    # Phase 1 had three signals and the naive profile named all three. Phase 2's
    # speech_rate and swear_density are the first that exist without being
    # weighted, and they would have been stored, declared in feature_schema.yaml,
    # and written into every vector as null — with §17's tuning input silently
    # absent. The same was true of stored EVENTS until commit 36.
    #
    # Weight 0 contributes exactly nothing to the composite (`composite_of`
    # skips these outright), so no score moves.
    weighted = {track.name for track in tracks}
    for name in pool.raw:
        if name not in weighted:
            tracks.append(pool.continuous(name, 0.0))
    for kind in pool.events:
        if kind not in weighted:
            tracks.append(pool.event(kind, 0.0))

    return tracks, missing


def load_events(ctx: StageContext) -> dict[str, list[float]]:
    rows = ctx.conn.execute(
        "SELECT kind, t FROM events WHERE stream_id = ? ORDER BY t", (ctx.stream_id,)
    ).fetchall()
    out: dict[str, list[float]] = {}
    for row in rows:
        out.setdefault(row["kind"], []).append(float(row["t"]))
    return out


def load_intervals(conn, stream_id: str, kind: str) -> list[tuple[float, float]] | None:
    """Event rows of one kind as spans, or None when nothing produces them.

    §6.4's `menu_screen_active` is a STATE, and `events.t_end` has carried
    "NULL for instantaneous" since `0001_init.sql` — so a span is already
    expressible and `feature_schema.yaml` does not have to move `menu_screen`
    out of its `events:` group.

    None rather than an empty list, deliberately: "§5.9's vision is Phase 7 and
    nothing writes these" is a different statement from "the operator was never
    in a menu", and only the first should read as a gate that could not be
    evaluated.
    """
    rows = conn.execute(
        "SELECT t, t_end FROM events WHERE stream_id = ? AND kind = ? ORDER BY t",
        (stream_id, kind),
    ).fetchall()
    if not rows:
        return None
    return [
        (float(row["t"]), float(row["t_end"] if row["t_end"] is not None else row["t"]))
        for row in rows
    ]


def composite_of(tracks: list[features.SignalTrack], length: int) -> np.ndarray:
    """§6.2 step 6: `composite[t] = sum(weight_i * z_i[t])`.

    Two things that look like defensive noise and are neither:

    **Unweighted tracks are skipped, not multiplied by zero.** A9 loads every
    stored signal at weight 0 so the feature vector is complete, and `0.0 * NaN`
    is NaN — so the moment a signal with gaps existed (`mic_f0`, Phase 3), one
    unweighted, unscored, purely-for-the-archive series turned the entire
    composite into NaN and the stream produced no candidates at all. Found by
    the export tests going from green to "nothing rated 2 or above".

    **A missing observation contributes zero.** In a sum of weighted z-scores,
    zero is the value that adds nothing — the same thing "no observation" should
    do. This is NOT a claim that the pitch was average: `feature_vector` still
    records null for that sample, so the archive says "not observed" while the
    score says "nothing added". Dropping the sample instead would make the
    composite depend on which signals happened to be defined at each instant.
    """
    total = np.zeros(length, dtype=np.float64)
    for track in tracks:
        if not track.weight:
            continue
        total += track.weight * np.nan_to_num(track.values, nan=0.0)
    return total


def smooth(composite: np.ndarray, sigma_s: float, grid_hz: float) -> np.ndarray:
    """§6.2 step 6: Gaussian, sigma = 2 s by default."""
    sigma = float(sigma_s) * float(grid_hz)
    if sigma <= 0:
        return composite
    return gaussian_filter1d(composite, sigma=sigma, mode="nearest")


# --------------------------------------------------------------------------
# the scoring pass
# --------------------------------------------------------------------------


@dataclass
class ProfileRun:
    """One profile's pass through §6.2 step 6, before anything is merged."""

    profile: object
    tracks: list[features.SignalTrack]
    smoothed: np.ndarray
    calibration: windows.Calibration
    built: list[windows.Window]
    missing: list[str]
    #: Each window's score as a share of this profile's best. Raw composites are
    #: not comparable across profiles — entertainment sums 20.7 of weight and
    #: gameplay 21.0, of which 4.2 has a producer — so every cross-profile
    #: comparison (which peak wins a merged moment) uses these instead.
    normalised: np.ndarray = field(default_factory=lambda: np.zeros(0))


@dataclass
class Moment(windows.Window):
    """A merged candidate: one row, every profile's opinion of it (§3.2).

    Subclasses `Window` so §6.6's spacing, §6.3's word snapping and the peak
    bookkeeping all apply unchanged — a moment IS a window, with the other
    profiles' scores attached.
    """

    scores: dict[str, float] = field(default_factory=dict)
    found_by: tuple[str, ...] = ()
    #: The normalised score of the profile whose peak this moment took, so a
    #: later profile only wins the peak by being more confident about it.
    best_normalised: float = 0.0


def score_stream(ctx: StageContext) -> ScoreResult:
    row = ctx.stream
    duration = row["duration_s"]
    if not duration:
        raise ScoreError("stream has no duration; run probe first")

    grid_hz = float(ctx.cfg.get("score.score_grid_hz"))
    timeline = grid.build(float(duration), grid_hz)

    baseline_s = float(ctx.cfg.get("score.rolling_baseline_window_s"))
    if duration < baseline_s:
        ctx.log(
            f"    WARNING  the stream is {duration:.0f}s but the rolling baseline window is "
            f"{baseline_s:.0f}s, so every sample is baselined against the whole signal. That is "
            f"global normalization, which §6.2 warns finds 'the loud game' rather than 'the loud "
            f"moment'. Fine for a test fixture; meaningless as a tuning signal."
        )

    pool = build_pool(ctx, timeline)

    # §6.4's gates read signals and events and never weights, so they are
    # computed once and the same two arrays are subtracted from every profile's
    # composite. commit 36 wrote `gates.compute` to make that true.
    penalties, gate_reasons = gates.compute(
        pool.raw, timeline, ctx.cfg,
        menu_intervals=load_intervals(ctx.conn, ctx.stream_id, gates.MENU),
    )
    penalty_tracks = gates.as_tracks(penalties)

    # §6.2 step 6: "FOR each profile in {entertainment, gameplay}".
    runs = [
        _run_profile(ctx, pool, profile, penalties, penalty_tracks, timeline,
                     grid_hz, float(duration))
        for profile in ctx.cfg.profiles
    ]
    if not runs[0].tracks:
        raise ScoreError(
            f"profile {ctx.cfg.profile.name!r} weights {sorted(ctx.cfg.profile.weights)} but none "
            f"of those signals exist for this stream. Has extraction run?"
        )

    # §6.2 step 8, with step 7 folded into it: a moment cannot carry both
    # profiles' scores until the windows that found it have been merged.
    moments = merge_across_profiles(runs)
    _clamp_merged(ctx, moments, float(duration))
    moments = snap_windows(ctx, moments, float(duration))
    agreement = _score_moments(ctx, runs, moments)

    # §6.6 AFTER the merge, which §6.2 step 6 does not say. Spacing exists to
    # stop "ten candidates from one 90-second stretch" in THE LIST THE OPERATOR
    # REVIEWS, and after step 8 that is one merged list — penalising per profile
    # first would let two profiles each contribute their own suppressed-but-kept
    # candidate into the same thirty seconds, which is the outcome §6.6 forbids.
    # With one profile the merge is a no-op and this is exactly what it was.
    factor = float(ctx.cfg.get("score.spacing.factor"))
    ranked = windows.apply_spacing(
        moments,
        window_s=float(ctx.cfg.get("score.spacing.window_s")),
        factor=factor,
        mode=str(ctx.cfg.get("score.spacing.mode")),
    )
    for moment in ranked:
        if moment.suppressed_by is not None:
            # The same factor on all three, so §7.4's three rankings cannot
            # disagree about which moments were suppressed.
            moment.scores = {k: v * factor for k, v in moment.scores.items()}

    result = write_candidates(ctx, ranked, runs)
    result.gate_reasons = gate_reasons
    result.penalty_shares = {p.name: p.share for p in penalties}
    result.rank_agreement = agreement
    result.found_by = _found_by_counts(ranked)
    result.counterfactual = counterfactual(
        ctx, composite_of(runs[0].tracks, timeline.size), penalties, grid_hz,
        float(duration),
    )
    return result


def _run_profile(
    ctx: StageContext, pool: Pool, profile, penalties, penalty_tracks,
    timeline: np.ndarray, grid_hz: float, duration: float,
) -> ProfileRun:
    """§6.2 step 6 for one profile: weight, penalise, smooth, peak, expand."""
    tracks, missing = tracks_for(pool, profile)

    # §6.2 step 6, in the order step 6 lists it: sum, THEN apply §6.4's gated
    # negatives, THEN smooth.
    raw_composite = composite_of(tracks, timeline.size)
    penalised = gates.apply(raw_composite, penalties)

    # After `composite_of`, never before it: a penalty track carries a negative
    # weight, so including it in the sum would apply the penalty twice. They are
    # appended so that A9's vector records the gate level under the key
    # `feature_schema.yaml` already declares, and so the breakdown still adds up
    # to the composite it is explaining.
    tracks = tracks + list(penalty_tracks)

    smoothed = smooth(penalised, ctx.cfg.get("score.smoothing_sigma_s"), grid_hz)

    calibration = calibrate_peaks(ctx, smoothed, duration)
    peaks = windows.find(smoothed, calibration.prominence,
                         float(ctx.cfg.get("score.min_peak_value")))
    peaks = windows.merge_peaks(smoothed, peaks,
                                float(ctx.cfg.get("score.window.hysteresis_enter")))
    built = windows.build(
        smoothed, timeline, peaks,
        exit_ratio=float(ctx.cfg.get("score.window.hysteresis_exit")),
        min_window_s=float(ctx.cfg.get("score.window.min_window_s")),
        max_window_s=float(ctx.cfg.get("score.window.max_window_s")),
        duration_s=duration,
    )
    return ProfileRun(
        profile=profile, tracks=tracks, smoothed=smoothed, calibration=calibration,
        built=built, missing=missing,
        normalised=combined_score.normalise(np.array([w.score for w in built])),
    )


def _clamp_merged(ctx: StageContext, moments: list[Moment], duration_s: float) -> None:
    """§6.3's bounds again, because a union can breach them.

    Each profile clamped its own windows to `max_window_s`, but two windows that
    overlap by a second and extend in opposite directions union to almost twice
    that — and §6.2 step 8 says "merge overlapping windows" without exempting
    the result from §6.3's range. A 110-second candidate is not a moment, it is
    two, and §7.1 reviews these at four seconds each.

    Clamped around the winning peak (`expand_around_peak`), which is the only
    variant that cannot move the peak outside its own window — §3.2's
    `CHECK (t_peak BETWEEN t_start AND t_end)` refuses that outright. Logged
    when it fires, because it is the merge undoing part of its own work.
    """
    min_window = float(ctx.cfg.get("score.window.min_window_s"))
    max_window = float(ctx.cfg.get("score.window.max_window_s"))

    clamped = 0
    for moment in moments:
        if moment.duration_s <= max_window:
            continue
        moment.t_start, moment.t_end = windows.clamp(
            moment.t_start, moment.t_end, moment.t_peak,
            min_window_s=min_window, max_window_s=max_window, duration_s=duration_s,
        )
        clamped += 1

    if clamped:
        ctx.log(
            f"    {clamped} merged window(s) exceeded max_window_s and were "
            f"clamped back around their peak"
        )


def _overlap(moment: "Moment", window: windows.Window) -> float:
    return min(moment.t_end, window.t_end) - max(moment.t_start, window.t_start)


def merge_across_profiles(runs: list[ProfileRun]) -> list[Moment]:
    """§6.2 step 8: "merge overlapping windows across profiles".

    ACROSS profiles, never within one. Two overlapping windows found by the SAME
    profile are §6.6's business — that has been the rule since Phase 1, and
    merging them here would silently change what a single-profile run produces.
    So a moment absorbs at most one window per profile, and a second gameplay
    window over the same stretch becomes its own moment for spacing to suppress.

    C2 decides the boundaries: `t_start` is the earliest and `t_end` the latest,
    because expanding is the direction in which a false positive is cheap.
    `t_peak` comes from the profile with the higher NORMALISED score — raw
    composites are not comparable across profiles, since one sums 20.7 of weight
    and the other 21.0 of which 4.2 has a producer.
    """
    primary = runs[0]
    moments = [
        Moment(index=w.index, t_peak=w.t_peak, t_start=w.t_start, t_end=w.t_end,
               value=w.value, score=w.score, prominence=w.prominence,
               found_by=(primary.profile.name,),
               best_normalised=float(primary.normalised[i]))
        for i, w in enumerate(primary.built)
    ]

    for run in runs[1:]:
        for i, window in enumerate(run.built):
            here = float(run.normalised[i])
            open_moments = [
                m for m in moments
                if run.profile.name not in m.found_by and _overlap(m, window) > 0
            ]
            if not open_moments:
                moments.append(Moment(
                    index=window.index, t_peak=window.t_peak,
                    t_start=window.t_start, t_end=window.t_end,
                    value=window.value, score=window.score,
                    prominence=window.prominence,
                    found_by=(run.profile.name,), best_normalised=here,
                ))
                continue

            best = max(open_moments, key=lambda m: _overlap(m, window))
            best.t_start = min(best.t_start, window.t_start)
            best.t_end = max(best.t_end, window.t_end)
            best.found_by = (*best.found_by, run.profile.name)
            if here > best.best_normalised:
                best.index, best.t_peak = window.index, window.t_peak
                best.value, best.prominence = window.value, window.prominence
                best.best_normalised = here

    return sorted(moments, key=lambda m: m.t_peak)


def _score_moments(ctx: StageContext, runs: list[ProfileRun],
                   moments: list[Moment]) -> dict:
    """§6.2 step 7: every profile's score for every merged moment, then §6.5.

    Each column is that profile's smoothed composite AT THE MERGED PEAK, not the
    score of whichever window it happened to find. The row describes one moment,
    and every column has to be describing the same instant or the two numbers
    are not comparable at all.
    """
    for moment in moments:
        moment.scores = {
            run.profile.name: float(run.smoothed[moment.index]) for run in runs
        }

    if not moments:
        return {}

    primary = runs[0].profile.name
    entertainment = np.array([m.scores[primary] for m in moments])

    if len(runs) == 1:
        # §6.5's own "casual/comedy: entertainment only". Combined mirrors the
        # single profile at its raw scale, which is exactly what it has meant
        # since Phase 1 — normalising here would rescale every stored score for
        # nothing, since a single ranking cannot disagree with itself.
        for moment, value in zip(moments, entertainment, strict=True):
            moment.score = float(value)
        return {}

    secondary = runs[1].profile.name
    gameplay = np.array([m.scores[secondary] for m in moments])
    values = combined_score.combined(
        combined_score.normalise(entertainment),
        combined_score.normalise(gameplay),
        float(ctx.cfg.get("score.combined.alpha")),
    )
    for moment, value in zip(moments, values, strict=True):
        moment.score = float(value)

    return combined_score.rank_agreement(
        values, entertainment,
        top_n=int(ctx.cfg.get("review.sections.combined_top_n")),
    )


def counterfactual(
    ctx: StageContext, raw_composite: np.ndarray, penalties: list[gates.Penalty],
    grid_hz: float, duration_s: float,
) -> dict | None:
    """What this stream looks like with §6.4 switched off — the number §17 needs.

    Counting peaks on the un-penalised composite at the PENALISED run's
    prominence would be a misleading answer, and measuring it that way was the
    first attempt: `windows.find` measures a peak above the higher of its two
    flanking cols, so digging a trough BETWEEN two peaks makes both of them more
    prominent, `calibrate_peaks` then bisects the threshold upward to hit §6.7's
    target — and at that raised threshold the un-penalised composite reported
    ZERO candidates. That is an artefact of calibrating on one curve and
    counting on another, not a suppression.

    So the counterfactual gets its own calibration, and the number reported is
    the one that means something: how many of the moments this stream would have
    produced without §6.4 sit inside a stretch the penalty holds down. That is
    "what the penalty removed", and it is what would show `afk_threshold_s` or
    `menu_grace_period_s` to be wrong.

    Only computed when something actually fired; peak-finding only, no window
    building, no spacing, no writes.
    """
    if not any(penalty.fired for penalty in penalties):
        return None

    unpenalised = smooth(raw_composite, ctx.cfg.get("score.smoothing_sigma_s"), grid_hz)
    baseline = calibrate_peaks(ctx, unpenalised, duration_s)
    enter = float(ctx.cfg.get("score.window.hysteresis_enter"))
    peaks = windows.merge_peaks(
        unpenalised,
        windows.find(unpenalised, baseline.prominence,
                     float(ctx.cfg.get("score.min_peak_value"))),
        enter,
    )

    held = np.zeros(raw_composite.size, dtype=bool)
    for penalty in penalties:
        held |= penalty.values > 0.0

    return {
        "candidates": len(peaks),
        "prominence": round(baseline.prominence, 6),
        "peaks_inside_a_penalty": sum(1 for peak in peaks if held[peak.index]),
    }


def word_boundaries(ctx: StageContext) -> tuple[np.ndarray, np.ndarray]:
    """Aligned word starts and ends across the whole stream, sorted.

    Only words forced alignment could place: one with no timestamp cannot be a
    boundary, and treating a null as zero would snap every window to the start
    of the stream.
    """
    starts: list[float] = []
    ends: list[float] = []
    for row in ctx.conn.execute(
        "SELECT words FROM segments WHERE stream_id = ?", (ctx.stream_id,)
    ):
        for word in json.loads(row["words"] or "[]"):
            if word.get("start") is None or word.get("end") is None:
                continue
            starts.append(float(word["start"]))
            ends.append(float(word["end"]))
    return np.array(sorted(starts)), np.array(sorted(ends))


def snap_windows(
    ctx: StageContext, built: list[windows.Window], duration_s: float
) -> list[windows.Window]:
    """§6.3's word-boundary snapping — a documented no-op until Phase 2.

    `score.window.snap_to_word_boundaries` has been config-only since commit 12
    for want of word timestamps. It does something now.
    """
    if not bool(ctx.cfg.get("score.window.snap_to_word_boundaries")):
        return built

    starts, ends = word_boundaries(ctx)
    if starts.size == 0:
        return built

    max_distance = float(ctx.cfg.get("score.window.snap_max_distance_s"))
    min_window = float(ctx.cfg.get("score.window.min_window_s"))
    max_window = float(ctx.cfg.get("score.window.max_window_s"))

    moved = 0
    for window in built:
        t_start, t_end = windows.snap_to_words(
            window.t_start, window.t_end,
            starts=starts, ends=ends, max_distance_s=max_distance,
            min_window_s=min_window, max_window_s=max_window, duration_s=duration_s,
        )
        if (t_start, t_end) != (window.t_start, window.t_end):
            moved += 1
            window.t_start, window.t_end = t_start, t_end

    if moved:
        ctx.log(f"    snapped {moved} of {len(built)} window(s) to word boundaries")
    return built


def calibrate_peaks(ctx: StageContext, composite: np.ndarray, duration_s: float):
    explicit = ctx.cfg.get("score.peak.prominence")
    min_value = float(ctx.cfg.get("score.min_peak_value"))

    enter = float(ctx.cfg.get("score.window.hysteresis_enter"))
    low, high = windows.target_range(
        duration_s, ctx.cfg.get("score.peak.target_candidates_per_hour"),
        ctx.cfg.get("score.peak.min_candidates"),
    )

    if explicit is not None or not ctx.cfg.get("score.peak.auto_calibrate"):
        prominence = float(explicit or 0.0)
        count = windows.candidate_count(composite, prominence, min_value, enter)
        return windows.Calibration(prominence, count, low, high, 0,
                                   low <= count <= high, "prominence pinned in config")

    return windows.calibrate(
        composite, target_low=low, target_high=high, min_value=min_value, enter=enter,
        max_iterations=int(ctx.cfg.get("score.peak.max_iterations")),
    )


# --------------------------------------------------------------------------
# writing candidates
# --------------------------------------------------------------------------


def write_candidates(
    ctx: StageContext, ranked: list[Moment], runs: list[ProfileRun],
) -> ScoreResult:
    """One row per moment, carrying every profile's score (§3.2).

    §3.2 gives `candidates` three score columns rather than three rows, and
    §7.4's four sections are orderings over one list — so a moment is one row,
    `profile` holds the profile-SET name, and
    `UNIQUE (stream_id, generation, profile, t_peak)` stays valid with no
    migration.

    THE BREAKDOWN IS THE PRIMARY PROFILE'S, AND ONLY ITS. `contributing_signals`
    is weight × value, so it is profile-dependent and a two-profile row has two
    of them. §6.5 calls entertainment "the primary profile"; before Phase 7 the
    gameplay breakdown is four rows (`marker_definite`, `mic_rms`, two laughs)
    that can be read straight off the entertainment one, and storing both would
    change a payload shape `review/queries.py`, `clipforge score --list` and the
    review UI's `?` panel all parse. All three SCORES go into the breakdown's
    context instead, so the panel can explain the number in the score box. A
    per-profile breakdown is what to build when the gameplay ranking becomes
    worth explaining, which is Phase 7.
    """
    config_version = ctx.cfg.version
    profile_set = ctx.cfg.profile_set
    schema = ctx.cfg.feature_schema
    primary = runs[0]
    secondary = runs[1].profile.name if len(runs) > 1 else None

    current = ctx.conn.execute(
        "SELECT generation, config_version FROM candidates "
        "WHERE stream_id = ? AND is_current = 1 LIMIT 1",
        (ctx.stream_id,),
    ).fetchone()

    # A new generation only when the configuration differs. Re-running with the
    # same weights replaces in place, so generation numbers stay meaningful when
    # comparing weight experiments months later.
    replacing = current is not None and current["config_version"] == config_version
    if replacing:
        generation = int(current["generation"])
    else:
        highest = ctx.conn.execute(
            "SELECT COALESCE(MAX(generation), 0) AS g FROM candidates WHERE stream_id = ?",
            (ctx.stream_id,),
        ).fetchone()["g"]
        generation = int(highest) + 1

    prior = [] if replacing else _rated_current(ctx)

    with db.transaction(ctx.conn):
        if replacing:
            ctx.conn.execute(
                "DELETE FROM candidates WHERE stream_id = ? AND generation = ?",
                (ctx.stream_id, generation),
            )
        else:
            ctx.conn.execute(
                "UPDATE candidates SET is_current = 0 WHERE stream_id = ?", (ctx.stream_id,)
            )

        inserted: list[tuple[int, Moment]] = []
        for moment in ranked:
            entertainment = moment.scores[primary.profile.name]
            # 0.0, not the entertainment score, when §6.5's second profile is
            # not running: an invented number in that column would be
            # indistinguishable from a measured one five months from now.
            gameplay = moment.scores[secondary] if secondary else 0.0

            # feature_vector holds exactly the schema's keys — A9's value is
            # that Phase 1 and Phase 3 vectors have identical shape. The dB
            # context a human needs to read the breakdown goes with the
            # breakdown. Every value in it is a z-score or a kernel level, so it
            # is the same whichever profile's tracks it is read from.
            vector = features.vector(schema, primary.tracks, moment.index)
            detail = features.breakdown(
                primary.tracks, moment.index,
                float(primary.smoothed[moment.index]), moment.suppressed_by,
            )
            explain = detail.to_json_dict()
            explain.update(features.unweighted_context(primary.tracks, moment.index))
            explain["_score_entertainment"] = round(entertainment, 6)
            explain["_score_gameplay"] = round(gameplay, 6)
            explain["_score_combined"] = round(moment.score, 6)
            explain["_found_by"] = "+".join(moment.found_by)

            cursor = ctx.conn.execute(
                """
                INSERT INTO candidates
                    (stream_id, generation, is_current, profile, t_start, t_end, t_peak,
                     score_entertainment, score_gameplay, score_combined,
                     contributing_signals, feature_vector, feature_schema_version,
                     config_version)
                VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ctx.stream_id, generation, profile_set,
                    moment.t_start, moment.t_end, moment.t_peak,
                    entertainment, gameplay, moment.score,
                    json.dumps(explain),
                    json.dumps(vector), schema.version, config_version,
                ),
            )
            inserted.append((int(cursor.lastrowid), moment))

        inherited = _inherit_ratings(ctx, inserted, prior) if prior else 0

        ctx.conn.execute(
            "UPDATE streams SET profile_used = ? WHERE id = ?", (profile_set, ctx.stream_id)
        )

    return ScoreResult(
        generation=generation, profile=profile_set, config_version=config_version,
        candidates=len(ranked), calibration=primary.calibration,
        calibrations={run.profile.name: run.calibration for run in runs},
        inherited_ratings=inherited, replaced_generation=replacing,
        missing_weights=primary.missing,
        missing_by_profile={run.profile.name: run.missing for run in runs},
    )


def _rated_current(ctx: StageContext) -> list[dict]:
    """Operator-rated candidates in the generation about to be superseded."""
    rows = ctx.conn.execute(
        """
        SELECT c.id, c.t_start, c.t_end, r.rating, r.tags, r.note,
               r.adjusted_start, r.adjusted_end
          FROM candidates c JOIN ratings r ON r.candidate_id = c.id
         WHERE c.stream_id = ? AND c.is_current = 1 AND r.rating_source = 'operator'
        """,
        (ctx.stream_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def _inherit_ratings(ctx: StageContext, inserted, prior: list[dict]) -> int:
    """Carry operator ratings onto the matching candidate in the new generation.

    §13.2 calls ratings irreplaceable, and §6.1 expects re-scoring to be routine;
    those two only coexist if a re-score does not cost a review session. Matching
    is by time overlap (IoU) — a moment is the same moment if the windows mostly
    agree, whatever the weights did to the peak.

    Carried rows are tagged `rating_source='inherited'` so §14's
    `signal_firing_rate_by_rating` never counts one twice as fresh evidence.
    """
    threshold = float(ctx.cfg.get("score.rating_inherit_min_overlap"))
    count = 0

    for candidate_id, window in inserted:
        best, best_overlap = None, 0.0
        for old in prior:
            overlap = _iou(window.t_start, window.t_end, old["t_start"], old["t_end"])
            if overlap > best_overlap:
                best, best_overlap = old, overlap

        if best is None or best_overlap < threshold:
            continue

        ctx.conn.execute(
            "INSERT OR IGNORE INTO ratings (candidate_id, rating, tags, note, "
            "adjusted_start, adjusted_end, rating_source, inherited_from) "
            "VALUES (?, ?, ?, ?, ?, ?, 'inherited', ?)",
            (candidate_id, best["rating"], best["tags"], best["note"],
             best["adjusted_start"], best["adjusted_end"], best["id"]),
        )
        count += 1

    return count


def _iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    overlap = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return overlap / union if union > 0 else 0.0


# --------------------------------------------------------------------------
# stage hooks
# --------------------------------------------------------------------------


def params(ctx: StageContext) -> dict:
    return {
        "config_version": ctx.cfg.version,
        "profile": ctx.cfg.profile_set,
        "signals": sorted(signals.kinds(ctx.conn, ctx.stream_id)),
        "events": ctx.conn.execute(
            "SELECT count(*) FROM events WHERE stream_id = ?", (ctx.stream_id,)
        ).fetchone()[0],
    }


def outputs(ctx: StageContext) -> list:
    return []  # writes candidate rows


def verify(ctx: StageContext) -> tuple[bool, str]:
    row = ctx.conn.execute(
        "SELECT count(*) AS n, MAX(config_version) AS v FROM candidates "
        "WHERE stream_id = ? AND is_current = 1",
        (ctx.stream_id,),
    ).fetchone()
    if not row["n"]:
        return False, "no current candidates"
    if row["v"] != ctx.cfg.version:
        return False, f"current candidates were scored by {row['v']}"
    return True, ""


def run(ctx: StageContext) -> None:
    result = score_stream(ctx)
    duration = float(ctx.stream["duration_s"])
    per_hour = result.candidates / (duration / 3600.0) if duration else 0.0

    verb = "replaced generation" if result.replaced_generation else "generation"
    ctx.log(
        f"    {result.candidates} candidates, {verb} {result.generation}, "
        f"profile {result.profile} ({result.config_version})"
    )
    for name, calibration in result.calibrations.items():
        ctx.log(
            f"    {name:<14} prominence {calibration.prominence:.4f}, "
            f"target {calibration.target_low}-{calibration.target_high}"
        )
        if not calibration.reached_target:
            ctx.log(f"    note: {name} calibration did not reach the target — "
                    f"{calibration.reason}")
    ctx.log(f"    {per_hour:.1f} candidates/hour after the merge")

    if result.inherited_ratings:
        ctx.log(f"    carried {result.inherited_ratings} operator rating(s) onto this generation")
    for name, missing in result.missing_by_profile.items():
        if missing:
            ctx.log(
                f"    note: {name} weights {missing} have no signal in this stream "
                f"(expected — those arrive in later phases)"
            )

    _log_negatives(ctx, result)
    _log_agreement(ctx, result)

    ctx.metric("candidates_per_hour_of_stream", round(per_hour, 3))
    ctx.metric("peak_prominence", round(result.calibration.prominence, 6),
               json.dumps({"reached_target": result.calibration.reached_target,
                           "reason": result.calibration.reason,
                           "by_profile": {name: c.prominence
                                          for name, c in result.calibrations.items()}}))


def _found_by_counts(moments: list[Moment]) -> dict:
    counts: dict[str, int] = {}
    for moment in moments:
        counts["+".join(moment.found_by)] = counts.get("+".join(moment.found_by), 0) + 1
    return dict(sorted(counts.items()))


def _log_agreement(ctx: StageContext, result: ScoreResult) -> None:
    """§6.5's combined ranking against the primary's — the number this phase
    cannot answer for itself.

    §6.5 justifies `score_combined` as the INTERSECTION of mechanics and
    personality. Before Phase 7 only 20% of `gameplay`'s weight has a producer
    and 71% of that is `marker_definite`, so the combined ranking may be an
    expensive restatement of the entertainment one — and §7.4 puts it first, at
    the top of the screen C4 calls the critical path.

    `combined_rank_agreement` is an INVENTED metric name (§14's table has no row
    for it), recorded on every run so the answer accumulates from the first real
    stream rather than being argued about later.
    """
    if not result.rank_agreement:
        return

    rho = result.rank_agreement.get("spearman")
    overlap = result.rank_agreement.get("top_n_overlap")
    top_n = result.rank_agreement.get("top_n")
    ctx.log(
        f"    combined vs {list(result.calibrations)[0]}: "
        f"rho {'—' if rho is None else f'{rho:+.3f}'}, "
        f"top-{top_n} overlap {'—' if overlap is None else f'{100 * overlap:.0f}%'}"
    )
    if rho is not None and rho > 0.95:
        ctx.log(
            "    note: the two rankings are nearly identical, which is what §6.5's "
            "combined section looks like before Phase 7 gives `gameplay` something "
            "of its own to say"
        )

    # WHAT rho DOES NOT SAY, and it is half the answer: two profiles can order
    # the same moments identically and still disagree about which moments there
    # ARE. §6.2 step 8's merge is what turns the second kind of disagreement
    # into candidates, so the counts travel with the correlation.
    found = ", ".join(f"{count} by {names}" for names, count in result.found_by.items())
    ctx.log(f"    moments: {found}")

    ctx.metric(
        "combined_rank_agreement",
        round(float(rho), 6) if rho is not None else 0.0,
        json.dumps({**result.rank_agreement,
                    "spearman_defined": rho is not None,
                    "profiles": list(result.calibrations),
                    "found_by": result.found_by}),
    )


def _log_negatives(ctx: StageContext, result: ScoreResult) -> None:
    """What §6.4 did, and — as loudly — what it could not evaluate.

    `negative_penalty_share` is an INVENTED metric name: §14's table has no
    negatives row. It is written on every run, including the runs where nothing
    fired and the runs where nothing could be evaluated, because C6 is the whole
    argument for §17 being able to tune `menu_grace_period_s` and
    `afk_threshold_s` later — a penalty that has never been observed to fire is
    a fact about this configuration, and it has to be in the record to be one.
    """
    for name, reason in sorted(result.gate_reasons.items()):
        ctx.log(f"    note: §6.4's {name} penalty was not evaluated — {reason}")

    fired = {name: share for name, share in result.penalty_shares.items() if share}
    for name, share in sorted(result.penalty_shares.items()):
        ctx.log(
            f"    §6.4 {name} penalty held {100 * share:.1f}% of the stream down"
            if share else
            f"    §6.4 {name} penalty evaluated and never fired"
        )
    if result.counterfactual is not None:
        counter = result.counterfactual
        ctx.log(
            f"    without §6.4 this stream calibrates to {counter['candidates']} "
            f"candidate(s) against {result.candidates}, and "
            f"{counter['peaks_inside_a_penalty']} of those peak(s) sit inside a "
            f"penalised stretch"
        )

    ctx.metric(
        "negative_penalty_share",
        round(max(fired.values()) if fired else 0.0, 6),
        json.dumps({
            "shares": {k: round(v, 6) for k, v in result.penalty_shares.items()},
            "not_evaluated": result.gate_reasons,
            "candidates": result.candidates,
            "without_negatives": result.counterfactual,
        }),
    )
