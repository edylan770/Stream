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
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter1d

from clipforge import db, signals
from clipforge.pipeline.context import StageContext
from clipforge.score import features, grid, kernels, windows

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


# --------------------------------------------------------------------------
# building the composite
# --------------------------------------------------------------------------


def build_tracks(ctx: StageContext, timeline: np.ndarray) -> tuple[list[features.SignalTrack], list[str]]:
    """Every weighted signal, resampled and normalized onto the grid.

    Continuous signals are z-scored (§6.2 step 3); event kernels are not. That
    asymmetry is deliberate and is what makes §6.5's weights comparable: both
    arrive in units where 1.0 is roughly one standard deviation of notability,
    z-scores by construction and kernels by unit-peak normalization.
    """
    profile = ctx.cfg.profile
    baseline_samples = grid.window_samples(
        ctx.cfg.get("score.rolling_baseline_window_s"), ctx.cfg.get("score.score_grid_hz")
    )
    std_floor = float(ctx.cfg.get("score.zscore_std_floor"))

    tracks: list[features.SignalTrack] = []
    missing: list[str] = []

    stored = signals.load_all(ctx.conn, ctx.stream_id)
    events = load_events(ctx)

    for name, weight in profile.weights.items():
        if name in stored:
            raw = grid.resample(stored[name], timeline)
            z, baseline, _ = grid.rolling_zscore(raw, baseline_samples, std_floor)
            tracks.append(features.SignalTrack(
                name=name, values=z, weight=float(weight),
                is_event=False, baseline=baseline, raw=raw,
                unit=str(stored[name].params.get("unit", "dB")),
            ))
        elif name in events or name in EVENT_SOURCE_KINDS:
            kernel = kernels.for_kind(name, timeline, events.get(name, []), ctx.cfg)
            tracks.append(features.SignalTrack(
                name=name, values=kernel, weight=float(weight), is_event=True,
            ))
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
    # absent.
    #
    # Weight 0 contributes exactly nothing to the composite, so no score moves.
    weighted = {track.name for track in tracks}
    for name, series in stored.items():
        if name in weighted:
            continue
        raw = grid.resample(series, timeline)
        z, baseline, _ = grid.rolling_zscore(raw, baseline_samples, std_floor)
        tracks.append(features.SignalTrack(
            name=name, values=z, weight=0.0, is_event=False, baseline=baseline, raw=raw,
            unit=str(series.params.get("unit", "dB")),
        ))

    return tracks, missing


def load_events(ctx: StageContext) -> dict[str, list[float]]:
    rows = ctx.conn.execute(
        "SELECT kind, t FROM events WHERE stream_id = ? ORDER BY t", (ctx.stream_id,)
    ).fetchall()
    out: dict[str, list[float]] = {}
    for row in rows:
        out.setdefault(row["kind"], []).append(float(row["t"]))
    return out


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

    tracks, missing = build_tracks(ctx, timeline)
    if not tracks:
        raise ScoreError(
            f"profile {ctx.cfg.profile.name!r} weights {sorted(ctx.cfg.profile.weights)} but none "
            f"of those signals exist for this stream. Has extraction run?"
        )

    raw_composite = composite_of(tracks, timeline.size)
    smoothed = smooth(raw_composite, ctx.cfg.get("score.smoothing_sigma_s"), grid_hz)

    calibration = calibrate_peaks(ctx, smoothed, float(duration))
    peaks = windows.find(smoothed, calibration.prominence,
                         float(ctx.cfg.get("score.min_peak_value")))
    peaks = windows.merge_peaks(smoothed, peaks,
                                float(ctx.cfg.get("score.window.hysteresis_enter")))

    built = windows.build(
        smoothed, timeline, peaks,
        exit_ratio=float(ctx.cfg.get("score.window.hysteresis_exit")),
        min_window_s=float(ctx.cfg.get("score.window.min_window_s")),
        max_window_s=float(ctx.cfg.get("score.window.max_window_s")),
        duration_s=float(duration),
    )
    built = snap_windows(ctx, built, float(duration))

    # §6.2 step 8 — "merge overlapping windows across profiles" — is a no-op in
    # Phase 1: there is one profile, same-bump peaks were already merged by
    # hysteresis_enter, and distinct overlapping windows are §6.6's business.
    ranked = windows.apply_spacing(
        built,
        window_s=float(ctx.cfg.get("score.spacing.window_s")),
        factor=float(ctx.cfg.get("score.spacing.factor")),
        mode=str(ctx.cfg.get("score.spacing.mode")),
    )

    return write_candidates(ctx, ranked, tracks, smoothed, calibration, missing)


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
    ctx: StageContext, ranked: list[windows.Window], tracks: list[features.SignalTrack],
    smoothed: np.ndarray, calibration: windows.Calibration, missing: list[str],
) -> ScoreResult:
    config_version = ctx.cfg.version
    profile = ctx.cfg.profile.name
    schema = ctx.cfg.feature_schema

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

        inserted: list[tuple[int, windows.Window]] = []
        for window in ranked:
            # feature_vector holds exactly the schema's keys — A9's value is
            # that Phase 1 and Phase 3 vectors have identical shape. The dB
            # context a human needs to read the breakdown goes with the
            # breakdown.
            vector = features.vector(schema, tracks, window.index)
            detail = features.breakdown(
                tracks, window.index, float(smoothed[window.index]), window.suppressed_by
            )
            explain = detail.to_json_dict()
            explain.update(features.unweighted_context(tracks, window.index))

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
                    ctx.stream_id, generation, profile,
                    window.t_start, window.t_end, window.t_peak,
                    # Phase 1 has one profile. The naive profile carries §6.5's
                    # entertainment weights, so its score belongs in that column;
                    # `profile` is what disambiguates. score_gameplay is 0.0
                    # rather than a faked §6.5 product, and combined mirrors
                    # entertainment because there is nothing to combine with.
                    window.score, 0.0, window.score,
                    json.dumps(explain),
                    json.dumps(vector), schema.version, config_version,
                ),
            )
            inserted.append((int(cursor.lastrowid), window))

        inherited = _inherit_ratings(ctx, inserted, prior) if prior else 0

        ctx.conn.execute(
            "UPDATE streams SET profile_used = ? WHERE id = ?", (profile, ctx.stream_id)
        )

    return ScoreResult(
        generation=generation, profile=profile, config_version=config_version,
        candidates=len(ranked), calibration=calibration, inherited_ratings=inherited,
        replaced_generation=replacing, missing_weights=missing,
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
        "profile": ctx.cfg.profile.name,
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
    ctx.log(
        f"    prominence {result.calibration.prominence:.4f}, "
        f"target {result.calibration.target_low}-{result.calibration.target_high}, "
        f"{per_hour:.1f}/hour"
    )

    if not result.calibration.reached_target:
        ctx.log(f"    note: calibration did not reach the target — {result.calibration.reason}")
    if result.inherited_ratings:
        ctx.log(f"    carried {result.inherited_ratings} operator rating(s) onto this generation")
    if result.missing_weights:
        ctx.log(
            f"    note: profile weights {result.missing_weights} have no signal in this stream "
            f"(expected in Phase 1 — those arrive in later phases)"
        )

    ctx.metric("candidates_per_hour_of_stream", round(per_hour, 3))
    ctx.metric("peak_prominence", round(result.calibration.prominence, 6),
               json.dumps({"reached_target": result.calibration.reached_target,
                           "reason": result.calibration.reason}))
