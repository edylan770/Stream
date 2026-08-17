"""Signals computed at score time from stored primitives (§5.4.1, §5.4.3).

§5.4.1 lists `mic_f0_variance`, `sudden_silence`, `overlap_speech` and
`input_stillness` in the same table as `mic_rms`, which reads as an instruction
to extract and store them. They are not stored, and the reason is C3:

> "Extraction is expensive and runs once; scoring is pure, cheap and re-runnable
> over the whole back catalogue."

Every signal in this module is a pure function of arrays that are already stored,
**plus a tunable** — the 5 s variance window, the 2 s silence lookback, the VAD
threshold, the stillness gate. Storing the result would freeze that tunable into
every stream ever processed and make retuning it cost a re-extraction of the
whole library. This codebase has drawn that line twice already: absolute dB is
stored rather than baselined (`extract/features.py`) and `events.t` holds the raw
press time rather than `t − 20` (`extract/markers.py`). Same boundary, same
reason.

A9 is unaffected. The feature vector is written at score time, so a derived
signal reaches it exactly like a stored one.

**A derivation whose inputs are absent produces nothing.** It does not produce
zeros. `input_stillness` on a stream with no input log is *unknown*, and a zero
there would read as "perfectly still" — which §6.4's AFK penalty would then act
on. Same rule the stage runner applies when it defers rather than fails.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

#: Reference for the semitone conversion. Arbitrary and irrelevant: a difference
#: of semitones is independent of it, and only differences are ever used.
SEMITONE_REF_HZ = 100.0


@dataclass(frozen=True)
class Derivation:
    """One derived signal, and what it needs to exist."""

    name: str
    #: Stored signal kinds this reads. Missing any one means the signal is not
    #: computed at all — never computed as zeros.
    requires: tuple[str, ...]
    compute: Callable[["Inputs"], np.ndarray]
    unit: str = ""
    #: True for §5.4.3 composites that are moments rather than levels; these
    #: become event times and go through `kernels.gaussian` like any other event.
    emits_events: bool = False


@dataclass
class Inputs:
    """Raw signal values already resampled onto the scoring grid."""

    timeline: np.ndarray
    grid_hz: float
    raw: dict[str, np.ndarray]
    cfg: object
    #: Filled as derivations run, so a later one can read an earlier one —
    #: `reaction_onset` is defined in terms of `sudden_silence` and
    #: `overlap_speech`, not in terms of the RMS underneath them.
    computed: dict[str, np.ndarray] = field(default_factory=dict)

    def get(self, name: str) -> np.ndarray:
        if name in self.computed:
            return self.computed[name]
        return self.raw[name]

    def has(self, name: str) -> bool:
        return name in self.computed or name in self.raw

    def samples(self, seconds: float) -> int:
        return max(int(round(float(seconds) * self.grid_hz)), 1)

    def opt(self, key: str) -> float:
        return float(self.cfg.get(f"score.derived.{key}"))


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def to_semitones(hz: np.ndarray) -> np.ndarray:
    """Hertz to a log scale, keeping NaN as NaN.

    CLAUDE.md's "dB are logarithms" rule, inverted. Pitch is log-distributed: a
    20 Hz rise at 100 Hz is three and a half semitones and the same rise at
    300 Hz is one, so a standard deviation taken over linear hertz understates
    exactly the high excursions §6.5 weights `mic_f0_variance` at 1.8 for.
    """
    values = np.asarray(hz, dtype=np.float64)
    out = np.full(values.shape, np.nan)
    voiced = np.isfinite(values) & (values > 0)
    out[voiced] = 12.0 * np.log2(values[voiced] / SEMITONE_REF_HZ)
    return out


def rolling_std(values: np.ndarray, window: int, min_observations: int) -> np.ndarray:
    """Centred rolling standard deviation over the observed samples only.

    NaN where fewer than `min_observations` samples were observed in the window.
    A "prosodic range" measured over two voiced frames is noise wearing the name
    of a signal, and zero would be a claim that the speaker was monotone.
    """
    n = values.size
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    data = np.asarray(values, dtype=np.float64)
    observed = np.isfinite(data)
    filled = np.where(observed, data, 0.0)

    cumsum = np.concatenate(([0.0], np.cumsum(filled)))
    cumsum2 = np.concatenate(([0.0], np.cumsum(np.square(filled))))
    cumcount = np.concatenate(([0.0], np.cumsum(observed.astype(np.float64))))

    half = max(int(window) // 2, 0)
    index = np.arange(n)
    lo = np.maximum(index - half, 0)
    hi = np.minimum(index + half + 1, n)

    count = cumcount[hi] - cumcount[lo]
    safe = np.maximum(count, 1.0)
    mean = (cumsum[hi] - cumsum[lo]) / safe
    variance = np.maximum((cumsum2[hi] - cumsum2[lo]) / safe - np.square(mean), 0.0)

    out = np.sqrt(variance)
    out[count < max(int(min_observations), 1)] = np.nan
    return out


def rolling_max(values: np.ndarray, window: int) -> np.ndarray:
    """Maximum over the `window` samples ENDING at each sample.

    Trailing rather than centred, deliberately: §5.4.1's silence signal is "RMS
    falls below floor *within 2 s of* high activity", and activity that has not
    happened yet cannot explain a silence now.
    """
    n = values.size
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    width = max(int(window), 1)
    padded = np.concatenate((np.full(width - 1, -np.inf), values))
    strided = np.lib.stride_tricks.sliding_window_view(padded, width)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmax(strided, axis=1)


def hold(active: np.ndarray, samples: int) -> np.ndarray:
    """Keep a boolean true for `samples` after it goes false.

    The hangover a speech gate needs: the gaps between words are below any
    energy threshold, and without this one sentence becomes nine "utterances"
    none of which survives `min_speech_s`.
    """
    if samples <= 0 or active.size == 0:
        return active.copy()
    out = active.copy()
    countdown = 0
    for i, value in enumerate(active):
        if value:
            countdown = int(samples)
        elif countdown > 0:
            out[i] = True
            countdown -= 1
    return out


def drop_short_runs(active: np.ndarray, samples: int) -> np.ndarray:
    """Remove true-runs shorter than `samples`. A click is not speech."""
    if samples <= 1 or active.size == 0:
        return active.copy()
    out = active.copy()
    start = None
    for i, value in enumerate(active):
        if value and start is None:
            start = i
        elif not value and start is not None:
            if i - start < samples:
                out[start:i] = False
            start = None
    if start is not None and active.size - start < samples:
        out[start:] = False
    return out


def ramp_up(active: np.ndarray, samples: int) -> np.ndarray:
    """0→1 over `samples` while `active` holds, back to 0 the moment it does not.

    §6.4 asks for penalties that ramp "gradually (not a step function)", and the
    same shape is what keeps `sudden_silence` from being a square wave that
    §6.2's Gaussian then has to smooth into something arbitrary.
    """
    if active.size == 0:
        return np.zeros(0, dtype=np.float64)
    steps = max(int(samples), 1)
    out = np.zeros(active.size, dtype=np.float64)
    run = 0
    for i, value in enumerate(active):
        run = run + 1 if value else 0
        out[i] = min(run / steps, 1.0) if run else 0.0
    return out


def speech_gate(
    rms: np.ndarray, inputs: Inputs, baseline_samples: int
) -> np.ndarray:
    """Deterministic voice activity from one isolated track's energy (§5.4.1).

    §5.4.1 says "VAD on both tracks". A model would mean silero, which lives
    behind the optional `asr` extra and is off by default — so `overlap_speech`
    would be missing on every stream that exists. C1 prefers the deterministic
    option anyway, and §4.2 makes it sound: the tracks are separately recorded
    and the operator wears headphones, so there is no acoustic path from the
    party's voice to the mic. Energy on the mic track IS the operator talking.

    The threshold is relative to the track's own rolling baseline, so it does
    not encode this microphone's gain — the same reason §6.2's z-score rolls.
    """
    from clipforge.score import grid

    if rms.size == 0:
        return np.zeros(0, dtype=bool)

    _z, baseline, _std = grid.rolling_zscore(rms, baseline_samples, std_floor=1e-9)
    loud = np.asarray(rms, dtype=np.float64) > baseline + inputs.opt("vad.margin_db")

    loud = hold(loud, inputs.samples(inputs.opt("vad.hangover_s")))
    return drop_short_runs(loud, inputs.samples(inputs.opt("vad.min_speech_s")))


# --------------------------------------------------------------------------
# the signals themselves
# --------------------------------------------------------------------------


def _mic_f0_variance(inputs: Inputs) -> np.ndarray:
    """§5.4.1: "rolling std of mic_f0 over 5 s. High prosodic range = comedic
    delivery. Distinct from peak pitch." """
    semitones = to_semitones(inputs.get("mic_f0"))
    return rolling_std(
        semitones,
        window=inputs.samples(inputs.opt("f0_variance_window_s")),
        min_observations=int(inputs.opt("f0_variance_min_observations")),
    )


def _sudden_silence(inputs: Inputs) -> np.ndarray:
    """§5.4.1: "RMS falls below floor within 2 s of high activity. Death, shock,
    concentration. Inverse of intuition — a strong signal."

    DEVIATION, and the same one already made about how RMS is stored: "floor"
    is read as a level relative to the rolling baseline, not an absolute dBFS
    number. An absolute floor bakes this microphone's gain and this room's noise
    into the signal, so the detector would need retuning on any setup change and
    a stream recorded quietly would be silent by definition.

    **The DROP is the signal, not the level.** The first cut tested `z <= quiet_z`
    with a negative threshold, and it never fired once — because on any real
    stream *silence is the baseline*. The operator is not talking most of the
    time, so the rolling mean sits near the quiet level and z during silence is
    about zero, never meaningfully negative. A threshold waiting for silence to
    fall below its own baseline waits forever. What §5.4.1 is describing is a
    fall FROM something, so that is what is measured: the recent peak minus the
    current level, with `quiet_z` demoted to a ceiling that says "and it really
    is quiet now" rather than doing the detecting itself.
    """
    from clipforge.score import grid

    baseline_samples = grid.window_samples(
        inputs.cfg.get("score.rolling_baseline_window_s"), inputs.grid_hz
    )
    z, _baseline, _std = grid.rolling_zscore(
        inputs.get("mic_rms"), baseline_samples,
        float(inputs.cfg.get("score.zscore_std_floor")),
    )

    recent_peak = rolling_max(z, inputs.samples(inputs.opt("silence.lookback_s")))
    was_loud = recent_peak >= inputs.opt("silence.activity_z")
    dropped = (recent_peak - z) >= inputs.opt("silence.drop_z")
    is_quiet = z <= inputs.opt("silence.quiet_z")

    return ramp_up(was_loud & dropped & is_quiet,
                   inputs.samples(inputs.opt("silence.ramp_s")))


def _overlap_speech(inputs: Inputs) -> np.ndarray:
    """§5.4.1: "VAD on both tracks, boolean AND. Everyone talking at once =
    something happened." """
    from clipforge.score import grid

    baseline_samples = grid.window_samples(
        inputs.cfg.get("score.rolling_baseline_window_s"), inputs.grid_hz
    )
    mic = speech_gate(inputs.get("mic_rms"), inputs, baseline_samples)
    party = speech_gate(inputs.get("party_rms"), inputs, baseline_samples)
    return (mic & party).astype(np.float64)


def _input_stillness(inputs: Inputs) -> np.ndarray:
    """§5.4.1: "inverse of input_rate, GATED. Sudden stillness after activity."

    Gated is the whole word: an unbroken hour of no input is someone who is not
    playing, where a second of stillness immediately after frantic input is a
    death or a shock. Same shape as `sudden_silence`, on a different signal.

    `input_coverage` is required, not optional. A stretch where the logger was
    not running has no input rate — that is *unknown*, not *still* — and
    treating it as stillness is how §6.4's AFK penalty would swallow a stream.
    """
    rate = np.asarray(inputs.get("input_rate"), dtype=np.float64)
    covered = np.asarray(inputs.get("input_coverage"), dtype=np.float64) > 0.5

    quiet = (rate <= inputs.opt("stillness.quiet_rate")) & covered
    busy = rolling_max(np.where(covered, rate, np.nan),
                       inputs.samples(inputs.opt("stillness.lookback_s")))
    was_busy = np.nan_to_num(busy, nan=0.0) >= inputs.opt("stillness.active_rate")

    return ramp_up(quiet & was_busy,
                   inputs.samples(inputs.opt("stillness.ramp_s")))


def _reaction_onset(inputs: Inputs) -> np.ndarray:
    """§5.4.3: "sudden_silence followed within 2 s by overlap_speech +
    party_rms spike. Classic 'oh my god' moment."

    Emitted as EVENT TIMES rather than a level. It names a moment, and §5.4.3's
    sibling `flick_signature` says "emit composite event" for the same shape of
    detection — so it goes through `kernels.gaussian` like every other event
    instead of inventing a second mechanism.

    **The overlap must BEGIN after the silence, not merely be present.** §5.4.3's
    word is "followed", and the difference is not pedantry. `sudden_silence` says
    the mic went quiet; `overlap_speech` says both mics are live. Those two are
    nearly contradictory, so the only way they hold at the same instant is inside
    the speech gate's `hangover_s` — the few hundred milliseconds it keeps a
    track "speaking" to bridge the gaps between words.

    MEASURED: testing for a *concurrent* overlap fired on the speech fixture at
    54.6 s, where the operator stops talking while the party carries on. Real
    moment, wrong mechanism — the firing existed entirely inside the hangover,
    so the detector's output depended on a constant chosen to stop one sentence
    being chopped into nine. Requiring a rising EDGE of overlap after the silence
    is what §5.4.3 actually describes: everyone goes quiet for a beat, and THEN
    everyone talks at once.
    """
    from clipforge.score import grid

    silence = inputs.get("sudden_silence")
    overlap = np.asarray(inputs.get("overlap_speech")) > 0.5

    baseline_samples = grid.window_samples(
        inputs.cfg.get("score.rolling_baseline_window_s"), inputs.grid_hz
    )
    party_z, _b, _s = grid.rolling_zscore(
        inputs.get("party_rms"), baseline_samples,
        float(inputs.cfg.get("score.zscore_std_floor")),
    )

    # Where the overlap STARTS, not where it holds.
    began = overlap & ~np.concatenate(([False], overlap[:-1]))
    window = inputs.samples(inputs.opt("reaction.window_s"))
    after_silence = rolling_max(
        (silence >= inputs.opt("reaction.silence_level")).astype(np.float64), window
    ) > 0.5

    fired = began & after_silence & (party_z >= inputs.opt("reaction.party_spike_z"))
    onsets = np.flatnonzero(fired)
    return inputs.timeline[onsets] if onsets.size else np.zeros(0, dtype=np.float64)


def _laugh_events(inputs: Inputs, source: str) -> np.ndarray:
    """§5.4.2's laugh events, thresholded out of §5.5's continuous score.

    The threshold is a §17-class tunable, so it lives at score time and not in
    the stored series — the same rule as everything else in this module. What
    extraction stores is the observation ("this much of the envelope's
    fluctuation was at laughter rates"); what scoring decides is whether that
    counts as a laugh.

    One event per run, at its start. §5.5 describes "rapid periodic bursts", so
    a three-second laugh is one moment; an event per sample would hand
    `score.events.combine: sum` thirty overlapping kernels for it.
    """
    score = np.asarray(inputs.get(source), dtype=np.float64)
    over = np.isfinite(score) & (score >= inputs.opt("laughter.threshold"))
    over = drop_short_runs(over, inputs.samples(inputs.opt("laughter.min_duration_s")))

    onsets = np.flatnonzero(over & ~np.concatenate(([False], over[:-1])))
    return inputs.timeline[onsets] if onsets.size else np.zeros(0, dtype=np.float64)


def _laugh_operator(inputs: Inputs) -> np.ndarray:
    return _laugh_events(inputs, "mic_laughter")


def _laugh_party(inputs: Inputs) -> np.ndarray:
    return _laugh_events(inputs, "party_laughter")


#: Every derived signal, in dependency order — `reaction_onset` reads two of the
#: others, so they have to exist first.
DERIVATIONS: tuple[Derivation, ...] = (
    # §5.4.2 names these events; §5.5 produces a continuous score. The threshold
    # between them is the tunable, hence score time.
    Derivation("laugh_operator", ("mic_laughter",), _laugh_operator,
               emits_events=True),
    Derivation("laugh_party", ("party_laughter",), _laugh_party,
               emits_events=True),
    Derivation("mic_f0_variance", ("mic_f0",), _mic_f0_variance, unit="st"),
    Derivation("sudden_silence", ("mic_rms",), _sudden_silence),
    Derivation("overlap_speech", ("mic_rms", "party_rms"), _overlap_speech),
    Derivation("input_stillness", ("input_rate", "input_coverage"), _input_stillness),
    Derivation(
        "reaction_onset",
        ("sudden_silence", "overlap_speech", "party_rms"),
        _reaction_onset,
        emits_events=True,
    ),
)


def compute(
    raw: dict[str, np.ndarray], timeline: np.ndarray, cfg
) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, list[float]]]:
    """Run every derivation whose inputs are present.

    Returns `(values, units, events)`. A derivation with a missing input is
    absent from all three rather than present and zero — see the module
    docstring.
    """
    grid_hz = float(cfg.get("score.score_grid_hz"))
    inputs = Inputs(timeline=timeline, grid_hz=grid_hz, raw=raw, cfg=cfg)

    values: dict[str, np.ndarray] = {}
    units: dict[str, str] = {}
    events: dict[str, list[float]] = {}

    for derivation in DERIVATIONS:
        if not all(inputs.has(name) for name in derivation.requires):
            continue
        produced = derivation.compute(inputs)
        if derivation.emits_events:
            events[derivation.name] = [float(t) for t in produced]
        else:
            inputs.computed[derivation.name] = produced
            values[derivation.name] = produced
            units[derivation.name] = derivation.unit

    return values, units, events


def missing_inputs(raw: dict[str, np.ndarray]) -> dict[str, list[str]]:
    """Which derivations could not run, and what each was short of.

    Reported rather than silently skipped: "input_stillness needs input_rate" is
    the sentence that explains why a §6.5 weight did nothing.

    Deliberately the same walk as `compute` — in declaration order, accumulating
    what has been produced — so the two cannot drift into disagreeing about what
    ran. A diagnostic that contradicts the thing it describes is worse than none.
    """
    produced = set(raw)
    out: dict[str, list[str]] = {}
    for derivation in DERIVATIONS:
        absent = sorted(n for n in derivation.requires if n not in produced)
        if absent:
            out[derivation.name] = absent
        else:
            produced.add(derivation.name)
    return out
