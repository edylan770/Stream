"""`audio_features` — frame RMS and pitch at 10 Hz (§5.1 stage 5, §5.4.1).

§5.4.1 specifies `librosa.feature.rms, hop 1600` and notes: *"Convert to dB.
Use delta vs. rolling baseline, not absolute."*

**What gets stored is absolute dBFS.** That note describes how the signal is
*consumed* — §6.2 step 3 z-scores it against a rolling window — not how it is
kept. `rolling_baseline_window_s` is a §17 tunable, so subtracting a baseline
here would freeze it into every stream ever processed, and retuning it would
mean re-extracting the entire back catalogue. C3 draws the line exactly here:
extraction is expensive and runs once, scoring is cheap and re-runnable.

**numpy rather than librosa, for RMS.** Frame RMS is a few lines over an array,
and librosa's only Phase 1 use would have been this one function at the cost of a
numba dependency. Phase 3's `mic_f0` is where librosa genuinely earns its place,
and that is where it now sits — `rms_series` is unchanged and still does not
import it.

**Pitch is tracked finely and stored coarsely, and that is not a preference.**
§5.4.1 puts every continuous signal at 10 Hz, but pyin's HMM sizes its transition
window from the hop: at 100 ms the window is 431 states against a 315-state pitch
grid and librosa raises outright. So pitch runs at `analysis_hop_s` (20 ms,
measured best on both accuracy and false-positive rate) and each 100 ms bucket
stores the median of the five frames inside it. `signal_series.params` records
both rates, so nothing downstream has to infer which one produced the number.

**Unvoiced is NaN, never zero.** Roughly a fifth of speech frames are genuinely
unvoiced — every s, f, t and k — and silence is unvoiced entirely. Zero would be
a pitch of 0 Hz, which is a measurement rather than an absence, and it would drag
every mean, z-score and rolling variance downstream toward it. The NaN is the
honest value and `score/grid.py` is what has to cope with it.

**No centering.** `librosa.feature.rms` defaults to `center=True`, padding the
signal so frame *i* is centred at *i·hop*. Rather than fabricate zeros at the
start, frame *i* here covers `[i·hop, i·hop+frame)` and `signal_series.t0`
records that its centre sits half a frame later. That column exists for this,
and it keeps the timestamp semantics in the data model rather than in a framing
convention that Phase 3 would have to rediscover.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.lib.stride_tricks import sliding_window_view

from clipforge import db, signals
from clipforge.pipeline.context import StageContext

#: role -> signal kind, per §5.4.1. `mixed` has no entry: it is never
#: extracted (§5.3 maps a:1/2/3) and nothing scores it.
SIGNAL_FOR_ROLE = {"mic": "mic_rms", "game": "game_rms", "party": "party_rms"}

#: role -> pitch kind. §5.4.1 declares `mic_f0` and `party_f0` and no game
#: equivalent, which is right: game audio has no voice, so a pitch track of an
#: explosion is a number with no referent.
F0_FOR_ROLE = {"mic": "mic_f0", "party": "party_f0"}

#: role -> laughter kind. §5.5: "run independently on the mic track and the
#: party track. Party laughter is the more valuable of the two because it is
#: external validation." No game entry, for the same reason as pitch.
LAUGHTER_FOR_ROLE = {"mic": "mic_laughter", "party": "party_laughter"}

#: Warn when more than this share of voiced pitch samples sits on the edge of
#: the configured search range. Not config: it is a display heuristic on a
#: diagnostic, like `doctor`'s stale-backup threshold, and the number it is
#: judging (`extract.f0.fmin_hz`/`fmax_hz`) is the tunable.
RAIL_WARN_FRACTION = 0.25

#: Samples read per iteration. Bounds peak memory: a 4-hour 16 kHz mono WAV is
#: 460 MB as int16 and 920 MB as float32, which is not something to load whole
#: on a machine that will also be running WhisperX.
READ_BLOCK_SAMPLES = 1 << 20  # ~65 s at 16 kHz


class FeatureError(RuntimeError):
    pass


@dataclass(frozen=True)
class Framing:
    """Frame geometry, derived from config and the file's own sample rate."""

    sample_rate: int
    frame: int
    hop: int
    signal_hz: float

    @property
    def t0(self) -> float:
        """Centre of frame 0. See the module docstring on centering."""
        return self.frame / (2.0 * self.sample_rate)

    def n_frames(self, n_samples: int) -> int:
        if n_samples < self.frame:
            return 0
        return (n_samples - self.frame) // self.hop + 1


def framing_for(sample_rate: int, *, signal_hz: float, frame_s: float) -> Framing:
    hop = int(round(sample_rate / signal_hz))
    frame = int(round(frame_s * sample_rate))
    if hop < 1:
        raise FeatureError(
            f"extract.signal_hz={signal_hz} is higher than the audio sample rate"
        )
    if frame < 1:
        raise FeatureError(f"extract.rms.frame_s={frame_s} rounds to zero samples")
    return Framing(sample_rate=sample_rate, frame=frame, hop=hop, signal_hz=signal_hz)


def frame_rms(buffer: np.ndarray, framing: Framing) -> np.ndarray:
    """RMS of every whole frame in `buffer`, as linear amplitude."""
    count = framing.n_frames(buffer.size)
    if count == 0:
        return np.zeros(0, dtype=np.float64)
    windows = sliding_window_view(buffer, framing.frame)[:: framing.hop][:count]
    return np.sqrt(np.mean(np.square(windows, dtype=np.float64), axis=1))


def to_db(rms: np.ndarray, db_floor: float) -> np.ndarray:
    """Linear amplitude to dBFS, floored.

    Digital silence is `20*log10(0) = -inf`, which propagates through every
    mean, z-score and plot downstream. Clamping the amplitude before the log
    keeps the result finite without a special case or a warning.
    """
    floor_amplitude = 10.0 ** (db_floor / 20.0)
    return 20.0 * np.log10(np.maximum(rms, floor_amplitude))


def stream_frame_rms(
    path: Path, *, signal_hz: float, frame_s: float, on_progress=None,
) -> tuple[np.ndarray, Framing, int]:
    """Stream a WAV and return its LINEAR RMS envelope, plus the framing used.

    Overlap-save: each read block is prefixed with the tail the previous one
    could not complete a frame from, so frames spanning a block boundary are
    computed exactly once and correctly.

    Factored out of `rms_series` because §5.5's laughter detector needs the same
    envelope at a much higher rate, and in LINEAR amplitude rather than dB —
    amplitude modulation is a multiplicative envelope, and taking the logarithm
    first turns the modulation the band-pass is looking for into a different
    shape. One streaming loop, two consumers.
    """
    with sf.SoundFile(path) as handle:
        if handle.channels != 1:
            raise FeatureError(
                f"{path.name} has {handle.channels} channels; audio_split writes mono"
            )
        framing = framing_for(handle.samplerate, signal_hz=signal_hz, frame_s=frame_s)

        chunks: list[np.ndarray] = []
        tail = np.zeros(0, dtype=np.float32)
        consumed = 0

        while True:
            block = handle.read(READ_BLOCK_SAMPLES, dtype="float32", always_2d=False)
            if block.size == 0:
                break
            buffer = np.concatenate((tail, block)) if tail.size else block

            count = framing.n_frames(buffer.size)
            if count:
                chunks.append(frame_rms(buffer, framing))
                tail = buffer[count * framing.hop:].copy()
            else:
                tail = buffer

            consumed += block.size
            if on_progress is not None:
                on_progress(consumed / handle.samplerate)

        total_frames = int(handle.frames)

    rms = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float64)
    return rms, framing, total_frames


def rms_series(
    path: Path, kind: str, *, signal_hz: float, frame_s: float, db_floor: float,
    role: str | None = None, on_progress=None,
) -> signals.Series:
    """Stream a WAV and produce its RMS envelope in dBFS."""
    rms, framing, total_frames = stream_frame_rms(
        path, signal_hz=signal_hz, frame_s=frame_s, on_progress=on_progress
    )
    return signals.Series(
        kind=kind,
        values=to_db(rms, db_floor),
        sample_rate_hz=signal_hz,
        t0=framing.t0,
        params={
            # §3.2 gained this column so that re-extracting at a different hop
            # is visible rather than a silent overwrite of incomparable data.
            "source": path.name,
            "role": role,
            # Explicit rather than left to a default, now that not every signal
            # is in decibels. Scoring reads this to label the review UI's `?`
            # panel, and a wrong label there is worse than none.
            "unit": "dB",
            "source_sample_rate": framing.sample_rate,
            "frame_samples": framing.frame,
            "hop_samples": framing.hop,
            "frame_s": frame_s,
            "db_floor": db_floor,
            "centered": False,
            "source_frames": total_frames,
        },
    )


# --------------------------------------------------------------------------
# laughter (§5.5)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LaughterSettings:
    """§5.5's envelope-periodicity heuristic, from config in one place."""

    band_hz: tuple[float, float]
    envelope_hz: float
    frame_s: float
    score_window_s: float
    floor_db: float
    filter_order: int
    min_depth: float

    @classmethod
    def from_config(cls, cfg) -> LaughterSettings:
        band = cfg.get("extract.laughter.band_hz")
        return cls(
            band_hz=(float(band[0]), float(band[1])),
            envelope_hz=float(cfg.get("extract.laughter.envelope_hz")),
            frame_s=float(cfg.get("extract.laughter.frame_s")),
            score_window_s=float(cfg.get("extract.laughter.score_window_s")),
            floor_db=float(cfg.get("extract.laughter.floor_db")),
            filter_order=int(cfg.get("extract.laughter.filter_order")),
            min_depth=float(cfg.get("extract.laughter.min_depth")),
        )

    def check_nyquist(self, rate: float, what: str) -> None:
        """Refuse a band the sample rate cannot represent.

        THE reason this signal does not reuse the stored `mic_rms`. §5.4 stores
        every continuous signal at 10 Hz, whose Nyquist is 5 Hz — below the
        band's own top edge. Two separate things go wrong there and both matter:
        content at 5-7 Hz has already ALIASED down to 5-3 Hz by the time it is
        sampled at 10 Hz, and `scipy.signal.butter` cannot even design the filter
        (it raises "critical frequencies must be 0 < Wn < 1"). So the failure is
        loud rather than silent, but only if something checks — hence this.
        """
        if self.band_hz[1] >= rate / 2.0:
            raise FeatureError(
                f"§5.5's laughter band tops out at {self.band_hz[1]:g} Hz, which is at "
                f"or above the Nyquist frequency of {what} ({rate:g} Hz -> "
                f"{rate / 2:g} Hz). Modulation in that band cannot be represented "
                f"there at all, let alone filtered. Raise "
                f"extract.laughter.envelope_hz above {2 * self.band_hz[1]:g} Hz."
            )


def band_share(
    envelope: np.ndarray, settings: LaughterSettings
) -> tuple[np.ndarray, np.ndarray]:
    """`(share, depth)` — how much of the envelope's fluctuation sits in §5.5's
    band, and how far that component actually swings.

    §5.5: "band-pass the RMS envelope at 4-7 Hz, threshold the energy. Cheap, no
    model, works surprisingly well."

    A SHARE rather than an absolute band energy, because absolute energy tracks
    how loud the moment was — which `mic_rms` already measures and §6.5 already
    weights. The question here is different: is the fluctuation *concentrated* at
    laughter rates? That is scale-free, so it needs no level calibration and does
    not double-count loudness.

    Zero-phase (`sosfiltfilt`): these become events, and an ordinary IIR pass
    would shift them later in time by its own group delay.
    """
    from scipy.signal import butter, sosfiltfilt

    settings.check_nyquist(settings.envelope_hz, "the laughter envelope")

    n = envelope.size
    window = max(int(round(settings.score_window_s * settings.envelope_hz)), 3)
    # `sosfiltfilt` pads by 3 * (order * 2), and refuses a signal shorter than
    # that. A clip too short to hold a few cycles has no periodicity to measure.
    if n < max(window, 12 * settings.filter_order + 1):
        blank = np.full(n, np.nan)
        return blank, blank.copy()

    nyquist = settings.envelope_hz / 2.0
    sos = butter(
        settings.filter_order,
        [settings.band_hz[0] / nyquist, settings.band_hz[1] / nyquist],
        btype="band", output="sos",
    )

    # Detrended LOCALLY, against a rolling mean rather than the file's own.
    #
    # MEASURED: subtracting the global mean made the score depend on what else
    # was in the recording. Every region of the laughter fixture is authored at
    # one level, and a global detrend still left the silence-to-region steps in
    # the denominator as "fluctuation" — so identical 5.5 Hz modulation scored
    # 0.77 in that file and 0.99 in a file containing nothing else. A signal
    # whose value at t depends on unrelated material elsewhere in the stream is
    # not a local measurement, and §6.2's rolling z-score would then be
    # normalising something that had already been normalised by the wrong thing.
    kernel = np.ones(window) / window
    local_mean = np.convolve(envelope, kernel, mode="same")
    detrended = envelope - local_mean

    filtered = sosfiltfilt(sos, detrended)

    band = np.sqrt(np.convolve(np.square(filtered), kernel, mode="same"))
    total = np.sqrt(np.convolve(np.square(detrended), kernel, mode="same"))

    share = np.divide(band, total, out=np.zeros_like(band), where=total > 1e-12)
    # Clipped, because a "share" above 1 is not a share. MEASURED on the noise
    # fixture, whose regions start and stop abruptly: the band-passed component
    # rings at a step and can locally exceed the detrended total it is a
    # fraction of, which reached 3.8. Nothing downstream was harmed — the
    # duration gate discards a transient — but a signal documented as 0..1 has
    # to actually be 0..1, or the threshold beside it stops meaning what it says.
    np.clip(share, 0.0, 1.0, out=share)

    # A SHARE ALONE IS NOT ENOUGH, and this was measured rather than reasoned.
    #
    # The share is scale-free, so a noise floor whose tiny random fluctuation
    # happens to sit in the band scores exactly as high as a real laugh. On the
    # speech fixture — which authors no laughter at all — that produced TEN
    # laugh events, most of them in the gaps between utterances.
    #
    # §5.5's own words are "rapid periodic BURSTS", and a burst has depth. So
    # the second half of the test is how far the band component actually swings
    # relative to the local level: modulation authored at depth 0.3 and 0.8
    # measured 0.21 and 0.55, while noise floor and out-of-band regions measured
    # 0.013-0.017 — more than an order of magnitude apart.
    depth = np.divide(band, local_mean, out=np.zeros_like(band),
                      where=local_mean > 1e-12)
    return share, depth


def laughter_series(
    path: Path, kind: str, *, signal_hz: float, settings: LaughterSettings,
    role: str | None = None, on_progress=None,
) -> signals.Series:
    """§5.5's envelope-periodicity score, on the standard signal grid.

    The envelope is computed HERE at `envelope_hz` rather than reusing the
    stored `mic_rms` — see `check_nyquist`. Everything else is arithmetic over
    that envelope, so re-tuning the band costs a re-extraction only because the
    envelope itself is not kept; the threshold that turns this into events is a
    score-time tunable and lives in `score/derived.py`.
    """
    envelope, framing, _total = stream_frame_rms(
        path, signal_hz=settings.envelope_hz, frame_s=settings.frame_s,
        on_progress=on_progress,
    )
    share, depth = band_share(envelope, settings)

    # Concentrated in the band AND actually swinging. Zero rather than NaN where
    # the modulation is too shallow: that is a measurement saying "nothing
    # periodic here", not an absence of measurement.
    score = np.where(depth >= settings.min_depth, share, 0.0)
    # ...but a comparison against NaN is False, so the gate would have quietly
    # converted "not measured" into "measured as zero" — which is a claim. Put
    # the absences back.
    score = np.where(np.isfinite(share) & np.isfinite(depth), score, np.nan)

    # Where there is no sound at all, the score is a ratio of two noise floors —
    # a number with no referent. NaN, not zero: commit 31's convention, and
    # `score/grid.py` knows what to do with it.
    floor = 10.0 ** (settings.floor_db / 20.0)
    score = np.where(envelope >= floor, score, np.nan)

    values = _downsample_mean(score, settings.envelope_hz, signal_hz)
    return signals.Series(
        kind=kind,
        values=values,
        sample_rate_hz=signal_hz,
        t0=framing.t0,
        params={
            "source": path.name,
            "role": role,
            "method": "envelope band share",
            # No unit: it is a ratio, and the review UI's context line is for
            # absolute levels in a unit a human already reads.
            "unit": "",
            "band_hz": list(settings.band_hz),
            "envelope_hz": settings.envelope_hz,
            "envelope_frame_s": settings.frame_s,
            "score_window_s": settings.score_window_s,
            "floor_db": settings.floor_db,
            "filter_order": settings.filter_order,
            "aggregate": "mean",
            "source_sample_rate": framing.sample_rate,
        },
    )


def _downsample_mean(values: np.ndarray, from_hz: float, to_hz: float) -> np.ndarray:
    """Average whole buckets down to the storage grid.

    Mean rather than max: the score already carries a `score_window_s` of
    smoothing, so neighbouring samples are near-duplicates and a max would only
    bias every bucket upward by the local noise.
    """
    step = from_hz / to_hz
    if not np.isclose(step, round(step)) or round(step) < 1:
        raise FeatureError(
            f"extract.laughter.envelope_hz ({from_hz:g}) must be a whole multiple "
            f"of extract.signal_hz ({to_hz:g}); it is {step:g}x"
        )
    step = int(round(step))
    usable = (values.size // step) * step
    if usable == 0:
        return np.zeros(0, dtype=np.float64)
    # Masked rather than `nanmean`, for the reason in `_bucket_median`: an
    # all-absent bucket is expected over silence, and silencing the warning it
    # raises would mean touching a process-wide filter from library code.
    buckets = np.ma.masked_invalid(values[:usable].reshape(-1, step))
    return np.ma.mean(buckets, axis=1).filled(np.nan)


# --------------------------------------------------------------------------
# pitch (§5.4.1's mic_f0 / party_f0)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class F0Settings:
    """Everything `f0_series` needs, read from config in one place."""

    fmin_hz: float
    fmax_hz: float
    analysis_rate_hz: int
    analysis_hop_s: float
    analysis_frame_s: float
    resolution: float
    chunk_s: float
    overlap_s: float

    @classmethod
    def from_config(cls, cfg) -> F0Settings:
        return cls(
            fmin_hz=float(cfg.get("extract.f0.fmin_hz")),
            fmax_hz=float(cfg.get("extract.f0.fmax_hz")),
            analysis_rate_hz=int(cfg.get("extract.f0.analysis_rate_hz")),
            analysis_hop_s=float(cfg.get("extract.f0.analysis_hop_s")),
            analysis_frame_s=float(cfg.get("extract.f0.analysis_frame_s")),
            resolution=float(cfg.get("extract.f0.resolution")),
            chunk_s=float(cfg.get("extract.f0.chunk_s")),
            overlap_s=float(cfg.get("extract.f0.overlap_s")),
        )

    def frames_per_bucket(self, signal_hz: float) -> int:
        """How many analysis frames land in one stored sample.

        Checked rather than assumed: a hop that does not divide the storage
        period leaves each bucket holding a different number of frames, so the
        series' declared sample rate stops describing its own contents.
        """
        period = 1.0 / float(signal_hz)
        count = period / self.analysis_hop_s
        if abs(count - round(count)) > 1e-9:
            raise FeatureError(
                f"extract.f0.analysis_hop_s={self.analysis_hop_s} does not divide "
                f"1/extract.signal_hz={period:g}. Each stored sample would aggregate "
                f"a different number of analysis frames."
            )
        return int(round(count))


def track_pitch(
    audio: np.ndarray, sample_rate: int, settings: F0Settings
) -> np.ndarray:
    """pyin over one contiguous span. NaN where unvoiced.

    Imported here rather than at module scope so that `rms_series` — every
    Phase 1 stream's only requirement — does not pay librosa's import cost, and
    so a broken numba install fails in the stage that needs it rather than at the
    top of a module the whole pipeline imports.
    """
    import librosa

    frame = int(round(sample_rate * settings.analysis_frame_s))
    hop = int(round(sample_rate * settings.analysis_hop_s))
    if audio.size < frame:
        return np.zeros(0, dtype=np.float64)

    f0, _voiced, _prob = librosa.pyin(
        audio, sr=sample_rate,
        fmin=settings.fmin_hz, fmax=settings.fmax_hz,
        frame_length=frame, hop_length=hop,
        resolution=settings.resolution,
        # Same convention as `rms_series`: no fabricated padding at the start.
        # Frame j covers [j*hop, j*hop+frame) and is timestamped at its centre.
        center=False,
    )
    return np.asarray(f0, dtype=np.float64)


def f0_series(
    path: Path, kind: str, *, signal_hz: float, settings: F0Settings,
    role: str | None = None, on_progress=None,
) -> signals.Series:
    """Stream a WAV and produce its pitch track, aggregated to `signal_hz`.

    Chunked for three independent reasons, any one of which would force it:
    pyin's observation matrix is `2 * n_pitch_bins * n_frames` float64 (3.6 GB
    over a 4-hour stream at a 20 ms hop); the stage must heartbeat or a run
    lasting a quarter of an hour looks crashed; and the operator deserves
    progress on the slowest thing in extraction.

    `overlap_s` of real audio is read either side of every chunk and then
    discarded, so no frame that survives was decoded by a Viterbi that started
    inside speech.
    """
    frames_per_bucket = settings.frames_per_bucket(signal_hz)

    with sf.SoundFile(path) as handle:
        if handle.channels != 1:
            raise FeatureError(
                f"{path.name} has {handle.channels} channels; audio_split writes mono"
            )
        source_rate = handle.samplerate
        total_samples = int(handle.frames)
        duration_s = total_samples / source_rate

        n_buckets = int(np.floor(duration_s * signal_hz))
        # One column per analysis frame, so the medians are a single vectorised
        # pass at the end. A 4-hour stream is 144k buckets of five, which is
        # 5.8 MB — where 144k Python lists would not be.
        collected = np.full((max(n_buckets, 0), frames_per_bucket), np.nan)
        # How many frames each bucket has taken so far. Carried ACROSS chunks:
        # `chunk_s` is not required to be a whole number of buckets, so the
        # bucket straddling a chunk boundary is filled by two calls and the
        # second must not overwrite the first.
        fill = np.zeros(max(n_buckets, 0), dtype=np.int64)

        overlap_samples = int(round(settings.overlap_s * source_rate))
        chunk_samples = max(int(round(settings.chunk_s * source_rate)), 1)

        for chunk_start in range(0, total_samples, chunk_samples):
            chunk_end = min(chunk_start + chunk_samples, total_samples)
            read_start = max(0, chunk_start - overlap_samples)
            read_end = min(total_samples, chunk_end + overlap_samples)

            handle.seek(read_start)
            block = handle.read(read_end - read_start, dtype="float32",
                                always_2d=False)
            if block.size == 0:
                continue

            audio, rate = _resample_for_pitch(block, source_rate, settings)
            pitch = track_pitch(audio, rate, settings)
            if pitch.size == 0:
                continue

            # Absolute time of each analysis frame's centre, in VOD seconds.
            hop = int(round(rate * settings.analysis_hop_s))
            frame = int(round(rate * settings.analysis_frame_s))
            times = (read_start / source_rate
                     + (np.arange(pitch.size) * hop + frame / 2.0) / rate)

            # Keep only frames belonging to this chunk, so the discarded overlap
            # cannot contribute twice or contribute a boundary artefact.
            keep = (times >= chunk_start / source_rate) & (times < chunk_end / source_rate)
            _accumulate(collected, fill, times[keep], pitch[keep], signal_hz)

            if on_progress is not None:
                on_progress(chunk_end / source_rate)

    values = _bucket_median(collected)
    return signals.Series(
        kind=kind,
        values=values,
        sample_rate_hz=signal_hz,
        # The centre of the bucket, matching rms_series' rule that a timestamp
        # describes the middle of what was measured.
        t0=0.5 / signal_hz,
        params={
            "source": path.name,
            "role": role,
            "method": "librosa.pyin",
            "unit": "Hz",
            "unvoiced": "nan",
            "source_sample_rate": source_rate,
            "analysis_rate_hz": settings.analysis_rate_hz,
            "analysis_hop_s": settings.analysis_hop_s,
            "analysis_frame_s": settings.analysis_frame_s,
            "frames_per_sample": frames_per_bucket,
            "aggregate": "median",
            "fmin_hz": settings.fmin_hz,
            "fmax_hz": settings.fmax_hz,
            "resolution": settings.resolution,
            "chunk_s": settings.chunk_s,
            "overlap_s": settings.overlap_s,
        },
    )


def _resample_for_pitch(
    block: np.ndarray, source_rate: int, settings: F0Settings
) -> tuple[np.ndarray, int]:
    """Decimate to the analysis rate. Measured to improve accuracy, not just speed."""
    target = int(settings.analysis_rate_hz)
    if target >= source_rate:
        return block.astype(np.float32), source_rate

    import librosa

    return (
        librosa.resample(block.astype(np.float32), orig_sr=source_rate,
                         target_sr=target, res_type="soxr_hq"),
        target,
    )


def _accumulate(
    collected: np.ndarray, fill: np.ndarray, times: np.ndarray,
    pitch: np.ndarray, signal_hz: float,
) -> None:
    """File each analysis frame into a free column of its storage bucket."""
    if times.size == 0 or collected.size == 0:
        return
    index = np.floor(times * signal_hz).astype(np.int64)
    inside = (index >= 0) & (index < collected.shape[0])
    index, values = index[inside], pitch[inside]
    if index.size == 0:
        return

    # Rank within this call's run of equal indices — times ascend, so a run is
    # contiguous — plus whatever earlier chunks already put in that bucket.
    order = np.arange(index.size)
    slot = fill[index] + (order - np.searchsorted(index, index, side="left"))
    room = slot < collected.shape[1]
    collected[index[room], slot[room]] = values[room]
    fill += np.bincount(index, minlength=fill.size)


def _bucket_median(collected: np.ndarray) -> np.ndarray:
    """Median of the voiced frames in each bucket; NaN when none were voiced.

    The all-unvoiced rows are excluded before the median rather than filtered
    out of the warning afterwards. `warnings.catch_warnings` mutates a
    PROCESS-WIDE filter and is documented as not thread-safe, and `review/jobs`
    runs stages on background threads — so suppressing there both leaks into
    other work and, worse, could hide a warning that mattered. Not raising it is
    cheaper than silencing it.
    """
    if collected.size == 0:
        return np.zeros(0, dtype=np.float64)
    out = np.full(collected.shape[0], np.nan)
    observed = np.isfinite(collected).any(axis=1)
    if observed.any():
        # `np.ma.median`, NOT `np.median`. The latter IGNORES a masked array's
        # mask — it warns "'partition' will ignore the 'mask'" and then takes
        # the median of the raw data, NaNs included, so every bucket holding one
        # unvoiced frame would come back unvoiced. Found by running the suite
        # with `-W error::RuntimeWarning`; the existing assertions were loose
        # enough to pass either way.
        out[observed] = np.ma.median(
            np.ma.masked_invalid(collected[observed]), axis=1
        ).filled(np.nan)
    return out


def voiced_fraction(series: signals.Series) -> float:
    """Share of samples that carry a pitch. Reported, and used by `verify`."""
    if len(series) == 0:
        return 0.0
    return float(np.mean(np.isfinite(series.values)))


def rail_fraction(series: signals.Series, settings: F0Settings) -> tuple[float, float]:
    """Share of voiced samples sitting at the bottom and top of the search range.

    MEASURED, and the reason this exists: content BELOW `fmin_hz` does not come
    back as "no pitch". A 32 Hz tone is reported as exactly 65.0 Hz on every
    frame, confidently voiced — pyin pins it to the lowest bin, because the grid
    it searches has no way to express "lower than this". The same must be assumed
    at the top.

    That matters twice over. §5.4.1's 65/400 range is a guess, and the falsifier
    written beside it in config is "f0 seen pinned at fmax during peaks" — which
    is only a falsifier if somebody is counting. And a rumble under the mic
    produces a *constant* 65 Hz, which is a flat signal rather than a loud one,
    so `mic_f0_variance` would quietly read it as the calmest stretch of the
    stream.
    """
    values = series.values[np.isfinite(series.values)]
    if values.size == 0:
        return 0.0, 0.0
    # Within one pitch bin of the rail: pyin quantises to `resolution` semitones,
    # so an exact equality test would miss by rounding.
    edge = 2.0 ** (settings.resolution / 12.0)
    at_min = float(np.mean(values <= settings.fmin_hz * edge))
    at_max = float(np.mean(values >= settings.fmax_hz / edge))
    return at_min, at_max


# --------------------------------------------------------------------------
# stage hooks
# --------------------------------------------------------------------------


def _roles(ctx: StageContext) -> dict[str, str]:
    """Extracted roles that have a signal defined, as role -> kind."""
    from clipforge.extract import audio

    return {
        role: SIGNAL_FOR_ROLE[role]
        for role in audio.roles_to_extract(
            _track_map(ctx), ctx.cfg.get("ingest.audio.extract_roles")
        )
        if role in SIGNAL_FOR_ROLE
    }


def _f0_roles(ctx: StageContext) -> dict[str, str]:
    """Roles that get a pitch track, as role -> kind.

    The intersection of three things: what §5.4.1 defines a pitch signal for,
    what this recording actually has, and what `extract.f0.roles` asks for. The
    last is a real knob rather than a formality — pitch is the most expensive
    thing in extraction and no §6.5 profile weights `party_f0`.
    """
    if not bool(ctx.cfg.get("extract.f0.enabled")):
        return {}
    wanted = set(ctx.cfg.get("extract.f0.roles") or [])
    return {
        role: F0_FOR_ROLE[role]
        for role in _roles(ctx)
        if role in F0_FOR_ROLE and role in wanted
    }


def _laughter_roles(ctx: StageContext) -> dict[str, str]:
    """Roles that get a laughter score, as role -> kind."""
    if not bool(ctx.cfg.get("extract.laughter.enabled")):
        return {}
    wanted = set(ctx.cfg.get("extract.laughter.roles") or [])
    return {
        role: LAUGHTER_FOR_ROLE[role]
        for role in _roles(ctx)
        if role in LAUGHTER_FOR_ROLE and role in wanted
    }


def _track_map(ctx: StageContext) -> dict:
    raw = ctx.stream["audio_track_map"]
    return json.loads(raw) if raw else {}


def params(ctx: StageContext) -> dict:
    return {
        "signal_hz": ctx.cfg.get("extract.signal_hz"),
        "rms": ctx.cfg.get("extract.rms"),
        "f0": ctx.cfg.get("extract.f0"),
        "laughter": ctx.cfg.get("extract.laughter"),
        "kinds": sorted(_roles(ctx).values()),
        "f0_kinds": sorted(_f0_roles(ctx).values()),
        "laughter_kinds": sorted(_laughter_roles(ctx).values()),
    }


def outputs(ctx: StageContext) -> list[Path]:
    return []  # writes signal_series rows, not files


def verify(ctx: StageContext) -> tuple[bool, str]:
    """Like probe, the product is rows, so file existence cannot check it."""
    expected = _roles(ctx)
    if not expected:
        return False, "no audio roles resolved (has audio_split run?)"

    present = set(signals.kinds(ctx.conn, ctx.stream_id))
    missing = sorted(set(expected.values()) - present)
    if missing:
        return False, f"no signal_series for {', '.join(missing)}"

    for kind in expected.values():
        series = signals.load(ctx.conn, ctx.stream_id, kind)
        if series is None or len(series) == 0:
            return False, f"{kind} is empty"
        if not math.isclose(series.sample_rate_hz, ctx.cfg.get("extract.signal_hz")):
            return False, f"{kind} was stored at {series.sample_rate_hz} Hz"

    # Pitch is checked for PRESENCE and shape, never for voicing. A `party.wav`
    # with nobody in Discord is legitimately unvoiced end to end, and a verify
    # that demanded voiced frames would re-run a 16-minute stage forever on a
    # solo stream.
    for kind in (*_f0_roles(ctx).values(), *_laughter_roles(ctx).values()):
        series = signals.load(ctx.conn, ctx.stream_id, kind)
        if series is None:
            return False, f"no signal_series for {kind}"
        if len(series) == 0:
            return False, f"{kind} is empty"
    return True, ""


def run(ctx: StageContext) -> None:
    roles = _roles(ctx)
    if not roles:
        raise FeatureError(
            "no tracks to analyse. audio_split found no usable roles, or "
            "ingest.audio.extract_roles excludes all of them."
        )

    signal_hz = float(ctx.cfg.get("extract.signal_hz"))
    frame_s = float(ctx.cfg.get("extract.rms.frame_s"))
    db_floor = float(ctx.cfg.get("extract.rms.db_floor"))

    produced = []
    for role, kind in roles.items():
        path = ctx.paths.audio(role)
        if not path.is_file():
            raise FeatureError(
                f"{path.name} is missing. audio_split should have written it; "
                f"re-run with --force audio_split."
            )

        series = rms_series(
            path, kind, signal_hz=signal_hz, frame_s=frame_s,
            db_floor=db_floor, role=role,
            on_progress=lambda _s: ctx.heartbeat(),
        )
        with db.transaction(ctx.conn):
            signals.store(ctx.conn, ctx.stream_id, series)
        produced.append(series)

    for series in produced:
        stats = signals.summarize(series)
        ctx.log(
            f"    {series.kind:<10} {len(series)} samples @ {series.sample_rate_hz:g} Hz, "
            f"median {stats['median']:.1f} dB, p95 {stats['p95']:.1f} dB, "
            f"max {stats['max']:.1f} dB"
        )
        ctx.metric(f"signal_samples.{series.kind}", float(len(series)))

    # After the RMS summary, so the log reads in the order the work happened.
    _run_laughter(ctx, signal_hz)
    _run_pitch(ctx, signal_hz)

    # A quiet mic here is the same diagnosis audio_split makes about a silent
    # file, one layer further on: nothing downstream will find anything, and
    # the cause is upstream.
    mic = next((s for s in produced if s.kind == "mic_rms"), None)
    if mic is not None and len(mic) and float(np.max(mic.values)) <= db_floor + 1.0:
        ctx.log(
            "    WARNING  mic_rms never rises above the noise floor. Scoring will "
            "find nothing. Check the track mapping in `clipforge status`."
        )


def _run_laughter(ctx: StageContext, signal_hz: float) -> None:
    """§5.5's envelope-periodicity score, per §4.2 track."""
    roles = _laughter_roles(ctx)
    if not roles:
        if bool(ctx.cfg.get("extract.laughter.enabled")):
            ctx.log("    laughter: no configured role is present in this recording")
        else:
            ctx.log("    laughter: disabled (extract.laughter.enabled)")
        return

    settings = LaughterSettings.from_config(ctx.cfg)
    # Fail here, before reading a four-hour WAV, rather than inside the filter
    # designer with a message about normalised critical frequencies.
    settings.check_nyquist(settings.envelope_hz, "the laughter envelope")

    for role, kind in roles.items():
        path = ctx.paths.audio(role)
        if not path.is_file():
            raise FeatureError(
                f"{path.name} is missing. audio_split should have written it; "
                f"re-run with --force audio_split."
            )

        series = laughter_series(
            path, kind, signal_hz=signal_hz, settings=settings, role=role,
            on_progress=lambda _s: ctx.heartbeat(),
        )
        with db.transaction(ctx.conn):
            signals.store(ctx.conn, ctx.stream_id, series)

        stats = signals.summarize(series)
        measured = stats.get("observed", 0)
        ctx.log(
            f"    {kind:<14} {len(series)} samples @ {series.sample_rate_hz:g} Hz, "
            f"{100 * measured / max(len(series), 1):.0f}% above the floor"
            + (f", median {stats['median']:.2f}, p95 {stats['p95']:.2f}"
               if measured else "")
        )
        ctx.metric(f"signal_samples.{kind}", float(len(series)))


def _run_pitch(ctx: StageContext, signal_hz: float) -> None:
    """§5.4.1's pitch tracks. The slowest thing in extraction, by a distance."""
    roles = _f0_roles(ctx)
    if not roles:
        if bool(ctx.cfg.get("extract.f0.enabled")):
            ctx.log("    f0: no configured role is present in this recording")
        else:
            ctx.log("    f0: disabled (extract.f0.enabled)")
        return

    settings = F0Settings.from_config(ctx.cfg)
    # Raises here, before a 16-minute run, rather than producing a series whose
    # declared rate does not describe its contents.
    settings.frames_per_bucket(signal_hz)
    _warm_up_pitch(settings)

    for role, kind in roles.items():
        path = ctx.paths.audio(role)
        if not path.is_file():
            raise FeatureError(
                f"{path.name} is missing. audio_split should have written it; "
                f"re-run with --force audio_split."
            )

        started = time.monotonic()
        series = f0_series(
            path, kind, signal_hz=signal_hz, settings=settings, role=role,
            on_progress=lambda _s: ctx.heartbeat(),
        )
        elapsed = time.monotonic() - started

        with db.transaction(ctx.conn):
            signals.store(ctx.conn, ctx.stream_id, series)

        voiced = voiced_fraction(series)
        stats = signals.summarize(series)
        median = stats.get("median")
        ctx.log(
            f"    {kind:<10} {len(series)} samples @ {series.sample_rate_hz:g} Hz, "
            f"{100 * voiced:.1f}% voiced"
            + (f", median {median:.0f} Hz" if median is not None else "")
            + f"  ({elapsed:.1f}s)"
        )
        ctx.metric(f"signal_samples.{kind}", float(len(series)))
        # §1.3 budgets 20-40 minutes for ALL unattended processing and this is
        # the largest single consumer after WhisperX. Recorded per role so the
        # operator can see what dropping one would buy. INVENTED metric name --
        # §14's table has no extraction-cost row beyond stage_duration_s, which
        # cannot attribute time within a stage.
        ctx.metric(
            f"f0_seconds_per_audio_hour.{role}",
            round(elapsed / max(len(series) / signal_hz / 3600.0, 1e-9), 1),
        )

        # The falsifier for fmin_hz/fmax_hz, made countable. Also INVENTED.
        at_min, at_max = rail_fraction(series, settings)
        ctx.metric(f"f0_rail_fraction.{role}", round(max(at_min, at_max), 4),
                   json.dumps({"at_fmin": round(at_min, 4),
                               "at_fmax": round(at_max, 4),
                               "fmin_hz": settings.fmin_hz,
                               "fmax_hz": settings.fmax_hz}))
        if at_max > RAIL_WARN_FRACTION:
            ctx.log(
                f"    WARNING  {100 * at_max:.0f}% of {kind}'s voiced samples sit at "
                f"fmax ({settings.fmax_hz:g} Hz). Pitch above the search range is "
                f"reported AS the range, so this is excitement being clipped. "
                f"Raise extract.f0.fmax_hz."
            )
        if at_min > RAIL_WARN_FRACTION:
            ctx.log(
                f"    WARNING  {100 * at_min:.0f}% of {kind}'s voiced samples sit at "
                f"fmin ({settings.fmin_hz:g} Hz). Content below the range is pinned "
                f"there rather than reported as unvoiced, so this usually means "
                f"rumble on the track rather than a very low voice."
            )

        if voiced == 0.0:
            # Role-aware, because the same observation means opposite things.
            # The first draft printed the party-track sentence under `mic_f0`,
            # which reads as a contradiction at the exact moment the operator
            # needs a straight answer.
            ctx.log(
                f"    note: {kind} found no voiced frames at all. "
                + ("Normal when nobody was in Discord."
                   if role == "party" else
                   "On the mic that is either genuinely no speech, or §4.2's "
                   "track mapping is wrong — check `clipforge status`.")
            )


def _warm_up_pitch(settings: F0Settings) -> None:
    """Pay numba's compile before anything is timed.

    MEASURED: the first pyin call in a process spends several seconds compiling,
    and it lands inside whichever role runs first — 7.5 s against 3.1 s for two
    identical 60 s tracks. That is noise at stream scale and ruinous at fixture
    scale, and `f0_seconds_per_audio_hour` exists precisely to answer "does pitch
    fit §1.3's budget", so it must not carry a one-off compile in the numerator
    for one role and not the other.
    """
    rate = int(settings.analysis_rate_hz)
    samples = int(rate * max(settings.analysis_frame_s * 3, 0.5))
    try:
        track_pitch(np.zeros(samples, dtype=np.float32), rate, settings)
    except Exception:  # noqa: BLE001 - a warm-up must never fail a run
        pass
