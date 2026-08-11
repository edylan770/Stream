"""`audio_features` — frame RMS at 10 Hz (§5.1 stage 5, §5.4.1).

§5.4.1 specifies `librosa.feature.rms, hop 1600` and notes: *"Convert to dB.
Use delta vs. rolling baseline, not absolute."*

**What gets stored is absolute dBFS.** That note describes how the signal is
*consumed* — §6.2 step 3 z-scores it against a rolling window — not how it is
kept. `rolling_baseline_window_s` is a §17 tunable, so subtracting a baseline
here would freeze it into every stream ever processed, and retuning it would
mean re-extracting the entire back catalogue. C3 draws the line exactly here:
extraction is expensive and runs once, scoring is cheap and re-runnable.

**numpy rather than librosa.** Frame RMS is a few lines over an array, and
librosa's only Phase 1 use would be this one function at the cost of a numba
dependency — heavy, and historically slow to support new Python releases, which
matters while the target machine's interpreter is still an open question.
Phase 3's `mic_f0` (pyin) is where librosa genuinely earns its place.

**No centering.** `librosa.feature.rms` defaults to `center=True`, padding the
signal so frame *i* is centred at *i·hop*. Rather than fabricate zeros at the
start, frame *i* here covers `[i·hop, i·hop+frame)` and `signal_series.t0`
records that its centre sits half a frame later. That column exists for this,
and it keeps the timestamp semantics in the data model rather than in a framing
convention that Phase 3 would have to rediscover.
"""

from __future__ import annotations

import math
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


def rms_series(
    path: Path, kind: str, *, signal_hz: float, frame_s: float, db_floor: float,
    role: str | None = None, on_progress=None,
) -> signals.Series:
    """Stream a WAV and produce its RMS envelope in dBFS.

    Overlap-save: each read block is prefixed with the tail the previous one
    could not complete a frame from, so frames spanning a block boundary are
    computed exactly once and correctly.
    """
    with sf.SoundFile(path) as handle:
        if handle.channels != 1:
            raise FeatureError(
                f"{path.name} has {handle.channels} channels; audio_split writes mono"
            )
        framing = framing_for(
            handle.samplerate, signal_hz=signal_hz, frame_s=frame_s
        )

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


def _track_map(ctx: StageContext) -> dict:
    import json

    raw = ctx.stream["audio_track_map"]
    return json.loads(raw) if raw else {}


def params(ctx: StageContext) -> dict:
    return {
        "signal_hz": ctx.cfg.get("extract.signal_hz"),
        "rms": ctx.cfg.get("extract.rms"),
        "kinds": sorted(_roles(ctx).values()),
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

    # A quiet mic here is the same diagnosis audio_split makes about a silent
    # file, one layer further on: nothing downstream will find anything, and
    # the cause is upstream.
    mic = next((s for s in produced if s.kind == "mic_rms"), None)
    if mic is not None and len(mic) and float(np.max(mic.values)) <= db_floor + 1.0:
        ctx.log(
            "    WARNING  mic_rms never rises above the noise floor. Scoring will "
            "find nothing. Check the track mapping in `clipforge status`."
        )
