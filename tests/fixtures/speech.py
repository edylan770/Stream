"""Rendering speech for the fixture, with ground truth by construction.

Phase 2 is the transcript layer and every other fixture is band-limited noise by
design, so there is nothing to transcribe. This produces authored dialogue whose
text and placement we *chose*, which is the only way a transcript test can
assert anything: nobody can say what the correct transcript of real footage is
without transcribing it by hand first, and then the test measures the hand
transcription.

**Piper, with the noise scales pinned to zero.** VITS synthesis samples noise
during inference, so the default settings give slightly different audio every
run. This fixture is the measuring instrument for everything in Phase 2 — its
`measured_dbfs` values are what tests compare pipeline output against — so it
has to be byte-identical on any machine that regenerates it. Zero noise costs a
little prosody and buys reproducibility. Verified: two renders of the same line
compare equal.

**The renderer is a seam.** Placement, levels, band-limiting and manifest shape
are ordinary code that should not need a 60 MB model to test, so
`ToneRenderer` stands in for everything except the synthesis itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from scipy.signal import resample_poly

#: Downloaded once, gitignored. ~60 MB each.
VOICES_DIR = Path(__file__).parent / "_voices"

#: One voice per speaker. Different sexes so the two tracks are unmistakably
#: different people — §5.8 assigns speakers by track energy, not by voice, but a
#: human checking the fixture should be able to hear which is which.
VOICES = {
    "A": "en_US-ryan-medium",      # the operator, on mic (§4.2 track 2)
    "B": "en_US-lessac-medium",    # the party, on Discord (§4.2 track 4)
}


class VoicesMissing(RuntimeError):
    """The Piper models are not downloaded."""


class SpeechRenderer(Protocol):
    def render(self, text: str, voice: str) -> tuple[int, np.ndarray]:
        """Return (sample_rate, mono float32 in [-1, 1])."""
        ...


# --------------------------------------------------------------------------
# the real one
# --------------------------------------------------------------------------


def voice_path(name: str, voices_dir: Path | None = None) -> Path:
    return (voices_dir or VOICES_DIR) / f"{name}.onnx"


def voices_available(voices_dir: Path | None = None) -> bool:
    return all(voice_path(name, voices_dir).is_file() for name in VOICES.values())


def download_voices(voices_dir: Path | None = None) -> list[Path]:
    """Fetch the models. Explicit, never automatic.

    Automatic download would hide a 120 MB network fetch inside a test run, and
    the suite's rule is that a fresh clone with no network still goes green —
    `needs_piper` skips instead.
    """
    from piper.download_voices import download_voice

    target = voices_dir or VOICES_DIR
    target.mkdir(parents=True, exist_ok=True)
    for name in VOICES.values():
        if not voice_path(name, target).is_file():
            download_voice(name, target)
    return [voice_path(name, target) for name in VOICES.values()]


@dataclass
class PiperRenderer:
    """Piper, pinned to deterministic synthesis."""

    voices_dir: Path = VOICES_DIR

    def __post_init__(self) -> None:
        self._cache: dict[str, object] = {}

    def _voice(self, name: str):
        if name not in self._cache:
            path = voice_path(name, self.voices_dir)
            if not path.is_file():
                raise VoicesMissing(
                    f"{path} is missing. Fetch the models once with:\n"
                    f"    python tests/fixtures/make_fixture.py --download-voices"
                )
            from piper import PiperVoice

            self._cache[name] = PiperVoice.load(path)
        return self._cache[name]

    def render(self, text: str, voice: str) -> tuple[int, np.ndarray]:
        from piper import SynthesisConfig

        chunks = list(
            self._voice(voice).synthesize(
                text,
                # noise_scale and noise_w_scale are what make VITS output vary
                # between runs. Zero removes the sampling entirely.
                SynthesisConfig(noise_scale=0.0, noise_w_scale=0.0),
            )
        )
        if not chunks:
            raise RuntimeError(f"piper produced no audio for {text!r}")
        audio = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
        return int(chunks[0].sample_rate), audio


# --------------------------------------------------------------------------
# the stand-in
# --------------------------------------------------------------------------


@dataclass
class ToneRenderer:
    """A fake voice: a two-formant buzz whose length tracks the text.

    Not speech and not trying to be. It exists so the placement, level and
    manifest machinery can be tested without a model, which is most of what
    this fixture's code actually does.
    """

    sample_rate: int = 22_050
    #: Seconds per character. Roughly conversational at ~13 characters/second.
    seconds_per_char: float = 0.075

    def render(self, text: str, voice: str) -> tuple[int, np.ndarray]:
        duration = max(0.4, len(text) * self.seconds_per_char)
        n = int(round(duration * self.sample_rate))
        t = np.arange(n, dtype=np.float64) / self.sample_rate

        # A different pitch per voice, so a test can tell them apart the way a
        # listener would.
        f0 = 110.0 if voice.endswith("A") or "ryan" in voice else 190.0
        signal = (
            0.6 * np.sin(2 * math.pi * f0 * t)
            + 0.3 * np.sin(2 * math.pi * 2 * f0 * t)
            + 0.1 * np.sin(2 * math.pi * 3 * f0 * t)
        )
        # Syllable-rate amplitude modulation, so speech_rate and VAD have
        # something with structure rather than a continuous drone.
        signal *= 0.5 + 0.5 * np.sin(2 * math.pi * 4.0 * t)
        return self.sample_rate, signal.astype(np.float32)


# --------------------------------------------------------------------------
# conditioning: what happens between the renderer and the track
# --------------------------------------------------------------------------


def resample(signal: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    """Piper renders at 22.05 kHz; the fixture is authored at 48 kHz."""
    if source_sr == target_sr:
        return signal.astype(np.float32)
    divisor = math.gcd(int(source_sr), int(target_sr))
    return resample_poly(
        signal, target_sr // divisor, source_sr // divisor
    ).astype(np.float32)


def band_limit(signal: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    """Remove everything above the cutoff, in the frequency domain.

    THIS IS NOT COSMETIC. §5.3 extracts audio at 16 kHz, whose Nyquist is 8 kHz,
    and the fixture's whole contract is that `measured_dbfs` in the manifest
    equals what the pipeline extracts. Energy above the cutoff is lost in that
    downsample, so leaving it in makes every measured level quietly wrong — the
    one thing a measuring instrument cannot be. The noise regions have been
    band-limited since commit 3 for exactly this reason; speech is no different.

    A brick wall rather than a designed filter, matching how the noise is built:
    no passband ripple to account for, and deterministic.
    """
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(signal.size, 1.0 / sample_rate)
    spectrum[freqs > cutoff_hz] = 0.0
    return np.fft.irfft(spectrum, n=signal.size).astype(np.float32)
