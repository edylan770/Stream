"""§8.2/§8.3 — loudness normalisation to a target, measured rather than hoped.

> | Loudness normalization | ffmpeg `loudnorm` to -14 LUFS | **Better** --
> avoids platform's worse normalization |

§8.3's burn-in applies `loudnorm` in ONE pass. That is a dynamic normaliser: it
adapts as it goes and lands near the target rather than on it. Two passes --
measure the whole clip, then apply with those measurements -- is the documented
recipe and is more accurate.

MEASURED on the speech fixture, by ENCODING each window both ways and reading
the resulting file on its own -- the only number that means anything is the one
in the file the operator would post. Target -14 LUFS, TP -1.5, LRA 11:

    window             length   source   one-pass  err    two-pass  err
    first utterance      8 s    -18.2     -14.3    0.30    -13.9    0.10
    three utterances    14 s    -18.5     -14.5    0.50    -14.0    0.00
    the overlap          8 s    -16.5     -16.4    2.40    -14.2    0.20
    a run of three      14 s    -24.0     -16.4    2.40    -15.4    1.40
    most of it          50 s    -19.9     -15.4    1.40    -14.3    0.30
    all of it           95 s    -13.3     -12.3    1.70    -13.5    0.50

    mean |error|:  one-pass 1.45 LU    two-pass 0.42 LU

**Two passes are better on average, and NOT reliably better on any one clip.**
Over nine jittered windows: one-pass mean 1.03 LU / worst 3.00; two-pass mean
0.63 / worst 1.70. Shifting a window by 0.2 s moved two-pass by 1.9 LU, because
pass 1's gated measurement over eight seconds is sensitive to which blocks
clear the gate. So two passes ship -- better on both statistics -- but the
result is verified afterwards and warned about rather than assumed, and
`verify_tolerance_lu` is set outside that measured spread instead of inside it.

Three things the measurement turned up that the plan had not anticipated, and
which are most of this module:

**1. Two passes fail CATASTROPHICALLY on a near-silent window, and loudnorm's
own numbers cannot tell you it is about to happen.** The fixture's first 8 s
are silence at a -35 dBFS noise floor. Two-pass landed at **-32.0 LUFS** against
a -14 target -- eighteen LU off -- because pass 1's measurements are
meaningless and pass 2 believes them. One-pass handled the same window to
0.30 LU, because it adapts as it goes.

A clip of gameplay with nobody talking is exactly this case, so it needs a
guard. The obvious guard is wrong:

    window            standalone ebur128    loudnorm input_i    input_lra
    silence 0-8 s           -36.2               -17.75            20.50
    silence 32-52 s         -36.8               -16.19            20.30
    first utterance         -18.3               -18.32             4.90
    the overlap             -16.5               -16.54            22.00

**loudnorm reports the silent window at -17.75 LUFS**, indistinguishable from
ordinary quiet speech, and its LRA does not separate them either. Its gating
threshold sits above the noise floor, so it averages only the loudest
fragments and reports a level the content does not have. A floor test on
`input_i` would sail straight past the one case it exists to catch -- which is
how this was found: the test asserting the guard fires on the fixture's
authored silence failed.

So the gate is a **separate, standalone `ebur128` pass** over the window, and
what it is compared against is **how much gain the target would need** rather
than an absolute floor -- gain is what does the damage, and a limit expressed
that way keeps its meaning if `target_lufs` moves. The fixture's silence needs
+22.2 dB; its quiet speech needs 4 to 10.

Measured and reproducible: the same window reads -36.2 alone and -17.8 when
`ebur128` is chained ahead of `loudnorm` in one graph, independent of how the
seek is done. Whatever ffmpeg negotiates between the two filters changes what
the first one sees, so they are not run in one chain.

Normalising such a clip would amplify its noise floor by twenty-odd dB, which
is worse than a quiet clip -- so it is left alone and says so.

**2. `linear=true` is requested and, here, never granted.** Every single run
came back `normalization_type: dynamic`. Linear normalisation is refused
whenever the required gain would push true peak past the ceiling, and on this
material it always does -- input TP between -0.96 and -3.09 dBTP needing about
+2.9 dB. The flag is still set, because on quieter material it would engage;
but nothing here may claim the result is a transparent linear gain.

**3. `loudnorm` outputs 192 kHz.** MEASURED: with no `aresample` after it the
encoder is handed `192000 Hz`, and AAC-LC tops out at 96 kHz. The rate is
therefore pinned in the same filter chain. Silent wrongness, cheap to prevent.

A fourth, recorded because it twice nearly sent this the other way. The first
attempt measured an *already rendered* clip rather than the master, and an
unrepresentative window, and concluded two-pass was worse. The second measured
with `ebur128` appended to the filter chain -- convenient, and wrong for the
same reason finding 1 describes. Both are why every number above comes from an
encoded file measured on its own.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from clipforge import ffmpeg
from clipforge.render import RenderError

#: What loudnorm resamples to internally. Anything downstream that cares about
#: sample rate has to say so, because this is not the input's rate.
LOUDNORM_INTERNAL_HZ = 192000

#: `Integrated loudness:` block of an `ebur128` summary.
_EBUR128_I = re.compile(r"Integrated loudness:\s*\n\s*I:\s*(-?[\d.]+|-inf)", re.M)


class LoudnessError(RenderError):
    pass


@dataclass(frozen=True)
class Measured:
    """What pass 1 found. The names are loudnorm's own, minus the `input_`."""

    i: float
    tp: float
    lra: float
    thresh: float
    #: loudnorm's own estimate of how far its one-pass result missed by. The
    #: documented recipe feeds this back as `offset`.
    target_offset: float

    @property
    def silent(self) -> bool:
        """True when there is effectively nothing here to normalise."""
        return self.i == float("-inf")


# --------------------------------------------------------------------------
# pass 1
# --------------------------------------------------------------------------


def measure_filter(cfg, before: str = "") -> str:
    """The analysis chain: whatever edits apply, then loudnorm in report mode.

    `before` is the rest of the audio chain -- muting and cuts, once §8.6 and
    §8.2's filler removal exist. Pass 1 has to see exactly what pass 2 will
    normalise, or the measurements describe audio that is never encoded.
    """
    target = _targets(cfg)
    analysis = (
        f"loudnorm=I={target['i']}:TP={target['tp']}:LRA={target['lra']}"
        f":print_format=json"
    )
    return f"{before},{analysis}" if before else analysis


def parse(stderr: str) -> Measured | None:
    """Pull pass 1's JSON out of ffmpeg's stderr.

    The LAST object in the stream: ffmpeg writes its own diagnostics around it,
    and a filter graph may contain more than one reporting filter.
    """
    end = stderr.rfind("}")
    start = stderr.rfind("{", 0, end)
    if end == -1 or start == -1:
        return None
    try:
        blob = json.loads(stderr[start:end + 1])
        return Measured(
            i=float(blob["input_i"]),
            tp=float(blob["input_tp"]),
            lra=float(blob["input_lra"]),
            thresh=float(blob["input_thresh"]),
            target_offset=float(blob["target_offset"]),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def measure(
    ffmpeg_bin: str,
    source: Path,
    *,
    cfg,
    clip_start: float,
    duration: float,
    audio_index: int,
    before: str = "",
) -> Measured | None:
    """Run pass 1 over the clip window. None when the JSON is unreadable.

    Decodes audio only -- no video filtering, no encode, output to null. On a
    45 s clip that is about a second.
    """
    argv = [
        ffmpeg_bin, "-hide_banner", "-nostdin",
        # A1 applies here too: this seeks the master.
        "-ss", f"{clip_start:.6f}",
        "-i", str(source),
        "-t", f"{duration:.6f}",
        "-map", f"0:a:{audio_index}",
        "-af", measure_filter(cfg, before),
        "-f", "null", "-",
    ]
    proc = ffmpeg.run(argv)
    return parse(proc.stderr or "")


# --------------------------------------------------------------------------
# pass 2
# --------------------------------------------------------------------------


def _targets(cfg) -> dict:
    return {
        "i": float(cfg.get("render.loudness.target_lufs", -14.0)),
        "tp": float(cfg.get("render.loudness.true_peak_db", -1.5)),
        "lra": float(cfg.get("render.loudness.lra", 11.0)),
    }


def apply_filter(cfg, measured: Measured | None) -> str:
    """The loudnorm to put in the encode, plus the sample-rate pin.

    With `measured`, this is pass 2 of the documented two-pass recipe. Without
    it -- an ffmpeg whose JSON could not be read -- it degrades to §8.3's
    single pass rather than refusing to render.
    """
    target = _targets(cfg)
    parts = [f"loudnorm=I={target['i']}:TP={target['tp']}:LRA={target['lra']}"]

    if measured is not None:
        parts[0] += (
            f":measured_I={measured.i}:measured_TP={measured.tp}"
            f":measured_LRA={measured.lra}:measured_thresh={measured.thresh}"
            f":offset={measured.target_offset}"
            # Requested, and on peaky material never granted -- loudnorm falls
            # back to dynamic when a linear gain would breach the TP ceiling.
            f":linear=true"
        )

    # MEASURED: without this the encoder is handed 192 kHz, which AAC-LC cannot
    # represent. Not optional, and not a detail loudnorm mentions.
    rate = int(cfg.get("render.loudness.sample_rate_hz", 48000))
    parts.append(f"aresample={rate}")
    return ",".join(parts)


def should_normalise(cfg, source_lufs: float | None) -> tuple[bool, str]:
    """Whether to normalise at all, and why not when not.

    `source_lufs` is a **standalone `ebur128`** measurement of the clip window,
    NOT loudnorm's `input_i`. See the module docstring: loudnorm reports the
    fixture's silent opening at -17.75 LUFS, which no threshold could
    distinguish from quiet speech, while ebur128 correctly reports -36.2.

    The test is on **how much gain the target would need**, not on an absolute
    floor. Gain is the thing that does the damage -- lifting a noise floor
    twenty-odd dB is what makes hiss audible -- and a limit expressed that way
    keeps its meaning when `target_lufs` changes, where a fixed floor would
    quietly become stricter or looser.

    MEASURED on the fixture: its authored silence reads -36.2 LUFS, needing
    +22.2 dB to reach -14; ordinary quiet speech in the same recording sits
    around -18 to -24, needing 4 to 10.
    """
    if source_lufs is None:
        return True, ""          # nothing measurable went wrong; carry on
    if source_lufs == float("-inf"):
        return False, "the clip is digital silence"

    target = _targets(cfg)["i"]
    limit = float(cfg.get("render.loudness.max_gain_db", 20.0))
    gain = target - source_lufs
    if gain > limit:
        return False, (
            f"the clip measures {source_lufs:.1f} LUFS and reaching "
            f"{target:g} would need +{gain:.1f} dB, past "
            f"render.loudness.max_gain_db ({limit:g}) -- that raises the noise "
            f"floor rather than the content"
        )
    return True, ""


def chain(cfg, measured: Measured | None, before: str = "",
          source_lufs: float | None = None) -> str | None:
    """The whole audio filter chain for the encode, or None to leave it alone."""
    if not bool(cfg.get("render.loudness.enabled", True)):
        return before or None
    if not should_normalise(cfg, source_lufs)[0]:
        return before or None
    applied = apply_filter(cfg, measured)
    return f"{before},{applied}" if before else applied


# --------------------------------------------------------------------------
# checking the result
# --------------------------------------------------------------------------


def source_loudness(
    ffmpeg_bin: str,
    source: Path,
    *,
    clip_start: float,
    duration: float,
    audio_index: int,
    before: str = "",
) -> float | None:
    """EBU R128 integrated loudness of the clip window, on its own.

    Deliberately a separate pass from `measure`. Chaining `ebur128` ahead of
    `loudnorm` in one graph changes what `ebur128` reports -- MEASURED, the
    fixture's silent opening reads -36.2 alone and -17.8 chained, whichever way
    the seek is done. Two decodes of a 45 s window cost about two seconds; a
    guard reading the wrong number costs a clip whose noise floor got raised
    twenty dB.
    """
    filters = f"{before},ebur128" if before else "ebur128"
    proc = ffmpeg.run(
        [ffmpeg_bin, "-hide_banner", "-nostdin",
         "-ss", f"{clip_start:.6f}", "-i", str(source),
         "-t", f"{duration:.6f}", "-map", f"0:a:{audio_index}",
         "-af", filters, "-f", "null", "-"],
        check=False,
    )
    return _integrated_from(proc.stderr or "")


def _integrated_from(stderr: str) -> float | None:
    match = _EBUR128_I.search(stderr)
    if not match:
        return None
    value = match.group(1)
    return float("-inf") if value == "-inf" else float(value)


def integrated(ffmpeg_bin: str, path: Path) -> float | None:
    """Integrated loudness of a finished file, per EBU R128.

    A separate measurement from loudnorm's own: the point is to check the file
    that was written, not to ask the filter whether it thinks it worked.
    """
    proc = ffmpeg.run(
        [ffmpeg_bin, "-hide_banner", "-nostdin", "-i", str(path),
         "-af", "ebur128", "-f", "null", "-"],
        check=False,
    )
    return _integrated_from(proc.stderr or "")


def off_target(cfg, loudness: float | None) -> str:
    """A warning line when the finished file missed, or '' when it did not.

    A warning, never a failure. The clip is postable either way, and whether
    1.5 LU matters is the operator's call -- but it should be a call, not a
    surprise.
    """
    if loudness is None:
        return ""
    target = _targets(cfg)["i"]
    tolerance = float(cfg.get("render.loudness.verify_tolerance_lu", 1.0))
    if loudness == float("-inf"):
        return "the finished clip is silent"
    delta = abs(loudness - target)
    if delta <= tolerance:
        return ""
    return (
        f"loudness came out {loudness:.1f} LUFS against a {target:g} target "
        f"(off by {delta:.1f} LU, tolerance {tolerance:g})"
    )
