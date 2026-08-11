"""Which video encoder the proxy stage should use.

**Measured, not assumed:** libx264 `veryfast` took 333 s to encode 682 s of
720p60 on the build machine — roughly 2× realtime, extrapolating to about two
hours for a 4-hour stream. §1.3 budgets 20–40 minutes for *all* unattended
processing, a budget Phase 2's WhisperX will also draw on. A faster CPU halves
that and still misses; a hardware encoder turns it into minutes.

Quality is the usual objection and it does not apply here. NVENC's H.264 is
meaningfully worse per bit than x264 — but §13.1 is explicit that the proxy
exists to be *scrubbed*, not watched, and it is never the source for an export.

**Listing an encoder is not detecting one.** ffmpeg builds ship `h264_nvenc`
whether or not the machine has an NVIDIA GPU; on the build machine it is listed
and fails with `Cannot load nvcuda.dll`. The only reliable probe is to encode a
fraction of a second and see whether it works.

(Not considered: an all-intra proxy codec like DNxHR LB or ProRes Proxy, which
is the usual editorial answer because every frame is a keyframe. §13.1 keeps
proxies forever and DNxHR LB at 720p is ~36 Mbps — about 65 GB per 4-hour
stream against the spec's ~4 GB, or 6.5 TB over 100 streams. §5.2's long-GOP
H.264 with A2's fixed GOP is the right trade for this project.)
"""

from __future__ import annotations

from dataclasses import dataclass

from clipforge import ffmpeg

#: Always available, always last: software encoding works everywhere.
FALLBACK = "libx264"

#: Cache probe results per (binary, codec). A test-encode costs a few hundred
#: milliseconds and the answer cannot change within a run.
_PROBE_CACHE: dict[tuple[str, str], bool] = {}


@dataclass(frozen=True)
class EncoderChoice:
    name: str
    preset: str
    args: list[str]
    source: str  # 'configured' | 'detected' | 'fallback'
    tried: tuple[str, ...] = ()

    @property
    def is_hardware(self) -> bool:
        return self.name != FALLBACK


def available_encoders(ffmpeg_bin: str) -> set[str]:
    """Encoders compiled into this build. Necessary, nowhere near sufficient."""
    proc = ffmpeg.run([ffmpeg_bin, "-hide_banner", "-encoders"], check=False)
    names = set()
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            names.add(parts[1])
    return names


def can_encode(ffmpeg_bin: str, codec: str, *, extra: list[str] | None = None) -> bool:
    """Test-encode a fraction of a second and see whether it actually works.

    This is the whole point of the module. `h264_nvenc` appears in `-encoders`
    on a machine with no NVIDIA driver and fails the moment it is opened.
    """
    key = (ffmpeg_bin, codec)
    if key in _PROBE_CACHE:
        return _PROBE_CACHE[key]

    argv = [
        ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-nostdin",
        "-f", "lavfi", "-i", "testsrc2=size=128x128:rate=30:duration=0.2",
        "-c:v", codec, *(extra or []), "-f", "null", "-",
    ]
    try:
        proc = ffmpeg.run(argv, check=False, timeout=60)
        ok = proc.returncode == 0 and "rror" not in proc.stderr
    except Exception:
        ok = False

    _PROBE_CACHE[key] = ok
    return ok


def settings_for(cfg, codec: str) -> tuple[str, list[str]]:
    """Preset and extra flags for one encoder.

    Falls back to the top-level `preset` so a codec absent from the `codecs`
    block still gets something sensible rather than crashing.
    """
    block = (cfg.get("ingest.proxy.codecs", {}) or {}).get(codec) or {}
    preset = str(block.get("preset") or cfg.get("ingest.proxy.preset"))
    return preset, [str(a) for a in (block.get("args") or [])]


def select(cfg, ffmpeg_bin: str) -> EncoderChoice:
    """Resolve `ingest.proxy.video_codec`, probing hardware if it says `auto`."""
    configured = str(cfg.get("ingest.proxy.video_codec"))

    if configured != "auto":
        preset, args = settings_for(cfg, configured)
        return EncoderChoice(configured, preset, args, "configured")

    compiled = available_encoders(ffmpeg_bin)
    tried: list[str] = []
    for candidate in cfg.get("ingest.proxy.codec_preference", []):
        candidate = str(candidate)
        if candidate == FALLBACK:
            break
        if candidate not in compiled:
            continue
        tried.append(candidate)
        preset, args = settings_for(cfg, candidate)
        # Probe with the real settings: an encoder can open fine and still
        # reject the rate control we intend to use.
        if can_encode(ffmpeg_bin, candidate, extra=["-preset", preset, *args]):
            return EncoderChoice(candidate, preset, args, "detected", tuple(tried))

    preset, args = settings_for(cfg, FALLBACK)
    return EncoderChoice(FALLBACK, preset, args, "fallback", tuple(tried))


def describe(choice: EncoderChoice) -> str:
    if choice.source == "configured":
        return f"{choice.name} (pinned in config)"
    if choice.source == "detected":
        return f"{choice.name} (hardware, detected)"
    if choice.tried:
        return f"{choice.name} (software; {', '.join(choice.tried)} present but not usable)"
    return f"{choice.name} (software; no hardware encoder compiled in)"
