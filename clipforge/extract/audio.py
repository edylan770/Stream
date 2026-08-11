"""`audio_split` — per-role WAVs from the master (§5.1 stage 4, §5.3).

Every Phase 1 signal comes from here, and §4.2 explains why the tracks are
separate: mic RMS is garbage if game audio is mixed in, because an explosion
registers as a mic spike. This stage is the point where that separation either
survives or is lost, so it is worth being fussy about what it produced.

Three things it checks that §5.3 does not mention:

- **Duration per track.** A truncated extraction is silent corruption. It shows
  up much later as candidates that simply stop appearing partway through a
  stream, which is nearly impossible to diagnose from the far end.
- **Digital silence.** A silent `mic.wav` means the track mapping is wrong, and
  nothing else in the pipeline will say so — scoring just quietly finds
  nothing. It warns rather than fails, because a silent `party.wav` is the
  ordinary state of a stream where nobody was in Discord.
- **All outputs or none.** `atomic_outputs` means a failure halfway through
  cannot leave two of three tracks in place, which would satisfy an existence
  check while being wrong.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from clipforge import ffmpeg
from clipforge.pipeline import progress
from clipforge.pipeline.atomic import atomic_outputs
from clipforge.pipeline.context import StageContext, master_identity


class AudioSplitError(RuntimeError):
    pass


@dataclass
class TrackSummary:
    role: str
    path: Path
    duration_s: float
    sample_rate: int
    channels: int
    peak_dbfs: float

    @property
    def is_silent(self) -> bool:
        return not math.isfinite(self.peak_dbfs)


# --------------------------------------------------------------------------
# role selection
# --------------------------------------------------------------------------


def roles_to_extract(track_map: dict, configured: list[str]) -> dict[str, int]:
    """Which roles this stream actually has, in the configured order.

    Intersection rather than assertion: §5.3's three tracks are what a properly
    configured OBS produces, but a single-track file legitimately has only
    `mic`, and demanding the others would make it unprocessable.
    """
    roles = (track_map or {}).get("roles") or {}
    return {role: int(roles[role]) for role in configured if role in roles}


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------


def build_command(
    ffmpeg_bin: str, master: Path, targets: list[tuple[str, int, Path]], *, cfg
) -> list[str]:
    """§5.3, as one invocation with several outputs.

    One command rather than one per track: ffmpeg demuxes the master once and
    writes all of them, which on a 50 GB file is the difference between one
    read pass and three.
    """
    argv = [ffmpeg_bin, "-hide_banner", "-nostdin", "-y", "-i", str(master)]
    for _role, index, destination in targets:
        argv += [
            "-map", f"0:a:{index}",
            "-ac", str(cfg.get("ingest.audio.channels")),
            "-ar", str(cfg.get("ingest.audio.sample_rate_hz")),
            "-c:a", str(cfg.get("ingest.audio.codec")),
            str(destination),
        ]
    argv += ["-progress", "pipe:2", "-nostats"]
    return argv


def summarize(role: str, path: Path, probe_blocks: int) -> TrackSummary:
    """Duration and peak level, without reading the whole file.

    `soundfile.info` gives the frame count from the header, and a handful of
    evenly spaced blocks is plenty to tell "has audio" from "is digital
    silence" — the only distinction this needs to make.
    """
    info = sf.info(path)
    duration = info.frames / float(info.samplerate) if info.samplerate else 0.0

    peak = 0.0
    if info.frames > 0 and probe_blocks > 0:
        block = max(1, min(info.samplerate, info.frames // max(probe_blocks, 1) or 1))
        with sf.SoundFile(path) as handle:
            for index in range(probe_blocks):
                start = int(index * (info.frames - block) / max(probe_blocks - 1, 1))
                handle.seek(max(0, min(start, max(info.frames - block, 0))))
                data = handle.read(block, dtype="float32", always_2d=False)
                if data.size:
                    peak = max(peak, float(np.max(np.abs(data))))

    peak_dbfs = 20.0 * math.log10(peak) if peak > 0 else float("-inf")
    return TrackSummary(
        role=role, path=path, duration_s=duration,
        sample_rate=int(info.samplerate), channels=int(info.channels),
        peak_dbfs=peak_dbfs,
    )


def assert_sound(summaries: list[TrackSummary], *, master_duration: float, cfg, log) -> None:
    """Raise unless the extraction actually represents the master."""
    max_delta = float(cfg.get("ingest.audio.verify.max_duration_delta_s"))
    expected_rate = int(cfg.get("ingest.audio.sample_rate_hz"))
    expected_channels = int(cfg.get("ingest.audio.channels"))
    silence_floor = float(cfg.get("ingest.audio.verify.silence_peak_dbfs"))

    for summary in summaries:
        delta = abs(summary.duration_s - master_duration)
        if delta > max_delta:
            raise AudioSplitError(
                f"{summary.path.name} is {summary.duration_s:.3f}s but the master is "
                f"{master_duration:.3f}s (off by {delta:.3f}s, limit {max_delta}s). "
                f"A truncated track silently removes every signal after the cut."
            )
        if summary.sample_rate != expected_rate or summary.channels != expected_channels:
            raise AudioSplitError(
                f"{summary.path.name} is {summary.sample_rate} Hz / "
                f"{summary.channels}ch, expected {expected_rate} Hz / {expected_channels}ch "
                f"(§5.3)"
            )

    for summary in summaries:
        if summary.peak_dbfs > silence_floor:
            continue
        if summary.role == "mic":
            log(
                f"    WARNING  mic.wav is digitally silent. Every Phase 1 signal comes "
                f"from this track, so scoring will find nothing. The track mapping is "
                f"almost certainly wrong — check `clipforge status` and set "
                f"ingest.audio.track_roles explicitly."
            )
        else:
            log(f"    note: {summary.path.name} is silent (normal if nothing used that source)")


# --------------------------------------------------------------------------
# stage hooks
# --------------------------------------------------------------------------


def _targets(ctx: StageContext) -> dict[str, int]:
    row = ctx.stream
    track_map = json.loads(row["audio_track_map"]) if row["audio_track_map"] else {}
    return roles_to_extract(track_map, ctx.cfg.get("ingest.audio.extract_roles"))


def params(ctx: StageContext) -> dict:
    return {
        **master_identity(ctx),
        "audio": ctx.cfg.get("ingest.audio"),
        "roles": _targets(ctx),
    }


def outputs(ctx: StageContext) -> list[Path]:
    return [ctx.paths.audio(role) for role in _targets(ctx)]


def verify(ctx: StageContext) -> tuple[bool, str]:
    """Cheap: existence and non-emptiness. The real checks run in `run`."""
    expected = outputs(ctx)
    if not expected:
        return False, "no audio roles resolved (has probe run?)"
    for path in expected:
        if not path.is_file():
            return False, f"missing output {path.name}"
        if path.stat().st_size == 0:
            return False, f"{path.name} is empty"
    return True, ""


def run(ctx: StageContext) -> None:
    row = ctx.stream
    master = ctx.master
    if not master.is_file():
        raise AudioSplitError(
            f"master not found: {master}. If the footage moved, "
            f"`clipforge db relink --from <old> --to <new>`."
        )

    roles = _targets(ctx)
    if not roles:
        raise AudioSplitError(
            "no audio roles to extract. probe found no usable tracks, or "
            "ingest.audio.extract_roles excludes all of them."
        )

    master_duration = float(row["duration_s"])
    ffmpeg_bin = ffmpeg.require("ffmpeg", ctx.cfg.ffmpeg_override)
    finals = [ctx.paths.audio(role) for role in roles]

    ctx.log(
        "    extracting "
        + ", ".join(f"{role}=a:{index}" for role, index in roles.items())
        + f" at {ctx.cfg.get('ingest.audio.sample_rate_hz')} Hz mono"
    )

    with atomic_outputs(finals) as temps:
        targets = [
            (role, index, tmp)
            for (role, index), tmp in zip(roles.items(), temps, strict=True)
        ]
        argv = build_command(ffmpeg_bin, master, targets, cfg=ctx.cfg)
        ffmpeg.run(argv, on_stderr_line=progress.reporter(ctx, master_duration))

        probe_blocks = int(ctx.cfg.get("ingest.audio.verify.silence_probe_blocks"))
        summaries = [summarize(role, tmp, probe_blocks) for role, _index, tmp in targets]
        # Assert before the rename: a truncated or wrongly-shaped track must
        # never exist at the final path for audio_features to consume.
        assert_sound(summaries, master_duration=master_duration, cfg=ctx.cfg, log=ctx.log)

    total_mb = sum(path.stat().st_size for path in finals) / 1_048_576
    ctx.metric("audio_split_mb", round(total_mb, 2))
    for summary in summaries:
        level = "silent" if summary.is_silent else f"peak {summary.peak_dbfs:.1f} dBFS"
        ctx.log(f"    {summary.role:<6} {summary.duration_s:.2f}s, {level}")
    ctx.log(f"    {total_mb:.1f} MB across {len(finals)} track(s)")
