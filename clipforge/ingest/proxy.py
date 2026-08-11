"""`proxy` — the scrubbing copy (§5.1 stage 3, §5.2).

§13.1 puts it plainly: proxies exist **for speed, not storage**. Scrubbing a
50 GB master in the review UI is unusable, and C4 makes the review UI the
critical path for the whole system. Everything here serves one property — that
seeking to an arbitrary timestamp is instant — and that property rests entirely
on A2's fixed GOP.

Which is why this stage checks its own work. `-sc_threshold 0` currently does
what §5.2 says on ffmpeg 9, but if a future version changes that mapping, x264
goes back to inserting keyframes on scene changes, seek granularity becomes
unpredictable, and *nothing visibly breaks*. The review UI just gets slower and
the operator concludes the tool is sluggish. A few hundred milliseconds of
verification after each encode converts that into an error message.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from clipforge import db, ffmpeg
from clipforge.ingest import encoders
from clipforge.ingest import probe as probe_stage
from clipforge.pipeline import progress
from clipforge.pipeline.atomic import atomic_output
from clipforge.pipeline.context import StageContext, master_identity


class ProxyError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# command construction
# --------------------------------------------------------------------------


def target_height(source_height: int, configured: int) -> int:
    """The proxy's height: never taller than the source, always even.

    Bare `scale=-2:720` from §5.2 *upscales* anything smaller, spending encode
    time to add no information. Doing the arithmetic here rather than with a
    `min(720,ih)` filtergraph expression avoids escaping the comma inside the
    function call, and avoids the odd heights that expression can produce —
    libx264 with yuv420p rejects those outright.
    """
    height = min(int(configured), int(source_height))
    return height - (height % 2)


def audio_map_index(track_map: dict) -> int:
    """Which audio stream the proxy carries.

    §5.2 hardcodes `0:a:0`, which assumes stream 0 is the mix. If an OBS layout
    orders the mic first, that literal makes the proxy — and therefore every
    review session run against it — mic-only. Prefer the track probe identified
    as `mixed`; otherwise take the lowest-indexed role we know about.
    """
    roles = (track_map or {}).get("roles") or {}
    if "mixed" in roles:
        return int(roles["mixed"])
    if roles:
        return min(int(index) for index in roles.values())
    return 0


def build_command(
    ffmpeg_bin: str,
    master: Path,
    destination: Path,
    *,
    cfg,
    source_height: int,
    rate: Fraction,
    audio_index: int,
    encoder: encoders.EncoderChoice | None = None,
) -> list[str]:
    """§5.2, with the deviations documented in config."""
    height = target_height(source_height, cfg.get("ingest.proxy.height"))
    if encoder is None:
        preset, codec_args = encoders.settings_for(cfg, encoders.FALLBACK)
        encoder = encoders.EncoderChoice(encoders.FALLBACK, preset, codec_args, "fallback")

    argv = [
        ffmpeg_bin, "-hide_banner", "-nostdin", "-y",
        # A1 is not in play here — the proxy is a full-file transcode with no
        # seek at all — but if one is ever added it goes before -i.
        "-i", str(master),
        "-map", "0:v:0", "-map", f"0:a:{audio_index}",
        "-c:v", encoder.name,
        "-preset", encoder.preset,
        "-b:v", str(cfg.get("ingest.proxy.video_bitrate")),
    ]

    if height != source_height:
        argv += ["-vf", f"scale=-2:{height}"]

    argv += [
        "-pix_fmt", str(cfg.get("ingest.proxy.pix_fmt")),
        # A2: fixed GOP is what makes seeking instant. `-g`/`-keyint_min` are
        # generic; disabling scene-cut detection is encoder-specific and comes
        # from the per-codec args (x264 wants -sc_threshold 0, NVENC wants
        # -no-scenecut 1).
        "-g", str(cfg.get("ingest.proxy.gop")),
        "-keyint_min", str(cfg.get("ingest.proxy.keyint_min")),
        *encoder.args,
    ]

    if cfg.get("ingest.proxy.force_cfr"):
        # Makes §5.2's "identical timebase" claim true rather than assumed.
        argv += ["-fps_mode", "cfr", "-r", f"{rate.numerator}/{rate.denominator}"]

    argv += [
        "-c:a", "aac", "-b:a", str(cfg.get("ingest.proxy.audio_bitrate")),
        "-movflags", "+faststart",
    ]
    # Escape hatch for encoder-specific flags: a hardware encoder needs rate
    # control options libx264 does not understand, and vice versa.
    argv += [str(a) for a in (cfg.get("ingest.proxy.extra_output_args") or [])]
    argv += ["-progress", "pipe:2", "-nostats", str(destination)]
    return argv


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------


@dataclass
class ProxyCheck:
    duration_s: float
    frames: int | None
    expected_frames: int
    keyframe_intervals: list[int]
    median_keyframe_interval: float | None


def count_frames(path: Path, ffprobe: str) -> int | None:
    """Frame count from the container.

    We write MP4, which records `nb_frames`, so this is exact and free. (The
    master often cannot answer the same question — MKV omits the field — which
    is why the *expected* count is derived from duration and rate instead.)
    """
    info = probe_stage.probe_json(
        path, "-select_streams", "v:0", "-show_entries", "stream=nb_frames", ffprobe=ffprobe
    )
    streams = info.get("streams") or []
    if not streams:
        return None
    raw = streams[0].get("nb_frames")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def keyframe_positions(path: Path, seconds: float, ffprobe: str) -> list[float]:
    """Timestamps of keyframes in the first `seconds` of the file."""
    proc = ffmpeg.run([
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-skip_frame", "nokey", "-show_entries", "frame=pts_time",
        "-of", "csv=p=0", "-read_intervals", f"%+{seconds}", str(path),
    ])
    values = []
    for line in proc.stdout.splitlines():
        token = line.strip().rstrip(",")
        if token and token != "N/A":
            try:
                values.append(float(token))
            except ValueError:
                continue
    return sorted(values)


def inspect(path: Path, rate: Fraction, expected_frames: int, sample_s: float,
            ffprobe: str) -> ProxyCheck:
    info = probe_stage.probe_json(path, "-show_format", ffprobe=ffprobe)
    duration = float((info.get("format") or {}).get("duration", 0.0))

    intervals: list[int] = []
    median: float | None = None
    if sample_s > 0:
        positions = keyframe_positions(path, sample_s, ffprobe)
        if len(positions) >= 3:
            step = float(rate)
            # Express gaps in frames so the check is directly comparable to the
            # configured GOP, whatever the frame rate is.
            intervals = [
                round((b - a) * step) for a, b in zip(positions, positions[1:], strict=False)
            ]
            ordered = sorted(intervals)
            median = float(ordered[len(ordered) // 2])

    return ProxyCheck(
        duration_s=duration,
        frames=count_frames(path, ffprobe),
        expected_frames=expected_frames,
        keyframe_intervals=intervals,
        median_keyframe_interval=median,
    )


def assert_sound(check: ProxyCheck, *, master_duration: float, gop: int, is_vfr: bool,
                 cfg, log) -> None:
    """Raise unless the proxy actually stands in for its master.

    Called from `run`, not from `verify`: the runner evaluates `verify` for
    every completed stage on every plan, so anything that shells out to ffprobe
    there would make `clipforge status` progressively slower with each stream.
    """
    max_duration_delta = float(cfg.get("ingest.proxy.verify.max_duration_delta_s"))
    delta = abs(check.duration_s - master_duration)
    if delta > max_duration_delta:
        raise ProxyError(
            f"proxy duration {check.duration_s:.3f}s differs from the master's "
            f"{master_duration:.3f}s by {delta:.3f}s (limit {max_duration_delta}s). "
            f"Every timestamp in the system assumes the two are interchangeable."
        )

    if check.frames is not None:
        frame_delta = abs(check.frames - check.expected_frames)
        max_frame_delta = int(cfg.get("ingest.proxy.verify.max_frame_delta"))
        if frame_delta > max_frame_delta:
            message = (
                f"proxy has {check.frames} frames, expected about "
                f"{check.expected_frames} from the master's duration and rate "
                f"(off by {frame_delta}, limit {max_frame_delta})"
            )
            if is_vfr:
                # Forcing CFR on variable-rate input necessarily changes the
                # count. Failing here would wedge every such stream forever.
                log(f"    note: {message} — expected for a VFR master, not treated as an error")
            else:
                raise ProxyError(message)

    if check.median_keyframe_interval is not None:
        ratio = float(cfg.get("ingest.proxy.verify.max_keyframe_interval_ratio"))
        if check.median_keyframe_interval > gop * ratio:
            raise ProxyError(
                f"keyframes are {check.median_keyframe_interval:.0f} frames apart, "
                f"expected {gop} (A2). Fixed GOP is what makes seeking instant, and "
                f"Phase 1 review scrubs the proxy directly rather than pre-rendering "
                f"previews. Check that -sc_threshold still reaches libx264 in this "
                f"ffmpeg build."
            )


# --------------------------------------------------------------------------
# stage hooks
# --------------------------------------------------------------------------


def params(ctx: StageContext) -> dict:
    """What a change to should force a re-encode.

    Note this hashes the *configured* `video_codec`, which may be the literal
    string `auto` — not the encoder `auto` resolves to on this machine. That is
    deliberate: recording the resolved value would invalidate every proxy the
    moment the library moves from the build machine to the streaming PC, and
    trigger a full back-catalogue re-encode. A proxy is a scrubbing aid, not a
    deliverable; a mix of NVENC and x264 proxies is fine, a hundred-stream
    rebuild is not. The resolved encoder goes to `tool_metrics` instead.
    """
    row = ctx.stream
    return {
        **master_identity(ctx),
        "proxy": ctx.cfg.get("ingest.proxy"),
        # The probed values the command is built from: re-probing to a different
        # rate or resolution must rebuild the proxy.
        "source": {
            "resolution": row["resolution"],
            "fps": f"{row['fps_num']}/{row['fps_den']}",
            "track_map": row["audio_track_map"],
        },
    }


def outputs(ctx: StageContext) -> list[Path]:
    return [ctx.paths.proxy]


def verify(ctx: StageContext) -> tuple[bool, str]:
    """Cheap by design — see `assert_sound`."""
    if not ctx.paths.proxy.is_file():
        return False, "missing output proxy.mp4"
    if ctx.paths.proxy.stat().st_size == 0:
        return False, "proxy.mp4 is empty"
    if not ctx.stream["proxy_path"]:
        return False, "proxy_path not recorded"
    return True, ""


def run(ctx: StageContext) -> None:
    row = ctx.stream
    master = ctx.master
    if not master.is_file():
        raise ProxyError(
            f"master not found: {master}. If the footage moved, "
            f"`clipforge db relink --from <old> --to <new>`."
        )

    if row["fps_num"] is None or not row["resolution"]:
        raise ProxyError("stream has not been probed")

    rate = Fraction(int(row["fps_num"]), int(row["fps_den"]))
    source_w, source_h = (int(v) for v in row["resolution"].split("x"))
    master_duration = float(row["duration_s"])
    track_map = json.loads(row["audio_track_map"]) if row["audio_track_map"] else {}

    ffmpeg_bin = ffmpeg.require("ffmpeg", ctx.cfg.ffmpeg_override)
    ffprobe_bin = ffmpeg.require("ffprobe", ctx.cfg.ffprobe_override)

    height = target_height(source_h, ctx.cfg.get("ingest.proxy.height"))
    audio_index = audio_map_index(track_map)
    encoder = encoders.select(ctx.cfg, ffmpeg_bin)
    ctx.log(
        f"    {source_w}x{source_h} -> scaled to {height}p, audio from a:{audio_index}, "
        f"{'CFR ' if ctx.cfg.get('ingest.proxy.force_cfr') else ''}"
        f"{rate.numerator}/{rate.denominator}"
    )
    ctx.log(f"    encoder {encoders.describe(encoder)}")

    with atomic_output(ctx.paths.proxy) as tmp:
        argv = build_command(
            ffmpeg_bin, master, tmp, cfg=ctx.cfg,
            source_height=source_h, rate=rate, audio_index=audio_index,
            encoder=encoder,
        )
        ffmpeg.run(argv, on_stderr_line=progress.reporter(ctx, master_duration))

        expected_frames = round(master_duration * float(rate))
        check = inspect(
            tmp, rate, expected_frames,
            float(ctx.cfg.get("ingest.proxy.verify.keyframe_sample_s")), ffprobe_bin,
        )
        # Assert before the rename: a proxy that fails its checks must never
        # exist at the final path for a later stage to pick up.
        assert_sound(
            check, master_duration=master_duration,
            gop=int(ctx.cfg.get("ingest.proxy.gop")),
            is_vfr=bool(row["is_vfr"]), cfg=ctx.cfg, log=ctx.log,
        )

    size_mb = ctx.paths.proxy.stat().st_size / 1_048_576
    with db.transaction(ctx.conn):
        ctx.conn.execute(
            "UPDATE streams SET proxy_path = ? WHERE id = ?",
            (ctx.paths.rel_proxy, ctx.stream_id),
        )
    ctx.metric("proxy_size_mb", round(size_mb, 2))
    ctx.metric("proxy_duration_delta_s", round(check.duration_s - master_duration, 4))
    # Which encoder actually produced this, since `auto` resolves per machine
    # and params_hash deliberately does not record the resolution (see params).
    ctx.metric("proxy_encoder", 1.0, json.dumps({"codec": encoder.name,
                                                 "preset": encoder.preset,
                                                 "source": encoder.source}))

    keyframes = (
        f", keyframes every {check.median_keyframe_interval:.0f} frames"
        if check.median_keyframe_interval is not None else ""
    )
    ctx.log(
        f"    {size_mb:.1f} MB, {check.duration_s:.2f}s, "
        f"{check.frames if check.frames is not None else '?'} frames{keyframes}"
    )
