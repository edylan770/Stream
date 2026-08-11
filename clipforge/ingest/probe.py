"""`probe` — ffprobe into the `streams` row (§5.1 stage 2).

Three things here are less obvious than they look.

**MKV reports neither `nb_frames` nor a per-stream `duration`.** Measured, not
assumed: on the fixture, ffprobe returns only `format.duration` for a Matroska
file, while the same query against an MP4 returns both. MKV is the container
§4.2/A7 mandates for real recordings, so nothing may depend on stream-level
duration or frame count.

**Constant-rate detection cannot use frame durations or PTS deltas.** See
`classify_frame_rate` — this is the subtle one, and getting it wrong would
label every OBS recording variable-rate.

**Track roles cannot be inferred from position alone.** §5.3 hardcodes
`0:a:1/2/3` against §4.2's layout. That is right for a correctly configured
OBS, wrong for anything else, and catastrophic when wrong: mistaking game audio
for the mic makes every RMS reading meaningless, and §4.2 calls the mistake
unrecoverable after the fact. Tags first, position second, config over both,
and an outright refusal when the file is genuinely ambiguous.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

from clipforge import db, ffmpeg
from clipforge.pipeline.context import StageContext, master_identity

#: Standard broadcast rates. A variable-rate file's average lands near but not
#: on one of these, and FCPXML needs a clean rational (§10.5) — `5994/100` is
#: not one, `60000/1001` is.
STANDARD_RATES: tuple[Fraction, ...] = (
    Fraction(24000, 1001),   # 23.976
    Fraction(24, 1),
    Fraction(25, 1),
    Fraction(30000, 1001),   # 29.97
    Fraction(30, 1),
    Fraction(50, 1),
    Fraction(60000, 1001),   # 59.94
    Fraction(60, 1),
    Fraction(120, 1),
)

#: Fractional tolerance for snapping an average rate onto a standard one.
RATE_SNAP_TOLERANCE = 0.005

#: Substrings OBS and friends put in MKV track titles, mapped to §4.2 roles.
#: Checked in order; first match wins.
TITLE_HINTS: tuple[tuple[str, str], ...] = (
    ("mic", "mic"),
    ("aux", "mic"),
    ("voice", "mic"),
    ("desktop", "game"),
    ("game", "game"),
    ("discord", "party"),
    ("party", "party"),
    ("chat", "party"),
    ("mix", "mixed"),
    ("master", "mixed"),
    ("track 1", "mixed"),
)

#: §4.2's mandated track order, by position, when tags say nothing.
POSITIONAL_ROLES = ("mixed", "mic", "game", "party")

SINGLE_TRACK_WARNING = (
    "This recording has ONE audio track, so mic RMS also hears the game. "
    "An explosion registers as a mic spike and no downstream signal can tell "
    "them apart (§4.2). Separate mic/game/party tracks in OBS before the next "
    "stream — this is unrecoverable after the fact, not something a later "
    "version can fix."
)


class ProbeError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# ffprobe
# --------------------------------------------------------------------------


def probe_json(path: Path, *args: str, ffprobe: str | None = None) -> dict[str, Any]:
    binary = ffprobe or ffmpeg.require("ffprobe")
    proc = ffmpeg.run([binary, "-v", "error", "-of", "json", *args, str(path)])
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ProbeError(f"ffprobe returned unparseable JSON for {path}: {exc}") from exc


def probe_container(path: Path, ffprobe: str | None = None) -> dict[str, Any]:
    return probe_json(path, "-show_streams", "-show_format", ffprobe=ffprobe)


def sample_pts(
    path: Path, *, start_s: float, seconds: float, ffprobe: str | None = None
) -> list[float]:
    """Presentation timestamps for a short window, in seconds."""
    binary = ffprobe or ffmpeg.require("ffprobe")
    interval = f"{start_s}%+{seconds}" if start_s else f"%+{seconds}"
    proc = ffmpeg.run([
        binary, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "frame=pts_time", "-of", "csv=p=0",
        "-read_intervals", interval, str(path),
    ])
    values = []
    for line in proc.stdout.splitlines():
        token = line.strip().rstrip(",")
        if token and token != "N/A":
            try:
                values.append(float(token))
            except ValueError:
                continue
    return values


# --------------------------------------------------------------------------
# frame rate
# --------------------------------------------------------------------------


def parse_rate(text: str | None) -> Fraction | None:
    if not text or text in ("0/0", "N/A"):
        return None
    try:
        value = Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None
    return value if value > 0 else None


def snap_to_standard(rate: Fraction) -> Fraction:
    """Pull a near-standard rate onto the exact one.

    A variable-rate file's `avg_frame_rate` comes back as something like
    `5994/100`. FCPXML needs `frameDuration` as an exact rational and every
    time on the timeline as an integer multiple of it, so an almost-right
    fraction produces a file Resolve rejects or silently rounds.

    Two rules matter, and the second is not obvious:

    1. An exact standard rate is returned untouched.
    2. Otherwise snap to the *nearest* candidate, not the first one in range.
       59.94 and 60 are only 0.1% apart, so any tolerance loose enough to
       catch a scruffy 59.94 also spans exact 60 — and "first match wins"
       would quietly rewrite genuine 60 fps footage as 59.94, putting every
       exported timeline on the wrong timebase and drifting about one frame
       per thousand.
    """
    if rate in STANDARD_RATES:
        return rate

    value = float(rate)
    nearest = min(STANDARD_RATES, key=lambda s: abs(value - float(s)))
    if abs(value - float(nearest)) <= float(nearest) * RATE_SNAP_TOLERANCE:
        return nearest
    return rate


@dataclass
class RateVerdict:
    """Whether the stream advances at a constant rate, and at what rate."""

    rate: Fraction
    is_vfr: bool
    reason: str
    max_residual_ticks: float = 0.0


def classify_frame_rate(
    r_frame_rate: Fraction | None,
    avg_frame_rate: Fraction | None,
    time_base: Fraction | None,
    pts: list[float],
) -> RateVerdict:
    """Decide CFR vs VFR by fitting PTS against a constant rate.

    Two obvious approaches are both wrong, and this was measured rather than
    reasoned about:

    *Comparing consecutive PTS deltas* flags every OBS recording. A CFR 30 fps
    MKV stores timestamps on a 1/1000 timebase, and 33.333 ms is not
    representable there, so the container alternates 33 and 34 ms. The deltas
    genuinely vary; the stream is genuinely constant-rate.

    *Trusting `frame=duration_time`* is worse. On that same file ffprobe
    reports a flat `0.033000` for every frame — not the real delta, and not
    even the true frame duration. It is a nominal value, so it can neither be
    confused by quantization nor detect real variable-rate content.

    What survives quantization is the *cumulative* fit: for a constant-rate
    stream, frame i sits at `start + i/rate` to within one timebase tick,
    forever. Variable-rate content drifts away without bound, because dropped
    or repeated frames accumulate.
    """
    rate = r_frame_rate or avg_frame_rate
    if rate is None:
        raise ProbeError("ffprobe reported no usable frame rate for the video stream")

    if len(pts) < 8:
        # Not enough evidence to fit. Fall back to the weak signal rather than
        # guessing, and say so.
        if r_frame_rate and avg_frame_rate and r_frame_rate != avg_frame_rate:
            return RateVerdict(snap_to_standard(avg_frame_rate), True,
                               "r_frame_rate != avg_frame_rate (too few frames to fit)")
        return RateVerdict(rate, False, "too few frames sampled to fit; assuming CFR")

    ordered = sorted(pts)
    start = ordered[0]
    step = 1.0 / float(rate)
    # One timebase tick is the finest distinction the container can express.
    # Allow a little over one so that ordinary rounding is never mistaken for
    # drift; default to a millisecond when the timebase is unknown.
    tick = float(time_base) if time_base else 0.001
    tolerance = max(tick * 1.5, step * 0.02)

    residuals = [abs(value - (start + index * step)) for index, value in enumerate(ordered)]
    worst = max(residuals)

    if worst <= tolerance:
        return RateVerdict(rate, False, "PTS follows a constant rate", worst / tick)
    return RateVerdict(
        snap_to_standard(avg_frame_rate or rate), True,
        f"PTS drifts from a constant rate by up to {worst * 1000:.1f} ms",
        worst / tick,
    )


# --------------------------------------------------------------------------
# audio track roles
# --------------------------------------------------------------------------


@dataclass
class TrackMap:
    roles: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    source: str = ""
    titles: dict[int, str] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {"roles": self.roles, "warnings": self.warnings,
             "source": self.source, "titles": {str(k): v for k, v in self.titles.items()}},
            indent=None,
        )


def role_from_title(title: str) -> str | None:
    lowered = title.lower()
    for needle, role in TITLE_HINTS:
        if needle in lowered:
            return role
    return None


def resolve_track_roles(
    audio_streams: list[dict[str, Any]], configured: dict[str, Any]
) -> TrackMap:
    """Map §4.2 roles onto ffmpeg audio stream indices.

    Order of authority: explicit config, then metadata tags, then position.
    Explicit config wins because it is the only one the operator can correct
    after seeing a warning.
    """
    count = len(audio_streams)
    if count == 0:
        raise ProbeError("no audio streams — every Phase 1 signal comes from audio")

    explicit = {role: int(index) for role, index in (configured or {}).items()
                if index is not None}
    if explicit:
        for role, index in explicit.items():
            if not 0 <= index < count:
                raise ProbeError(
                    f"config maps {role} to audio stream {index}, but the file has "
                    f"{count} audio stream(s) (0-{count - 1})"
                )
        return TrackMap(roles=explicit, source="config",
                        titles=_titles(audio_streams))

    titles = _titles(audio_streams)
    tagged: dict[str, int] = {}
    for index, title in titles.items():
        role = role_from_title(title)
        if role and role not in tagged:
            tagged[role] = index

    # Tags are only trustworthy if they identified the tracks that matter.
    if "mic" in tagged and len(tagged) >= 2:
        return TrackMap(roles=tagged, source="metadata tags", titles=titles)

    if count == 1:
        return TrackMap(
            roles={"mic": 0}, warnings=[SINGLE_TRACK_WARNING],
            source="single track (degraded)", titles=titles,
        )

    if count >= 4:
        return TrackMap(
            roles={role: index for index, role in enumerate(POSITIONAL_ROLES)},
            source="§4.2 positional order", titles=titles,
        )

    # Two or three tracks. §4.2's layout expects four, so any guess here is a
    # coin flip between "track 1 is the mix" and "track 1 is the mic" — and
    # picking wrong silently poisons every RMS reading in the stream.
    raise ProbeError(
        f"{count} audio streams and no usable track titles. §4.2 expects four "
        f"(mixed/mic/game/party) and guessing which is the mic would silently "
        f"contaminate every signal. Set ingest.audio.track_roles in "
        f"clipforge/config/local.yaml, e.g.\n"
        f"    ingest:\n      audio:\n        track_roles:\n"
        f"          mic: 1\n          game: 0\n"
        + (f"  Titles found: {titles}" if titles else "  No track titles present.")
    )


def _titles(audio_streams: list[dict[str, Any]]) -> dict[int, str]:
    titles = {}
    for index, stream in enumerate(audio_streams):
        tags = stream.get("tags") or {}
        title = tags.get("title") or tags.get("handler_name") or ""
        if title and title.lower() != "soundhandler":
            titles[index] = title
    return titles


# --------------------------------------------------------------------------
# the stage
# --------------------------------------------------------------------------


@dataclass
class ProbeResult:
    duration_s: float
    width: int
    height: int
    rate: Fraction
    is_vfr: bool
    rate_reason: str
    track_map: TrackMap
    video_codec: str
    container: str


def analyse(master: Path, cfg, ffprobe: str | None = None) -> ProbeResult:
    info = probe_container(master, ffprobe)
    streams = info.get("streams") or []
    fmt = info.get("format") or {}

    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise ProbeError(f"{master.name} has no video stream")

    # Duration: MKV carries this only at container level. Prefer the stream's
    # value where it exists (more precise), fall back to the container's.
    duration = _first_float(video.get("duration"), fmt.get("duration"))
    if duration is None or duration <= 0:
        raise ProbeError(f"could not determine a duration for {master.name}")

    sample_seconds = cfg.get("ingest.probe.rate_sample_seconds")
    pts = sample_pts(master, start_s=0, seconds=sample_seconds, ffprobe=ffprobe)
    # A second window well into the file: a stream can start CFR and degrade
    # once the encoder is under load, which is exactly when OBS drops frames.
    if duration > sample_seconds * 4:
        pts_mid = sample_pts(
            master, start_s=round(duration / 2, 3), seconds=sample_seconds, ffprobe=ffprobe
        )
    else:
        pts_mid = []

    verdict = classify_frame_rate(
        parse_rate(video.get("r_frame_rate")),
        parse_rate(video.get("avg_frame_rate")),
        parse_rate(video.get("time_base")),
        pts,
    )
    if not verdict.is_vfr and len(pts_mid) >= 8:
        mid = classify_frame_rate(
            parse_rate(video.get("r_frame_rate")),
            parse_rate(video.get("avg_frame_rate")),
            parse_rate(video.get("time_base")),
            pts_mid,
        )
        if mid.is_vfr:
            verdict = RateVerdict(mid.rate, True, f"mid-file: {mid.reason}",
                                  mid.max_residual_ticks)

    audio = [s for s in streams if s.get("codec_type") == "audio"]
    track_map = resolve_track_roles(audio, cfg.get("ingest.audio.track_roles", {}))

    return ProbeResult(
        duration_s=duration,
        width=int(video["width"]),
        height=int(video["height"]),
        rate=verdict.rate,
        is_vfr=verdict.is_vfr,
        rate_reason=verdict.reason,
        track_map=track_map,
        video_codec=video.get("codec_name", "?"),
        container=fmt.get("format_name", "?"),
    )


def _first_float(*values) -> float | None:
    for value in values:
        if value in (None, "", "N/A"):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


# -- stage hooks ------------------------------------------------------------


def params(ctx: StageContext) -> dict:
    return {
        **master_identity(ctx),
        "track_roles": ctx.cfg.get("ingest.audio.track_roles", {}),
        "rate_sample_seconds": ctx.cfg.get("ingest.probe.rate_sample_seconds"),
    }


def outputs(ctx: StageContext) -> list[Path]:
    return []  # writes database rows, not files


def verify(ctx: StageContext) -> tuple[bool, str]:
    """Probe's product is columns, so file existence cannot check it."""
    row = ctx.stream
    if row["duration_s"] is None:
        return False, "duration not recorded"
    if row["fps_num"] is None or row["fps_den"] is None:
        return False, "frame rate not recorded"
    if not row["audio_track_map"]:
        return False, "audio track map not recorded"
    return True, ""


def run(ctx: StageContext) -> None:
    master = ctx.master
    if not master.is_file():
        raise ProbeError(
            f"master not found: {master}. If the footage moved, "
            f"`clipforge db relink --from <old> --to <new>`."
        )

    result = analyse(master, ctx.cfg, ffmpeg.require("ffprobe", ctx.cfg.ffprobe_override))
    ctx.heartbeat()

    with db.transaction(ctx.conn):
        ctx.conn.execute(
            """
            UPDATE streams
               SET duration_s = ?, fps = ?, fps_num = ?, fps_den = ?,
                   resolution = ?, is_vfr = ?, audio_track_map = ?
             WHERE id = ?
            """,
            (
                result.duration_s,
                float(result.rate),
                result.rate.numerator,
                result.rate.denominator,
                f"{result.width}x{result.height}",
                1 if result.is_vfr else 0,
                result.track_map.to_json(),
                ctx.stream_id,
            ),
        )

    ctx.log(
        f"    {result.container} {result.video_codec} {result.width}x{result.height} "
        f"@ {result.rate.numerator}/{result.rate.denominator} "
        f"({float(result.rate):.3f} fps), {result.duration_s:.2f}s"
    )
    ctx.log(f"    tracks: {result.track_map.roles} (from {result.track_map.source})")

    if result.is_vfr:
        ctx.log(
            f"    WARNING  variable frame rate: {result.rate_reason}.\n"
            f"             The proxy will be forced to CFR so timestamps stay usable, but\n"
            f"             §5.2's claim that proxy and master share a timebase does not hold\n"
            f"             for this file. Frame-count verification drops to advisory and\n"
            f"             FCPXML export will use the master's own grid."
        )
    for warning in result.track_map.warnings:
        ctx.log(f"    WARNING  {warning}")
