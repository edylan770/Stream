"""§8 — turning an approved moment into a postable file.

> **Auto-finish every approved clip to postable quality. Then selectively
> upgrade.** [...] most clips do not deserve 30 minutes of attention. Spending
> equal effort on all five is how the pipeline dies.

The library half of `clipforge render`. Kept apart from the CLI for the reason
`register` was split in commit 14a: the app shell needs to call this without a
parser, and the UI's job runner will.

**This is a command, not a §5.1 stage**, for two reasons. A stage re-runs
unattended whenever its inputs change, and burning encoder minutes on a 50 GB
master because a re-score bumped a sequence number would be wrong. And it
*cannot* be a stage anyway: it operates on approved moments, which do not exist
until after review.

**One decode, one encode.** The source is read once, `split` forks it inside
the filter graph, the regions are composed, and captions are burned last —
after every scale, because libass draws onto the frame it is given.

**libx264, deliberately not `encoders.select`.** That detector exists to push a
three-hour proxy through a throughput bottleneck, and it was measured doing so:
libx264 333 s against h264_qsv 139.5 s. A 45-second deliverable has the
opposite priority — it is encoded once and watched many times — and a hardware
encoder at equal bitrate is visibly worse than x264 at CRF 18. §8.3 asks for
x264 by name.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

from clipforge import db, ffmpeg, paths
from clipforge.render import RenderError
from clipforge.ingest import probe as probe_stage
from clipforge.ingest import proxy as proxy_stage
from clipforge.pipeline.atomic import atomic_output
from clipforge.render import (
    ass,
    crop,
    edits,
    loudness,
    presets,
    selection,
    timeline,
    words,
)
from clipforge.render.crop import Template
from clipforge.render.selection import Moment
from clipforge.render.timeline import EditPlan

#: The label the video graph ends on, so the caller maps the same thing whether
#: or not captions were burned.
VIDEO_LABEL = "vout"

CAPTIONS_NAME = "captions.ass"


# --------------------------------------------------------------------------
# what to render
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Options:
    min_rating: int = 2
    template: str | None = None
    out_dir: Path | None = None
    limit: int | None = None
    stills: bool = False
    dry_run: bool = False
    allow_vfr: bool = False
    preset: str | None = None
    #: §8.6: "produce a dual export -- clean version for platforms that require
    #: it, unmuted version elsewhere." Only meaningful with muting on.
    dual: bool = False
    #: Set by the dual pass to force muting off for the second file.
    mute_override: bool | None = None


@dataclass
class Clip:
    """One finished file, and what went into it."""

    index: int
    moment: Moment
    plan: EditPlan
    destination: Path
    caption_lines: int = 0
    caption_words: int = 0
    interpolated: int = 0
    duration_s: float = 0.0
    size_bytes: int = 0
    #: Integrated loudness of the finished file, per EBU R128. None when it was
    #: not measured (a still, or verification turned off).
    loudness_lufs: float | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def candidate_id(self) -> int | None:
        return self.moment.candidate_ids[0] if self.moment.candidate_ids else None


@dataclass
class Result:
    clips: list[Clip] = field(default_factory=list)
    template: Template | None = None
    check: crop.Check | None = None
    preset: presets.Preset | None = None
    graph: str = ""
    skipped: int = 0


# --------------------------------------------------------------------------
# geometry and audio
# --------------------------------------------------------------------------


def source_for(cfg, stream, stream_paths: paths.StreamPaths) -> Path:
    """Master or proxy, per `render.source`.

    Separate from `export.source` because they are different decisions: an
    FCPXML references the master for an editor to conform against, while a
    burned-in short *is* the deliverable and wants the quality source even when
    the FCPXML path has been pointed at a proxy.
    """
    which = str(cfg.get("render.source", "master")).lower()
    if which == "proxy":
        if not stream["proxy_path"]:
            raise RenderError(
                f"{stream['id']} has no proxy. Run `clipforge run {stream['id']}`."
            )
        return stream_paths.absolute(stream["proxy_path"])
    if which != "master":
        raise RenderError(
            f"render.source must be 'master' or 'proxy', got {which!r}")
    return Path(stream["master_path"])


def audio_stream_index(cfg, stream) -> int | None:
    """Which audio stream becomes the clip's audio.

    §8.3's burn-in has a bare `-af`, so ffmpeg takes the first audio stream --
    which is the mix only by luck of the OBS layout. A short carrying game
    audio and no voice is a silent failure that survives all the way to upload.

    `proxy.audio_map_index` already solves exactly this for §5.2 (prefer the
    probed `mixed` role, else the lowest-indexed role found), so it is reused
    rather than re-derived.
    """
    setting = cfg.get("render.audio.source", "auto")
    if setting not in (None, "auto", ""):
        return int(setting)
    raw = stream["audio_track_map"]
    if not raw:
        return None
    track_map = json.loads(raw)
    if not (track_map.get("roles") or {}):
        return None
    return proxy_stage.audio_map_index(track_map)


def clip_window(cfg, moment: Moment, duration_s: float) -> EditPlan:
    """The approved window plus handles, clamped to the source."""
    handles = float(cfg.get("render.handles_s", 0.0))
    start = max(moment.t_start - handles, 0.0)
    end = min(moment.t_end + handles, duration_s)
    if end - start <= timeline.EPSILON:
        raise RenderError(
            f"moment {moment.t_start:.2f}-{moment.t_end:.2f}s falls outside the "
            f"source's {duration_s:.2f}s"
        )
    return timeline.build(start, end)


# --------------------------------------------------------------------------
# captions
# --------------------------------------------------------------------------


def caption_script(cfg, conn, stream_id: str, stream, plan: EditPlan,
                   template: Template) -> tuple[ass.Script | None, words.WordLoad]:
    """The ASS document for one clip, or None when captions are off."""
    load = words.load(conn, stream_id, plan, stream["audio_track_map"])
    if not bool(cfg.get("render.captions.enabled", True)) or not load.words:
        return None, load

    script = ass.build_from_config(
        cfg, load.words,
        width=template.output[0],
        height=template.output[1],
        duration=plan.duration,
        roles=(words.MIC, words.PARTY),
    )
    return script, load


# --------------------------------------------------------------------------
# the command
# --------------------------------------------------------------------------


def audio_filters(cfg, plan: EditPlan,
                  mute: list[tuple[float, float]] | None = None) -> str:
    """Everything that happens to the audio BEFORE loudness normalisation.

    Cuts first, then muting: the mute ranges are already in output time, which
    is the clock that exists AFTER the cuts, so muting first would place them
    against a timeline that no longer exists by the time audio is written.

    Commit 26 built this seam and applies the same chain in loudness pass 1, so
    the measurement describes the audio that is actually encoded. Measuring
    un-muted audio and normalising muted audio would be wrong by however much
    the muting removed, and nothing downstream could tell.
    """
    return edits.audio_chain(plan, mute or [])


def build_command(
    ffmpeg_bin: str,
    source: Path,
    destination: Path,
    *,
    cfg,
    plan: EditPlan,
    graph: str,
    audio_index: int | None,
    still: bool = False,
    audio_chain: str | None = None,
    preset: presets.Preset | None = None,
) -> list[str]:
    """The whole encode, as a list. Pure -- no I/O, so it is testable.

    A1: `-ss` goes before `-i`, and `ffmpeg.run` refuses the command if it does
    not. On a 50 GB master that is the difference between milliseconds and
    minutes.
    """
    argv: list[str] = [
        ffmpeg_bin, "-hide_banner", "-nostdin", "-y",
        "-ss", f"{plan.clip_start:.6f}",
        "-i", str(source),
        "-t", f"{plan.source_duration:.6f}",
        "-filter_complex", graph,
        "-map", f"[{VIDEO_LABEL}]",
    ]

    if still:
        # A still exists to check crop geometry, so it is deliberately silent,
        # single-frame, and un-captioned -- see `render_stream`.
        argv += ["-frames:v", "1", "-an", str(destination)]
        return argv

    def setting(key, fallback):
        return presets.setting(cfg, preset, key, fallback)

    if audio_index is None:
        argv += ["-an"]
    else:
        argv += ["-map", f"0:a:{audio_index}"]
        if audio_chain:
            argv += ["-af", audio_chain]
        argv += [
            "-c:a", str(setting("audio_codec", "aac")),
            "-b:a", str(setting("audio_bitrate", "192k")),
        ]

    argv += [
        "-c:v", str(setting("codec", "libx264")),
        "-crf", str(setting("crf", 18)),
        "-preset", str(setting("preset", "slow")),
        "-pix_fmt", str(setting("pix_fmt", "yuv420p")),
        # Short-form players seek from the first byte; without this the moov
        # atom lands at the end and the first play stalls while it downloads.
        "-movflags", "+faststart",
        "-progress", "pipe:2", "-nostats",
        str(destination),
    ]
    return argv


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------


@dataclass
class ClipCheck:
    width: int = 0
    height: int = 0
    duration_s: float = 0.0
    pix_fmt: str = ""
    has_audio: bool = False


def inspect(path: Path, ffprobe: str) -> ClipCheck:
    info = probe_stage.probe_container(path, ffprobe=ffprobe)
    duration = float((info.get("format") or {}).get("duration", 0.0))
    streams = info.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    return ClipCheck(
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        duration_s=duration,
        pix_fmt=str(video.get("pix_fmt") or ""),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
    )


def assert_sound(check: ClipCheck, *, template: Template, plan: EditPlan,
                 want_audio: bool, cfg) -> None:
    """Raise unless the file is what was asked for.

    The proxy stage's precedent: the expensive checks live beside the encode,
    not in a `verify` hook the runner would evaluate on every `status` call.
    """
    expected_w, expected_h = template.output
    if (check.width, check.height) != (expected_w, expected_h):
        raise RenderError(
            f"encoded {check.width}x{check.height}, expected "
            f"{expected_w}x{expected_h} from template {template.name!r}"
        )

    tolerance = float(cfg.get("render.verify.max_duration_delta_s", 0.5))
    delta = abs(check.duration_s - plan.duration)
    if delta > tolerance:
        raise RenderError(
            f"clip is {check.duration_s:.2f}s, expected {plan.duration:.2f}s "
            f"(off by {delta:.2f}s, tolerance {tolerance}s)"
        )

    wanted_fmt = str(cfg.get("render.video.pix_fmt", "yuv420p"))
    if check.pix_fmt and check.pix_fmt != wanted_fmt:
        raise RenderError(
            f"clip is {check.pix_fmt}, expected {wanted_fmt} -- half the "
            f"platforms cannot decode anything else"
        )

    if want_audio and not check.has_audio:
        raise RenderError(
            "clip has no audio stream. A silent short is a failure that "
            "survives all the way to upload, so it is caught here."
        )


# --------------------------------------------------------------------------
# top level
# --------------------------------------------------------------------------


def _filename(stream_id: str, index: int, moment: Moment, suffix: str,
              tag: str = "") -> str:
    total = int(moment.t_start)
    return (f"{stream_id}_{index + 1:02d}_{total // 60}m{total % 60:02d}s"
            f"{tag}{suffix}")


def render_stream(
    cfg,
    conn: sqlite3.Connection,
    stream_id: str,
    options: Options,
    log=print,
) -> Result:
    """Render every approved moment. Returns what it did."""
    stream = conn.execute(
        "SELECT * FROM streams WHERE id = ?", (stream_id,)
    ).fetchone()
    if stream is None:
        raise RenderError(f"no stream {stream_id!r}")
    if not stream["duration_s"] or not stream["resolution"]:
        raise RenderError(f"{stream_id} has not been probed")

    if stream["is_vfr"] and not options.allow_vfr:
        raise RenderError(
            f"{stream_id} was probed as variable frame rate. A constant "
            f"`-t` over variable frames drifts, so the clip would not be the "
            f"length it claims. Render from the proxy, which is forced to "
            f"CFR:\n    clipforge render {stream_id} --set render.source=proxy\n"
            f"or pass --allow-vfr if you know this recording is really constant."
        )

    moments = selection.approved_moments(
        conn, stream_id, min_rating=options.min_rating)
    if not moments:
        raise RenderError(
            f"{stream_id} has nothing rated {options.min_rating} or above yet. "
            f"Review it first: `clipforge review {stream_id}`."
        )
    if options.limit is not None:
        moments = moments[:options.limit]

    stream_paths = paths.StreamPaths(cfg.data_root, stream_id).ensure()
    source = source_for(cfg, stream, stream_paths)
    if not source.is_file():
        raise RenderError(
            f"source not found: {source}. If the footage moved, "
            f"`clipforge db relink --from <old> --to <new>`."
        )

    width, height = (int(v) for v in str(stream["resolution"]).split("x"))
    template = crop.resolve(
        crop.get_template(cfg, options.template), width, height)
    check = crop.validate(
        template, width, height,
        upscale_warn_factor=float(cfg.get("render.crop.upscale_warn_factor", 2.0)),
    )

    preset = None if options.stills else presets.load(cfg, options.preset)
    result = Result(template=template, check=check, preset=preset)
    for line in crop.describe(template, check):
        log(f"  {line}")
    if preset is not None:
        log(f"  preset    {preset.name}"
            + (f"  ({preset.note})" if preset.note else ""))

    audio_index = None if options.stills else audio_stream_index(cfg, stream)
    if audio_index is None and not options.stills:
        log("  WARNING  no audio track map -- rendering silent")

    out_dir = options.out_dir or stream_paths.exports_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".png" if options.stills else         f".{presets.setting(cfg, preset, 'container', 'mp4')}"

    ffmpeg_bin = ffmpeg.require("ffmpeg", cfg.ffmpeg_override)
    ffprobe_bin = ffmpeg.require("ffprobe", cfg.ffprobe_override)
    duration_s = float(stream["duration_s"])

    for index, moment in enumerate(moments):
        plan = clip_window(cfg, moment, duration_s)

        # §8.2's cuts FIRST, because everything after them lives on the clock
        # they create: the caption times, the mute ranges, and the loudness
        # measurement all have to describe the clip that actually gets written.
        cut_plan, mute = _plan_edits(cfg, conn, stream_id, stream, plan,
                                     options, log)
        if cut_plan.cuts:
            plan = timeline.build(plan.clip_start, plan.clip_end, cut_plan.cuts)

        destination = out_dir / _filename(
            stream_id, index, moment, suffix, _dual_tag(cfg, options))
        clip = Clip(index=index, moment=moment, plan=plan, destination=destination)

        script, load = caption_script(
            cfg, conn, stream_id, stream, plan, template)
        # A still exists to check crop geometry. It is not captioned: seeking
        # to a representative frame means the output clock no longer matches
        # the caption times, and aligning the two would be work in service of
        # a picture whose whole job is to show where the rectangles are.
        if options.stills:
            script = None
        clip.caption_words = len(load.words)
        clip.interpolated = load.interpolated
        if script is not None:
            clip.caption_lines = script.lines

        graph = crop.filter_complex(
            template,
            default_fit=str(cfg.get("render.crop.fit", crop.FILL)),
            pad_color=str(cfg.get("render.crop.pad_color", "#000000")),
            captions=CAPTIONS_NAME if script is not None else None,
            label=VIDEO_LABEL,
            trim=edits.trim_chain(plan, audio=False, entry="[0:v]", label="vcut"),
        )
        result.graph = graph

        long = presets.too_long(preset, plan.duration)
        if long:
            clip.warnings.append(long)
            log(f"  WARNING  {long}")

        if options.dry_run:
            log(f"  [{index + 1}] {destination.name}  "
                f"{plan.clip_start:.2f}-{plan.clip_end:.2f}s "
                f"({plan.duration:.2f}s, {clip.caption_words} word(s))")
            result.clips.append(clip)
            continue

        audio_chain = _audio_chain(
            ffmpeg_bin, source, cfg=cfg, plan=plan, audio_index=audio_index,
            mute=mute, log=log) if not options.stills else None

        _encode(ffmpeg_bin, source, destination, cfg=cfg, plan=plan, graph=graph,
                audio_index=audio_index, script=script, still=options.stills,
                audio_chain=audio_chain, preset=preset, log=log)

        if not options.stills:
            found = inspect(destination, ffprobe_bin)
            assert_sound(found, template=template, plan=plan,
                         want_audio=audio_index is not None, cfg=cfg)
            clip.duration_s = found.duration_s
            if audio_index is not None:
                clip.loudness_lufs = loudness.integrated(ffmpeg_bin, destination)
                missed = loudness.off_target(cfg, clip.loudness_lufs)
                if missed:
                    clip.warnings.append(missed)
                    log(f"  WARNING  {missed}")
        clip.size_bytes = destination.stat().st_size
        result.clips.append(clip)
        log(f"  [{index + 1}/{len(moments)}] {destination.name}  "
            f"{clip.size_bytes / 1_000_000:.1f} MB, "
            f"{clip.caption_lines} caption line(s)")

    return result


def _dual_tag(cfg, options: Options) -> str:
    """§8.6's "dual export": the pair has to be tellable apart on disk.

    Only when --dual is in play. A single render keeps the name it always had,
    so turning muting on does not rename every file the operator already has.
    """
    if not options.dual:
        return ""
    muted = cfg.get("render.mute.enabled", False)         if options.mute_override is None else options.mute_override
    return "_clean" if muted else "_unmuted"


def _plan_edits(cfg, conn, stream_id, stream, plan, options, log):
    """§8.2's cuts and §8.6's mute ranges, both off unless asked for."""
    track_map = stream["audio_track_map"]
    spans = None
    if bool(cfg.get("render.filler.enabled", False)) or             bool(cfg.get("render.mute.enabled", False)):
        spans = edits.spans_in(conn, stream_id, plan, track_map)

    cut_plan = edits.filler_cuts(cfg, conn, stream_id, plan, track_map, spans)

    # The dual export's second file is the unmuted one (§8.6), so muting is
    # forced off for it rather than the config being rewritten mid-run.
    muted = cfg.get("render.mute.enabled", False)         if options.mute_override is None else options.mute_override
    mute: list[tuple[float, float]] = []
    if muted:
        after_cuts = timeline.build(plan.clip_start, plan.clip_end, cut_plan.cuts)             if cut_plan.cuts else plan
        mute = edits.mute_ranges(cfg, conn, stream_id, after_cuts, track_map, spans)

    for line in edits.describe(cut_plan, mute):
        log(f"  {line}")
    return cut_plan, mute


def _audio_chain(ffmpeg_bin, source: Path, *, cfg, plan, audio_index, log,
                 mute=None):
    """The audio filter chain, including §8.3's loudness normalisation.

    Pass 1 runs here, over the same pre-loudnorm chain pass 2 will use, so the
    measurement describes the audio that actually gets encoded.
    """
    if audio_index is None:
        return None

    before = audio_filters(cfg, plan, mute)
    if not bool(cfg.get("render.loudness.enabled", True)):
        return before or None

    # The silence gate first, and from a STANDALONE ebur128 -- loudnorm's own
    # `input_i` reports the fixture's silent opening at -17.75 LUFS, which no
    # floor test could tell from quiet speech. See render/loudness.py.
    source_lufs = loudness.source_loudness(
        ffmpeg_bin, source, clip_start=plan.clip_start,
        duration=plan.source_duration, audio_index=audio_index, before=before,
    )
    allowed, why = loudness.should_normalise(cfg, source_lufs)
    if not allowed:
        log(f"  loudness  not normalised -- {why}")
        return before or None

    measured = None
    if bool(cfg.get("render.loudness.two_pass", True)):
        measured = loudness.measure(
            ffmpeg_bin, source, cfg=cfg, clip_start=plan.clip_start,
            duration=plan.source_duration, audio_index=audio_index,
            before=before,
        )
        if measured is None:
            # An ffmpeg whose JSON could not be read. §8.3's single pass still
            # works and is close enough to be worth shipping over refusing.
            log("  WARNING  could not read loudnorm's measurement; "
                "falling back to §8.3's single pass")

    if source_lufs is not None:
        log(f"  loudness  source {source_lufs:.1f} LUFS -> target "
            f"{cfg.get('render.loudness.target_lufs')}")
    return loudness.chain(cfg, measured, before, source_lufs=source_lufs)


def _encode(ffmpeg_bin, source: Path, destination: Path, *, cfg, plan, graph,
            audio_index, script, still, log, audio_chain=None, preset=None) -> None:
    """One file, written to a temp path and atomically renamed (A10).

    The ASS file lives in a scratch directory and ffmpeg is run from it, so the
    filter graph names it bare. MEASURED in commit 23: an absolute Windows path
    splits at its drive colon inside a filter description, and no escaping of
    it survives an apostrophe in a parent directory.
    """
    from clipforge.pipeline import progress

    with TemporaryDirectory(prefix="clipforge-render-") as work:
        work_dir = Path(work)
        if script is not None:
            (work_dir / CAPTIONS_NAME).write_text(script.render(), encoding="utf-8")

        with atomic_output(destination) as temp:
            argv = build_command(
                ffmpeg_bin, source, temp, cfg=cfg, plan=plan, graph=graph,
                audio_index=audio_index, still=still,
                audio_chain=audio_chain, preset=preset,
            )
            reporter = None if still else progress.reporter(
                _LogOnly(log), plan.duration)
            ffmpeg.run(argv, cwd=str(work_dir), on_stderr_line=reporter)


class _LogOnly:
    """`progress.reporter` wants a stage context; a command only has a log."""

    def __init__(self, log):
        self.log = log

    def heartbeat(self) -> None:
        return None


# --------------------------------------------------------------------------
# recording
# --------------------------------------------------------------------------


def record(conn: sqlite3.Connection, cfg, stream_id: str, result: Result,
           stream_paths: paths.StreamPaths, options: Options) -> None:
    """§3.2's `exports` / `export_items`, one row per clip.

    `kind='clip'` -- the existing CHECK constraint already admits it, so there
    is no migration. The render params hash goes to `tool_metrics` rather than
    a column: `render:` sits outside `VERSIONED_SUBTREES` by design, so nothing
    else would ever detect that a file on disk is stale against the settings
    that produced it (C6 -- log it now, it is free).
    """
    if not result.clips:
        return

    payload = {
        "template": result.template.name if result.template else None,
        "output": list(result.template.output) if result.template else None,
        "crop": cfg.get("render.crop"),
        "captions": cfg.get("render.captions"),
        "video": cfg.get("render.video"),
        "handles_s": cfg.get("render.handles_s"),
    }
    import hashlib

    from clipforge.config import canonical_json
    params_hash = hashlib.sha256(
        canonical_json(payload).encode()).hexdigest()[:16]

    with db.transaction(conn):
        for clip in result.clips:
            try:
                relative = stream_paths.rel_export(clip.destination.name) \
                    if clip.destination.is_relative_to(stream_paths.exports_dir) \
                    else str(clip.destination)
            except ValueError:
                relative = str(clip.destination)

            cursor = conn.execute(
                "INSERT INTO exports (candidate_id, stream_id, kind, path, "
                "config_version, item_count) VALUES (?, ?, 'clip', ?, ?, 1)",
                (clip.candidate_id, stream_id, relative, cfg.version),
            )
            conn.execute(
                "INSERT INTO export_items (export_id, seq, candidate_id, "
                "t_start, t_end, role, label) VALUES (?, 0, ?, ?, ?, 'clip', ?)",
                (cursor.lastrowid, clip.candidate_id, clip.plan.clip_start,
                 clip.plan.clip_end, clip.destination.name),
            )

        conn.execute(
            "INSERT INTO tool_metrics (stream_id, metric, value, meta) "
            "VALUES (?, 'render_clips', ?, ?)",
            (stream_id, float(len(result.clips)), json.dumps({
                "params_hash": params_hash,
                "template": payload["template"],
                "min_rating": options.min_rating,
                "caption_lines": sum(c.caption_lines for c in result.clips),
                "interpolated_words": sum(c.interpolated for c in result.clips),
                "uncovered_px": result.check.uncovered if result.check else 0,
            })),
        )


def frame_rate(stream) -> Fraction:
    """The source's rate, for anything that needs it downstream."""
    return Fraction(int(stream["fps_num"] or 30), int(stream["fps_den"] or 1))
