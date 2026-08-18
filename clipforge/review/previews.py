"""`previews` — §5.1 stage 15, §7.2's precomputed review assets.

> "**Every expensive operation happens during unattended batch processing.
> Nothing expensive happens while the operator is waiting.**"

That sentence is the whole stage. C4 makes the review screen the critical path
and §7.1 gives it four seconds per candidate; an asset that has to be produced
while the operator holds `j` has already failed.

It lives under `review/` rather than `extract/` because nothing outside the
review screen reads these files. They are not signals and they are not
deliverables, and deleting the whole directory costs one re-run of a stage that
depends on nothing expensive.

THREE THINGS §7.2 GETS WRONG, AND ONE IT GETS RIGHT

**Assets are named by their WINDOW, never by `candidate_id`.** §7.2's own
command writes `previews/{candidate_id}.webm`, and candidates here are
append-only generations (`0001_init.sql`) — a re-score with different weights
mints generation N+1 with entirely new ids. Under §7.2's naming, §6.1's promise
that re-scoring is "free and infinitely repeatable" would quietly become "free,
plus a full re-encode of every preview in the library". The clip depends only on
`t_peak` and the strip only on the window, so those are what the names are made
of. Several candidate rows across several generations then share one file, and a
re-score that changed only weights regenerates NOTHING.

**There is no waveform PNG, deliberately.** §7.2 asks for one ("mic + party RMS
over the window, ~10 KB"), but `review/queries.py` already ships a downsampled
envelope in the candidate payload and `review.js` draws it as inline SVG —
vector, theme-aware, no files, no stage cost, and already on screen. A PNG would
be a second thing to keep in sync, 120 more files per stream, and orphaned by
every re-score. What was genuinely missing is the SECOND TRACK, and that is
fixed in the payload where the data already lives.

**The 2 s clip SHIFTS at the ends of a recording, it does not shorten.**
`-ss {t_peak-1}` is negative for a peak inside the first second, and there may be
under two seconds of footage after a peak near the end. C2 says expand rather
than contract, so the window slides to stay two seconds long wherever the
footage allows and shortens only when the recording itself is shorter.

**A1 is what §7.2 gets right**, emphatically ("Hardcode the fast form; this must
never be gotten wrong") — and `ffmpeg.run` refuses a command that gets it wrong
anyway.

THE COST, MEASURED, BECAUSE §7.2'S OWN SETTINGS DO NOT FIT THE BUDGET

Encoding five 2 s clips from real 720p footage (`Testvid.mp4`), SSIM against a
lossless reference of the same scaled window, extrapolated to §7.1's 120
candidates:

    variant                     SSIM      bytes    s/clip   120 candidates
    §7.2 verbatim (vp9)        0.9638    249 KB     6.78      13.6 min
    vp9 -cpu-used 5            0.9572    248 KB     3.01       6.0 min
    vp9 -cpu-used 8 realtime   0.9564    320 KB     0.71       1.4 min
    h264 crf 28                0.9193    132 KB     0.56       1.1 min
    h264 crf 23                0.9416    250 KB     0.60       1.2 min

§1.3 budgets 20–40 minutes for ALL unattended processing and §5.4's pitch stage
already spends ~16 of it. **§7.2's literal command would spend another 13.6 on
an asset whose entire purpose is saving the operator seek latency**, so it is
not the default. `-cpu-used 8 -deadline realtime` is: 9.5× faster for 0.007 of
SSIM, at 28% more bytes on a tier §13.1 measures in megabytes.

**VP9 itself is kept, and that is the measured half.** At a matched ~250 KB,
h264 scores 0.9416 against vp9's 0.9564–0.9638 — so §7.2 is right about the
codec and wrong only about the speed preset. Worth knowing that the SYNTHETIC
fixture says the opposite: on `fixture_long` the same command runs at 2.04 s per
clip because that source is 640×360, which understates the real cost by 3.4×.
The numbers above are from real footage for exactly that reason.

§7.2's size estimates are close but low: it says ~150 KB per clip and ~30 KB per
strip, and measured here they are ~250–320 KB and ~16 KB. Its bottom line
(~25 MB per stream for 120 candidates) survives — this comes out near 40 MB,
which §13.1 still calls negligible.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from clipforge import db, ffmpeg
from clipforge.pipeline.atomic import atomic_output
from clipforge.pipeline.context import StageContext


class PreviewError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# names: content-addressed by window, so a re-score reuses what it can
# --------------------------------------------------------------------------


def _ms(seconds: float) -> int:
    """Times become names, so they are quantised once and only here.

    Milliseconds because §7.3 nudges in 0.5 s steps and `score.score_grid_hz` is
    10 Hz — nothing downstream distinguishes finer than this, and rounding stops
    float noise in `t_peak` minting a second name for one moment.
    """
    return int(round(float(seconds) * 1000.0))


def clip_name(t_peak: float, container: str = "webm") -> str:
    """The 2 s preview is centred on the peak and ignores the window entirely.

    So a boundary nudge, or a re-score that moved the boundaries but not the
    peak, reuses the file that already exists.
    """
    return f"p{_ms(t_peak)}.{container}"


def strip_name(t_start: float, t_end: float) -> str:
    return f"t{_ms(t_start)}_{_ms(t_end)}.jpg"


def clip_window(t_peak: float, duration_s: float,
                stream_duration_s: float) -> tuple[float, float]:
    """Where the 2 s clip actually starts, and how long it runs.

    C2 in miniature: at the head or tail of a recording the window SLIDES to
    stay `duration_s` long, and shortens only when the recording is shorter than
    the clip. A peak at t=0.4 s still gets two seconds of context, where §7.2's
    bare `t_peak - 1` would ask ffmpeg to seek to −0.6.
    """
    span = max(0.0, float(stream_duration_s))
    want = max(0.0, float(duration_s))
    if span <= want:
        return 0.0, span
    start = float(t_peak) - want / 2.0
    start = min(max(start, 0.0), span - want)
    return start, want


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def clip_command(ffmpeg_bin: str, proxy: Path, destination: Path, *,
                 start_s: float, duration_s: float, cfg) -> list[str]:
    """§7.2's command, with the speed preset the measurement chose.

    `-an` is §7.2's own: the clip loops silently on focus (§7.3) and `space`
    plays the real window from the proxy, so muxing audio would be bytes nothing
    plays. `-cpu-used`/`-deadline` are libvpx options and are emitted only for
    it, so pointing `previews.clip.codec` at something else cannot produce a
    command that will not run.
    """
    codec = str(cfg.get("previews.clip.codec"))
    argv = [
        ffmpeg_bin, "-y", "-hide_banner", "-v", "error",
        # A1, and §7.2 is emphatic about it: input seeking uses the container
        # index and takes milliseconds regardless of file size.
        "-ss", f"{start_s:.3f}",
        "-i", str(proxy),
        "-t", f"{duration_s:.3f}",
        "-vf", f"scale={int(cfg.get('previews.clip.width'))}:-2",
        "-c:v", codec,
        "-crf", str(int(cfg.get("previews.clip.crf"))),
        # `-b:v 0` is what makes -crf a quality target rather than a cap.
        "-b:v", "0",
        "-an",
    ]
    if codec.startswith("libvpx"):
        argv += [
            "-row-mt", "1",
            "-cpu-used", str(int(cfg.get("previews.clip.cpu_used"))),
            "-deadline", str(cfg.get("previews.clip.deadline")),
        ]
    argv += [str(a) for a in (cfg.get("previews.clip.extra_args") or [])]
    argv.append(str(destination))
    return argv


def strip_plan(window_s: float, fps: float, frames: int) -> tuple[int, int]:
    """(select step, tile columns) for the thumb strip.

    §7.2 wants "5 frames evenly spaced across window". Selecting every Nth
    decoded frame in ONE pass is what makes this cheap: MEASURED at 0.93 s per
    strip against 3.30 s for five separate `-ss` seeks, because five seeks are
    five process launches. A window holding fewer than `frames` frames — which
    §7.3's nudges can produce — yields a narrower strip rather than a tile
    stalled waiting for frames that will never arrive.
    """
    total = max(1, int(round(max(0.0, window_s) * max(fps, 1e-9))))
    columns = max(1, min(int(frames), total))
    return max(1, total // columns), columns


def strip_command(ffmpeg_bin: str, proxy: Path, destination: Path, *,
                  t_start: float, window_s: float, fps: float, cfg) -> list[str]:
    frames = int(cfg.get("previews.thumbstrip.frames"))
    step, columns = strip_plan(window_s, fps, frames)
    width = int(cfg.get("previews.thumbstrip.width"))
    # The comma inside mod() is escaped because ffmpeg splits filter arguments
    # on it before the expression parser ever sees the string.
    select = "select='not(mod(n\\," + str(step) + "))'"
    return [
        ffmpeg_bin, "-y", "-hide_banner", "-v", "error",
        "-ss", f"{t_start:.3f}",
        "-i", str(proxy),
        "-t", f"{window_s:.3f}",
        "-vf", f"{select},scale={width}:-2,tile={columns}x1",
        # The chain emits one tiled image; without this ffmpeg keeps muxing
        # further tiles over it.
        "-frames:v", "1",
        "-fps_mode", "vfr",
        "-q:v", str(int(cfg.get("previews.thumbstrip.quality"))),
        str(destination),
    ]


def probe_ok(path: Path, ffprobe_bin: str) -> tuple[bool, str]:
    """That the file just written actually decodes.

    The lesson `ingest/proxy.py` learned the hard way: an encoder that silently
    changes behaviour produces a file that exists, has a plausible size, and
    shows the operator a black rectangle. Cheap to check, invisible if skipped.
    """
    if not path.is_file() or path.stat().st_size == 0:
        return False, "empty"
    result = ffmpeg.run(
        [ffprobe_bin, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,width,height",
         "-of", "default=nw=1:nk=1", str(path)],
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or "ffprobe failed").strip().splitlines()
        return False, detail[-1] if detail else "ffprobe failed"
    fields = [line for line in (result.stdout or "").splitlines() if line.strip()]
    if len(fields) < 3:
        return False, "no decodable video stream"
    return True, ""


# --------------------------------------------------------------------------
# stage hooks
# --------------------------------------------------------------------------


def _cfg(ctx: StageContext, key: str, default=None):
    return ctx.cfg.get(f"previews.{key}", default)


def params(ctx: StageContext) -> dict:
    """What a change to should force a re-encode.

    The candidate set is deliberately NOT in it. Staleness against scoring is
    already the DAG's job — `previews` requires `score`, so a re-score marks
    this stale on its own. Hashing the candidate ids here as well would rebuild
    every asset whenever a rating touched a row that has nothing to do with the
    pixels.
    """
    return {"previews": ctx.cfg.get("previews")}


def available(ctx: StageContext) -> tuple[bool, str]:
    if not _cfg(ctx, "enabled", True):
        return False, "previews.enabled is false"
    return True, ""


def outputs(ctx: StageContext) -> list[Path]:
    """Nothing fixed: the names come from the windows scoring produced.

    `verify` carries the whole check instead.
    """
    return []


def verify(ctx: StageContext) -> tuple[bool, str]:
    rows = _current(ctx.conn, ctx.stream_id)
    if not rows:
        # No candidates is not a failed previews stage; it is a stream that
        # scored nothing, which `score` has already reported.
        return True, ""
    missing = 0
    for row in rows:
        for column in ("preview_path", "thumbstrip_path"):
            rel = row[column]
            if not rel or not ctx.paths.absolute(rel).is_file():
                missing += 1
    if missing:
        return False, f"{missing} preview asset(s) missing for {len(rows)} candidate(s)"
    return True, ""


def _current(conn: sqlite3.Connection, stream_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT id, t_start, t_end, t_peak, preview_path, thumbstrip_path "
        "FROM candidates WHERE stream_id = ? AND is_current = 1 ORDER BY t_peak",
        (stream_id,),
    ).fetchall()


@dataclass
class Result:
    clips_written: int = 0
    clips_reused: int = 0
    strips_written: int = 0
    strips_reused: int = 0
    candidates: int = 0
    bytes_total: int = 0
    failures: list[str] = field(default_factory=list)


def run(ctx: StageContext) -> None:
    row = ctx.stream
    proxy = ctx.paths.proxy
    if not proxy.is_file():
        raise PreviewError(
            f"no proxy at {proxy}. §7.2 generates previews from the proxy rather "
            f"than the master: it is CFR with a fixed GOP (A2), so an input seek "
            f"lands within one GOP of the peak."
        )
    if row["fps_num"] is None:
        raise PreviewError("stream has not been probed")

    fps = float(Fraction(int(row["fps_num"]), int(row["fps_den"])))
    stream_duration = float(row["duration_s"] or 0.0)
    candidates = _current(ctx.conn, ctx.stream_id)
    if not candidates:
        ctx.log("    no current candidates -- nothing to preview")
        return

    ffmpeg_bin = ffmpeg.require("ffmpeg", ctx.cfg.ffmpeg_override)
    ffprobe_bin = ffmpeg.require("ffprobe", ctx.cfg.ffprobe_override)
    container = str(_cfg(ctx, "clip.container"))
    ctx.paths.previews_dir.mkdir(parents=True, exist_ok=True)

    result = Result(candidates=len(candidates))
    updates: list[tuple[str, str, int]] = []

    for index, cand in enumerate(candidates):
        ctx.heartbeat()
        t_peak = float(cand["t_peak"])
        t_start, t_end = float(cand["t_start"]), float(cand["t_end"])

        clip_rel = ctx.paths.rel_preview(clip_name(t_peak, container))
        strip_rel = ctx.paths.rel_preview(strip_name(t_start, t_end))
        clip_abs = ctx.paths.absolute(clip_rel)
        strip_abs = ctx.paths.absolute(strip_rel)

        if clip_abs.is_file() and clip_abs.stat().st_size:
            result.clips_reused += 1
        else:
            start_s, duration_s = clip_window(
                t_peak, float(_cfg(ctx, "clip.duration_s")), stream_duration)
            if _encode(ctx, clip_abs,
                       lambda out, s=start_s, d=duration_s: clip_command(
                           ffmpeg_bin, proxy, out, start_s=s, duration_s=d,
                           cfg=ctx.cfg),
                       ffprobe_bin, result, f"clip @ {t_peak:.2f}s"):
                result.clips_written += 1

        if strip_abs.is_file() and strip_abs.stat().st_size:
            result.strips_reused += 1
        elif _encode(ctx, strip_abs,
                     lambda out, a=t_start, b=t_end: strip_command(
                         ffmpeg_bin, proxy, out, t_start=a,
                         window_s=max(b - a, 0.0), fps=fps, cfg=ctx.cfg),
                     ffprobe_bin, result, f"strip @ {t_start:.2f}-{t_end:.2f}s"):
            result.strips_written += 1

        if clip_abs.is_file() and strip_abs.is_file():
            updates.append((clip_rel, strip_rel, int(cand["id"])))

        if index and index % 25 == 0:
            ctx.log(f"    {index}/{len(candidates)} candidates")

    with db.transaction(ctx.conn):
        ctx.conn.executemany(
            # waveform_path is deliberately left NULL -- see the module
            # docstring. The column stays because §3.2 declares it and some
            # later asset may want it.
            "UPDATE candidates SET preview_path = ?, thumbstrip_path = ? WHERE id = ?",
            updates,
        )

    for path in ctx.paths.previews_dir.glob("*"):
        if path.is_file():
            result.bytes_total += path.stat().st_size

    _report(ctx, result)


def _encode(ctx: StageContext, destination: Path, build: Callable[[Path], list[str]],
            ffprobe_bin: str, result: Result, label: str) -> bool:
    """One asset, written atomically (A10) and checked before it is published.

    `build` takes the temp path rather than this rewriting the last element of a
    finished command. A10 means the encoder must be pointed at the temp file,
    and patching `argv[-1]` would silently write to the wrong place the first
    time a command grows a trailing flag.

    A failure here does NOT abort the stage. A preview is a review convenience;
    losing one costs a seek, where aborting costs every asset after it and
    leaves the stage permanently un-done. The count goes to the log and to
    `tool_metrics`, so a systematic failure is visible rather than inferred.
    """
    try:
        with atomic_output(destination) as tmp:
            ffmpeg.run(build(tmp))
            ok, why = probe_ok(tmp, ffprobe_bin)
            if not ok:
                raise PreviewError(f"{destination.name} did not decode: {why}")
        return True
    except (ffmpeg.FFmpegError, PreviewError, OSError) as exc:
        result.failures.append(f"{label}: {exc}")
        return False


def _report(ctx: StageContext, result: Result) -> None:
    ctx.log(
        f"    {result.candidates} candidate(s): "
        f"{result.clips_written} clip(s) + {result.strips_written} strip(s) written, "
        f"{result.clips_reused + result.strips_reused} reused"
    )
    ctx.log(f"    previews/ is {result.bytes_total / 1_048_576:.1f} MB")
    if result.failures:
        ctx.log(f"    WARNING  {len(result.failures)} asset(s) failed:")
        for line in result.failures[:5]:
            ctx.log(f"      {line}")

    # Both names are INVENTED -- §14's table has no previews row. Recorded
    # because they are the numbers that say whether §7.2 still fits §1.3's
    # budget once real candidate counts exist, which no fixture can answer.
    ctx.metric("preview_bytes_total", float(result.bytes_total))
    ctx.metric("preview_assets_failed", float(len(result.failures)))


def prune(conn: sqlite3.Connection, paths, *, dry_run: bool = False) -> list[Path]:
    """Delete assets no candidate row names any more.

    Content-addressed names mean a re-score usually reuses most files, but one
    that moves every window leaves the old strips behind forever. Referenced by
    ANY generation, not only the current one: §6.1 keeps old generations so the
    operator's ratings survive, and `render/selection.py` reads across them.
    """
    referenced = set()
    for row in conn.execute(
        "SELECT preview_path, thumbstrip_path, waveform_path FROM candidates "
        "WHERE stream_id = ?", (paths.stream_id,)
    ):
        referenced.update(str(value) for value in row if value)

    removed = []
    for path in sorted(paths.previews_dir.glob("*")):
        if not path.is_file():
            continue
        if paths.rel_preview(path.name) in referenced:
            continue
        removed.append(path)
        if not dry_run:
            path.unlink()
    return removed
