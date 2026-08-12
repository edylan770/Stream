"""`clipforge export` — approved moments as an FCPXML timeline (§10.5).

Deliberately a command rather than a §5.1 pipeline stage. A stage runs
unattended whenever its inputs change; an export is a decision taken after
reviewing, and re-running it silently because a re-score bumped a sequence
number would be wrong.

§10.5's reasoning, which is why this produces a text file and not video:

> no re-encoding, no keyframe-boundary artifacts, frame-accurate cuts, and a
> 40-clip compilation "renders" in milliseconds because it is a text file.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path

from clipforge import config, db, paths
from clipforge.pipeline.atomic import atomic_output
from clipforge.render import fcpxml, selection
from clipforge.render.timebase import DOWN, TimeBase


class ExportError(RuntimeError):
    pass


def add_arguments(parser) -> None:
    parser.add_argument("stream_id")
    config.add_config_arguments(parser)
    parser.add_argument(
        "--min-rating", type=int, default=2, choices=(0, 1, 2),
        help="lowest rating to include (default 2, 'clip it'; 1 adds 'maybe')",
    )
    parser.add_argument("--out", help="output path (default: the stream's exports/)")
    parser.add_argument(
        "--allow-vfr", action="store_true",
        help="export a variable-rate master anyway (see the refusal for why not)",
    )


def resolve_source(cfg, stream, stream_paths: paths.StreamPaths) -> Path:
    """Master or proxy, per `export.source`."""
    which = str(cfg.get("export.source", "master")).lower()
    if which == "proxy":
        if not stream["proxy_path"]:
            raise ExportError(f"{stream['id']} has no proxy. Run `clipforge run {stream['id']}`.")
        return stream_paths.absolute(stream["proxy_path"])
    if which != "master":
        raise ExportError(f"export.source must be 'master' or 'proxy', got {which!r}")
    return Path(stream["master_path"])


def check_rate_is_constant(stream, allow_vfr: bool) -> None:
    """A VFR master has no frameDuration, so every time in the file is a guess.

    §5.2 claims the proxy shares the master's timebase; commit 7 established
    that is false for a variable-rate source, which is why `is_vfr` is probed
    and why the proxy forces CFR. Writing a constant grid over variable frames
    produces a file Resolve accepts and quietly misaligns, which is the worst
    of the available outcomes.
    """
    if not stream["is_vfr"] or allow_vfr:
        return
    raise ExportError(
        f"{stream['id']} was probed as variable frame rate. FCPXML times are integer "
        f"multiples of one frameDuration, which a VFR source does not have — the file "
        f"would import and be subtly misaligned. Export from the proxy instead, which "
        f"is forced to CFR:\n"
        f"    clipforge export {stream['id']} --set export.source=proxy\n"
        f"or pass --allow-vfr if you know this recording is really constant."
    )


def main(args) -> int:
    cfg = config.from_args(args)
    if not cfg.db_path.exists():
        print(f"no database at {cfg.db_path}. Run `clipforge db init`.")
        return 1

    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    try:
        return _export(cfg, conn, args)
    except ExportError as exc:
        print(f"error: {exc}")
        return 1
    finally:
        conn.close()


def _export(cfg, conn: sqlite3.Connection, args) -> int:
    stream = conn.execute(
        "SELECT * FROM streams WHERE id = ?", (args.stream_id,)
    ).fetchone()
    if stream is None:
        raise ExportError(f"no stream {args.stream_id!r}")

    check_rate_is_constant(stream, args.allow_vfr)
    timebase = TimeBase.from_stream(stream)

    moments = selection.approved_moments(conn, args.stream_id, min_rating=args.min_rating)
    if not moments:
        raise ExportError(
            f"{args.stream_id} has nothing rated {args.min_rating} or above yet. "
            f"Review it first: `clipforge review {args.stream_id}`."
        )

    stream_paths = paths.StreamPaths(cfg.data_root, args.stream_id).ensure()
    source = resolve_source(cfg, stream, stream_paths)
    if not source.is_file():
        raise ExportError(
            f"source not found: {source}. If the footage moved, "
            f"`clipforge db relink --from <old> --to <new>`."
        )

    if not stream["duration_s"] or not stream["resolution"]:
        raise ExportError(f"{args.stream_id} has not been probed")

    # Floor: the last frame index that certainly exists in the file.
    source_frames = timebase.frame_at(float(stream["duration_s"]), DOWN)
    width, height = (int(v) for v in str(stream["resolution"]).split("x"))

    def label(index, moment):
        return f"{index + 1:02d} · {_clock(moment.t_start)}"

    clips = fcpxml.plan_clips(moments, timebase, source_frames, label)
    if not clips:
        raise ExportError("every approved moment fell outside the source's duration")

    tree = fcpxml.build(
        source=source,
        timebase=timebase,
        source_frames=source_frames,
        clips=clips,
        resolution=(width, height),
        project_name=stream["title"] or args.stream_id,
        audio_sources=_audio_sources(stream),
        version=str(cfg.get("export.fcpxml_version", "1.10")),
    )

    destination = (
        Path(args.out).expanduser().resolve() if args.out
        else stream_paths.exports_dir / _filename(args.stream_id, args.min_rating)
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = fcpxml.to_bytes(tree)
    with atomic_output(destination) as temp:
        temp.write_bytes(payload)

    _record(conn, cfg, args, stream_paths, destination, moments, clips, timebase)
    _report(args.stream_id, destination, moments, clips, timebase, source)
    return 0


def _audio_sources(stream) -> int:
    """How many audio streams the asset declares.

    From the probed §4.2 track map. Getting this low means Resolve links fewer
    tracks than exist, which is visible and fixable in the timeline; getting it
    high makes Resolve report missing media, so the count is clamped to what was
    actually found.
    """
    raw = stream["audio_track_map"]
    if not raw:
        return 1
    roles = json.loads(raw).get("roles", {})
    return max(len(roles), 1)


def _filename(stream_id: str, min_rating: int) -> str:
    tag = {2: "approved", 1: "approved+maybe", 0: "all"}[min_rating]
    return f"{stream_id}_{tag}_{date.today().isoformat()}.fcpxml"


def _clock(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 60}m{total % 60:02d}s"


def _record(conn, cfg, args, stream_paths, destination, moments, clips, timebase) -> None:
    """§3.2's exports/export_items. `t_start`/`t_end` are the FRAME-SNAPPED
    values actually written, not the candidate's, so the row describes the file
    rather than the intention behind it."""
    try:
        relative = stream_paths.rel_export(destination.name) if destination.is_relative_to(
            stream_paths.exports_dir) else str(destination)
    except ValueError:
        relative = str(destination)

    with db.transaction(conn):
        cursor = conn.execute(
            "INSERT INTO exports (stream_id, kind, path, config_version, item_count) "
            "VALUES (?, 'fcpxml', ?, ?, ?)",
            (args.stream_id, relative, cfg.version, len(clips)),
        )
        export_id = cursor.lastrowid

        for seq, (clip, moment) in enumerate(zip(clips, moments, strict=False)):
            conn.execute(
                "INSERT INTO export_items "
                "(export_id, seq, candidate_id, t_start, t_end, role, label) "
                "VALUES (?, ?, ?, ?, ?, 'clip', ?)",
                (export_id, seq,
                 moment.candidate_ids[0] if moment.candidate_ids else None,
                 float(timebase.seconds(clip.source_start)),
                 float(timebase.seconds(clip.source_end)),
                 clip.name),
            )

        # The schema's own description: "set when export rounds the boundaries
        # onto the frame grid, so the UI and the timeline never disagree about
        # where a cut is."
        contributing = [cid for moment in moments for cid in moment.candidate_ids]
        conn.executemany(
            "UPDATE candidates SET frame_snapped = 1 WHERE id = ?",
            [(cid,) for cid in contributing],
        )

        conn.execute(
            "INSERT INTO tool_metrics (stream_id, metric, value, meta) VALUES (?, ?, ?, ?)",
            (args.stream_id, "export_items", float(len(clips)),
             json.dumps({"kind": "fcpxml", "min_rating": args.min_rating,
                         "rescued": sum(1 for m in moments if m.rescued)})),
        )


def _report(stream_id, destination, moments, clips, timebase, source) -> None:
    total = sum(clip.frames for clip in clips)
    rescued = [m for m in moments if m.rescued]

    print(f"exported  {destination}")
    print(f"  source    {source}")
    print(f"  rate      {timebase.describe()}  frameDuration {timebase.frame_duration_value}")
    print(f"  clips     {len(clips)}")
    print(f"  timeline  {total} frames, {float(timebase.seconds(total)):.2f}s")

    if rescued:
        # The whole reason this reads across generations.
        print(f"\n  {len(rescued)} moment(s) came from a superseded scoring generation —")
        print("  a re-score had moved on from them, and reading is_current would have")
        print("  dropped them from this export:")
        for moment in rescued:
            print(f"    {_clock(moment.t_start)}–{_clock(moment.t_end)}  "
                  f"generation {moment.generation}, rated {moment.rated_at}")

    print(f"\nimport into Resolve: File > Import > Timeline, then pick {destination.name}")
