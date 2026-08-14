"""`clipforge render` — approved moments as postable files (§8).

The counterpart to `clipforge export`: that one writes a timeline for an editor
to conform, this one writes the finished thing. §10.5 is explicit that YouTube
assembly must never extract clip files — but it is equally explicit about the
exception:

> Only extract standalone files for short-form, where a discrete upload
> artifact is genuinely required.

Two flags exist because the crop coordinates are placeholders until a real OBS
layout is measured, and "edit numbers, wait for an encode, look" is the slow
way to fix that:

* `--dry-run` prints the resolved geometry and the filter graph, encoding
  nothing.
* `--stills` writes one PNG per moment through the identical graph. A still
  costs about a second; the encode costs a minute.
"""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

from clipforge import config, db, paths
from clipforge.render import clips


def add_arguments(parser) -> None:
    parser.add_argument("stream_id")
    config.add_config_arguments(parser)
    parser.add_argument(
        "--min-rating", type=int, default=2, choices=(0, 1, 2),
        help="lowest rating to render (default 2, 'clip it'; 1 adds 'maybe')",
    )
    parser.add_argument(
        "--template",
        help="crop template name (default: render.crop.template)",
    )
    parser.add_argument(
        "--preset",
        help="encode preset (default: render.presets.default). They are nearly "
             "identical today -- see the config",
    )
    parser.add_argument("--out", help="output directory (default: the stream's exports/)")
    parser.add_argument(
        "--limit", type=int,
        help="render only the first N approved moments",
    )
    parser.add_argument(
        "--stills", action="store_true",
        help="one PNG per moment instead of a clip, to check the crop fast",
    )
    parser.add_argument(
        "--dual", action="store_true",
        help="§8.6's dual export: a muted and an unmuted copy of each clip. "
             "Needs render.mute.enabled",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the geometry and filter graph, encode nothing",
    )
    parser.add_argument(
        "--allow-vfr", action="store_true",
        help="render a variable-rate master anyway (see the refusal for why not)",
    )


def main(args) -> int:
    cfg = config.from_args(args)
    if not cfg.db_path.exists():
        print(f"no database at {cfg.db_path}. Run `clipforge db init`.")
        return 1

    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    try:
        return _render(cfg, conn, args)
    except clips.RenderError as exc:
        print(f"error: {exc}")
        return 1
    finally:
        conn.close()


def _render(cfg, conn: sqlite3.Connection, args) -> int:
    options = clips.Options(
        min_rating=args.min_rating,
        template=args.template,
        out_dir=Path(args.out).expanduser().resolve() if args.out else None,
        limit=args.limit,
        stills=args.stills,
        dry_run=args.dry_run,
        allow_vfr=args.allow_vfr,
        preset=args.preset,
        dual=args.dual,
    )

    print(f"rendering {args.stream_id}")
    result = clips.render_stream(cfg, conn, args.stream_id, options, log=print)

    if args.dual and not args.dry_run:
        if not cfg.get("render.mute.enabled", False):
            print("  --dual with muting off would write the same file twice; "
                  "set render.mute.enabled=true")
        else:
            print("\nrendering the unmuted copy (§8.6)")
            unmuted = clips.render_stream(
                cfg, conn, args.stream_id,
                replace(options, mute_override=False), log=print)
            result.clips.extend(unmuted.clips)

    if args.dry_run:
        print("\nfilter graph:")
        for step in result.graph.split(";"):
            print(f"  {step}")
        print(f"\n{len(result.clips)} clip(s) would be written. Nothing encoded.")
        return 0

    stream_paths = paths.StreamPaths(cfg.data_root, args.stream_id)
    if not args.stills:
        clips.record(conn, cfg, args.stream_id, result, stream_paths, options)

    _report(result, args)
    return 0


def _report(result: clips.Result, args) -> None:
    total = sum(clip.size_bytes for clip in result.clips)
    kind = "still" if args.stills else "clip"
    print(f"\n{len(result.clips)} {kind}(s), {total / 1_000_000:.1f} MB total")

    if result.clips:
        print(f"  in        {result.clips[0].destination.parent}")
    if result.template:
        print(f"  template  {result.template.name} "
              f"-> {result.template.output[0]}x{result.template.output[1]}")
    if result.preset:
        print(f"  preset    {result.preset.name}")

    measured = [c.loudness_lufs for c in result.clips if c.loudness_lufs is not None]
    if measured:
        print(f"  loudness  {', '.join(f'{v:.1f}' for v in measured)} LUFS")

    for clip in result.clips:
        for warning in clip.warnings:
            print(f"  WARNING   {clip.destination.name}: {warning}")

    interpolated = sum(clip.interpolated for clip in result.clips)
    if interpolated:
        # A caption track that is mostly guesswork should say so rather than
        # being inferred from the captions looking slightly off.
        print(f"  {interpolated} word(s) had no forced alignment and were placed "
              f"by interpolation")

    if args.stills:
        print("\nStills are un-captioned by design -- they exist to check the crop.")
        print("Adjust the coordinates in crop_templates.yaml and run this again.")
