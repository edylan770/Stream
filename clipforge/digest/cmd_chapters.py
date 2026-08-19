"""`clipforge chapters <stream_id>` — §9.3's boundaries, and where they came from.

Chapters are not a table: §3.2 declares none and §9.2 nests them inside the
digest JSON, which is where the `digest` stage will store them. This command
exists so they can be looked at before that stage is built, and afterwards when
a digest looks wrong and the question is whether the segmentation or the prompt
caused it.

**The report leads with which of §9.3's four inputs contributed and which did
not.** On a stream processed with shipped defaults three of the four produce
nothing — transcription is off, no OBS log is attached, and timed game data has
no producer anywhere in the system — so "chapter segmentation" without that
caveat would read as four sources agreeing when it is one splitting on silence.
"""

from __future__ import annotations

from clipforge import config, db
from clipforge.digest import chapters


def add_arguments(parser) -> None:
    parser.add_argument("stream_id")
    parser.add_argument(
        "--all-boundaries", action="store_true",
        help="also show candidates that were merged away or left unused",
    )
    config.add_config_arguments(parser)


def main(args) -> int:
    cfg = config.from_args(args)
    if not cfg.db_path.exists():
        print(f"no database at {cfg.db_path}")
        return 1

    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    try:
        try:
            result = chapters.segment(conn, args.stream_id, cfg)
        except chapters.ChapterError as exc:
            print(exc)
            return 1
    finally:
        conn.close()

    _report(args.stream_id, result, args)
    return 0


def _report(stream_id: str, result: chapters.Result, args) -> None:
    print(f"{stream_id}: {len(result.chapters)} chapter(s)")
    print()

    print("§9.3's four inputs:")
    for name in chapters.INPUTS:
        if name in result.inputs:
            print(f"  - {name:<10} nothing: {result.inputs[name]}")
        else:
            count = sum(1 for b in result.boundaries if b.source == name)
            print(f"  + {name:<10} proposed {count} boundary/boundaries")
    print()

    for chapter in result.chapters:
        opened = chapter.opened_by
        why = "the stream starts" if opened is None else (
            f"{opened.source} — {opened.detail}"
            + (f" (also seen by {', '.join(opened.corroborated_by)})"
               if opened.corroborated_by else ""))
        print(f"  {chapter.index:>2}. {_clock(chapter.t_start)} – {_clock(chapter.t_end)}"
              f"  {chapter.duration_s / 60:5.1f} min   {why}")

    if args.all_boundaries:
        print()
        print("every candidate boundary:")
        for boundary in result.boundaries:
            mark = "used" if boundary.accepted else "  --"
            print(f"  {mark}  {_clock(boundary.t)}  {boundary.source:<10} "
                  f"strength {boundary.strength:.2f}  {boundary.detail}")

    if result.unmet_targets:
        print()
        print("§9.3's 10–30 min target is guidance, and here it was not met:")
        for note in result.unmet_targets:
            print(f"  - {note}")

    if result.contributing == ["silence"]:
        print()
        print("Only the silence input contributed, which is what a stream processed")
        print("with shipped defaults looks like. Turn on extract.whisperx.enabled for")
        print("the embedding shift; attach an OBS log for scene changes.")


def _clock(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:d}:{(total % 3600) // 60:02d}:{total % 60:02d}"
