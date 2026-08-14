"""`clipforge hook` — §8.5's hook options, via a paste round trip.

Three modes of one command, because they are three steps of one job:

    clipforge hook <id>                       write the prompt
    clipforge hook <id> --apply reply.json    validate what came back
    clipforge hook <id> --pick <export> 2     store the one you want

Nothing here reaches the network. §2.2 puts reasoning through a frontier model
and §12.4 prices it at cents per stream, but until there is a key the operator
is the transport -- and every §12 rule is enforced on the reply regardless of
how it arrived. See `render/hooks.py`.
"""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

from clipforge import config, db, paths
from clipforge.pipeline.atomic import atomic_output
from clipforge.render import hooks


def add_arguments(parser) -> None:
    parser.add_argument("stream_id")
    config.add_config_arguments(parser)
    parser.add_argument(
        "--apply", metavar="FILE",
        help="validate a model reply (paste it into a file first)",
    )
    parser.add_argument(
        "--pick", nargs=2, metavar=("EXPORT_ID", "CHOICE"),
        help="store a hook: CHOICE is an option number from --apply, or literal text",
    )
    parser.add_argument(
        "--copy", action="store_true",
        help="also put the prompt on the clipboard",
    )
    parser.add_argument("--out", help="where to write the prompt")


def main(args) -> int:
    cfg = config.from_args(args)
    if not cfg.db_path.exists():
        print(f"no database at {cfg.db_path}. Run `clipforge db init`.")
        return 1

    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    try:
        if args.pick:
            return _pick(cfg, conn, args)
        if args.apply:
            return _apply(cfg, conn, args)
        return _prompt(cfg, conn, args)
    except hooks.HookError as exc:
        print(f"error: {exc}")
        return 1
    finally:
        conn.close()


def _clips(conn, stream_id: str):
    if conn.execute("SELECT 1 FROM streams WHERE id = ?", (stream_id,)).fetchone() is None:
        raise hooks.HookError(f"no stream {stream_id!r}")
    return hooks.load_clips(conn, stream_id)


def _prompt(cfg, conn: sqlite3.Connection, args) -> int:
    clips = _clips(conn, args.stream_id)
    source = hooks.source_for(cfg)
    usable, why = source.available(cfg)
    if not usable:
        print(f"  {source.name}: {why}")

    text = hooks.build_prompt(cfg, clips, args.stream_id)

    stream_paths = paths.StreamPaths(cfg.data_root, args.stream_id).ensure()
    destination = (
        Path(args.out).expanduser().resolve() if args.out
        else stream_paths.exports_dir /
        f"{args.stream_id}_hooks_{date.today().isoformat()}.md"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with atomic_output(destination) as temp:
        temp.write_text(text, encoding="utf-8")

    print(f"prompt written for {len(clips)} clip(s)")
    print(f"  {destination}")
    if args.copy:
        print("  copied to the clipboard" if hooks.to_clipboard(text)
              else "  could not reach the clipboard; open the file instead")

    print("\nPaste it into a frontier model, save the reply to a file, then:")
    print(f"    clipforge hook {args.stream_id} --apply <that file>")
    return 0


def _apply(cfg, conn: sqlite3.Connection, args) -> int:
    clips = _clips(conn, args.stream_id)
    path = Path(args.apply).expanduser()
    if not path.is_file():
        raise hooks.HookError(f"no reply file at {path}")

    result = hooks.validate(hooks.parse_reply(path.read_text(encoding="utf-8")), clips)
    hooks.record(conn, args.stream_id, result)

    if result.dropped:
        # §12.2 drops silently as far as the model is concerned. The operator
        # is not the model: they need to know four of five were discarded.
        print(f"dropped {len(result.dropped)} entr(ies):")
        for reason, detail in result.dropped:
            print(f"  {reason:<12} {detail}")
        print()

    if not result.proposals:
        print("nothing usable in that reply.")
        print(f"  llm_invalid_id_rate {result.invalid_id_rate:.2f} "
              f"(recorded to tool_metrics)")
        return 1

    by_id = {clip.export_id: clip for clip in clips}
    for proposal in result.proposals:
        clip = by_id[proposal.export_id]
        print(f"export {proposal.export_id}  {Path(clip.path).name}")
        print(f'  quote: "{proposal.quote.strip()}"')
        for index, option in enumerate(proposal.options, start=1):
            print(f"    {index}. {option}")
        # §8.5: the operator picks and rewrites. Nothing is stored here.
        print(f"  pick one:  clipforge hook {args.stream_id} "
              f"--pick {proposal.export_id} <number|your own text>")
        print()

    print(f"llm_invalid_id_rate {result.invalid_id_rate:.2f} "
          f"({result.returned} returned, {len(result.proposals)} accepted)")
    return 0


def _pick(cfg, conn: sqlite3.Connection, args) -> int:
    raw_id, choice = args.pick
    try:
        export_id = int(raw_id)
    except ValueError:
        raise hooks.HookError(f"{raw_id!r} is not an export id") from None

    # A bare number means "the option `--apply` printed", which requires the
    # reply to resolve it against -- the options are never stored, because
    # storing four hooks nobody chose is how `hook_text` stops meaning "the
    # hook". Anything else is the operator's own words, which §8.5 expects at
    # least as often as a straight pick.
    text = choice
    if choice.strip().isdigit():
        if not args.apply:
            raise hooks.HookError(
                f"option {choice} needs the reply it came from -- the options "
                f"are not stored. Either pass it:\n"
                f"    clipforge hook {args.stream_id} --apply <reply> "
                f"--pick {export_id} {choice}\n"
                f"or give the text directly:\n"
                f'    clipforge hook {args.stream_id} --pick {export_id} "your hook"'
            )
        text = _option_from_reply(conn, args, export_id, int(choice))

    hooks.store(conn, export_id, text)
    print(f"export {export_id} hook_text set to {text!r}")
    return 0


def _option_from_reply(conn, args, export_id: int, index: int) -> str:
    path = Path(args.apply).expanduser()
    if not path.is_file():
        raise hooks.HookError(f"no reply file at {path}")

    result = hooks.validate(
        hooks.parse_reply(path.read_text(encoding="utf-8")),
        _clips(conn, args.stream_id),
    )
    proposal = next((p for p in result.proposals if p.export_id == export_id), None)
    if proposal is None:
        raise hooks.HookError(
            f"that reply has no accepted options for export {export_id}")
    if not 1 <= index <= len(proposal.options):
        raise hooks.HookError(
            f"option {index} is out of range: export {export_id} has "
            f"{len(proposal.options)}")
    return proposal.options[index - 1]
