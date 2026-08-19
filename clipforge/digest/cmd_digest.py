"""`clipforge digest` — §9.4's summaries, via a paste round trip.

Two modes of one command, because they are two steps of one job:

    clipforge digest <id>                      write the prompt
    clipforge digest <id> --apply reply.json   validate it and store a new version

The deterministic digest is not made here — it is the `digest` pipeline stage,
which runs as part of `clipforge run` and needs no model at all. This command
only adds what a model can add: titles, summaries, themes and §9.5's open loops.

Nothing here reaches the network. §2.2 puts reasoning through a frontier model
and §12.4 prices it at cents per stream, but until there is a key the operator is
the transport — and every §12 rule is enforced on the reply regardless of how it
arrived. Same shape as `clipforge hook`; see `clipforge/llm.py`.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path

from clipforge import config, db, llm, paths
from clipforge.digest import build, prompt
from clipforge.pipeline.atomic import atomic_output


def add_arguments(parser) -> None:
    parser.add_argument("stream_id")
    config.add_config_arguments(parser)
    parser.add_argument(
        "--apply", metavar="FILE",
        help="validate a model reply and store it as a new digest version",
    )
    parser.add_argument(
        "--copy", action="store_true", help="also put the prompt on the clipboard")
    parser.add_argument("--out", help="where to write the prompt")


def main(args) -> int:
    cfg = config.from_args(args)
    if not cfg.db_path.exists():
        print(f"no database at {cfg.db_path}. Run `clipforge db init`.")
        return 1

    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    try:
        if args.apply:
            return _apply(cfg, conn, args)
        return _prompt(cfg, conn, args)
    except prompt.DigestPromptError as exc:
        print(f"error: {exc}")
        return 1
    finally:
        conn.close()


def _load(conn: sqlite3.Connection, stream_id: str) -> tuple[dict, int]:
    """The newest digest, and its version.

    Everything downstream is pinned to this version — see `prompt.build_prompt`.
    """
    if conn.execute("SELECT 1 FROM streams WHERE id = ?",
                    (stream_id,)).fetchone() is None:
        raise prompt.DigestPromptError(f"no stream {stream_id!r}")

    row = build.latest(conn, stream_id)
    if row is None:
        raise prompt.DigestPromptError(
            f"{stream_id} has no digest yet. The deterministic one is built by "
            f"the pipeline, which needs no model:\n"
            f"    clipforge run {stream_id} --only digest"
        )
    return json.loads(row["content"]), int(row["version"])


def _prompt(cfg, conn: sqlite3.Connection, args) -> int:
    content, version = _load(conn, args.stream_id)
    views = prompt.chapter_views(conn, args.stream_id, content)
    text = prompt.build_prompt(cfg, content, version, views)

    stream_paths = paths.StreamPaths(cfg.data_root, args.stream_id).ensure()
    destination = (
        Path(args.out).expanduser().resolve() if args.out
        else stream_paths.exports_dir /
        f"{args.stream_id}_digest_v{version}_{date.today().isoformat()}.md"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with atomic_output(destination) as temp:
        temp.write_text(text, encoding="utf-8")

    with_text = sum(1 for view in views if view.lines)
    print(f"prompt written for digest v{version}: {len(views)} chapter(s), "
          f"{with_text} with a transcript")
    print(f"  {destination}")
    if with_text == 0:
        # The default case today, and worth saying plainly rather than letting
        # the operator paste a prompt that cannot produce anything.
        print("\n  NOTE: no chapter has a transcript, so there is nothing for a "
              "model to summarise.\n  Turn on extract.whisperx.enabled and "
              "re-run this stream first.")
    if args.copy:
        print("  copied to the clipboard" if llm.to_clipboard(text)
              else "  could not reach the clipboard; open the file instead")

    print("\nPaste it into a frontier model, save the reply to a file, then:")
    print(f"    clipforge digest {args.stream_id} --apply <that file>")
    return 0


def _apply(cfg, conn: sqlite3.Connection, args) -> int:
    content, version = _load(conn, args.stream_id)
    views = prompt.chapter_views(conn, args.stream_id, content)

    path = Path(args.apply).expanduser()
    if not path.is_file():
        raise prompt.DigestPromptError(f"no reply file at {path}")

    result = prompt.validate(
        llm.parse_reply(path.read_text(encoding="utf-8")), views, version=version)
    llm.record(conn, args.stream_id, result, accepted=len(result.chapters))

    if result.dropped:
        # §12.2 drops silently as far as the model is concerned. The operator is
        # not the model: they need to know four of six were discarded.
        print(f"dropped {len(result.dropped)} item(s):")
        for reason, detail in result.dropped:
            print(f"  {reason:<12} {detail}")
        print()

    if not result.chapters:
        print("nothing usable in that reply.")
        print(f"  llm_invalid_id_rate {result.invalid_id_rate:.2f} "
              f"(recorded to tool_metrics)")
        return 1

    merged = prompt.reduce_into(content, result, conn, args.stream_id)
    digest = build.Digest(content=merged,
                          markdown=build.render_markdown(merged))

    source = str(cfg.get("digest.source", "manual"))
    with db.transaction(conn):
        # A NEW version. §9.1 keeps every one forever, so v1's deterministic
        # digest survives whatever a model said about it.
        new_version = build.write(conn, args.stream_id, digest,
                                  model_used=f"{source}:pasted")
        loops = prompt.store_open_loops(conn, args.stream_id,
                                        merged.get("open_loops") or [])

    for chapter in merged["chapters"]:
        if chapter.get("title"):
            print(f"  {chapter['index']}. {chapter['title']}")

    print(f"\ndigest v{new_version} written "
          f"({len(result.chapters)} chapter(s) summarised, "
          f"{len(merged.get('themes_observed') or [])} theme(s), "
          f"{loops} open loop(s))")
    print(f"  v{version} is unchanged — §9.1 keeps every version")
    print(f"llm_invalid_id_rate {result.invalid_id_rate:.2f} "
          f"({result.returned} returned, {len(result.chapters)} accepted)")
    return 0
