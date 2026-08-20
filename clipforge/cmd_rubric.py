"""`clipforge rubric` — read and append the written rubric.

The review UI is where a rubric normally gets written, right after the session
that produced the opinion. This exists for the other cases: reading an old
version, diffing two, and exporting the plain-markdown mirror §13.2's tier 2
asks for.

Every write appends. There is no `--replace` and no `--delete`, because a digest
made under v3 has to stay explicable once v4 exists.
"""

from __future__ import annotations

import difflib
from pathlib import Path

from clipforge import config, db, rubric
from clipforge.pipeline.atomic import atomic_output


def add_arguments(parser) -> None:
    config.add_config_arguments(parser)
    parser.add_argument("--show", nargs="?", const="current", metavar="VERSION",
                        help="print one version (default: the current one)")
    parser.add_argument("--list", action="store_true",
                        help="every version, with what stood behind it")
    parser.add_argument("--write", metavar="FILE",
                        help="append a new version from a file, or - for stdin")
    parser.add_argument("--diff", nargs=2, metavar=("A", "B"),
                        help="unified diff between two versions")
    parser.add_argument("--export", nargs="?", const="", metavar="DIR",
                        help="write the markdown mirror (default: rubric.export_dir)")


def main(args) -> int:
    cfg = config.from_args(args)
    if not cfg.db_path.exists():
        print(f"no database at {cfg.db_path}. Run `clipforge db init`.")
        return 1

    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    try:
        if args.write:
            return _write(cfg, conn, args.write)
        if args.diff:
            return _diff(conn, args.diff)
        if args.export is not None:
            return _export(cfg, conn, args.export)
        if args.list:
            return _list(conn)
        return _show(conn, args.show or "current")
    except ValueError as exc:
        print(f"error: {exc}")
        return 1
    finally:
        conn.close()


def _resolve(conn, which: str) -> rubric.Rubric | None:
    if which in ("current", "latest"):
        return rubric.current(conn)
    try:
        wanted = int(str(which).lstrip("v"))
    except ValueError:
        raise ValueError(f"not a version: {which!r}. Use a number, or 'current'.") from None
    return rubric.get(conn, wanted)


def _show(conn, which: str) -> int:
    entry = _resolve(conn, which)
    if entry is None:
        print("no rubric has been written yet.")
        print("  Write one after a review session — the review summary screen has a box,")
        print("  or `clipforge rubric --write notes.md`.")
        return 0
    print(f"v{entry.version}  {entry.created_at}  ({entry.evidence})")
    print()
    print(entry.text)
    return 0


def _list(conn) -> int:
    entries = rubric.versions(conn)
    if not entries:
        print("no rubric has been written yet.")
        return 0
    for entry in entries:
        first = entry.text.strip().splitlines()[0] if entry.text.strip() else ""
        head = first if len(first) <= 60 else first[:57] + "..."
        print(f"  v{entry.version:<3} {entry.created_at:<20} {entry.evidence:<28} {head}")
    return 0


def _write(cfg, conn, source: str) -> int:
    import sys

    text = sys.stdin.read() if source == "-" else Path(source).read_text(encoding="utf-8")
    with db.transaction(conn):
        entry = rubric.write(conn, text)

    print(f"wrote v{entry.version} ({entry.evidence})")
    over = rubric.oversized(entry, cfg)
    if over:
        print(f"  note: {over} character(s) over rubric.warn_chars "
              f"({cfg.get('rubric.warn_chars')}). Nothing is truncated — every prompt "
              f"that reads the rubric will carry all of it.")
    if entry.n_ratings_at_write == 0:
        print("  note: no operator ratings exist yet, so this is written from intent "
              "rather than from a review session. Recorded as such.")
    return 0


def _diff(conn, pair: list[str]) -> int:
    left, right = (_resolve(conn, which) for which in pair)
    for which, entry in zip(pair, (left, right)):
        if entry is None:
            print(f"no rubric version {which}")
            return 1

    lines = difflib.unified_diff(
        left.text.splitlines(), right.text.splitlines(),
        fromfile=f"v{left.version}", tofile=f"v{right.version}", lineterm="",
    )
    body = list(lines)
    if not body:
        print(f"v{left.version} and v{right.version} are identical")
        return 0
    print("\n".join(body))
    return 0


def _export(cfg, conn, destination: str) -> int:
    entries = rubric.versions(conn)
    if not entries:
        print("no rubric has been written yet — nothing to export.")
        return 0

    directory = Path(destination) if destination else cfg.resolve(cfg.get("rubric.export_dir"))
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "rubric.md"
    with atomic_output(target) as tmp:
        tmp.write_text(rubric.to_markdown(entries), encoding="utf-8")

    print(f"wrote {target} ({len(entries)} version(s))")
    print("  §13.2's tier 2: plain markdown, so the judgement survives a schema change")
    print("  or a rewrite of this application. The database stays the source of truth.")
    return 0
