"""`clipforge db ...` — init, inspect, and relink the database."""

from __future__ import annotations

import sqlite3

from clipforge import config, db


def add_arguments(parser) -> None:
    config.add_config_arguments(parser)
    actions = parser.add_subparsers(dest="db_action", metavar="<action>")

    init = actions.add_parser("init", help="create the database and apply migrations")
    init.set_defaults(db_action="init")

    schema = actions.add_parser("schema", help="print the live schema")
    schema.add_argument("--table", help="limit to one table")
    schema.set_defaults(db_action="schema")

    info = actions.add_parser("info", help="path, schema version, and row counts")
    info.set_defaults(db_action="info")

    relink = actions.add_parser(
        "relink",
        help="rewrite master_path prefixes after moving footage to another drive",
    )
    relink.add_argument("--from", dest="from_prefix", required=True)
    relink.add_argument("--to", dest="to_prefix", required=True)
    relink.add_argument("--dry-run", action="store_true")
    relink.set_defaults(db_action="relink")


def main(args) -> int:
    cfg = config.from_args(args)
    action = getattr(args, "db_action", None) or "info"

    if action == "init":
        return _init(cfg)

    if not cfg.db_path.exists():
        print(f"no database at {cfg.db_path}. Run `clipforge db init`.")
        return 1

    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    try:
        if action == "schema":
            return _schema(conn, getattr(args, "table", None))
        if action == "info":
            return _info(cfg, conn)
        if action == "relink":
            return _relink(conn, args.from_prefix, args.to_prefix, args.dry_run)
    finally:
        conn.close()

    print(f"unknown action: {action}")
    return 2


def _init(cfg) -> int:
    existed = cfg.db_path.exists()
    conn = db.connect(cfg.db_path)
    try:
        before = db.user_version(conn)
        applied = db.migrate(conn)
        after = db.user_version(conn)
        tables = db.table_names(conn)
    finally:
        conn.close()

    print(f"database   {cfg.db_path}")
    if not existed:
        print(f"created    schema version {after}, {len(tables)} tables")
    elif applied:
        print(f"migrated   {before} -> {after} (applied {', '.join(map(str, applied))})")
    else:
        print(f"up to date at schema version {after}, {len(tables)} tables")
    return 0


def _schema(conn: sqlite3.Connection, table: str | None) -> int:
    query = (
        "SELECT type, name, sql FROM sqlite_master "
        "WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%'"
    )
    params: tuple[str, ...] = ()
    if table:
        query += " AND (name = ? OR tbl_name = ?)"
        params = (table, table)
    query += " ORDER BY CASE type WHEN 'table' THEN 0 ELSE 1 END, name"

    rows = conn.execute(query, params).fetchall()
    if not rows:
        print("nothing found" + (f" for table {table!r}" if table else ""))
        return 1
    for row in rows:
        print(f"{row['sql']};\n")
    return 0


def _info(cfg, conn: sqlite3.Connection) -> int:
    tables = db.table_names(conn)
    print(f"database        {cfg.db_path}")
    print(f"schema version  {db.user_version(conn)} (latest {db.latest_version()})")
    print(f"journal mode    {conn.execute('PRAGMA journal_mode').fetchone()[0]}")
    print(f"foreign keys    {'on' if conn.execute('PRAGMA foreign_keys').fetchone()[0] else 'OFF'}")
    print(f"size            {cfg.db_path.stat().st_size / 1024:.1f} KiB")
    print()

    width = max(len(name) for name in tables)
    populated = 0
    for name in tables:
        count = conn.execute(f"SELECT count(*) FROM {name}").fetchone()[0]  # noqa: S608
        if count:
            populated += 1
        print(f"  {name.ljust(width)}  {count}")
    print(f"\n{len(tables)} tables, {populated} non-empty")
    return 0


def _relink(conn: sqlite3.Connection, from_prefix: str, to_prefix: str, dry_run: bool) -> int:
    """Rewrite `streams.master_path` prefixes.

    Masters are the one path the database stores absolutely — they live outside
    `data_root`, often on a drive that gets a different letter on another
    machine. Everything else is relative and needs no repair.
    """
    rows = conn.execute("SELECT id, master_path FROM streams ORDER BY id").fetchall()
    changes = [
        (row["id"], row["master_path"], to_prefix + row["master_path"][len(from_prefix) :])
        for row in rows
        if row["master_path"].startswith(from_prefix)
    ]

    if not changes:
        print(f"no master_path starts with {from_prefix!r} ({len(rows)} streams checked)")
        return 1

    for stream_id, old, new in changes:
        print(f"{stream_id}\n  {old}\n  -> {new}")

    if dry_run:
        print(f"\n{len(changes)} would change (dry run)")
        return 0

    with db.transaction(conn):
        conn.executemany(
            "UPDATE streams SET master_path = ? WHERE id = ?",
            [(new, stream_id) for stream_id, _, new in changes],
        )
    print(f"\n{len(changes)} updated")
    return 0
