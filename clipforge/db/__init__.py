"""Database access, schema, and migrations (spec §3).

§2.3 lists a `db/schema.sql` alongside migrations. Keeping both means keeping
them in sync by hand, and they drift. Here `migrations/` is the single source of
truth and `clipforge db schema` dumps the live shape when you want to read it.

A5: WAL is mandatory — the review UI reads while batch processing writes.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

#: Wait this long for a writer before raising `database is locked`. Batch
#: processing holds brief write transactions; the review UI should queue rather
#: than fail.
BUSY_TIMEOUT_MS = 10_000


class MigrationError(RuntimeError):
    pass


def migration_files() -> list[tuple[int, Path]]:
    """Migrations as (version, path), ordered. Filenames are `NNNN_name.sql`."""
    found: list[tuple[int, Path]] = []
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        prefix = path.stem.split("_", 1)[0]
        if not prefix.isdigit():
            raise MigrationError(f"migration {path.name} does not start with a number")
        found.append((int(prefix), path))

    versions = [version for version, _ in found]
    if versions != sorted(set(versions)):
        raise MigrationError(f"migration versions are duplicated or out of order: {versions}")
    if versions and versions != list(range(1, len(versions) + 1)):
        raise MigrationError(f"migration versions must run 1..N with no gaps: {versions}")
    return found


def latest_version() -> int:
    files = migration_files()
    return files[-1][0] if files else 0


def connect(db_path: Path, *, create_parents: bool = True) -> sqlite3.Connection:
    """Open the database with the pragmas this project depends on."""
    if create_parents:
        db_path.parent.mkdir(parents=True, exist_ok=True)

    # isolation_level=None disables the sqlite3 module's legacy habit of
    # opening a transaction behind your back before DML. Without it, an
    # explicit BEGIN raises "cannot start a transaction within a transaction"
    # the moment anything has already been inserted, and transaction boundaries
    # depend on which statement ran last. This module controls them explicitly;
    # `transaction()` is the only way to get one.
    conn = sqlite3.connect(db_path, timeout=BUSY_TIMEOUT_MS / 1000, isolation_level=None)
    conn.row_factory = sqlite3.Row
    # WAL is persistent in the file header; setting it every open is harmless
    # and means a database created by another tool still gets it.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    # Durable enough for a single-writer local pipeline, and much faster than
    # FULL for the many small writes the score stage makes.
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def user_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def migrate(conn: sqlite3.Connection) -> list[int]:
    """Apply pending migrations. Returns the versions applied (empty = current)."""
    current = user_version(conn)
    applied: list[int] = []

    for version, path in migration_files():
        if version <= current:
            continue
        sql = path.read_text(encoding="utf-8")
        # The transaction control goes INSIDE the script. executescript() issues
        # an implicit COMMIT when a transaction is already open, so wrapping the
        # call in an outer BEGIN would silently defeat the atomicity it looks
        # like it provides. A migration either lands whole or not at all —
        # otherwise a failure halfway leaves half a schema and user_version
        # still reading 0, and the retry dies on "table already exists".
        # PRAGMA takes no bound parameters, hence the interpolation; `version`
        # comes from a filename we already validated as digits.
        script = f"BEGIN;\n{sql}\nPRAGMA user_version={version};\nCOMMIT;"
        try:
            conn.executescript(script)
        except sqlite3.Error as exc:
            conn.rollback()
            raise MigrationError(f"migration {path.name} failed: {exc}") from exc
        applied.append(version)

    return applied


def open_db(db_path: Path, *, migrate_to_latest: bool = True) -> sqlite3.Connection:
    """Connect and bring the schema up to date."""
    conn = connect(db_path)
    if migrate_to_latest:
        migrate(conn)
    else:
        current = user_version(conn)
        expected = latest_version()
        if current != expected:
            conn.close()
            raise MigrationError(
                f"database at {db_path} is at schema version {current}, expected {expected}. "
                f"Run `clipforge backup` and then `clipforge db init`."
            )
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Explicit write transaction that rolls back on any exception."""
    conn.execute("BEGIN")
    try:
        yield conn
    except BaseException:
        conn.rollback()
        raise
    conn.commit()


def table_names(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [row["name"] for row in rows]
