"""§13.2's backup of the irreplaceable tier.

> **Tier 1 — SQLite database. THE IRREPLACEABLE TIER.** Five months of ratings,
> digests, embeddings, and idea history cannot be reconstructed. Signals can be
> re-extracted from footage; the operator's judgment calls cannot.

§13.2's recipe is four lines of bash — `sqlite3 ... VACUUM INTO`, `gzip`, upload
to B2, retain 30 daily + 12 monthly. What follows is that, adjusted for the five
places the literal script does not survive contact with this project.

**No `sqlite3` binary.** Nothing here depends on one; every other database
access in the codebase goes through Python's `sqlite3` module, and requiring a
CLI tool for the one job that must never be skipped is a way to have it skipped.
MEASURED: the venv is Python 3.12.10 with SQLite 3.49.1, well past `VACUUM
INTO`'s 3.27 floor. The floor is still checked, because a machine with an older
interpreter would otherwise fail with a bare syntax error.

**The source is opened READ ONLY, over a `file:...?mode=ro` URI.** Not
`db.connect`, which issues `PRAGMA journal_mode=WAL` — and MEASURED, that
statement rewrites the file header of a database that is not already in WAL. A
backup that modifies the thing it is backing up is the one bug this whole module
exists to avoid, and `CLAUDE.md`'s rule 5 exists because it happened once.
MEASURED: `VACUUM INTO` works fine over a read-only connection, and the source
file's SHA-256 is unchanged across a backup.

**It is safe against a live database, and that was measured rather than
trusted.** A5 puts the database in WAL so the review UI can write while the
pipeline runs, which means a backup taken at 04:00 may run against a connection
mid-session. MEASURED: with 626 KB of committed, deliberately un-checkpointed
WAL outstanding, the copy contained every one of those rows. Had it not, this
module would be worse than nothing — a file that looks like a backup and is
missing the newest ratings.

**A same-day repeat is refused rather than silently failing.** §13.2's own
`clipforge_$(date +%F).db` collides on the second run of a day, and MEASURED,
`VACUUM INTO` then raises `output file already exists` — so the spec's script
fails at exactly the moment someone runs it by hand to be safe. Here that is an
explicit refusal with `--force` to override. (MEASURED, and worth knowing: the
refusal only covers a NON-EMPTY target. A zero-byte file is accepted and
overwritten, so every write goes through `atomic_output`, which unlinks first.)

**No upload.** C5 — no B2 account exists and zero streams do. But a `.gz` beside
its source is not protection against the disk dying, and this module says so out
loud rather than letting the word "backup" imply otherwise. `backup.mirror_dir`
copies the finished file to a second plain directory — an external drive, or a
folder some other program already syncs. No credentials, no SDK, no new
dependency.

**§13.3 is not a one-off.** *"An untested backup is not a backup."* The spec
suggests testing the restore path once, early, and then trusting it. Trusting it
for five months is the failure this is meant to prevent, so `verify()` ships
here and runs on demand against any backup: decompress to a scratch directory,
open it with the application's own `db.open_db(migrate_to_latest=False)` — which
already refuses a schema version it does not expect — run SQLite's own integrity
check, and compare every table's row count against the manifest written
alongside the backup.

The manifest is what makes verification mean anything. Comparing a month-old
backup against the live database would fail for the honest reason that the live
database has moved on; comparing against hardcoded counts would pass a broken
backup. The counts are recorded at the moment of the copy and checked against
themselves — the same rule every numeric test in this project follows.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import sqlite3
import tempfile
import time
from dataclasses import dataclass, field
from datetime import date as date_cls
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from clipforge import __version__, db
from clipforge.pipeline import atomic

#: `VACUUM INTO` landed in SQLite 3.27 (2019). Checked rather than assumed so an
#: older interpreter fails with a sentence instead of a syntax error.
MIN_SQLITE = (3, 27, 0)

#: §13.2's naming, with the date it was taken. The suffix depends on `compress`.
STEM_PREFIX = "clipforge_"
_NAME = re.compile(r"^clipforge_(\d{4})-(\d{2})-(\d{2})\.db(\.gz)?$")

#: §14 has no backup metric. This name is INVENTED; nothing in the spec uses it.
METRIC = "backup_duration_s"


class BackupError(RuntimeError):
    """A backup could not be taken, or a backup could not be verified."""


# --------------------------------------------------------------------------
# reading the source without touching it
# --------------------------------------------------------------------------


def read_only_uri(path: Path) -> str:
    """A `file:` URI that opens `path` read-only.

    `?` and `#` are URI syntax and appear in real Windows filenames, so the path
    is percent-encoded. The drive letter's colon and the separators are not:
    SQLite accepts `file:C:/Users/...` and mangling those breaks it.
    """
    return f"file:{quote(path.as_posix(), safe='/:')}?mode=ro"


def open_source(path: Path) -> sqlite3.Connection:
    """Open the live database read-only.

    Deliberately NOT `db.connect`. That sets `PRAGMA journal_mode=WAL`, which
    MEASURED rewrites the file header when the database is not already in WAL —
    a write to the one file that must not be written to here.
    """
    if not path.exists():
        raise BackupError(f"no database at {path}. Run `clipforge db init`.")
    if sqlite3.sqlite_version_info < MIN_SQLITE:
        raise BackupError(
            f"SQLite {sqlite3.sqlite_version} cannot VACUUM INTO; "
            f"§13.2 needs {'.'.join(map(str, MIN_SQLITE))} or newer"
        )
    conn = sqlite3.connect(read_only_uri(path), uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def vacuum_into(source: Path, destination: Path) -> int:
    """Copy `source` to `destination`, compacted. Returns the bytes written.

    The whole read path in one function, so the test that asserts the source is
    byte-identical afterwards exercises the code that runs rather than a
    reconstruction of it.
    """
    conn = open_source(source)
    try:
        # A bound parameter, not an f-string: the destination is a real path
        # that may contain a quote. VERIFIED to work on this build.
        conn.execute("VACUUM INTO ?", (str(destination),))
    finally:
        conn.close()
    return destination.stat().st_size


def manifest(conn: sqlite3.Connection, source: Path) -> dict:
    """What the backup contains. `conn` is the COPY; `source` names the original.

    Row counts per table plus the schema version. `verify` checks the restored
    copy against this rather than against the live database, which has legitimately
    moved on, or against literals, which would pass a broken backup.

    **Read from the copy, not from the source.** A5 puts the database in WAL so
    the review UI can write while other things run, so anything committed between
    a source-side count and the `VACUUM INTO` would be in the backup and not in
    the manifest — and `--verify` would report a problem on a backup that is
    perfectly good. A nightly check that cries wolf stops being read, which is
    the failure §13.3 exists to prevent. Reading the copy makes the manifest true
    by construction instead of by winning a race. It cannot be closed with a
    transaction either: SQLite refuses to VACUUM from inside one.
    """
    tables = db.table_names(conn)
    return {
        "clipforge_version": __version__,
        "taken_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": str(source),
        "source_bytes": source.stat().st_size,
        "schema_version": db.user_version(conn),
        "sqlite_version": sqlite3.sqlite_version,
        # Interpolated because a table name cannot be a bound parameter. The
        # names come from sqlite_master via db.table_names, never from input.
        "tables": {
            name: int(conn.execute(f"SELECT count(*) FROM {name}").fetchone()[0])  # noqa: S608
            for name in tables
        },
    }


def manifest_of(db_file: Path, source: Path) -> dict:
    """`manifest`, reading a database file that is already a copy.

    Opened with a plain connection rather than `db.connect`: this is a private
    scratch file, and setting `journal_mode=WAL` on it would leave `-wal`/`-shm`
    siblings beside a file that is about to be compressed and renamed.
    """
    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row
    try:
        return manifest(conn, source)
    finally:
        conn.close()


# --------------------------------------------------------------------------
# naming and listing
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Entry:
    """One backup file on disk."""

    path: Path
    day: date_cls
    bytes: int

    @property
    def sidecar(self) -> Path:
        return sidecar_for(self.path)

    def age_days(self, today: date_cls | None = None) -> int:
        return ((today or date_cls.today()) - self.day).days


def directory(cfg) -> Path:
    return cfg.resolve(cfg.get("backup.dir"))


def filename(day: date_cls, compress: bool) -> str:
    return f"{STEM_PREFIX}{day.isoformat()}.db" + (".gz" if compress else "")


def sidecar_for(path: Path) -> Path:
    """The manifest beside a backup. `x.db.gz` -> `x.json`."""
    name = path.name
    for suffix in (".db.gz", ".db"):
        if name.endswith(suffix):
            return path.parent / (name[: -len(suffix)] + ".json")
    return path.with_suffix(".json")


def list_backups(where: Path) -> list[Entry]:
    """Every backup in a directory, newest first.

    Only files matching the exact naming pattern are returned — which is also
    what makes `prune` safe: it can never delete something it cannot name.
    """
    if not where.is_dir():
        return []
    found: list[Entry] = []
    for path in where.iterdir():
        match = _NAME.match(path.name)
        if not match or not path.is_file():
            continue
        try:
            day = date_cls(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            continue  # 2026-02-30: a name that looks right and is not a date
        found.append(Entry(path=path, day=day, bytes=path.stat().st_size))
    return sorted(found, key=lambda e: (e.day, e.path.name), reverse=True)


# --------------------------------------------------------------------------
# retention (§13.2: "30 daily + 12 monthly")
# --------------------------------------------------------------------------


def plan_prune(entries: list[Entry], keep_daily: int, keep_monthly: int) -> list[Entry]:
    """Which backups to delete. §13.2 gives the counts and not the rule.

    Newest first: keep the newest `keep_daily`. Of what is left, keep the newest
    file in each distinct year-month, for the newest `keep_monthly` months.
    Delete the rest.

    The newest backup is never deletable, whatever the settings say — a
    `keep_daily: 0` typo must not be able to empty the directory.
    """
    ordered = sorted(entries, key=lambda e: (e.day, e.path.name), reverse=True)
    if not ordered:
        return []

    keep = {id(e) for e in ordered[: max(keep_daily, 1)]}

    monthly: dict[tuple[int, int], Entry] = {}
    for entry in ordered[max(keep_daily, 1) :]:
        monthly.setdefault((entry.day.year, entry.day.month), entry)
    for month in sorted(monthly, reverse=True)[: max(keep_monthly, 0)]:
        keep.add(id(monthly[month]))

    return [e for e in ordered if id(e) not in keep]


# --------------------------------------------------------------------------
# taking a backup
# --------------------------------------------------------------------------


@dataclass
class Result:
    path: Path
    sidecar: Path
    manifest: dict
    stored_bytes: int
    vacuumed_bytes: int
    duration_s: float
    compressed: bool
    mirrored: Path | None = None
    pruned: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    dry_run: bool = False

    @property
    def source_bytes(self) -> int:
        return int(self.manifest["source_bytes"])

    @property
    def rows(self) -> int:
        return sum(self.manifest["tables"].values())


def take(
    cfg,
    *,
    day: date_cls | None = None,
    force: bool = False,
    prune: bool = True,
    dry_run: bool = False,
) -> Result:
    """§13.2's nightly job, once.

    VACUUM INTO a temp beside the destination, gzip it, atomically rename, write
    the manifest, mirror it, prune, and record the metric. A10's temp-then-rename
    applies here for the same reason it applies to a proxy: a backup interrupted
    halfway must not leave a file that looks complete.
    """
    source = cfg.db_path
    day = day or date_cls.today()
    compress = bool(cfg.get("backup.compress", True))
    where = directory(cfg)
    final = where / filename(day, compress)
    warnings: list[str] = []

    # Opened and closed before anything else so a missing database reports
    # itself as one, rather than as whatever the collision check says next.
    # The real manifest is read from the COPY further down; this one exists
    # only for --dry-run, which has no copy to read.
    conn = open_source(source)
    try:
        recorded = manifest(conn, source) if dry_run else {}
    finally:
        conn.close()

    collision = final.exists() and not force
    if collision and not dry_run:
        raise BackupError(
            f"{final.name} already exists. §13.2 names backups by date, so a second "
            f"run on the same day would overwrite the first. Use --force to replace it."
        )

    if dry_run:
        entries = list_backups(where)
        if collision:
            # A dry run that raised would be useless on the one day you most
            # want to ask: the day a backup already exists.
            warnings.append(f"{final.name} already exists — a real run needs --force")
        return Result(
            path=final, sidecar=sidecar_for(final), manifest=recorded,
            stored_bytes=0, vacuumed_bytes=0, duration_s=0.0, compressed=compress,
            pruned=[e.path for e in plan_prune(
                entries,
                int(cfg.get("backup.keep_daily")),
                int(cfg.get("backup.keep_monthly")),
            )] if prune else [],
            warnings=warnings, dry_run=True,
        )

    where.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    scratch = atomic.temp_path_for(where / f"{STEM_PREFIX}{day.isoformat()}.vacuum.db")
    scratch.unlink(missing_ok=True)

    try:
        vacuumed = vacuum_into(source, scratch)
        # From the copy, so the manifest describes what was actually captured
        # rather than what the source held a moment earlier.
        recorded = manifest_of(scratch, source)

        with atomic.atomic_output(final) as tmp:
            if compress:
                # filename='' and mtime=0 so two backups of identical content
                # produce identical bytes. Without them the gzip header carries
                # the temp file's name and the wall clock, and every nightly
                # differs from the last even when nothing in the database did —
                # which defeats any mirror that deduplicates, and makes
                # comparing two backups impossible.
                with open(tmp, "wb") as sink:
                    with gzip.GzipFile(filename="", mode="wb", fileobj=sink, mtime=0) as out:
                        with open(scratch, "rb") as raw:
                            shutil.copyfileobj(raw, out, 1024 * 1024)
            else:
                shutil.copy2(scratch, tmp)
    finally:
        scratch.unlink(missing_ok=True)

    stored = final.stat().st_size
    duration = time.perf_counter() - started

    recorded = {
        **recorded,
        "backup": final.name,
        "compressed": compress,
        "vacuumed_bytes": vacuumed,
        "stored_bytes": stored,
    }
    side = sidecar_for(final)
    try:
        with atomic.atomic_output(side) as tmp:
            tmp.write_text(json.dumps(recorded, indent=2, sort_keys=True), encoding="utf-8")
    except OSError as exc:
        warnings.append(
            f"backup written but its manifest was not ({exc}) — `--verify` can still "
            f"check integrity and schema version, but not row counts"
        )

    result = Result(
        path=final, sidecar=side, manifest=recorded, stored_bytes=stored,
        vacuumed_bytes=vacuumed, duration_s=duration, compressed=compress,
        warnings=warnings,
    )

    result.mirrored = _mirror(cfg, final, side, warnings)
    if prune:
        result.pruned = _prune(cfg, where, warnings)
    _record_metric(cfg, result, warnings)
    return result


def _mirror(cfg, final: Path, side: Path, warnings: list[str]) -> Path | None:
    """Copy the finished backup to a second directory, if one is configured.

    NOT §13.2's B2/S3 upload. This is the honest version of it available without
    an account: a plain directory that is either on other hardware or is synced
    by something else. A copy on the same disk protects against a bad migration
    and a mistaken DELETE; it does not protect against the disk.
    """
    configured = cfg.get("backup.mirror_dir", None)
    if not configured:
        return None

    target = cfg.resolve(configured)
    try:
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(final, target / final.name)
        if side.exists():
            shutil.copy2(side, target / side.name)
    except OSError as exc:
        # Never fatal: the local backup succeeded, and failing the whole job
        # because an external drive is unplugged would mean no backup at all.
        warnings.append(f"mirror to {target} failed: {exc}")
        return None
    return target / final.name


def _prune(cfg, where: Path, warnings: list[str]) -> list[Path]:
    removed: list[Path] = []
    doomed = plan_prune(
        list_backups(where),
        int(cfg.get("backup.keep_daily")),
        int(cfg.get("backup.keep_monthly")),
    )
    for entry in doomed:
        try:
            entry.path.unlink()
            removed.append(entry.path)
            entry.sidecar.unlink(missing_ok=True)
        except OSError as exc:
            warnings.append(f"could not remove {entry.path.name}: {exc}")
    return removed


def _record_metric(cfg, result: Result, warnings: list[str]) -> None:
    """C6: log it from day one. §14 names no backup metric, so this one is invented.

    Written after the copy, so it describes a backup that the backup itself does
    not contain. Never fatal — a database that cannot be written to is a reason
    to keep the backup, not to throw it away.
    """
    try:
        conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    except Exception as exc:  # MigrationError, or a locked/read-only file
        warnings.append(f"backup taken; its metric was not recorded ({exc})")
        return
    try:
        with db.transaction(conn):
            conn.execute(
                "INSERT INTO tool_metrics (metric, value, meta) VALUES (?, ?, ?)",
                (METRIC, round(result.duration_s, 4), json.dumps({
                    "path": result.path.name,
                    "source_bytes": result.source_bytes,
                    "vacuumed_bytes": result.vacuumed_bytes,
                    "stored_bytes": result.stored_bytes,
                    "rows": result.rows,
                    "schema_version": result.manifest["schema_version"],
                    "pruned": len(result.pruned),
                    "mirrored": bool(result.mirrored),
                })),
            )
    except sqlite3.Error as exc:
        warnings.append(f"backup taken; its metric was not recorded ({exc})")
    finally:
        conn.close()


# --------------------------------------------------------------------------
# §13.3 — the restore path, tested
# --------------------------------------------------------------------------


@dataclass
class VerifyResult:
    path: Path
    ok: bool
    restored_bytes: int
    schema_version: int | None
    tables: dict[str, int]
    expected: dict[str, int] | None
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def rows(self) -> int:
        return sum(self.tables.values())


def verify(path: Path) -> VerifyResult:
    """§13.3: restore to a scratch directory, point the application at it.

    > "An untested backup is not a backup."

    Four checks, in the order that a broken backup fails them: it decompresses,
    SQLite's own `integrity_check` passes, the application's `open_db` accepts
    the schema version, and every table holds the number of rows the manifest
    recorded when the copy was taken.
    """
    if not path.exists():
        raise BackupError(f"no backup at {path}")

    problems: list[str] = []
    notes: list[str] = []
    expected: dict[str, int] | None = None

    side = sidecar_for(path)
    if side.exists():
        try:
            expected = json.loads(side.read_text(encoding="utf-8")).get("tables")
        except (OSError, ValueError) as exc:
            notes.append(f"manifest {side.name} unreadable ({exc}); row counts unchecked")
    else:
        notes.append(f"no manifest beside {path.name}; row counts unchecked")

    scratch = Path(tempfile.mkdtemp(prefix="clipforge-restore-"))
    restored = scratch / "restored.db"
    try:
        try:
            if path.name.endswith(".gz"):
                with gzip.open(path, "rb") as raw, open(restored, "wb") as out:
                    shutil.copyfileobj(raw, out, 1024 * 1024)
            else:
                shutil.copy2(path, restored)
        except (OSError, EOFError, gzip.BadGzipFile) as exc:
            return VerifyResult(
                path=path, ok=False, restored_bytes=0, schema_version=None,
                tables={}, expected=expected,
                problems=[f"{path.name} could not be decompressed: {exc}"], notes=notes,
            )

        size = restored.stat().st_size
        tables: dict[str, int] = {}
        version: int | None = None

        try:
            probe = sqlite3.connect(restored)
            probe.row_factory = sqlite3.Row
            integrity = probe.execute("PRAGMA integrity_check").fetchone()[0]
            probe.close()
            if integrity != "ok":
                problems.append(f"integrity_check said {integrity!r}")
        except sqlite3.Error as exc:
            problems.append(f"not a readable database: {exc}")
            return VerifyResult(
                path=path, ok=False, restored_bytes=size, schema_version=None,
                tables={}, expected=expected, problems=problems, notes=notes,
            )

        # "Point the application at it": open_db refuses a schema version this
        # build does not expect, which is the check that matters after a
        # migration lands between the backup and the restore.
        try:
            conn = db.open_db(restored, migrate_to_latest=False)
        except db.MigrationError as exc:
            problems.append(str(exc))
            return VerifyResult(
                path=path, ok=False, restored_bytes=size, schema_version=None,
                tables={}, expected=expected, problems=problems, notes=notes,
            )
        try:
            version = db.user_version(conn)
            for name in db.table_names(conn):
                tables[name] = int(
                    conn.execute(f"SELECT count(*) FROM {name}").fetchone()[0]  # noqa: S608
                )
        finally:
            conn.close()

        if expected is not None:
            for name, count in sorted(expected.items()):
                if name not in tables:
                    problems.append(f"table {name} is missing from the restored copy")
                elif tables[name] != count:
                    problems.append(
                        f"{name}: {tables[name]} rows restored, manifest recorded {count}"
                    )
            for name in sorted(set(tables) - set(expected)):
                notes.append(f"{name} is in the copy and not in the manifest")

        return VerifyResult(
            path=path, ok=not problems, restored_bytes=size, schema_version=version,
            tables=tables, expected=expected, problems=problems, notes=notes,
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# --------------------------------------------------------------------------
# the nightly schedule
# --------------------------------------------------------------------------


def schedule_command(cfg, task_name: str = "ClipForge nightly backup") -> str:
    """The `schtasks` line for §13.2's "nightly, automated".

    Printed, never run: registering a scheduled task is a standing change to the
    machine and belongs to the operator's keystroke, not to a tool that was asked
    to take one backup.

    It goes through `clipforge.ps1` rather than `clipforge.exe` because that
    launcher does a `Push-Location` to the project root, and
    `config.find_project_root` walks up from the working directory — a task run
    from `C:\\Windows\\System32` would otherwise resolve `./data` somewhere else
    entirely and quietly back up nothing.
    """
    launcher = cfg.project_root / "clipforge.ps1"
    at = str(cfg.get("backup.schedule_time", "04:00"))
    run = (
        f'powershell -NoProfile -ExecutionPolicy Bypass -File "{launcher}" backup'
    )
    return f'schtasks /Create /TN "{task_name}" /TR \'{run}\' /SC DAILY /ST {at} /F'


def newest(where: Path) -> Entry | None:
    found = list_backups(where)
    return found[0] if found else None


def age_of_newest(cfg) -> tuple[Entry | None, int | None]:
    """The newest backup and its age in days, for `clipforge doctor`."""
    entry = newest(directory(cfg))
    return entry, (entry.age_days() if entry else None)


def humanise(size: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GiB"


def on_same_volume(a: Path, b: Path) -> bool:
    """Whether two paths are on the same drive.

    The one thing that decides whether a mirror is protection against hardware
    failure or only against a mistake, and it is worth saying which.
    """
    try:
        return os.path.splitdrive(a.resolve())[0].lower() == \
            os.path.splitdrive(b.resolve())[0].lower()
    except OSError:
        return True
