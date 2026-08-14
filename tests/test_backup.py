"""§13.2's backup and §13.3's restore test.

Every database here is built from the real migrations inside `tmp_path`.
**`data/clipforge.db` is never opened** — CLAUDE.md rule 5 exists because it was
written to once, and a backup test is exactly the shape of code that would do it
again.

The numeric assertions read `manifest.json`-style recorded state (the backup's
own sidecar) and the config object. Nothing here compares against a literal row
count, because a literal would pass a backup that had silently lost rows.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sqlite3
import sys
from datetime import date as date_cls
from datetime import timedelta
from pathlib import Path

import pytest

from clipforge import cli, config, db
from clipforge.db import backup

# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def _cfg(tmp_path, **overrides):
    """A config whose data_root and db live entirely inside tmp_path."""
    flags = [
        f"paths.data_root={(tmp_path / 'data').as_posix()}",
        f"paths.db={(tmp_path / 'data' / 'clipforge.db').as_posix()}",
        f"backup.dir={(tmp_path / 'data' / 'backups').as_posix()}",
    ]
    flags += [f"{k}={v}" for k, v in overrides.items()]
    return config.load(overrides=flags)


def _populate(conn, streams: int = 2, metrics: int = 5) -> None:
    with db.transaction(conn):
        for i in range(streams):
            conn.execute(
                "INSERT INTO streams (id, date, master_path) VALUES (?, ?, ?)",
                (f"s{i}", "2026-08-14", f"X:/rec{i}.mkv"),
            )
        for i in range(metrics):
            conn.execute(
                "INSERT INTO tool_metrics (metric, value) VALUES (?, ?)",
                ("stage_duration_s", float(i)),
            )


@pytest.fixture
def cfg(tmp_path):
    made = _cfg(tmp_path)
    conn = db.open_db(made.db_path)
    try:
        _populate(conn)
    finally:
        conn.close()
    return made


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# the four measurements, as assertions
# --------------------------------------------------------------------------


def test_the_copy_leaves_the_source_byte_identical(cfg, tmp_path):
    """MEASURED, and the reason `open_source` exists.

    `db.connect` issues `PRAGMA journal_mode=WAL`, which rewrites the header of
    a database that is not already in WAL. The read path must never do that to
    the one file §13.2 calls irreplaceable.
    """
    # Checkpoint first so the main file is the whole story.
    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()

    before = _sha(cfg.db_path)
    written = backup.vacuum_into(cfg.db_path, tmp_path / "copy.db")
    assert written > 0
    assert _sha(cfg.db_path) == before


def test_take_changes_the_source_only_by_appending_its_own_metric(cfg):
    """The honest whole-operation statement.

    The copy itself does not touch the source; `take` then deliberately appends
    one `tool_metrics` row (C6). Nothing else in the database may move.
    """
    def counts():
        conn = db.open_db(cfg.db_path, migrate_to_latest=False)
        try:
            return {n: conn.execute(f"SELECT count(*) FROM {n}").fetchone()[0]
                    for n in db.table_names(conn)}
        finally:
            conn.close()

    before = counts()
    backup.take(cfg)
    after = counts()

    assert after["tool_metrics"] == before["tool_metrics"] + 1
    assert {k: v for k, v in after.items() if k != "tool_metrics"} == \
           {k: v for k, v in before.items() if k != "tool_metrics"}

    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    try:
        newest = conn.execute(
            "SELECT metric FROM tool_metrics ORDER BY id DESC LIMIT 1"
        ).fetchone()[0]
    finally:
        conn.close()
    assert newest == backup.METRIC


def test_the_source_is_opened_read_only(cfg):
    """The connection itself must refuse writes, not merely decline to make them."""
    conn = backup.open_source(cfg.db_path)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO tool_metrics (metric, value) VALUES ('x', 1)")
    finally:
        conn.close()


def test_committed_but_uncheckpointed_wal_rows_are_in_the_copy(cfg):
    """A5 puts the database in WAL so the review UI writes while the pipeline runs.

    A nightly backup will therefore often run against a live connection with
    outstanding WAL frames. MEASURED that they are captured — if they were not,
    this module would produce a file that looks like a backup and is missing the
    newest ratings, which is worse than no backup at all.
    """
    writer = db.connect(cfg.db_path)
    writer.execute("PRAGMA wal_autocheckpoint=0")  # never checkpoint behind us
    try:
        with db.transaction(writer):
            for i in range(40):
                writer.execute(
                    "INSERT INTO tool_metrics (metric, value) VALUES (?, ?)",
                    ("uncheckpointed", float(i)),
                )
        wal = cfg.db_path.parent / (cfg.db_path.name + "-wal")
        assert wal.exists() and wal.stat().st_size > 0, "no WAL to test against"

        result = backup.take(cfg)
    finally:
        writer.close()

    check = backup.verify(result.path)
    assert check.ok, check.problems
    assert check.tables["tool_metrics"] == result.manifest["tables"]["tool_metrics"]
    # And the rows are the ones that were only in the WAL.
    assert check.tables["tool_metrics"] >= 40


def test_the_manifest_describes_the_copy_not_the_source(cfg, monkeypatch):
    """A5 puts the database in WAL so the review UI writes while other things run.

    Reading the row counts from the source and then making the copy leaves a gap:
    anything committed in between is in the copy and not in the manifest, and
    `--verify` then reports a problem on a backup that is perfectly good. A 04:00
    job that cries wolf is a check that stops being read, which is the whole
    failure §13.3 exists to prevent.

    It cannot be closed with a transaction — SQLite refuses to VACUUM inside one
    — so the manifest is read from the copy instead, where it is true by
    construction. This test drives a writer straight into the gap.
    """
    real = backup.vacuum_into

    def racing(source, destination):
        writer = db.open_db(source, migrate_to_latest=False)
        try:
            with db.transaction(writer):
                writer.execute(
                    "INSERT INTO tool_metrics (metric, value) VALUES ('raced', 1)"
                )
        finally:
            writer.close()
        return real(source, destination)

    monkeypatch.setattr(backup, "vacuum_into", racing)
    result = backup.take(cfg)

    check = backup.verify(result.path)
    assert check.ok, check.problems
    assert check.tables == result.manifest["tables"]


def test_a_second_backup_on_the_same_day_is_refused_and_force_replaces_it(cfg):
    """§13.2's own script fails here.

    Its filename is `clipforge_$(date +%F).db` and MEASURED, `VACUUM INTO`
    raises `output file already exists` on a non-empty target — so running the
    spec's snippet twice in a day fails at the moment someone runs it by hand to
    be safe. Refusing explicitly, with a way through, is the difference.
    """
    first = backup.take(cfg)
    with pytest.raises(backup.BackupError, match="already exists"):
        backup.take(cfg)

    # Add a row so the replacement is distinguishable from the original.
    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    with db.transaction(conn):
        conn.execute("INSERT INTO tool_metrics (metric, value) VALUES ('after', 1)")
    conn.close()

    second = backup.take(cfg, force=True)
    assert second.path == first.path
    assert second.manifest["tables"]["tool_metrics"] > first.manifest["tables"]["tool_metrics"]


def test_a_zero_byte_leftover_does_not_break_the_backup(cfg):
    """MEASURED: `VACUUM INTO` refuses a non-empty target but ACCEPTS a
    zero-byte one, so a crashed run's leftover temp is not self-protecting.
    `atomic_output` unlinks first, which is what makes this safe."""
    where = backup.directory(cfg)
    where.mkdir(parents=True, exist_ok=True)
    day = date_cls.today()
    from clipforge.pipeline import atomic

    atomic.temp_path_for(
        where / f"{backup.STEM_PREFIX}{day.isoformat()}.vacuum.db"
    ).write_bytes(b"")
    atomic.temp_path_for(where / backup.filename(day, True)).write_bytes(b"")

    result = backup.take(cfg)
    assert backup.verify(result.path).ok


# --------------------------------------------------------------------------
# the round trip (§13.3)
# --------------------------------------------------------------------------


def test_backup_then_verify_round_trip(cfg):
    result = backup.take(cfg)

    assert result.path.exists()
    assert result.sidecar.exists()
    assert result.path.name.endswith(".db.gz")

    check = backup.verify(result.path)
    assert check.ok, check.problems
    assert check.problems == []
    # Every table, compared against the manifest recorded at copy time.
    assert check.tables == result.manifest["tables"]
    assert check.schema_version == db.latest_version()


def test_verify_compares_against_the_manifest_not_the_live_database(cfg):
    """An old backup is legitimately behind the live database.

    Comparing the two would fail for an honest reason and train the operator to
    ignore the check. The manifest is the fixed point.
    """
    result = backup.take(cfg)

    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    with db.transaction(conn):
        for i in range(20):
            conn.execute(
                "INSERT INTO tool_metrics (metric, value) VALUES (?, ?)", ("later", float(i))
            )
    live = conn.execute("SELECT count(*) FROM tool_metrics").fetchone()[0]
    conn.close()

    check = backup.verify(result.path)
    assert check.ok, check.problems
    assert check.tables["tool_metrics"] < live


def test_verify_fails_on_a_truncated_archive(cfg):
    result = backup.take(cfg)
    raw = result.path.read_bytes()
    result.path.write_bytes(raw[: len(raw) // 2])

    check = backup.verify(result.path)
    assert not check.ok
    assert any(result.path.name in p for p in check.problems)


def test_verify_fails_when_rows_are_missing(cfg):
    """The check that actually matters: a file that opens fine and is short.

    Rebuild the archive from a database with rows deleted, leaving the original
    manifest in place. Integrity and schema both pass; the row counts do not.
    """
    result = backup.take(cfg)

    tampered = cfg.db_path.parent / "tampered.db"
    conn = db.open_db(tampered)
    _populate(conn, streams=1, metrics=1)
    conn.close()
    with open(tampered, "rb") as raw, gzip.open(result.path, "wb") as out:
        out.write(raw.read())

    check = backup.verify(result.path)
    assert not check.ok
    assert any("manifest recorded" in p for p in check.problems)


def test_verify_fails_on_a_schema_version_this_build_does_not_expect(cfg):
    """§13.3 is "point the application at it", and this is what that catches.

    `db.open_db(migrate_to_latest=False)` already refuses an unexpected version.
    Reusing it means the restore test checks the same thing the app checks.
    """
    result = backup.take(cfg)

    future = cfg.db_path.parent / "future.db"
    conn = db.open_db(future)
    conn.execute(f"PRAGMA user_version={db.latest_version() + 5}")
    conn.close()
    with open(future, "rb") as raw, gzip.open(result.path, "wb") as out:
        out.write(raw.read())

    check = backup.verify(result.path)
    assert not check.ok
    assert any("schema version" in p for p in check.problems)


def test_verify_without_a_manifest_still_checks_what_it_can(cfg):
    result = backup.take(cfg)
    result.sidecar.unlink()

    check = backup.verify(result.path)
    assert check.ok
    assert check.expected is None
    assert any("no manifest" in n for n in check.notes)


def test_an_uncompressed_backup_round_trips(tmp_path):
    made = _cfg(tmp_path, **{"backup.compress": "false"})
    conn = db.open_db(made.db_path)
    _populate(conn)
    conn.close()

    result = backup.take(made)
    assert result.path.name.endswith(".db")
    assert not result.path.name.endswith(".gz")
    assert backup.verify(result.path).ok


# --------------------------------------------------------------------------
# retention (§13.2: 30 daily + 12 monthly)
# --------------------------------------------------------------------------


def _fake_backups(where, days: int, today: date_cls | None = None) -> list:
    where.mkdir(parents=True, exist_ok=True)
    today = today or date_cls.today()
    made = []
    for offset in range(days):
        day = today - timedelta(days=offset)
        path = where / backup.filename(day, True)
        path.write_bytes(b"x" * 16)
        backup.sidecar_for(path).write_text("{}", encoding="utf-8")
        made.append(path)
    return made


def test_retention_keeps_the_configured_daily_and_monthly_counts(cfg):
    """Read from config, never from literals — the rule CLAUDE.md sets for every
    numeric assertion in this project."""
    keep_daily = int(cfg.get("backup.keep_daily"))
    keep_monthly = int(cfg.get("backup.keep_monthly"))
    where = backup.directory(cfg)
    _fake_backups(where, days=400)

    entries = backup.list_backups(where)
    assert len(entries) == 400

    doomed = backup.plan_prune(entries, keep_daily, keep_monthly)
    survivors = [e for e in entries if e not in doomed]

    assert len(survivors) <= keep_daily + keep_monthly
    # The newest `keep_daily` are all present.
    assert [e.day for e in survivors[:keep_daily]] == [e.day for e in entries[:keep_daily]]
    # What survives beyond them is at most one per month.
    months = [(e.day.year, e.day.month) for e in survivors[keep_daily:]]
    assert len(months) == len(set(months))
    assert len(months) <= keep_monthly


def test_the_newest_backup_is_never_prunable(cfg):
    where = backup.directory(cfg)
    _fake_backups(where, days=5)
    entries = backup.list_backups(where)

    # Even with settings that say to keep nothing.
    doomed = backup.plan_prune(entries, keep_daily=0, keep_monthly=0)
    assert entries[0] not in doomed
    assert len(doomed) == len(entries) - 1


def test_prune_never_touches_a_file_it_cannot_name(cfg):
    where = backup.directory(cfg)
    _fake_backups(where, days=3)
    stranger = where / "important-notes.txt"
    stranger.write_text("do not delete me", encoding="utf-8")
    odd = where / "clipforge_2026-02-30.db.gz"  # right shape, not a date
    odd.write_bytes(b"x")

    backup.take(cfg, force=True)

    assert stranger.exists()
    assert odd.exists()


def test_pruning_removes_the_manifest_alongside(cfg):
    where = backup.directory(cfg)
    _fake_backups(where, days=200)
    before = backup.list_backups(where)

    result = backup.take(cfg, force=True)
    assert result.pruned

    for path in result.pruned:
        assert not path.exists()
        assert not backup.sidecar_for(path).exists()
    assert len(backup.list_backups(where)) < len(before)


def test_no_prune_keeps_everything(cfg):
    where = backup.directory(cfg)
    _fake_backups(where, days=200)
    result = backup.take(cfg, force=True, prune=False)
    assert result.pruned == []
    assert len(backup.list_backups(where)) == 200


# --------------------------------------------------------------------------
# mirroring, instrumentation, dry run
# --------------------------------------------------------------------------


def test_the_mirror_receives_the_backup_and_its_manifest(tmp_path):
    mirror = tmp_path / "elsewhere"
    made = _cfg(tmp_path, **{"backup.mirror_dir": mirror.as_posix()})
    conn = db.open_db(made.db_path)
    _populate(conn)
    conn.close()

    result = backup.take(made)
    assert result.mirrored is not None
    assert (mirror / result.path.name).exists()
    assert (mirror / result.sidecar.name).exists()
    assert backup.verify(mirror / result.path.name).ok


def test_a_failing_mirror_warns_and_does_not_lose_the_backup(tmp_path):
    """An unplugged external drive must not mean no backup at all."""
    blocked = tmp_path / "not-a-dir"
    blocked.write_text("I am a file", encoding="utf-8")
    made = _cfg(tmp_path, **{"backup.mirror_dir": blocked.as_posix()})
    conn = db.open_db(made.db_path)
    _populate(conn)
    conn.close()

    result = backup.take(made)
    assert result.path.exists()
    assert result.mirrored is None
    assert any("mirror" in w for w in result.warnings)
    assert backup.verify(result.path).ok


def test_the_metric_row_lands(cfg):
    result = backup.take(cfg)

    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    try:
        rows = conn.execute(
            "SELECT value, meta FROM tool_metrics WHERE metric = ?", (backup.METRIC,)
        ).fetchall()
    finally:
        conn.close()

    assert len(rows) == 1
    assert rows[0]["value"] >= 0
    meta = json.loads(rows[0]["meta"])
    assert meta["path"] == result.path.name
    assert meta["stored_bytes"] == result.stored_bytes
    assert meta["schema_version"] == db.latest_version()


def test_a_dry_run_writes_nothing(cfg):
    where = backup.directory(cfg)
    _fake_backups(where, days=200)
    before = {p.name for p in where.iterdir()}

    result = backup.take(cfg, dry_run=True)

    assert result.dry_run
    assert result.pruned, "a 200-day directory has something to prune"
    assert {p.name for p in where.iterdir()} == before
    assert result.manifest["tables"]
    # Today's backup is among the 200 fakes, so a real run would refuse. A dry
    # run must say that rather than raise — the day a backup already exists is
    # the day you most want to ask what a run would do.
    assert any("--force" in w for w in result.warnings)


def test_the_gzip_header_carries_no_name_and_no_clock(cfg):
    """Two backups of identical content must be identical bytes.

    gzip stamps the source filename and the wall clock into its header by
    default, so without pinning them every nightly differs from the last even
    when the database did not — which defeats any mirror that deduplicates and
    makes comparing two backups impossible.

    Asserted on the header rather than on two whole files, because two whole
    files legitimately differ: `take` appends a metric row to the source between
    them, so the second backup really does have one row more.
    """
    first = backup.take(cfg)
    header = first.path.read_bytes()[:10]

    flags = header[3]
    mtime = int.from_bytes(header[4:8], "little")
    assert flags & 0x08 == 0, "FNAME is set — the temp file's name is in the header"
    assert mtime == 0, "MTIME is set — every nightly would differ from the last"

    second = backup.take(cfg, force=True)
    assert second.path.read_bytes()[:10] == header


# --------------------------------------------------------------------------
# naming, listing, and the printed schedule
# --------------------------------------------------------------------------


def test_backup_names_carry_their_date_and_pair_with_a_manifest():
    day = date_cls(2026, 8, 14)
    assert backup.filename(day, compress=True) == "clipforge_2026-08-14.db.gz"
    assert backup.filename(day, compress=False) == "clipforge_2026-08-14.db"
    assert backup.sidecar_for(
        Path("x/clipforge_2026-08-14.db.gz")
    ).name == "clipforge_2026-08-14.json"
    assert backup.sidecar_for(
        Path("x/clipforge_2026-08-14.db")
    ).name == "clipforge_2026-08-14.json"


def test_a_read_only_uri_survives_an_awkward_path():
    """`?` and `#` are URI syntax and appear in real filenames."""
    uri = backup.read_only_uri(Path(r"C:\Users\O'Brien\a #1 (why?)\clipforge.db"))
    assert uri.startswith("file:C:/")
    assert uri.endswith("?mode=ro")
    assert "%23" in uri and "%3F" in uri
    assert uri.count("?") == 1


def test_list_backups_is_newest_first(cfg):
    where = backup.directory(cfg)
    _fake_backups(where, days=4)
    entries = backup.list_backups(where)
    assert [e.day for e in entries] == sorted((e.day for e in entries), reverse=True)
    assert entries[0].age_days() == 0


def test_the_schedule_command_is_printed_not_run(cfg, capsys):
    """Creating a scheduled task is a standing change to the machine."""
    command = backup.schedule_command(cfg)
    assert command.startswith("schtasks /Create")
    # Through the launcher, which Push-Locations to the project root — a task
    # run from System32 would otherwise resolve ./data somewhere else.
    assert "clipforge.ps1" in command
    assert str(cfg.get("backup.schedule_time")) in command


# --------------------------------------------------------------------------
# CLI wiring
# --------------------------------------------------------------------------


def test_the_command_is_registered_and_takes_config_flags():
    assert "backup" in cli.COMMANDS
    parser = cli.build_parser("backup")
    args = parser.parse_args(["backup", "--set", "backup.keep_daily=3"])
    assert args.overrides == ["backup.keep_daily=3"]


def test_the_cli_takes_lists_and_verifies_a_backup(cfg, capsys, monkeypatch):
    from clipforge.db import cmd_backup

    parser = cli.build_parser("backup")
    flags = [
        f"paths.data_root={cfg.data_root.as_posix()}",
        f"paths.db={cfg.db_path.as_posix()}",
        f"backup.dir={backup.directory(cfg).as_posix()}",
    ]

    def run(extra):
        argv = ["backup"]
        for flag in flags:
            argv += ["--set", flag]
        return cmd_backup.main(parser.parse_args(argv + extra))

    assert run([]) == 0
    assert "backup" in capsys.readouterr().out

    assert run(["--list"]) == 0
    listed = capsys.readouterr().out
    assert "clipforge_" in listed and "retention" in listed

    assert run(["--verify"]) == 0
    assert "§13.3 satisfied" in capsys.readouterr().out

    # A same-day repeat is a refusal with a non-zero exit, not a traceback.
    assert run([]) == 1
    assert "--force" in capsys.readouterr().out


def test_the_cli_reports_a_bad_backup_with_a_failing_exit_code(cfg, capsys):
    from clipforge.db import cmd_backup

    result = backup.take(cfg)
    result.path.write_bytes(b"not a gzip file at all")

    parser = cli.build_parser("backup")
    argv = ["backup", "--verify", str(result.path)]
    for flag in (f"paths.db={cfg.db_path.as_posix()}",
                 f"backup.dir={backup.directory(cfg).as_posix()}"):
        argv += ["--set", flag]

    assert cmd_backup.main(parser.parse_args(argv)) == 1
    assert "NOT restorable" in capsys.readouterr().out


def test_output_survives_a_codepage_that_cannot_hold_a_section_sign(cfg, tmp_path):
    """MEASURED: this is the one command that runs unattended.

    Task Scheduler redirects stdout to a file, Python then encodes it with the
    locale codepage, and on one that cannot hold `§` or an em dash every print
    raised — *after* the backup was written. `--verify` verified a good file and
    then died reporting that it had, exiting 1. A nightly job that says it
    failed when it succeeded is worse than one that fails.
    """
    import subprocess

    env = dict(os.environ, PYTHONIOENCODING="ascii")
    # conftest's hermetic_config only reaches this process; a subprocess would
    # still pick up whatever the developer has in $CLIPFORGE_CONFIG.
    env.pop(config.ENV_CONFIG, None)
    flags = [
        "--set", f"paths.db={cfg.db_path.as_posix()}",
        "--set", f"backup.dir={backup.directory(cfg).as_posix()}",
    ]
    base = [sys.executable, "-m", "clipforge.cli", "backup"]

    for extra in ([], ["--verify"], ["--list"], ["--schedule"], []):
        done = subprocess.run(  # noqa: S603
            base + flags + extra, env=env, capture_output=True, text=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        assert "UnicodeEncodeError" not in done.stderr, (extra, done.stderr[-400:])

    # The last entry repeats a same-day backup: a refusal, exit 1, still no
    # traceback — the error message carries a § too.
    done = subprocess.run(  # noqa: S603
        base + flags, env=env, capture_output=True, text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )
    assert done.returncode == 1
    assert "already exists" in done.stdout
    assert "Traceback" not in done.stderr


def test_doctor_reports_the_newest_backup(cfg):
    from clipforge import doctor

    absent = doctor._check_backups(cfg)
    assert not absent.ok and not absent.required
    assert "none in" in absent.detail

    backup.take(cfg)
    present = doctor._check_backups(cfg)
    assert present.ok
    assert "today" in present.detail


def test_doctor_warns_when_the_newest_backup_is_stale(cfg):
    from clipforge import doctor

    where = backup.directory(cfg)
    where.mkdir(parents=True, exist_ok=True)
    old = where / backup.filename(date_cls.today() - timedelta(days=30), True)
    old.write_bytes(b"x")

    check = doctor._check_backups(cfg)
    assert not check.ok
    assert "may have stopped" in check.detail
