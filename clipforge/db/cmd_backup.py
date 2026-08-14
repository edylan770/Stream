"""`clipforge backup` — §13.2's nightly job, and §13.3's restore test.

A top-level command rather than a `clipforge db` action, because the thing that
types it most is Windows Task Scheduler and `clipforge backup` is what goes in
that box. The logic lives in `db/backup.py` so the tests never shell out — the
same library-core/CLI split `ingest/register.py` uses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from clipforge import config
from clipforge.db import backup


def _survive_a_redirected_stdout() -> None:
    """Never let a §-sign decide whether the nightly backup reports success.

    MEASURED, and the reason this exists. Every other command here is typed by a
    person at a console; this one is run by Task Scheduler with its output
    redirected to a file, and Python then encodes stdout with the locale's code
    page rather than UTF-8. On a codepage that cannot hold `§` or an em dash,
    `print` raises UnicodeEncodeError — **after** the backup has been written —
    so the job exits 1 and the operator is told a successful backup failed.

    (On this machine the piped default is cp1252, which holds both, so it does
    not fire here. It fires on a machine whose locale differs, or under any
    PYTHONIOENCODING that is stricter — measured with `ascii`, where `--verify`
    verified the file correctly and then died printing that it had.)

    `errors="replace"` costs a `?` in place of a section sign in a log file, and
    buys an exit code that means what it says.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError, OSError):
            # Not a reconfigurable text stream — pytest's capture, a pipe
            # someone replaced. Nothing to do, and nothing worth failing over.
            pass


def add_arguments(parser) -> None:
    config.add_config_arguments(parser)
    parser.add_argument(
        "--list", action="store_true", dest="do_list",
        help="show existing backups, newest first",
    )
    parser.add_argument(
        "--verify", nargs="?", const="", metavar="PATH",
        help="§13.3: restore a backup to a scratch directory and check it "
             "(default: the newest)",
    )
    parser.add_argument(
        "--schedule", action="store_true",
        help="print the schtasks line for a nightly run (does not create it)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="replace today's backup if one already exists",
    )
    parser.add_argument(
        "--no-prune", action="store_true",
        help="keep every backup, ignoring backup.keep_daily / keep_monthly",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="say what would be written and what retention would delete",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")


def main(args) -> int:
    _survive_a_redirected_stdout()
    cfg = config.from_args(args)
    where = backup.directory(cfg)

    if args.schedule:
        return _schedule(cfg)
    if args.do_list:
        return _list(cfg, where, args.json)
    if args.verify is not None:
        return _verify(cfg, where, args.verify, args.json)

    try:
        result = backup.take(
            cfg, force=args.force, prune=not args.no_prune, dry_run=args.dry_run
        )
    except backup.BackupError as exc:
        print(f"backup failed: {exc}")
        return 1

    if args.json:
        print(json.dumps({
            "path": str(result.path),
            "sidecar": str(result.sidecar),
            "dry_run": result.dry_run,
            "source_bytes": result.source_bytes,
            "vacuumed_bytes": result.vacuumed_bytes,
            "stored_bytes": result.stored_bytes,
            "duration_s": round(result.duration_s, 4),
            "rows": result.rows,
            "schema_version": result.manifest["schema_version"],
            "mirrored": str(result.mirrored) if result.mirrored else None,
            "pruned": [p.name for p in result.pruned],
            "warnings": result.warnings,
        }, indent=2))
        return 0

    return _report(cfg, result)


# --------------------------------------------------------------------------


def _report(cfg, result: backup.Result) -> int:
    if result.dry_run:
        print(f"would write   {result.path}")
        print(f"              {result.rows} rows across "
              f"{len(result.manifest['tables'])} tables, schema version "
              f"{result.manifest['schema_version']}")
        for path in result.pruned:
            print(f"would delete  {path.name}  (retention)")
        if not result.pruned:
            print("would delete  nothing")
        # Printed here as well as below, because the dry-run branch returns
        # early — and the warning it most often carries is "today's backup
        # already exists", which is the one thing a dry run is asked.
        for warning in result.warnings:
            print(f"warning       {warning}")
        return 0

    saved = 1 - (result.stored_bytes / result.source_bytes) if result.source_bytes else 0
    print(f"backup        {result.path}")
    print(f"              {backup.humanise(result.source_bytes)} live -> "
          f"{backup.humanise(result.stored_bytes)} stored "
          f"({saved:.0%} smaller) in {result.duration_s:.2f}s")
    print(f"contents      {result.rows} rows across {len(result.manifest['tables'])} "
          f"tables, schema version {result.manifest['schema_version']}")

    if result.mirrored:
        same = backup.on_same_volume(result.path, result.mirrored)
        print(f"mirror        {result.mirrored}"
              + ("  — SAME DRIVE as the original, so this is not protection "
                 "against that drive failing" if same else ""))
    else:
        print("mirror        none configured (backup.mirror_dir)")

    for path in result.pruned:
        print(f"pruned        {path.name}")

    for warning in result.warnings:
        print(f"warning       {warning}")

    # Said plainly rather than left implied. §13.2's own plan ends with an
    # upload for exactly this reason.
    if not result.mirrored:
        print("\nThis backup is on the same disk as the database it came from. That "
              "covers a bad migration or a mistaken delete, not the disk dying. "
              "Set backup.mirror_dir to a second drive or a synced folder.")
    return 0


def _list(cfg, where: Path, as_json: bool) -> int:
    entries = backup.list_backups(where)
    if as_json:
        print(json.dumps([{
            "path": str(e.path), "date": e.day.isoformat(),
            "bytes": e.bytes, "age_days": e.age_days(),
            "manifest": e.sidecar.exists(),
        } for e in entries], indent=2))
        return 0

    print(f"directory     {where}")
    if not entries:
        print("\nNo backups yet. Run `clipforge backup`.")
        return 1

    total = sum(e.bytes for e in entries)
    print(f"              {len(entries)} backups, {backup.humanise(total)} total\n")
    for entry in entries:
        age = entry.age_days()
        when = "today" if age == 0 else f"{age}d ago"
        flag = "" if entry.sidecar.exists() else "   (no manifest — row counts unverifiable)"
        print(f"  {entry.path.name:<32} {backup.humanise(entry.bytes):>10}  {when}{flag}")

    keep_daily = int(cfg.get("backup.keep_daily"))
    keep_monthly = int(cfg.get("backup.keep_monthly"))
    doomed = backup.plan_prune(entries, keep_daily, keep_monthly)
    print(f"\nretention     {keep_daily} daily + {keep_monthly} monthly (§13.2)"
          + (f" — {len(doomed)} would be pruned on the next run" if doomed else ""))
    return 0


def _verify(cfg, where: Path, given: str, as_json: bool) -> int:
    if given:
        path = Path(given).expanduser()
        if not path.is_absolute() and not path.exists():
            path = where / given
    else:
        entry = backup.newest(where)
        if entry is None:
            print(f"no backups in {where}. Run `clipforge backup`.")
            return 1
        path = entry.path

    try:
        result = backup.verify(path)
    except backup.BackupError as exc:
        print(f"verify failed: {exc}")
        return 1

    if as_json:
        print(json.dumps({
            "path": str(result.path), "ok": result.ok,
            "schema_version": result.schema_version,
            "restored_bytes": result.restored_bytes, "rows": result.rows,
            "tables": result.tables, "problems": result.problems, "notes": result.notes,
        }, indent=2))
        return 0 if result.ok else 1

    print(f"verifying     {result.path.name}")
    print(f"restored      {backup.humanise(result.restored_bytes)}, "
          f"{result.rows} rows across {len(result.tables)} tables, "
          f"schema version {result.schema_version}")
    for note in result.notes:
        print(f"note          {note}")
    for problem in result.problems:
        print(f"PROBLEM       {problem}")

    if result.ok:
        checked = "" if result.expected is None else ", every row count matches its manifest"
        print(f"\nOK — it decompresses, passes integrity_check, opens with this build's "
              f"schema{checked}. §13.3 satisfied for this file.")
        return 0
    print("\nThis backup is NOT restorable as-is. §13.2 calls this tier "
          "irreplaceable; take a fresh one and find out why this failed.")
    return 1


def _schedule(cfg) -> int:
    command = backup.schedule_command(cfg)
    at = cfg.get("backup.schedule_time", "04:00")
    print("§13.2 asks for a nightly, automated backup. On Windows that is Task")
    print("Scheduler. This command is PRINTED and not run — creating a scheduled")
    print("task is a standing change to the machine and belongs to your keystroke.\n")
    print("Run this once, in a normal (non-elevated) PowerShell:\n")
    print(f"  {command}\n")
    print(f"It runs daily at {at} as you, so it needs you logged in — which is the")
    print("right trade for a home machine. Check it afterwards with:\n")
    print('  schtasks /Query /TN "ClipForge nightly backup"\n')
    print("and remove it with /Delete. To take one right now: clipforge backup")
    return 0
