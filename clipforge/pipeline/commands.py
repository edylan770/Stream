"""`clipforge run` and `clipforge status`."""

from __future__ import annotations

import json
import sqlite3

from clipforge import config, db
from clipforge.pipeline import runner, stages

#: How each decision renders. The marker column is what you scan.
MARKERS = {
    runner.RUN: "RUN ",
    runner.SKIP: "skip",
    runner.BLOCKED: "BLOCK",
    runner.DEFERRED: "  - ",
    runner.EXTERNAL_DONE: "done",
}


def _open(args) -> tuple[config.Config, sqlite3.Connection]:
    cfg = config.from_args(args)
    if not cfg.db_path.exists():
        raise SystemExit(f"no database at {cfg.db_path}. Run `clipforge db init`.")
    return cfg, db.open_db(cfg.db_path, migrate_to_latest=False)


def _require_stream(conn: sqlite3.Connection, stream_id: str) -> None:
    row = conn.execute("SELECT id FROM streams WHERE id = ?", (stream_id,)).fetchone()
    if row is None:
        known = [
            r["id"] for r in conn.execute("SELECT id FROM streams ORDER BY id LIMIT 20")
        ]
        raise SystemExit(
            f"no stream {stream_id!r}."
            + (f" Known: {', '.join(known)}" if known else " Register one first.")
        )


def _print_plan(decisions: list[runner.Decision]) -> None:
    width = max(len(d.stage) for d in decisions)
    print(f"  {'':5} {'stage'.ljust(width)}  reason")
    for decision in decisions:
        marker = MARKERS.get(decision.action, decision.action)
        print(f"  {marker:5} {decision.stage.ljust(width)}  {decision.reason}")


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


#: The CLI dispatcher maps one command name to one module exposing
#: add_arguments/main, so `run` and `status` get thin adapter modules
#: (clipforge/pipeline/cmd_run.py, cmd_status.py) that point back here.
def add_run_arguments(parser) -> None:
    parser.add_argument("stream_id")
    config.add_config_arguments(parser)
    parser.add_argument("--only", metavar="STAGE", help="run just this stage")
    parser.add_argument("--from", dest="from_stage", metavar="STAGE",
                        help="run this stage and everything downstream of it")
    parser.add_argument("--force", metavar="STAGE", action="append",
                        help="re-run STAGE even if up to date; 'all' forces everything")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and exit without running anything")
    parser.add_argument("--no-cascade", action="store_true",
                        help="do not re-run stages whose dependencies re-ran")


def run_main(args) -> int:
    cfg, conn = _open(args)
    try:
        _require_stream(conn, args.stream_id)

        wanted, force = _resolve_selection(args)
        engine = runner.Runner(
            cfg=cfg, conn=conn, stream_id=args.stream_id,
            force=force, cascade=not args.no_cascade,
        )

        print(f"stream  {args.stream_id}")
        print(f"config  {cfg.version}")
        print()

        reclaimed = engine.reclaim_crashed()
        if reclaimed:
            print(f"  reclaimed {len(reclaimed)} stage(s) left running by a dead process: "
                  f"{', '.join(reclaimed)}")
        swept = engine.sweep()
        if swept:
            print(f"  swept {len(swept)} orphaned temp file(s) from an interrupted run")
        if reclaimed or swept:
            print()

        decisions = engine.plan(wanted)
        _print_plan(decisions)
        print()

        if args.dry_run:
            planned = sum(1 for d in decisions if d.will_run)
            print(f"dry run: {planned} stage(s) would run")
            return 0

        blocked = [d for d in decisions if d.action == runner.BLOCKED]
        if blocked and not any(d.will_run for d in decisions):
            print(f"nothing to do. {blocked[0].stage}: {blocked[0].reason}")
            return 1

        results = engine.execute(decisions)
        print()

        failed = [r for r in results if r.status == "failed"]
        ran = [r for r in results if r.status == "done"]
        total = sum(r.duration_s for r in results)
        print(f"{len(ran)} stage(s) in {total:.1f}s"
              + (f", {len(failed)} failed" if failed else ""))
        return 1 if failed else 0
    finally:
        conn.close()


def _resolve_selection(args) -> tuple[list[str] | None, set[str]]:
    """Turn --only/--from/--force into a stage set and a force set."""
    force: set[str] = set()
    for name in args.force or []:
        if name == "all":
            force |= set(stages.STAGES)
        else:
            force.add(stages.get(name).name)

    if args.only:
        spec = stages.get(args.only)
        # Dependencies are included so the plan can report them, but they are
        # not force-run: --only means "this stage", and an unmet dependency
        # should say so rather than quietly pulling in a four-hour encode.
        return sorted(stages.ancestors(spec.name) | {spec.name}), force

    if args.from_stage:
        spec = stages.get(args.from_stage)
        wanted = stages.ancestors(spec.name) | {spec.name} | stages.descendants(spec.name)
        force.add(spec.name)
        return sorted(wanted), force

    return None, force


# --------------------------------------------------------------------------
# status
# --------------------------------------------------------------------------


def add_status_arguments(parser) -> None:
    parser.add_argument("stream_id", nargs="?", help="omit to list all streams")
    config.add_config_arguments(parser)
    parser.add_argument("--all", action="store_true",
                        help="include stages belonging to phases that are not built")


def status_main(args) -> int:
    cfg, conn = _open(args)
    try:
        if not args.stream_id:
            return _list_streams(conn)

        _require_stream(conn, args.stream_id)
        stream = conn.execute(
            "SELECT * FROM streams WHERE id = ?", (args.stream_id,)
        ).fetchone()

        print(f"stream      {stream['id']}")
        print(f"master      {stream['master_path']}")
        duration = stream["duration_s"]
        print(f"duration    {duration:.1f}s" if duration else "duration    (not probed)")
        if stream["resolution"]:
            fps = f"{stream['fps_num']}/{stream['fps_den']}" if stream["fps_num"] else stream["fps"]
            vfr = "  VFR" if stream["is_vfr"] else ""
            print(f"video       {stream['resolution']} @ {fps}{vfr}")
        if stream["audio_track_map"]:
            _print_track_map(stream["audio_track_map"])
        print(f"markers     {stream['marker_time_base']} time base"
              + (f", anchor {stream['record_start_epoch_ms']}"
                 if stream["record_start_epoch_ms"] else ", no anchor"))
        print()

        engine = runner.Runner(cfg=cfg, conn=conn, stream_id=args.stream_id)
        decisions = {d.stage: d for d in engine.plan()}

        def hidden_by_default(spec) -> bool:
            # A phase-1 stage that is not built yet is imminent and worth
            # showing; a phase-5 stage is not. Collapsing both into "deferred"
            # and hiding them together would misreport how far along this is.
            return decisions[spec.name].action == runner.DEFERRED and spec.phase > 1

        rows = []
        for spec in stages.iter_specs():
            if not args.all and hidden_by_default(spec):
                continue
            row = engine.row(spec.name)
            when = row["finished_at"] if row is not None and row["finished_at"] else ""
            rows.append((spec, decisions[spec.name], row, when))

        width = max(len(spec.name) for spec, _, _, _ in rows)
        print(f"  {'':6} {'stage'.ljust(width)}  {'state':<8} {'finished':<20} next run")
        for spec, decision, row, when in rows:
            state = row["status"] if row is not None else "-"
            marker = MARKERS.get(decision.action, decision.action)
            print(f"  {marker:6} {spec.name.ljust(width)}  {state:<8} {when:<20} "
                  f"{decision.reason}")

        hidden = sum(1 for s in stages.iter_specs() if hidden_by_default(s))
        if hidden and not args.all:
            print(f"\n  {hidden} stage(s) from phases 2+ hidden; --all to show")
        _print_metrics(conn, args.stream_id)
        return 0
    finally:
        conn.close()


def _print_track_map(raw: str) -> None:
    data = json.loads(raw)
    roles = data.get("roles", data)
    mapping = ", ".join(f"{role}=a:{index}" for role, index in sorted(roles.items()))
    print(f"audio       {mapping}")
    for warning in data.get("warnings", []):
        print(f"  WARNING   {warning}")


def _print_metrics(conn: sqlite3.Connection, stream_id: str) -> None:
    rows = conn.execute(
        "SELECT value, meta FROM tool_metrics "
        "WHERE stream_id = ? AND metric = 'stage_duration_s' ORDER BY id",
        (stream_id,),
    ).fetchall()
    if not rows:
        return
    total = sum(r["value"] for r in rows)
    print(f"\n  {len(rows)} stage run(s) recorded, {total:.1f}s of processing")


def _list_streams(conn: sqlite3.Connection) -> int:
    rows = conn.execute(
        """
        SELECT s.id, s.date, s.duration_s,
               (SELECT COUNT(*) FROM pipeline_stages p
                 WHERE p.stream_id = s.id AND p.status = 'done') AS done
          FROM streams s ORDER BY s.date DESC, s.id
        """
    ).fetchall()
    if not rows:
        print("no streams registered. `clipforge register --master <file>`")
        return 0
    width = max(len(r["id"]) for r in rows)
    print(f"  {'stream'.ljust(width)}  {'date':<12} {'duration':>10}  stages done")
    for row in rows:
        duration = f"{row['duration_s']:.0f}s" if row["duration_s"] else "-"
        print(f"  {row['id'].ljust(width)}  {row['date']:<12} {duration:>10}  {row['done']}")
    return 0
