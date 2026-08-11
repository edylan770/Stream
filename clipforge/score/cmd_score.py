"""`clipforge score` — re-score a stream without touching extraction.

§6.1: *"Scoring is a pure function over stored signals. It must be re-runnable
across the entire back catalog in seconds."* The pipeline's `run` correctly
skips a stage whose parameters have not changed; this command exists because
"re-run it anyway" is the whole point of the extraction/scoring boundary, and
having to remember `--force score` would put friction on the one operation the
architecture is built to make free.
"""

from __future__ import annotations

from clipforge import config, db
from clipforge.pipeline import runner, stages


def add_arguments(parser) -> None:
    parser.add_argument("stream_id", nargs="?", help="omit to score every stream")
    config.add_config_arguments(parser)
    parser.add_argument(
        "--list", action="store_true", help="show the resulting candidates"
    )


def main(args) -> int:
    cfg = config.from_args(args)
    if not cfg.db_path.exists():
        print(f"no database at {cfg.db_path}. Run `clipforge db init`.")
        return 1

    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    try:
        if args.stream_id:
            targets = [args.stream_id]
            if conn.execute(
                "SELECT 1 FROM streams WHERE id = ?", (args.stream_id,)
            ).fetchone() is None:
                print(f"no stream {args.stream_id!r}")
                return 1
        else:
            targets = [r["id"] for r in conn.execute("SELECT id FROM streams ORDER BY date, id")]
            if not targets:
                print("no streams registered")
                return 0

        print(f"profile  {cfg.profile.name}")
        print(f"config   {cfg.version}")
        print()

        failures = 0
        for stream_id in targets:
            print(f"{stream_id}")
            engine = runner.Runner(
                cfg=cfg, conn=conn, stream_id=stream_id,
                force={"score"}, log=lambda m: print(m),
            )
            engine.reclaim_crashed()

            decisions = engine.plan()
            blocked = next(
                (d for d in decisions if d.stage == "score" and d.action == runner.BLOCKED), None
            )
            if blocked is not None:
                print(f"  cannot score: {blocked.reason}")
                failures += 1
                continue

            results = engine.execute([d for d in decisions if d.stage == "score"])
            if results and results[0].status == "failed":
                failures += 1
            elif args.list:
                _list(conn, stream_id)
            print()

        return 1 if failures else 0
    finally:
        conn.close()


def _list(conn, stream_id: str) -> None:
    rows = conn.execute(
        "SELECT t_start, t_end, t_peak, score_combined, contributing_signals "
        "FROM candidates WHERE stream_id = ? AND is_current = 1 "
        "ORDER BY score_combined DESC",
        (stream_id,),
    ).fetchall()
    if not rows:
        return

    import json

    print(f"    {'#':>3}  {'window':<18} {'peak':>8} {'score':>8}  top contributors")
    for index, row in enumerate(rows, start=1):
        signals_json = json.loads(row["contributing_signals"] or "{}")
        top = [
            f"{name} {value:+.2f}"
            for name, value in list(signals_json.items())[:3]
            if not name.startswith("_")
        ]
        window = f"{row['t_start']:.1f}-{row['t_end']:.1f}s"
        print(f"    {index:>3}  {window:<18} {row['t_peak']:>7.1f}s "
              f"{row['score_combined']:>8.3f}  {', '.join(top)}")
