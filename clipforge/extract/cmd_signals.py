"""`clipforge signals` — look at what extraction actually produced.

Scoring consumes these arrays and emits candidates; when the candidates look
wrong, the first question is always whether the signal underneath them is
wrong. Without this you would be reading BLOBs out of SQLite by hand.
"""

from __future__ import annotations

import sqlite3

from clipforge import config, db, signals


def add_arguments(parser) -> None:
    parser.add_argument("stream_id")
    parser.add_argument("--kind", help="limit to one signal, e.g. mic_rms")
    parser.add_argument(
        "--at", type=float, metavar="SECONDS",
        help="print the value at this timestamp instead of summary stats",
    )
    parser.add_argument(
        "--window", type=float, default=0.0, metavar="SECONDS",
        help="with --at, summarise +/- this many seconds around it",
    )
    parser.add_argument("--params", action="store_true", help="show extraction parameters")
    config.add_config_arguments(parser)


def main(args) -> int:
    cfg = config.from_args(args)
    if not cfg.db_path.exists():
        print(f"no database at {cfg.db_path}. Run `clipforge db init`.")
        return 1

    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    try:
        if conn.execute(
            "SELECT 1 FROM streams WHERE id = ?", (args.stream_id,)
        ).fetchone() is None:
            print(f"no stream {args.stream_id!r}")
            return 1

        series = signals.load_all(conn, args.stream_id)
        if args.kind:
            series = {k: v for k, v in series.items() if k == args.kind}
            if not series:
                available = signals.kinds(conn, args.stream_id)
                print(f"no signal {args.kind!r}. Available: {', '.join(available) or 'none'}")
                return 1

        if not series:
            print(f"{args.stream_id} has no signals yet. Run `clipforge run {args.stream_id}`.")
            return 1

        if args.at is not None:
            return _at(series, args.at, args.window)
        return _summary(conn, args.stream_id, series, args.params)
    finally:
        conn.close()


def _summary(conn: sqlite3.Connection, stream_id: str, series: dict, show_params: bool) -> int:
    duration = conn.execute(
        "SELECT duration_s FROM streams WHERE id = ?", (stream_id,)
    ).fetchone()[0]

    width = max(len(k) for k in series)
    print(f"{'signal'.ljust(width)}  {'samples':>8} {'rate':>6} {'t0':>6}  "
          f"{'min':>7} {'p05':>7} {'med':>7} {'p95':>7} {'max':>7}")
    for kind, data in series.items():
        stats = signals.summarize(data)
        if stats["n"] == 0:
            print(f"{kind.ljust(width)}  {'0':>8}  (empty)")
            continue
        print(
            f"{kind.ljust(width)}  {stats['n']:>8} {data.sample_rate_hz:>5g}H "
            f"{data.t0:>6.3f}  {stats['min']:>7.1f} {stats['p05']:>7.1f} "
            f"{stats['median']:>7.1f} {stats['p95']:>7.1f} {stats['max']:>7.1f}"
        )

    # Coverage: a series noticeably shorter than the stream means extraction
    # stopped early, which otherwise shows up much later as candidates that
    # stop appearing partway through.
    if duration:
        print()
        for kind, data in series.items():
            if len(data) == 0:
                continue
            covered = data.t0 + data.duration_s
            gap = duration - covered
            note = "" if abs(gap) < 1.0 else f"   <-- {gap:+.1f}s vs stream duration"
            print(f"  {kind.ljust(width)} covers {covered:.2f}s of {duration:.2f}s{note}")

    if show_params:
        print()
        for kind, data in series.items():
            print(f"  {kind}")
            for key, value in sorted(data.params.items()):
                print(f"    {key:<20} {value}")
    return 0


def _at(series: dict, t: float, window: float) -> int:
    import numpy as np

    width = max(len(k) for k in series)
    if window <= 0:
        print(f"at t={t:.3f}s")
        for kind, data in series.items():
            index = data.index_at(t)
            print(f"  {kind.ljust(width)} {data.values[index]:>8.2f} dB "
                  f"(sample {index}, centred {data.time_of(index):.3f}s)")
        return 0

    print(f"over t={t - window:.3f}..{t + window:.3f}s")
    for kind, data in series.items():
        chunk = data.slice(t - window, t + window)
        if chunk.size == 0:
            print(f"  {kind.ljust(width)} (no samples in range)")
            continue
        values = chunk.astype(np.float64)
        print(
            f"  {kind.ljust(width)} n={values.size:<5} "
            f"min {values.min():>7.2f}  median {np.median(values):>7.2f}  "
            f"max {values.max():>7.2f} dB"
        )
    return 0
