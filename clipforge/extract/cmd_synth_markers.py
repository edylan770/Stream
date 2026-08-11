"""`clipforge synth-markers` — fabricate marker presses for testing.

§15 Phase 0 (the F1/F2 daemon) is not built, and Phase 1 has to be exercisable
without it. This writes JSONL in exactly §4.3's format against the stream's own
anchor, so `marker_events` parses it on the same code path the real daemon will
eventually feed — including the epoch→VOD conversion, which is the one piece of
arithmetic in the system that everything else depends on.
"""

from __future__ import annotations

import json
from datetime import datetime

from clipforge import config, db, paths
from clipforge.extract import markers
from clipforge.pipeline import runner


def add_arguments(parser) -> None:
    parser.add_argument("stream_id")
    parser.add_argument(
        "--at", type=float, action="append", required=True, metavar="SECONDS",
        help="VOD time of the moment; repeatable",
    )
    parser.add_argument(
        "--kind", default="definite", choices=("maybe", "definite"),
        help="F1 = maybe, F2 = definite (§4.3)",
    )
    parser.add_argument(
        "--delay", type=float, default=None, metavar="SECONDS",
        help="reaction delay to simulate; the press lands this far AFTER --at "
             "(default: score.markers.retro_offset_s, i.e. §4.3's 5-15 s range)",
    )
    parser.add_argument(
        "--replace", action="store_true", help="overwrite markers.jsonl instead of appending"
    )
    config.add_config_arguments(parser)


def main(args) -> int:
    cfg = config.from_args(args)
    if not cfg.db_path.exists():
        print(f"no database at {cfg.db_path}. Run `clipforge db init`.")
        return 1

    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    try:
        row = conn.execute(
            "SELECT * FROM streams WHERE id = ?", (args.stream_id,)
        ).fetchone()
        if row is None:
            print(f"no stream {args.stream_id!r}")
            return 1

        anchor = row["record_start_epoch_ms"]
        if anchor is None:
            anchor = _synthesize_anchor(conn, args.stream_id)
            print(f"stream had no anchor; created one at {anchor} "
                  f"({datetime.fromtimestamp(anchor / 1000).isoformat(timespec='seconds')})")

        delay = args.delay
        if delay is None:
            delay = float(cfg.get("score.markers.retro_offset_s"))

        kind = f"marker_{args.kind}"
        stream_paths = paths.StreamPaths(cfg.data_root, args.stream_id).ensure()
        path = stream_paths.markers_jsonl

        lines = []
        for moment in sorted(args.at):
            press = moment + delay
            if row["duration_s"] and press > row["duration_s"]:
                print(f"  warning: a press for t={moment} lands at {press:.1f}s, past the "
                      f"{row['duration_s']:.1f}s recording — marker_events will drop it")
            lines.append(json.dumps({
                "epoch_ms": int(anchor) + int(round(press * 1000)),
                "kind": kind,
            }))

        existing = ""
        if path.is_file() and not args.replace:
            existing = path.read_text(encoding="utf-8")
            if existing and not existing.endswith("\n"):
                existing += "\n"
        path.write_text(existing + "\n".join(lines) + "\n", encoding="utf-8")

        print(f"{'wrote' if args.replace else 'appended'} {len(lines)} {kind} press(es) to {path}")
        for moment in sorted(args.at):
            print(f"  moment {moment:.1f}s -> pressed {moment + delay:.1f}s "
                  f"(reaction delay {delay:.1f}s)")
        print(f"\nnext: clipforge run {args.stream_id} --force marker_events")
        return 0
    finally:
        conn.close()


def _synthesize_anchor(conn, stream_id: str) -> int:
    """Give an anchorless stream a plausible one and switch it to epoch time.

    Registering arbitrary footage leaves `record_start_epoch_ms` NULL, which is
    correct — there was no OBS session. But §4.3 markers are epoch-based by
    definition, so testing them needs an anchor to convert against.
    """
    anchor = int(datetime.now().timestamp() * 1000)
    with db.transaction(conn):
        conn.execute(
            "UPDATE streams SET record_start_epoch_ms = ?, marker_time_base = 'epoch' "
            "WHERE id = ?",
            (anchor, stream_id),
        )
    return anchor
