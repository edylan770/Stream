"""`clipforge review` — start the review server."""

from __future__ import annotations

from clipforge import config, db
from clipforge.review import app as review_app


def add_arguments(parser) -> None:
    parser.add_argument(
        "stream_id", nargs="?",
        help="open this stream directly; omit to choose from a list",
    )
    config.add_config_arguments(parser)
    parser.add_argument("--host", help="default: review.host (loopback)")
    parser.add_argument("--port", type=int, help="default: review.port")
    parser.add_argument("--no-open", action="store_true", help="do not open a browser")


def main(args) -> int:
    cfg = config.from_args(args)
    if not cfg.db_path.exists():
        print(f"no database at {cfg.db_path}. Run `clipforge db init`.")
        return 1

    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    try:
        if args.stream_id:
            row = conn.execute(
                "SELECT proxy_path, (SELECT COUNT(*) FROM candidates c "
                "WHERE c.stream_id = s.id AND c.is_current = 1) AS n "
                "FROM streams s WHERE id = ?",
                (args.stream_id,),
            ).fetchone()
            if row is None:
                print(f"no stream {args.stream_id!r}")
                return 1
            if not row["n"]:
                print(f"{args.stream_id} has no candidates yet. "
                      f"Run `clipforge run {args.stream_id}` first.")
                return 1
            if not row["proxy_path"]:
                print(f"{args.stream_id} has no proxy, so there is nothing to play. "
                      f"Run `clipforge run {args.stream_id}`.")
                return 1

        ready = conn.execute(
            "SELECT COUNT(*) FROM streams s WHERE s.proxy_path IS NOT NULL AND EXISTS "
            "(SELECT 1 FROM candidates c WHERE c.stream_id = s.id AND c.is_current = 1)"
        ).fetchone()[0]
        if not ready:
            print("no stream has candidates and a proxy yet.")
            print("Register a recording and run the pipeline first:")
            print("  clipforge register --master <file>")
            print("  clipforge run <stream_id>")
            return 1
    finally:
        conn.close()

    review_app.serve(cfg, host=args.host, port=args.port, open_browser=not args.no_open)
    return 0
