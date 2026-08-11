"""`clipforge metrics` — is review actually fast? (§7.5, §14)

C4 sets a hard target of 120 candidates in under 8 minutes and says that if it
is exceeded, the UI gets fixed *before any other feature anywhere in the
system*. That only means something if the number is measured, which is the
entire point of §7.5's instrumentation hooks.
"""

from __future__ import annotations

import json

from clipforge import config, db
from clipforge.review import queries


def add_arguments(parser) -> None:
    parser.add_argument("stream_id", nargs="?", help="omit for every stream")
    config.add_config_arguments(parser)
    parser.add_argument("--json", action="store_true")


def main(args) -> int:
    cfg = config.from_args(args)
    if not cfg.db_path.exists():
        print(f"no database at {cfg.db_path}. Run `clipforge db init`.")
        return 1

    target_ms = float(cfg.get("review.target_ms_per_candidate"))
    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    try:
        if args.stream_id:
            ids = [args.stream_id]
            if conn.execute(
                "SELECT 1 FROM streams WHERE id = ?", (args.stream_id,)
            ).fetchone() is None:
                print(f"no stream {args.stream_id!r}")
                return 1
        else:
            ids = [r["id"] for r in conn.execute("SELECT id FROM streams ORDER BY date, id")]

        report = {sid: queries.review_metrics(conn, sid) for sid in ids}

        if args.json:
            print(json.dumps(report, indent=2))
            return 0

        for stream_id, m in report.items():
            _print(stream_id, m, target_ms)
            print()

        _stage_durations(conn, ids)
        return 0
    finally:
        conn.close()


def _print(stream_id: str, m: dict, target_ms: float) -> None:
    print(stream_id)
    print(f"  candidates      {m['candidates']}")
    print(f"  rated           {m['rated']}"
          + (f"  ({m['rated'] / m['candidates']:.0%})" if m["candidates"] else ""))

    if m["rated"]:
        labels = {0: "skip", 1: "maybe", 2: "clip it"}
        counts = "  ".join(
            f"{labels[k]}={m['by_rating'].get(k, 0)}" for k in (2, 1, 0)
        )
        print(f"  ratings         {counts}")
        # §14: "approval_rate — approved / total. Is the threshold correct?"
        if m["approval_rate"] is not None:
            print(f"  approval rate   {m['approval_rate']:.1%}")

    median = m["median_review_ms"]
    if median is None:
        print("  review speed    not measured yet — rate some candidates in the UI")
        return

    # Median, not mean: leave the tab open over lunch and one candidate reads
    # forty minutes, which would swamp an average and hide the real figure.
    verdict = "within target" if median <= target_ms else "OVER TARGET"
    print(f"  per candidate   {median / 1000:.2f}s median  ({verdict}, "
          f"target {target_ms / 1000:.1f}s)")
    if m["mean_review_ms"] and m["mean_review_ms"] > median * 2:
        print(f"                  mean is {m['mean_review_ms'] / 1000:.1f}s — "
              f"some candidates sat idle, which is why the median is the headline")

    for session in m["sessions"]:
        meta = json.loads(session["meta"] or "{}")
        reviewed = meta.get("reviewed", 0)
        seconds = session["value"]
        rate = f", {seconds / reviewed:.1f}s each" if reviewed else ""
        print(f"  session         {seconds / 60:.1f} min for {reviewed} candidates{rate}"
              f"  [{session['recorded_at']}]")

    if median > target_ms:
        print("\n  §7.1: if review exceeds the target, fix the UI before adding any "
              "feature anywhere in the system.")


def _stage_durations(conn, ids: list[str]) -> None:
    """§14's stage_duration_s — where unattended processing actually goes."""
    rows = conn.execute(
        """
        SELECT stream_id, meta, value FROM tool_metrics
         WHERE metric = 'stage_duration_s' AND stream_id IN (%s)
        """ % ",".join("?" * len(ids)),
        ids,
    ).fetchall()
    if not rows:
        return

    totals: dict[str, float] = {}
    for row in rows:
        stage = json.loads(row["meta"] or "{}").get("stage", "?")
        totals[stage] = totals.get(stage, 0.0) + float(row["value"])

    print("processing time by stage")
    width = max(len(s) for s in totals)
    for stage, seconds in sorted(totals.items(), key=lambda kv: -kv[1]):
        print(f"  {stage.ljust(width)}  {seconds:>8.1f}s")
