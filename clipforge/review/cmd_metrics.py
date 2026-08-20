"""`clipforge metrics` — is review actually fast? (§7.5, §14)

C4 sets a hard target of 120 candidates in under 8 minutes and says that if it
is exceeded, the UI gets fixed *before any other feature anywhere in the
system*. That only means something if the number is measured, which is the
entire point of §7.5's instrumentation hooks.
"""

from __future__ import annotations

import json

from clipforge import config, db, tuning
from clipforge.review import queries


def add_arguments(parser) -> None:
    parser.add_argument("stream_id", nargs="?", help="omit for every stream")
    config.add_config_arguments(parser)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--tuning", action="store_true",
        help="§14's weight-tuning table: which signals discriminate (§17)")
    parser.add_argument(
        "--record", action="store_true",
        help="with --tuning, write the figures to tool_metrics (§17 reads them there)")


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
        study = tuning.collect(conn, cfg, ids) if args.tuning else None

        if args.json:
            # Shape UNCHANGED without --tuning: a flat {stream_id: metrics}, so
            # anything already parsing this keeps working. Asking for the tuning
            # table is asking for a second section, and only then does the
            # payload need somewhere to put it.
            payload: dict = report if study is None else {
                "streams": report, "tuning": _tuning_json(cfg, *study)}
            print(json.dumps(payload, indent=2))
            return 0

        for stream_id, m in report.items():
            _print(stream_id, m, target_ms)
            print()

        _stage_durations(conn, ids)

        if study is not None:
            print()
            _tuning(cfg, *study)
            if args.record:
                with db.transaction(conn):
                    written = tuning.record(conn, *study)
                print()
                print(f"  recorded        {written} row(s) into tool_metrics")
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
        # An inherited row is the operator's opinion carried onto this
        # generation's candidate by a re-score, not one made about it. Counted
        # (the review screen shows it as rated) but never silently.
        carried = (m.get("by_source") or {}).get("inherited", 0)
        if carried:
            print(f"  carried over    {carried} of them came from an earlier "
                  f"generation via a re-score")

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

    _nudges(m)

    if median > target_ms:
        print("\n  §7.1: if review exceeds the target, fix the UI before adding any "
              "feature anywhere in the system.")


def _nudges(m: dict) -> None:
    """§17's input for `min_window_s` / `max_window_s` (GUESSES gap 1).

    §17 says to tune those two against "how often the operator nudges boundaries
    during review". A count alone cannot do it — it does not say which way to
    move a bound — so what is printed is the direction, and then the two
    figures that actually decide: how many nudged windows had come out sitting
    exactly on a clamp.
    """
    n = m.get("nudges") or {}
    if not n.get("nudged"):
        return

    print(f"  windows nudged  {n['nudged']} ({n['presses']} keypresses)"
          f"  ·  {n['extended']} extended, {n['trimmed']} trimmed")
    print(f"  mean move       start {n['mean_start_delta_s']:+.2f}s  "
          f"end {n['mean_end_delta_s']:+.2f}s  "
          f"length {n['mean_length_delta_s']:+.2f}s")

    if n["was_at_min"]:
        print(f"  at min_window_s {n['was_at_min']} of them had been clamped to "
              f"min_window_s — §17 says tune it against exactly this")
    if n["was_at_max"]:
        print(f"  at max_window_s {n['was_at_max']} of them had been clamped to "
              f"max_window_s — §17 says tune it against exactly this")
    if n["dropped_peak"]:
        print(f"  peak dropped    {n['dropped_peak']} window(s) no longer contain "
              f"the peak they were detected from")


def _tuning(cfg, corpus, stats, markers) -> None:
    """§14's `signal_firing_rate_by_rating`, and the two marker figures.

    THE CORPUS LINE COMES FIRST AND ALWAYS PRINTS. §17's procedure ("adjust
    weights toward signals that discriminate") is only sound over enough
    ratings to tell discrimination from luck, and the failure mode this guards
    against is a table of four-decimal numbers built from three opinions, which
    reads as evidence and is not.
    """
    print("tuning — §14's weight-tuning input")
    labels = {2: "clip it", 1: "maybe", 0: "skip"}
    spread = "  ".join(f"{labels[k]}={corpus.by_rating.get(k, 0)}" for k in (2, 1, 0))
    print(f"  corpus          {corpus.streams} stream(s), {corpus.rated_streams} rated"
          f"  ·  {corpus.moments} moment(s) ({spread})")
    if corpus.schema_versions and corpus.schema_versions != [cfg.feature_schema.version]:
        print(f"  schema          vectors from feature_schema_version "
              f"{corpus.schema_versions}; the table reads today's declared keys, so a "
              f"signal absent from an older vector simply has fewer observations")
    if corpus.without_vector:
        print(f"  no vector       {corpus.without_vector} moment(s) had no feature "
              f"vector to read")

    _markers(markers)

    ok, why = tuning.rankable(corpus, cfg)
    if not ok:
        print(f"  not enough      {why}")
        return

    print()
    width = max(len(s.name) for s in stats)
    group_width = max(len(s.group) for s in stats)
    print(f"  {'signal'.ljust(width)}  {'group'.ljust(group_width)}   sep"
          f"  n(skip)  n(clip)   fired   fired")
    for stat in stats:
        head = f"  {stat.name.ljust(width)}  {stat.group.ljust(group_width)}"
        if stat.separation is None:
            print(f"{head}     —  {stat.reason}")
            continue
        arrow = "   ← discriminates the WRONG way" if stat.separation < 0.5 else ""
        print(f"{head}  {stat.separation:.2f}  {stat.n_skip:>7}  {stat.n_clip:>7}"
              f"  {stat.rate_skip:>5.0%}  {stat.rate_clip:>5.0%}{arrow}")

    print()
    print("  sep is P(a `clip it` moment outscores a `skip` one): 0.5 "
          "discriminates nothing, 1.0 perfectly.")
    print(f"  Ranked on it rather than on the two rate columns because those depend on "
          f"tuning.firing_threshold_z ({cfg.get('tuning.firing_threshold_z')}), which is "
          f"a guess; sep depends on no threshold at all.")


def _markers(markers) -> None:
    """§14's `marker_precision` and `marker_recall_proxy`, corpus-wide.

    `marker_recall_proxy` is the one §14 singles out: it "directly measures how
    many good clips the operator misses live, which is the exact worry that
    motivated automatic detection in the first place".
    """
    anchored = sum(m.anchored for m in markers)
    anchored_approved = sum(m.anchored_approved for m in markers)
    approved = sum(m.approved for m in markers)
    unmarked = sum(m.approved_unmarked for m in markers)
    if not anchored and not approved:
        return

    if anchored:
        print(f"  marker prec.    {anchored_approved / anchored:.0%}  "
              f"({anchored_approved} of {anchored} marker-anchored moment(s) approved)")
    if approved:
        print(f"  marker recall   {unmarked / approved:.0%}  "
              f"({unmarked} of {approved} approved moment(s) had no press inside them)")
    print("                  press_inside only — not §7.4's looser reading, which moves "
          "when a marker weight moves. See clipforge/moments.py")


def _tuning_json(cfg, corpus, stats, markers) -> dict:
    ok, why = tuning.rankable(corpus, cfg)
    return {
        "corpus": {
            "streams": corpus.streams, "rated_streams": corpus.rated_streams,
            "moments": corpus.moments, "by_rating": corpus.by_rating,
            "without_vector": corpus.without_vector,
            "schema_versions": corpus.schema_versions,
        },
        "rankable": ok,
        "reason": why,
        "signals": [
            {"signal": s.name, "group": s.group, "separation": s.separation,
             "reason": s.reason, "n_skip": s.n_skip, "n_clip": s.n_clip,
             "rate_skip": s.rate_skip, "rate_clip": s.rate_clip}
            for s in stats
        ],
        "markers": [
            {"stream_id": m.stream_id, "anchored": m.anchored,
             "anchored_approved": m.anchored_approved, "approved": m.approved,
             "approved_unmarked": m.approved_unmarked,
             "precision": m.precision, "recall_proxy": m.recall_proxy,
             "anchoring": "press_inside"}
            for m in markers
        ],
    }

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
