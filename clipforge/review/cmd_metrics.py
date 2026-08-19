"""`clipforge metrics` — is review actually fast? (§7.5, §14)

C4 sets a hard target of 120 candidates in under 8 minutes and says that if it
is exceeded, the UI gets fixed *before any other feature anywhere in the
system*. That only means something if the number is measured, which is the
entire point of §7.5's instrumentation hooks.
"""

from __future__ import annotations

import json

from clipforge import config, db
from clipforge.review import queries, tuning


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
            print(json.dumps({
                "streams": report,
                "tuning": tuning.tuning_metrics(conn, cfg, ids).to_json(),
            }, indent=2))
            return 0

        for stream_id, m in report.items():
            _print(stream_id, m, target_ms)
            print()

        _stage_durations(conn, ids)
        _tuning(conn, cfg, ids)
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


#: How many signals to print. The full schema is 33 and most of them are null
#: on a Phase 1 stream; the ones that separate the verdicts are what §17 acts on.
TOP_SIGNALS = 12


def _tuning(conn, cfg, ids: list[str]) -> None:
    """§14's three weight-tuning metrics (GUESSES gaps 2 and 3).

    THE SAMPLE SIZE IS PRINTED BEFORE ANY RATE, and below
    `metrics.min_samples_for_rate` no rate is printed at all. A fraction over
    n=1 looks exactly like a fraction over n=1000, and §17's whole procedure is
    someone changing a weight because of a number on this screen.
    """
    result = tuning.tuning_metrics(conn, cfg, ids)

    print()
    print("weight tuning (§14, §17)")
    print(f"  rated           {result.rated} operator rating(s) across "
          f"{result.streams} stream(s)"
          f"  ·  {result.approved} approved, {result.rejected} rejected")

    if not result.rated:
        print("  nothing to tune on yet — rate some candidates in the review UI")
        return

    if not result.enough:
        # Deliberately blunt. This is the screen someone would change a weight
        # from, and the honest answer today is "you cannot yet".
        print(f"  NOT ENOUGH YET  §17 wants ~10 streams; rates need "
              f"{result.min_samples} approved and {result.min_samples} rejected "
              f"(have {result.approved}/{result.rejected}).")
        print("                  counts below are real; the fractions they imply "
              "are not, and are withheld.")

    _markers(result)
    _signals(result)


def _markers(result: tuning.Tuning) -> None:
    m = result.markers
    print(f"  markers         {m['anchored']} marker-anchored candidate(s), "
          f"{m['anchored_approved']} of them approved")

    if not result.enough:
        return
    precision = m[tuning.MARKER_PRECISION]
    recall = m[tuning.MARKER_RECALL]
    if precision is not None:
        print(f"  marker precision {precision:.1%} — of what you marked live, "
              f"how much survived review (§17's retro_offset_s)")
    if recall is not None:
        # §14 calls this the valuable one, and it is the number that says
        # whether automatic detection is earning its keep at all.
        print(f"  missed live      {recall:.1%} of approved moments had no marker "
              f"— what the detector caught and you did not")


def _signals(result: tuning.Tuning) -> None:
    """§14's signal_firing_rate_by_rating, strongest separation first.

    `separation` leads because it needs no threshold: it is the difference of a
    signal's mean value on approved versus rejected moments. A positive number
    means the signal is higher on the moments you kept, which is the definition
    of a signal worth more weight.

    Ranked by |separation| rather than by separation, because a signal that is
    reliably LOWER on approved moments is just as informative — it wants a
    negative weight, or it is a §6.4 gate.
    """
    rows = [r for r in result.signals if r["separation"] is not None]
    if not rows:
        print("  signals         no signal was observed on both an approved and "
              "a rejected candidate yet")
        return

    print(f"  signals         separation = mean(approved) - mean(rejected); "
          f"fired at z >= {result.threshold}")
    width = max(len(r["signal"]) for r in rows[:TOP_SIGNALS])
    for row in rows[:TOP_SIGNALS]:
        line = (f"    {row['signal'].ljust(width)}  {row['separation']:+7.3f}"
                f"   n={row['n_approved']}/{row['n_rejected']}")
        if result.enough and row["lift"] is not None:
            line += (f"   fired {row['fired_rate_approved']:.0%} vs "
                     f"{row['fired_rate_rejected']:.0%}")
        print(line)

    if len(rows) > TOP_SIGNALS:
        print(f"    … {len(rows) - TOP_SIGNALS} more; --json for all of them")


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
