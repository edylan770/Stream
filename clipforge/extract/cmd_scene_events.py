"""`clipforge scene-events --check <log>` — the one command to run on a real log.

This exists because `clipforge/extract/obs_log.py` was written without ever
seeing an OBS log, and a wrong regex there fails *silently*: zero events, which
looks exactly like a session in which nobody switched scenes.

So the diagnostic's job is to make a dead pattern impossible to miss. It prints:

  * how many lines each pattern claimed — a zero is the whole story
  * the share of lines with no parseable timestamp, checked first because the
    `elapsed` pattern failing makes every other number meaningless
  * the recording spans found, and whether one can be picked without guessing
  * the scene timeline, in recording-relative seconds
  * **every line mentioning a scene or a recording that no pattern claimed**

It takes a file rather than a stream id on purpose: the log to check is on the
streaming PC, and this has to be runnable there before anything is registered.
And it prints a report rather than requiring the log to be copied anywhere,
because OBS logs carry machine paths, hardware details and sometimes stream
URLs — the report is the thing that is safe to send on.
"""

from __future__ import annotations

from pathlib import Path

from clipforge import config
from clipforge.extract import obs_log

#: Unmatched lines are the most useful output and also the most numerous on a
#: log whose patterns are wrong. Enough to see the shape, not a second copy of
#: the file.
SUSPECT_LIMIT = 25


def add_arguments(parser) -> None:
    parser.add_argument(
        "log", nargs="?",
        help="an OBS log file (%%APPDATA%%\\obs-studio\\logs\\*.txt)",
    )
    parser.add_argument(
        "--check", metavar="LOG",
        help="alias for the positional argument, so the documented form "
             "`scene-events --check <log>` works",
    )
    parser.add_argument(
        "--master", metavar="NAME",
        help="the recording's filename, to pick which span in the log is it "
             "(only useful once patterns.recording_path is set)",
    )
    parser.add_argument(
        "--all-suspects", action="store_true",
        help=f"print every unmatched line, not the first {SUSPECT_LIMIT}",
    )
    config.add_config_arguments(parser)


def main(args) -> int:
    target = args.check or args.log
    if not target:
        print("give a log file: clipforge scene-events --check <path>")
        return 2

    path = Path(target).expanduser()
    if not path.is_file():
        print(f"not a file: {path}")
        return 1

    cfg = config.from_args(args)
    try:
        patterns = obs_log.Patterns.from_config(
            cfg.get("extract.scene_events.patterns") or {})
    except obs_log.ObsLogError as exc:
        print(f"config error: {exc}")
        return 1

    parsed = obs_log.parse(obs_log.read(path), patterns)
    return _report(parsed, patterns, path, args)


def _report(parsed, patterns, path: Path, args) -> int:
    print(f"{path.name}: {parsed.lines} non-blank line(s)")
    print()
    print("EVERY PATTERN BELOW IS UNVALIDATED -- that is what this command is for.")
    print("Fix them in clipforge/config/clipforge.yaml -> extract.scene_events.patterns")
    print()

    # The elapsed prefix first and on its own: if it is wrong, nothing else
    # below means anything, and reporting it alongside the others invites
    # chasing a scene pattern that was never the problem.
    timestamped = parsed.lines - parsed.without_timestamp
    share = (timestamped / parsed.lines * 100.0) if parsed.lines else 0.0
    print(f"  elapsed          {timestamped:>5} line(s) timestamped  ({share:.0f}%)")
    if timestamped == 0:
        print("                   ^^ NOTHING matched. Every other number below is")
        print("                      meaningless until this pattern is right.")
    elif share < 50:
        print("                   ^^ under half. Suspect this pattern first.")

    for key in ("scene_switch", "recording_start", "recording_stop", "recording_path"):
        count = parsed.hits.get(key, 0)
        note = ""
        if key == "recording_path" and getattr(patterns, "recording_path") is None:
            note = "  (not configured -- a log with several recordings is refused)"
        elif count == 0:
            note = "  <-- DEAD PATTERN"
        print(f"  {key:<16} {count:>5} line(s){note}")

    print()
    print(f"recordings found: {len(parsed.recordings)}")
    for index, rec in enumerate(parsed.recordings, start=1):
        stop = f"{rec.stop_elapsed:.3f}" if rec.stop_elapsed is not None else "never stopped"
        length = (f"{rec.duration_s / 60.0:.1f} min"
                  if rec.duration_s is not None else "unknown length")
        print(f"  {index}. starts {rec.start_elapsed:.3f}s  stops {stop}  ({length})"
              + (f"  path={rec.path}" if rec.path else ""))

    try:
        chosen = obs_log.select_recording(parsed, args.master)
    except obs_log.ObsLogError as exc:
        print()
        print(f"CANNOT CHOOSE A RECORDING: {exc}")
        print()
        _suspects(parsed, args)
        return 1

    print(f"  -> would use the one starting at {chosen.start_elapsed:.3f}s")

    duration = chosen.duration_s if chosen.duration_s is not None else 0.0
    spans = obs_log.scene_spans(parsed, chosen, duration or 0.0)
    print()
    print(f"scene timeline ({len(spans)} span(s), seconds into the recording):")
    if not spans:
        print("  none. Either the scene_switch pattern is dead (see above), or")
        print("  the session genuinely never changed scene.")
    for span in spans:
        print(f"  {span.t:9.2f} - {span.t_end:9.2f}  "
              f"{span.duration_s / 60.0:5.1f} min  {span.scene}")

    print()
    _suspects(parsed, args)
    return 0


def _suspects(parsed, args) -> None:
    """The most useful output when something is wrong, so it goes last."""
    if not parsed.unmatched_suspects:
        print("no unmatched lines mention a scene or a recording.")
        return

    total = len(parsed.unmatched_suspects)
    shown = parsed.unmatched_suspects if args.all_suspects \
        else parsed.unmatched_suspects[:SUSPECT_LIMIT]
    print(f"{total} line(s) mention a scene or a recording and matched NOTHING.")
    print("If a pattern above is dead, its real wording is probably in here:")
    for number, line in shown:
        print(f"  {number:>6}: {line}")
    if len(shown) < total:
        print(f"  ... and {total - len(shown)} more (--all-suspects)")
