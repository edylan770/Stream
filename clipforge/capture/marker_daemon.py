"""§4.3's marker hotkeys — the highest-weighted signal in the system.

Two keys, one JSONL line each:

| Key | Meaning | Weight |
|---|---|---|
| F1 | "maybe" — something might have happened | Moderate |
| F2 | "definitely" — that was good | High |

The retro offset is **not** applied here. §4.3 says to default the window anchor
to `t − 20s`, but that is a §17 tunable, so the file records the observation and
scoring decides what it implies — the same reasoning that keeps `events.t` raw
in `extract/markers.py`.

**The hook never suppresses.** §4.5: capture processes must never be able to
interrupt OBS, and by extension the game. F1 is help or ping in most titles,
including Marvel Rivals — a daemon that swallowed the keypress would break the
thing it exists to document. `pynput`'s listener passes keys through unless
explicitly told not to, and it is never told not to.

Hotkeys are configurable for the same reason: §4.3's F1/F2 are the default, not
a requirement, and a game that needs those keys should cost a flag rather than
the whole signal.

Run it:

    python -m clipforge.capture.marker_daemon --dir D:/capture
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from clipforge.capture import contract

#: §4.3's bindings, as pynput names them.
DEFAULT_BINDINGS = {"f1": contract.MARKER_MAYBE, "f2": contract.MARKER_DEFINITE}

#: Two presses closer together than this are one press — key repeat, or a
#: nervous double-tap. Not in the spec; §6.5's marker kernel combines with `max`
#: so duplicates are harmless to scoring, but they inflate marker_precision
#: (§14) and make the log harder to read.
DEFAULT_DEBOUNCE_S = 0.35


@dataclass
class MarkerDaemon:
    """Presses in, JSONL out. No keyboard, no threads, entirely testable."""

    sink: contract.JsonlSink
    clock: Callable[[], int] = contract.now_epoch_ms
    debounce_s: float = DEFAULT_DEBOUNCE_S
    log: Callable[[str], None] = print
    counts: dict[str, int] = field(default_factory=dict)
    _last: dict[str, int] = field(default_factory=dict, repr=False)

    def press(self, kind: str) -> bool:
        """Record one press. Returns False when debounced."""
        when = self.clock()
        previous = self._last.get(kind)
        if previous is not None and (when - previous) < self.debounce_s * 1000:
            return False

        self._last[kind] = when
        self.sink.write(contract.marker_record(when, kind))
        self.counts[kind] = self.counts.get(kind, 0) + 1
        self.log(f"  {kind:<16} {when}   ({self.counts[kind]} this session)")
        return True


def markers_path(directory: Path, today: date | None = None) -> Path:
    """A file per day, in the capture directory.

    Not beside the recording: this daemon has no idea where OBS is writing, and
    §4.5 forbids making it depend on OBS to find out. Presses outside any
    recording convert to negative or out-of-range VOD times at ingest and are
    dropped there, so a whole day in one file costs nothing.
    """
    stamp = (today or date.today()).isoformat()
    return Path(directory) / f"markers-{stamp}.jsonl"


def listen(daemon: MarkerDaemon, bindings: dict[str, str]) -> int:
    """Bind the hotkeys and block. The only part that needs a real keyboard."""
    try:
        from pynput import keyboard
    except ImportError:
        print(
            "pynput is not installed. Install the capture extra:\n"
            '    pip install "clipforge[capture]"',
            file=sys.stderr,
        )
        return 1

    hotkeys = {f"<{key}>": (lambda k=kind: daemon.press(k)) for key, kind in bindings.items()}
    listener = keyboard.GlobalHotKeys(hotkeys)

    print("marker daemon running")
    for key, kind in bindings.items():
        print(f"  {key.upper():<4} -> {kind}")
    print(f"\nwriting {daemon.sink.path}")
    print("Keys are NOT suppressed — the game still sees them. Ctrl-C to stop.\n")

    listener.start()
    try:
        listener.join()
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
        daemon.sink.close()
    total = sum(daemon.counts.values())
    print(f"\nstopped. {total} marker(s) written to {daemon.sink.path}")
    return 0


def parse_bindings(raw: list[str] | None) -> dict[str, str]:
    """`--bind f9=marker_definite`, for a game that needs F1."""
    bindings = dict(DEFAULT_BINDINGS)
    for item in raw or []:
        key, _, kind = item.partition("=")
        key, kind = key.strip().lower(), kind.strip()
        if not key or kind not in contract.MARKER_KINDS:
            raise SystemExit(
                f"--bind expects <key>=<kind> with kind in {contract.MARKER_KINDS}, got {item!r}"
            )
        bindings = {k: v for k, v in bindings.items() if v != kind}
        bindings[key] = kind
    return bindings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="marker_daemon",
        description="§4.3's F1/F2 marker hotkeys -> JSONL.",
    )
    parser.add_argument("--dir", default=".", help="capture directory (default: here)")
    parser.add_argument("--bind", action="append", metavar="KEY=KIND",
                        help="rebind a hotkey, e.g. f9=marker_definite")
    parser.add_argument("--debounce", type=float, default=DEFAULT_DEBOUNCE_S,
                        help="seconds; two presses closer than this are one")
    args = parser.parse_args(argv)

    bindings = parse_bindings(args.bind)
    with contract.JsonlSink(markers_path(Path(args.dir))) as sink:
        return listen(MarkerDaemon(sink=sink, debounce_s=args.debounce), bindings)


if __name__ == "__main__":
    sys.exit(main())
