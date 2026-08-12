"""§4.4's input activity logger — rates only, never keystrokes.

> Low-level keyboard/mouse hook (`pynput` on Windows). Writes JSONL at 10 Hz
> aggregate rather than per-event to keep files small.

Derived downstream: mouse velocity spikes (flicks), input rate spikes (panic),
sudden stillness (shock/death). §4.4 calls this "arguably the second-strongest
gameplay signal after markers".

**IT MUST NEVER BE POSSIBLE FOR THIS TO RECORD WHICH KEY WAS PRESSED.**

A global keyboard hook sees everything typed into every window on the machine,
including passwords typed into a browser while OBS is idle. §4.4's own JSONL
example is rates only, so aggregate-only is what the spec asks for — but that
has to be a property of the code rather than a habit, because a debug print
added in a hurry six months from now would quietly turn this into a keylogger.

The guarantee is structural: `Aggregator.key()` takes **no arguments**. The
pynput adapter is a one-line lambda that discards the key object before calling
it, so no function in this module can see a key identity even if it wanted to,
and nothing downstream can be handed one.

Run it:

    python -m clipforge.capture.input_logger --dir D:/capture
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from clipforge.capture import contract


@dataclass
class Aggregator:
    """Counts events into fixed windows and emits §4.4 records.

    Two clocks on purpose. `epoch` stamps the record, because A8 requires
    wall-clock milliseconds that are comparable to the anchor. `elapsed` drives
    the window, because it must be monotonic — an NTP correction mid-stream
    would otherwise stretch or collapse a bucket and put a spurious spike in the
    one signal that is supposed to detect spikes.
    """

    sink: contract.JsonlSink
    hz: float = contract.INPUT_HZ
    epoch: Callable[[], int] = contract.now_epoch_ms
    elapsed: Callable[[], float] = time.monotonic
    windows: int = 0

    _keys: int = field(default=0, repr=False)
    _clicks: int = field(default=0, repr=False)
    _distance: float = field(default=0.0, repr=False)
    _last_xy: tuple[int, int] | None = field(default=None, repr=False)
    _window_start: float | None = field(default=None, repr=False)

    @property
    def period_s(self) -> float:
        return 1.0 / self.hz

    # -- event intake ------------------------------------------------------

    def key(self) -> None:
        """A key went down. WHICH key is deliberately not a parameter."""
        self._keys += 1

    def click(self) -> None:
        self._clicks += 1

    def move(self, x: int, y: int) -> None:
        """Accumulate travelled distance, not position.

        Distance rather than a position sample because §5.4.1 wants px/s and
        polling positions at 10 Hz would miss the fast part of a flick entirely
        — the whole point of the signal.
        """
        if self._last_xy is not None:
            dx = x - self._last_xy[0]
            dy = y - self._last_xy[1]
            self._distance += math.hypot(dx, dy)
        self._last_xy = (x, y)

    # -- windowing ---------------------------------------------------------

    def tick(self) -> bool:
        """Emit a record if a window has elapsed. Returns whether it did."""
        now = self.elapsed()
        if self._window_start is None:
            self._window_start = now
            return False

        span = now - self._window_start
        if span < self.period_s:
            return False

        self.sink.write(contract.input_record(
            self.epoch(),
            keys_per_s=self._keys / span,
            mouse_vel_px_s=self._distance / span,
            clicks_per_s=self._clicks / span,
        ))
        self._keys = self._clicks = 0
        self._distance = 0.0
        self._window_start = now
        self.windows += 1
        return True


def input_path(directory: Path, today: date | None = None) -> Path:
    """A file per day, alongside the marker log and for the same reason."""
    stamp = (today or date.today()).isoformat()
    return Path(directory) / f"input-{stamp}.jsonl"


def listen(aggregator: Aggregator) -> int:
    """Hook the keyboard and mouse and pump the aggregator."""
    try:
        from pynput import keyboard, mouse
    except ImportError:
        print(
            "pynput is not installed. Install the capture extra:\n"
            '    pip install "clipforge[capture]"',
            file=sys.stderr,
        )
        return 1

    # `_key` and `_button` are swallowed HERE and never passed on. This is the
    # line that makes the privacy guarantee structural rather than aspirational.
    keys = keyboard.Listener(on_press=lambda _key: aggregator.key())
    mice = mouse.Listener(
        on_move=lambda x, y: aggregator.move(x, y),
        on_click=lambda _x, _y, _button, pressed: aggregator.click() if pressed else None,
    )

    print("input logger running")
    print(f"  {aggregator.hz:g} Hz aggregate -> {aggregator.sink.path}")
    print("  rates only: this records HOW MUCH you typed, never WHAT.")
    print("Ctrl-C to stop.\n")

    keys.start()
    mice.start()
    try:
        while True:
            aggregator.tick()
            time.sleep(aggregator.period_s / 4)
    except KeyboardInterrupt:
        pass
    finally:
        keys.stop()
        mice.stop()
        aggregator.sink.close()
    print(f"\nstopped. {aggregator.windows} window(s) written.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="input_logger",
        description="§4.4's 10 Hz input activity aggregate. Records rates, never keys.",
    )
    parser.add_argument("--dir", default=".", help="capture directory (default: here)")
    parser.add_argument("--hz", type=float, default=contract.INPUT_HZ)
    args = parser.parse_args(argv)

    with contract.JsonlSink(input_path(Path(args.dir))) as sink:
        return listen(Aggregator(sink=sink, hz=args.hz))


if __name__ == "__main__":
    sys.exit(main())
