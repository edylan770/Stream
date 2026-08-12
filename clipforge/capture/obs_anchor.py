"""§4.1's anchor: the wall-clock moment OBS began recording.

> One number per stream. No drift. No manual alignment. This is the entire sync
> solution.

Everything live-captured — every marker press, every input sample — becomes a
VOD time through this one integer. Get it wrong and every one of them is wrong
by the same amount, silently, forever.

**The anchor is written beside the recording.** §4.1 specifies the mechanism and
never says where the file goes. `RecordStateChanged` carries `outputPath`, and
`ingest.register.find_capture_file` already looks beside the master, so writing
it there means `clipforge register --master <recording>` works with no flags at
all — and stopping and restarting OBS mid-session produces one correct anchor
per recording instead of one file overwriting another.

**Two sources, per §4.1.** The WebSocket event is preferred. The hotkey fallback
exists for a machine where obs-websocket is unavailable, and records
`written_at_epoch_ms` separately so the gap between them is visible as the error
bar it is.

Run it:

    python -m clipforge.capture.obs_anchor
    python -m clipforge.capture.obs_anchor --hotkey --output "D:/rec/stream.mkv"
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from clipforge.capture import contract

#: OBS reports this as `output_state` when recording has actually begun.
STATE_STARTED = "OBS_WEBSOCKET_OUTPUT_STARTED"


@dataclass
class AnchorWriter:
    """Turns a record-start notification into `anchor.json`.

    Separated from the WebSocket client so the interesting half is testable
    without OBS: the tests call `record_started` directly.
    """

    clock: Callable[[], int] = contract.now_epoch_ms
    log: Callable[[str], None] = print
    #: Where to write when OBS reports no path (it should always report one).
    fallback_dir: Path | None = None
    written: list[Path] = field(default_factory=list)

    def record_started(self, output_path: str | None, epoch_ms: int | None = None) -> Path | None:
        """Called the instant recording begins. Returns the anchor written."""
        when = self.clock() if epoch_ms is None else epoch_ms
        target = self._anchor_path(output_path)
        if target is None:
            self.log(
                "  WARNING  OBS reported a recording start with no output path and no "
                "--fallback-dir is set, so there is nowhere to put the anchor. "
                "Markers from this recording cannot be converted (§4.1)."
            )
            return None

        contract.write_anchor(target, when, contract.SOURCE_OBS_WEBSOCKET)
        self.written.append(target)
        self.log(f"  anchor    {when}  ->  {target}")
        return target

    def _anchor_path(self, output_path: str | None) -> Path | None:
        if output_path:
            return Path(output_path).expanduser().parent / contract.ANCHOR_FILENAME
        if self.fallback_dir:
            return Path(self.fallback_dir) / contract.ANCHOR_FILENAME
        return None


def handle_event(writer: AnchorWriter, event) -> Path | None:
    """Adapt one `RecordStateChanged` event.

    The timestamp is taken here, on arrival, rather than from anything in the
    event: OBS does not report when it started, so the honest reading is "when
    this process learned of it". The delivery latency is tens of milliseconds
    against §4.3's 5-15 second reaction window, so it does not matter — but it
    is why `source` is recorded rather than assumed.
    """
    state = getattr(event, "output_state", None)
    active = getattr(event, "output_active", None)
    if state != STATE_STARTED and not active:
        return None
    return writer.record_started(getattr(event, "output_path", None))


# --------------------------------------------------------------------------
# the real loop
# --------------------------------------------------------------------------


def watch(host: str, port: int, password: str | None, writer: AnchorWriter) -> int:
    """Connect to obs-websocket and write an anchor per recording."""
    try:
        import obsws_python as obs
    except ImportError:
        print(
            "obsws-python is not installed. Install the capture extra:\n"
            '    pip install "clipforge[capture]"\n'
            "or use the fallback: --hotkey --output <recording path>",
            file=sys.stderr,
        )
        return 1

    try:
        client = obs.EventClient(host=host, port=port, password=password or "")
    except Exception as exc:  # noqa: BLE001 — any connection failure reads the same
        print(f"cannot reach obs-websocket at {host}:{port} — {exc}", file=sys.stderr)
        print("Enable it in OBS: Tools > WebSocket Server Settings.", file=sys.stderr)
        return 1

    client.callback.register(lambda event: handle_event(writer, event))
    print(f"watching obs-websocket at {host}:{port}")
    print("anchor.json is written beside each recording as it starts. Ctrl-C to stop.\n")

    try:
        import threading
        threading.Event().wait()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        try:
            client.disconnect()
        except Exception:  # noqa: BLE001 — nothing useful to do while exiting
            pass
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="obs_anchor",
        description="Write §4.1's record-start anchor beside each OBS recording.",
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=4455)
    parser.add_argument("--password", default=None)
    parser.add_argument(
        "--hotkey", action="store_true",
        help="§4.1's fallback: write the anchor NOW, for a recording started by hand",
    )
    parser.add_argument("--output", help="the recording path (required with --hotkey)")
    parser.add_argument(
        "--fallback-dir",
        help="where to write when OBS reports no output path",
    )
    args = parser.parse_args(argv)

    writer = AnchorWriter(
        fallback_dir=Path(args.fallback_dir) if args.fallback_dir else None
    )

    if args.hotkey:
        if not args.output:
            parser.error("--hotkey needs --output <the recording path>")
        when = contract.now_epoch_ms()
        target = Path(args.output).expanduser().parent / contract.ANCHOR_FILENAME
        contract.write_anchor(target, when, contract.SOURCE_HOTKEY_SCRIPT)
        print(f"anchor    {when}  ->  {target}")
        print("\nNOTE: the hotkey fallback records when THIS RAN, not when OBS started.")
        print("Run it as close to the record hotkey as you can (§4.1).")
        return 0

    return watch(args.host, args.port, args.password, writer)


if __name__ == "__main__":
    sys.exit(main())
