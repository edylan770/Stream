"""Turn ffmpeg's `-progress` stream into a heartbeat and a percentage.

The heartbeat is what lets the next invocation tell a crashed run from a long
one (see `runner.reclaim_crashed`). A 4-hour master spends long enough in both
the proxy encode and the audio split that the distinction is not academic.
"""

from __future__ import annotations

import re
from collections.abc import Callable

#: ffmpeg emits `out_time_us=<int>` among its key=value progress lines.
_OUT_TIME_US = re.compile(r"^out_time_us=(\d+)$")

#: Report at most this often, in percent. Enough to see it moving, few enough
#: that a long encode does not bury the rest of the run's output.
STEP_PCT = 10


def reporter(ctx, total_s: float, *, step_pct: int = STEP_PCT) -> Callable[[str], None]:
    """Build an ffmpeg stderr line handler for `ffmpeg.run(on_stderr_line=...)`."""
    state = {"pct": -step_pct}

    def handle(line: str) -> None:
        match = _OUT_TIME_US.match(line.strip())
        if not match:
            return
        ctx.heartbeat()
        if total_s <= 0:
            return
        elapsed = int(match.group(1)) / 1_000_000.0
        pct = int(min(100.0, elapsed / total_s * 100))
        if pct >= state["pct"] + step_pct:
            state["pct"] = pct
            ctx.log(f"      {pct}%")

    return handle
