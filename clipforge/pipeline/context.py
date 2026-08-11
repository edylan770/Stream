"""What a stage is handed when it runs.

Kept separate from both the registry and the runner so that stage modules can
import it without creating a cycle: `runner` imports `stages` and `context`,
stage modules import `context`, and `stages` imports neither.
"""

from __future__ import annotations

import os
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from clipforge.config import Config
from clipforge.paths import StreamPaths

#: Do not write a heartbeat more often than this. A proxy encode emits progress
#: lines several times a second and the heartbeat only needs to prove liveness.
HEARTBEAT_INTERVAL_S = 5.0


class StreamNotFound(LookupError):
    pass


@dataclass
class StageContext:
    """Everything a stage needs, and nothing it does not."""

    cfg: Config
    conn: sqlite3.Connection
    stream_id: str
    log: Callable[[str], None] = print
    _stage: str = ""
    _last_heartbeat: float = field(default=0.0, repr=False)

    @property
    def stream(self) -> sqlite3.Row:
        """The `streams` row, read fresh.

        Deliberately not cached: `probe` mutates this row, and the stages after
        it need the duration and track map it wrote, not the empty values that
        were there when the run started.
        """
        row = self.conn.execute(
            "SELECT * FROM streams WHERE id = ?", (self.stream_id,)
        ).fetchone()
        if row is None:
            raise StreamNotFound(
                f"no stream {self.stream_id!r}. Register it with "
                f"`clipforge register --master <file>`."
            )
        return row

    @property
    def paths(self) -> StreamPaths:
        return StreamPaths(self.cfg.data_root, self.stream_id)

    @property
    def master(self) -> Path:
        return Path(self.stream["master_path"])

    def heartbeat(self) -> None:
        """Prove this run is alive, so a crash can be told from a long encode."""
        now = time.monotonic()
        if now - self._last_heartbeat < HEARTBEAT_INTERVAL_S:
            return
        self._last_heartbeat = now
        self.conn.execute(
            "UPDATE pipeline_stages SET heartbeat_at = datetime('now') "
            "WHERE stream_id = ? AND stage = ?",
            (self.stream_id, self._stage),
        )

    def metric(self, name: str, value: float, meta: str | None = None) -> None:
        """§14 instrumentation."""
        self.conn.execute(
            "INSERT INTO tool_metrics (stream_id, metric, value, meta) VALUES (?, ?, ?, ?)",
            (self.stream_id, name, value, meta),
        )


def master_identity(ctx: StageContext) -> dict[str, Any]:
    """How a stage identifies the master file for `params_hash`.

    **Name and size, never mtime or full path.** Modification time does not
    survive a copy and the absolute path changes the moment footage moves to
    another drive — including either one would mean that carrying the library
    to the streaming PC invalidates every stage and triggers a full
    re-extraction of the entire back catalogue. That is precisely the migration
    the relative-path design exists to make free.

    Size still catches the case that matters: this stream now points at a
    different file. Anything subtler is what `--force` is for.
    """
    master = ctx.master
    try:
        size = master.stat().st_size
    except OSError:
        size = None
    return {"master_name": master.name, "master_size": size}


def running_process_is_us(host: str | None, pid: int | None) -> bool:
    import socket

    return host == socket.gethostname() and pid == os.getpid()
