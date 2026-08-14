"""Running the pipeline from the browser, in the background.

A proxy encode is minutes to hours (§1.3 budgets 20-40 min of unattended
processing for a 4-hour stream), so the request that starts it cannot be the
request that waits for it. A job is a thread plus a log buffer; the browser
polls.

**Polling, not SSE.** `Runner.log` is already a `Callable[[str], None]` and
`pipeline.progress` already pushes ffmpeg percentages through it, so a job needs
only a buffer and a cursor. Stage transitions and 10%-step encode progress do
not need sub-second latency, and polling survives a page reload with no
reconnect logic and no long-lived connection to manage.

**Two locks, because there are two ways to collide.**

`Runner.reclaim_crashed` treats *any* `running` row as a dead run and marks it
failed. That is right for a crash and catastrophic for a live sibling: a second
run started while the first is mid-encode declares the first one's stage dead
and starts over on top of the file it is still writing. The runner's docstring
calls single-writer "the documented assumption"; nothing enforced it.

So: the registry refuses a second job for a stream this process is already
running, and `start` additionally refuses when the database shows a `running`
stage with a fresh heartbeat belonging to *another* process — a terminal
`clipforge run`, typically. The second check cannot be airtight (the other
process may start a microsecond later) but it turns the common case from silent
artifact corruption into a message.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

from clipforge import db, paths
from clipforge.config import Config
from clipforge.pipeline import runner
from clipforge.pipeline.context import running_process_is_us

RUNNING = "running"
DONE = "done"
FAILED = "failed"

#: What a job is doing. `run` is §5.1's pipeline; `render` is §8's auto-finish.
#:
#: There is still only ONE job per stream, not one per kind. A stream is being
#: worked on or it is not — the same single-writer rule `live_elsewhere`
#: enforces against other processes — and two concurrent jobs would mean the
#: run view tracking two logs and two cursors for one screen.
KIND_RUN = "run"
KIND_RENDER = "render"

#: Log lines kept per job. A full run emits a few hundred; the cap exists so a
#: pathological stage cannot grow the buffer without bound. Dropped lines are
#: counted, and the cursor is absolute, so a client that falls behind sees a gap
#: rather than silently re-reading the wrong lines.
MAX_LINES = 5000


class JobBusy(RuntimeError):
    """Something is already running this stream."""


@dataclass
class Job:
    id: str
    stream_id: str
    kind: str = KIND_RUN
    state: str = RUNNING
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    error: str | None = None
    plan: list[dict] = field(default_factory=list)
    stages_run: list[dict] = field(default_factory=list)

    _lines: list[str] = field(default_factory=list, repr=False)
    _dropped: int = field(default=0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # -- written from the job thread ---------------------------------------

    def append(self, line: str) -> None:
        with self._lock:
            self._lines.append(str(line))
            if len(self._lines) > MAX_LINES:
                overflow = len(self._lines) - MAX_LINES
                del self._lines[:overflow]
                self._dropped += overflow

    def finish(self, state: str, error: str | None = None) -> None:
        with self._lock:
            self.state = state
            self.error = error
            self.finished_at = time.time()

    # -- read from request threads -----------------------------------------

    @property
    def live(self) -> bool:
        return self.state == RUNNING

    def snapshot(self, since: int = 0) -> dict:
        """State plus only the lines the caller has not seen.

        `next_cursor` is an absolute line index, so trimming the buffer cannot
        make an old cursor point at the wrong line — it points before the
        window, and the reply says how many were dropped.
        """
        with self._lock:
            first = self._dropped
            start = max(int(since) - first, 0)
            lines = self._lines[start:]
            return {
                "id": self.id,
                "stream_id": self.stream_id,
                "kind": self.kind,
                "state": self.state,
                "error": self.error,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "elapsed_s": round((self.finished_at or time.time()) - self.started_at, 1),
                "plan": list(self.plan),
                "stages_run": list(self.stages_run),
                "lines": lines,
                "next_cursor": first + len(self._lines),
                "dropped": max(first - int(since), 0),
            }


def decision_json(decision: runner.Decision) -> dict:
    return {
        "stage": decision.stage,
        "action": decision.action,
        "reason": decision.reason,
        "phase": decision.phase,
        "will_run": decision.will_run,
    }


def live_elsewhere(conn, stream_id: str) -> str | None:
    """A stage another process is working on right now, if any.

    Keyed on heartbeat age rather than on the `running` status alone: a status
    left behind by a crash outlives the process forever, and treating that as
    live would wedge the stream permanently — which is exactly what
    `reclaim_crashed` exists to undo.
    """
    rows = conn.execute(
        """
        SELECT stage, host, pid,
               CAST((julianday('now') - julianday(heartbeat_at)) * 86400.0 AS INTEGER)
                   AS age_s
          FROM pipeline_stages
         WHERE stream_id = ? AND status = 'running'
        """,
        (stream_id,),
    ).fetchall()

    for row in rows:
        age = row["age_s"]
        if age is None or age >= runner.LIVE_HEARTBEAT_S:
            continue
        if running_process_is_us(row["host"], row["pid"]):
            continue
        return f"{row['stage']} (host {row['host']}, pid {row['pid']}, {age}s ago)"
    return None


class JobRegistry:
    """Every pipeline run this process has started."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._by_stream: dict[str, str] = {}
        self._lock = threading.Lock()

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def for_stream(self, stream_id: str) -> Job | None:
        """The most recent job for a stream, running or finished."""
        with self._lock:
            job_id = self._by_stream.get(stream_id)
            return self._jobs.get(job_id) if job_id else None

    def start(self, cfg: Config, stream_id: str, kind: str = KIND_RUN) -> Job:
        with self._lock:
            existing = self._jobs.get(self._by_stream.get(stream_id, ""))
            if existing is not None and existing.live:
                raise JobBusy(
                    f"{stream_id} is already running (job {existing.id})."
                )

            # Checked while holding the registry lock so two requests cannot
            # both pass it and then both start.
            conn = db.open_db(cfg.db_path, migrate_to_latest=False)
            try:
                busy = live_elsewhere(conn, stream_id)
            finally:
                conn.close()
            if busy is not None:
                raise JobBusy(
                    f"another clipforge process is already working on {stream_id}: {busy}. "
                    f"This design assumes a single writer — starting a second run would "
                    f"mark that stage dead and re-run it on top of the file it is writing."
                )

            job = Job(id=uuid.uuid4().hex[:12], stream_id=stream_id, kind=kind)
            self._jobs[job.id] = job
            self._by_stream[stream_id] = job.id

        work = _RUNNERS.get(kind)
        if work is None:
            raise ValueError(f"unknown job kind {kind!r}")
        thread = threading.Thread(
            target=work, args=(job, cfg),
            name=f"clipforge-{kind}-{stream_id}", daemon=True,
        )
        thread.start()
        return job


def _run_job(job: Job, cfg: Config) -> None:
    """The job thread.

    Opens its own connection: `db.connect` leaves `check_same_thread` at its
    default, so a connection created for a request cannot be used here — and
    should not be, since this outlives the request that started it.
    """
    conn = None
    try:
        conn = db.open_db(cfg.db_path, migrate_to_latest=False)
        engine = runner.Runner(cfg=cfg, conn=conn, stream_id=job.stream_id, log=job.append)

        reclaimed = engine.reclaim_crashed()
        if reclaimed:
            job.append(
                f"  reclaimed {len(reclaimed)} stage(s) left running by a dead process: "
                f"{', '.join(reclaimed)}"
            )
        swept = engine.sweep()
        if swept:
            job.append(f"  swept {len(swept)} orphaned temp file(s) from an interrupted run")

        decisions = engine.plan()
        job.plan = [decision_json(d) for d in decisions]
        planned = [d.stage for d in decisions if d.will_run]
        job.append(
            f"  {len(planned)} stage(s) to run: {', '.join(planned)}"
            if planned else "  everything up to date"
        )

        results = engine.execute(decisions)
        job.stages_run = [
            {"stage": r.stage, "status": r.status,
             "duration_s": round(r.duration_s, 1), "error": r.error}
            for r in results
        ]

        failed = [r for r in results if r.status == "failed"]
        if failed:
            job.finish(FAILED, f"{failed[0].stage}: {failed[0].error}")
        else:
            job.finish(DONE)
    except BaseException as exc:  # noqa: BLE001 — a job thread must never die silently
        job.append(f"  {type(exc).__name__}: {exc}")
        job.finish(FAILED, f"{type(exc).__name__}: {exc}")
    finally:
        if conn is not None:
            conn.close()


def _render_job(job: Job, cfg: Config) -> None:
    """§8's auto-finish, in the background.

    Deliberately the same machinery as a pipeline run — buffer, absolute
    cursor, one-writer guard — because from the browser's point of view they
    are the same thing: something long is happening and the log is how you
    watch it. The run view's polling reads this with no change.

    Defaults only. `clipforge render` has `--template`, `--stills` and
    `--dry-run`, and those belong on the command line: they are the loop for
    fitting crop coordinates to a real layout, which is iteration, not a
    one-click action.
    """
    from clipforge.render import clips

    conn = None
    try:
        conn = db.open_db(cfg.db_path, migrate_to_latest=False)
        options = clips.Options()
        job.append(f"  rendering {job.stream_id} (rating >= {options.min_rating})")

        result = clips.render_stream(cfg, conn, job.stream_id, options,
                                     log=job.append)
        stream_paths = paths.StreamPaths(cfg.data_root, job.stream_id)
        clips.record(conn, cfg, job.stream_id, result, stream_paths, options)

        total = sum(clip.size_bytes for clip in result.clips)
        job.append(
            f"  {len(result.clips)} clip(s), {total / 1_000_000:.1f} MB, "
            f"template {result.template.name if result.template else '?'}"
        )
        if result.clips:
            job.append(f"  in {result.clips[0].destination.parent}")
        job.finish(DONE)
    except BaseException as exc:  # noqa: BLE001 — a job thread must never die silently
        job.append(f"  {type(exc).__name__}: {exc}")
        job.finish(FAILED, f"{type(exc).__name__}: {exc}")
    finally:
        if conn is not None:
            conn.close()


#: Kind -> the function that does the work. Consulted in `start`, so an unknown
#: kind fails there rather than starting a thread that does nothing.
_RUNNERS = {KIND_RUN: _run_job, KIND_RENDER: _render_job}
