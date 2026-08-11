"""The stage runner — idempotency, resumability, and staleness (C7, §5.1, A10).

§5.1 states the rule as:

> "Each stage checks `pipeline_stages` for `status='done'` and skips."

That is necessary and not sufficient, and building to it literally produces a
pipeline that lies. Three things go wrong:

1. **The output may not be there.** A `done` row says a stage once succeeded,
   not that its artifact still exists. Delete `data/` and every stage still
   reports done.
2. **The settings may have changed.** Re-encoding a proxy at a different
   bitrate must re-run. `done` cannot express that.
3. **An upstream stage may have re-run.** This is the dangerous one. Fix the
   track mapping, re-run `probe`, and `proxy` — built from the old mapping —
   stays `done` forever. The pipeline reports success while serving artifacts
   derived from superseded inputs, which is exactly the outcome C7 exists to
   prevent.

So a stage skips only when it is `done`, its `params_hash` matches, its output
verifies, and no dependency has completed more recently than it has. That last
comparison is keyed on `completed_seq` rather than `finished_at`, because
whole-second timestamps cannot separate two stages that finish in the same
second — see migration 0002.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from clipforge import db
from clipforge.config import Config, canonical_json
from clipforge.pipeline import stages
from clipforge.pipeline.atomic import sweep_orphans
from clipforge.pipeline.context import StageContext
from clipforge.pipeline.stages import StageSpec

#: A `running` row whose heartbeat is fresher than this probably belongs to a
#: live process, not a crashed one. Single-writer is the documented assumption;
#: this makes violating it visible instead of silent.
LIVE_HEARTBEAT_S = 60

RUN = "run"
SKIP = "skip"
BLOCKED = "blocked"
DEFERRED = "deferred"
EXTERNAL_DONE = "external"


@dataclass
class Decision:
    """What will happen to one stage, and why."""

    stage: str
    action: str
    reason: str
    phase: int = 1

    @property
    def will_run(self) -> bool:
        return self.action == RUN


@dataclass
class StageResult:
    stage: str
    status: str
    duration_s: float
    error: str | None = None


@dataclass
class Runner:
    cfg: Config
    conn: sqlite3.Connection
    stream_id: str
    force: set[str] = field(default_factory=set)
    cascade: bool = True
    log: Callable[[str], None] = print

    def __post_init__(self) -> None:
        stages.validate_graph()
        self.ctx = StageContext(
            cfg=self.cfg, conn=self.conn, stream_id=self.stream_id, log=self.log
        )

    # -- pipeline_stages bookkeeping ---------------------------------------

    def row(self, stage: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM pipeline_stages WHERE stream_id = ? AND stage = ?",
            (self.stream_id, stage),
        ).fetchone()

    def _ensure_row(self, stage: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO pipeline_stages (stream_id, stage, status, attempt_count) "
            "VALUES (?, ?, 'pending', 0)",
            (self.stream_id, stage),
        )

    def _next_seq(self) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(completed_seq), 0) + 1 AS n FROM pipeline_stages"
        ).fetchone()
        return int(row["n"])

    def _write_done(self, stage: str, params_hash: str) -> None:
        self._ensure_row(stage)
        self.conn.execute(
            """
            UPDATE pipeline_stages
               SET status        = 'done',
                   finished_at   = datetime('now'),
                   started_at    = COALESCE(started_at, datetime('now')),
                   heartbeat_at  = datetime('now'),
                   params_hash   = ?,
                   completed_seq = ?,
                   host          = ?,
                   pid           = ?,
                   error         = NULL
             WHERE stream_id = ? AND stage = ?
            """,
            (params_hash, self._next_seq(), socket.gethostname(), os.getpid(),
             self.stream_id, stage),
        )

    def mark_external_done(self, stage: str, params_hash: str = "") -> None:
        """Complete a stage from a command rather than from `run`.

        `register_stream` is §5.1's first stage but cannot be a runnable one:
        it needs the master path before any `streams` row exists to read it
        from. `clipforge register` does the work and calls this.
        """
        with db.transaction(self.conn):
            self._write_done(stage, params_hash)

    def _fail(self, stage: str, error: str) -> None:
        with db.transaction(self.conn):
            self.conn.execute(
                "UPDATE pipeline_stages SET status = 'failed', error = ?, "
                "finished_at = datetime('now') WHERE stream_id = ? AND stage = ?",
                (error, self.stream_id, stage),
            )

    # -- crash recovery ----------------------------------------------------

    def reclaim_crashed(self) -> list[str]:
        """Treat any `running` row as a dead run.

        A crashed process cannot clean up after itself, so `running` outlives it
        forever. Without this, one kill wedges the stream permanently.
        """
        rows = self.conn.execute(
            """
            SELECT stage, host, pid, heartbeat_at,
                   CAST((julianday('now') - julianday(heartbeat_at)) * 86400.0 AS INTEGER)
                       AS heartbeat_age_s
              FROM pipeline_stages
             WHERE stream_id = ? AND status = 'running'
            """,
            (self.stream_id,),
        ).fetchall()

        reclaimed = []
        for row in rows:
            age = row["heartbeat_age_s"]
            if age is not None and age < LIVE_HEARTBEAT_S:
                self.log(
                    f"  WARNING  {row['stage']} is marked running with a {age}s-old "
                    f"heartbeat (host {row['host']}, pid {row['pid']}). Another "
                    f"clipforge may be working on this stream right now — this design "
                    f"assumes a single writer and does not defend against two."
                )
            with db.transaction(self.conn):
                self.conn.execute(
                    "UPDATE pipeline_stages SET status = 'failed', "
                    "error = 'process exited without finishing this stage', "
                    "attempt_count = attempt_count + 1 "
                    "WHERE stream_id = ? AND stage = ?",
                    (self.stream_id, row["stage"]),
                )
            reclaimed.append(row["stage"])
        return reclaimed

    def sweep(self) -> list[Path]:
        """Remove temp files left behind by dead runs (A10's missing half)."""
        return sweep_orphans(self.ctx.paths.dir)

    # -- planning ----------------------------------------------------------

    def params_hash(self, spec: StageSpec) -> str:
        payload = {"stage": spec.name, "params": spec.params(self.ctx)}
        return hashlib.sha256(canonical_json(payload).encode()).hexdigest()[:16]

    def plan(self, wanted: list[str] | None = None) -> list[Decision]:
        """Decide what happens to every stage, in dependency order."""
        ordered = stages.resolve_order(wanted or stages.default_plan())
        decisions: list[Decision] = []
        will_run: set[str] = set()
        satisfied: set[str] = set()

        for name in ordered:
            spec = stages.get(name)
            decision = self._decide(spec, will_run, satisfied)
            decisions.append(decision)
            if decision.will_run:
                will_run.add(name)
                satisfied.add(name)
            elif decision.action in (SKIP, EXTERNAL_DONE):
                satisfied.add(name)

        return decisions

    def _decide(self, spec: StageSpec, will_run: set[str], satisfied: set[str]) -> Decision:
        row = self.row(spec.name)

        unmet = [d for d in spec.requires if d not in satisfied]
        if unmet:
            return Decision(
                spec.name, BLOCKED, f"waiting on {', '.join(sorted(unmet))}", spec.phase
            )

        if spec.external:
            if row is not None and row["status"] == "done":
                return Decision(spec.name, EXTERNAL_DONE, "completed by command", spec.phase)
            return Decision(
                spec.name, BLOCKED, f"not registered: run `{spec.external_hint}`", spec.phase
            )

        if not spec.implemented:
            return Decision(spec.name, DEFERRED, f"not built (phase {spec.phase})", spec.phase)

        if spec.name in self.force:
            return Decision(spec.name, RUN, "forced", spec.phase)

        if row is None:
            return Decision(spec.name, RUN, "never run", spec.phase)

        if row["status"] != "done":
            detail = f"last status {row['status']}"
            if row["error"]:
                detail += f": {row['error'][:60]}"
            return Decision(spec.name, RUN, detail, spec.phase)

        if row["params_hash"] != self.params_hash(spec):
            return Decision(spec.name, RUN, "parameters changed", spec.phase)

        ok, why = spec.verify(self.ctx)
        if not ok:
            return Decision(spec.name, RUN, why or "output failed verification", spec.phase)

        if self.cascade:
            rerunning = sorted(d for d in spec.requires if d in will_run)
            if rerunning:
                return Decision(
                    spec.name, RUN, f"{', '.join(rerunning)} is re-running", spec.phase
                )
            stale = self._stale_dependency(spec, row)
            if stale:
                return Decision(spec.name, RUN, f"{stale} completed more recently", spec.phase)

        return Decision(spec.name, SKIP, "up to date", spec.phase)

    def _stale_dependency(self, spec: StageSpec, row: sqlite3.Row) -> str | None:
        """The make rule: a dependency that finished after me makes me stale."""
        mine = row["completed_seq"]
        if mine is None:
            return None
        for dep in spec.requires:
            dep_row = self.row(dep)
            if dep_row is not None and dep_row["completed_seq"] is not None:
                if dep_row["completed_seq"] > mine:
                    return dep
        return None

    # -- execution ---------------------------------------------------------

    def execute(self, decisions: list[Decision]) -> list[StageResult]:
        """Run the stages the plan selected, stopping at the first failure.

        Continuing past a failure would run stages against inputs that are not
        there, producing a second, less informative error.
        """
        results: list[StageResult] = []
        for decision in decisions:
            if not decision.will_run:
                continue
            result = self._run_stage(stages.get(decision.stage))
            results.append(result)
            if result.status == "failed":
                break
        return results

    def _run_stage(self, spec: StageSpec) -> StageResult:
        self.ctx._stage = spec.name
        self.ctx._last_heartbeat = 0.0
        params_hash = self.params_hash(spec)

        with db.transaction(self.conn):
            self._ensure_row(spec.name)
            self.conn.execute(
                """
                UPDATE pipeline_stages
                   SET status = 'running', started_at = datetime('now'),
                       heartbeat_at = datetime('now'), host = ?, pid = ?,
                       attempt_count = attempt_count + 1, error = NULL
                 WHERE stream_id = ? AND stage = ?
                """,
                (socket.gethostname(), os.getpid(), self.stream_id, spec.name),
            )

        self.log(f"  {spec.name} ...")
        started = time.perf_counter()

        try:
            spec.load().run(self.ctx)
        except Exception as exc:
            elapsed = time.perf_counter() - started
            self._fail(spec.name, f"{type(exc).__name__}: {exc}")
            self._record_duration(spec.name, elapsed, "failed")
            self.log(f"  {spec.name} FAILED after {elapsed:.1f}s\n    {exc}")
            return StageResult(spec.name, "failed", elapsed, str(exc))

        # Verify before claiming success: a stage that returns without writing
        # its output must not leave a `done` row behind for the next run to
        # trust.
        ok, why = spec.verify(self.ctx)
        elapsed = time.perf_counter() - started
        if not ok:
            self._fail(spec.name, f"completed but output did not verify: {why}")
            self._record_duration(spec.name, elapsed, "failed")
            self.log(f"  {spec.name} FAILED verification after {elapsed:.1f}s ({why})")
            return StageResult(spec.name, "failed", elapsed, why)

        with db.transaction(self.conn):
            self._write_done(spec.name, params_hash)
        self._record_duration(spec.name, elapsed, "done")
        self.log(f"  {spec.name} done in {elapsed:.1f}s")
        return StageResult(spec.name, "done", elapsed)

    def _record_duration(self, stage: str, seconds: float, status: str) -> None:
        """§14 `stage_duration_s`.

        Only for stages that actually ran. Recording skips as zero-second
        durations would drown the bottleneck analysis this metric exists for.
        """
        row = self.row(stage)
        meta = json.dumps(
            {"stage": stage, "status": status, "attempt": row["attempt_count"] if row else None}
        )
        self.ctx.metric("stage_duration_s", seconds, meta)
