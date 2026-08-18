"""Running the pipeline in a background thread (the app shell's engine).

Numbers come from the fixture's `manifest.json`, never from constants, and the
set of stages a run should execute is computed from the stage registry rather
than listed here — otherwise building Phase 2 would leave this file asserting a
Phase 1 pipeline and passing.
"""

from __future__ import annotations

import threading
import time

import pytest

from clipforge import config, db
from clipforge.ingest import register
from clipforge.pipeline import stages
from clipforge.review import jobs


def make_workspace(root, fixture_dir, manifest):
    cfg = config.load(overrides=[f"paths.data_root={(root / 'data').as_posix()}"])
    db.open_db(cfg.db_path).close()

    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    try:
        result = register.register_stream(
            cfg, conn, register.RegisterSpec(master=fixture_dir / manifest["video"])
        )
    finally:
        conn.close()
    return cfg, result.stream_id


@pytest.fixture
def registered(tmp_path, fixture_dir, manifest):
    """A registered stream with nothing processed yet."""
    return make_workspace(tmp_path, fixture_dir, manifest)


@pytest.fixture(scope="module")
def completed(tmp_path_factory, fixture_dir, manifest):
    """One real end-to-end run, through a job, reused by the assertions below."""
    cfg, stream_id = make_workspace(tmp_path_factory.mktemp("jobs"), fixture_dir, manifest)
    registry = jobs.JobRegistry()
    job = wait_for(registry.start(cfg, stream_id))
    return cfg, stream_id, job


def wait_for(job, timeout_s: float = 600.0):
    deadline = time.monotonic() + timeout_s
    while job.live and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not job.live, f"job did not finish within {timeout_s}s"
    return job


def open_db(cfg):
    return db.open_db(cfg.db_path, migrate_to_latest=False)


# --------------------------------------------------------------------------
# a job runs the pipeline
# --------------------------------------------------------------------------


def test_the_job_completes(completed):
    _cfg, _stream_id, job = completed
    assert job.state == jobs.DONE, job.error


def test_it_runs_exactly_the_stages_that_are_built(completed):
    """Computed from the registry, so a new stage updates this test rather than
    passing while asserting an older pipeline.

    The predicate used to be `spec.phase == 1`, which held only while phase 1
    was the only phase with modules. It is now "implemented, and not deferred or
    blocked in this run" — which is the thing actually worth asserting: nothing
    that could have run was silently skipped, and everything that did not run
    said why. `previews` is what broke the old form, by being the first
    phase-3 stage that runs on a default stream.
    """
    _cfg, _stream_id, job = completed
    excused = {
        entry["stage"] for entry in job.plan
        if entry["action"] in ("deferred", "blocked")
    }
    expected = {
        spec.name for spec in stages.iter_specs()
        if spec.module is not None and spec.name not in excused
    }
    assert {entry["stage"] for entry in job.stages_run} == expected
    assert all(entry["status"] == "done" for entry in job.stages_run)
    # A stage excused without a reason is the failure this guards against: a
    # silent skip and a stated deferral look identical in the stages_run set.
    assert all(entry["reason"] for entry in job.plan
               if entry["stage"] in excused)


def test_the_plan_is_captured_with_its_reasons(completed):
    """§5.1's decision table is the most useful diagnostic in the pipeline, and
    the run view renders it — so the job has to keep it."""
    _cfg, _stream_id, job = completed
    assert job.plan
    assert all(entry["reason"] for entry in job.plan)
    assert {entry["stage"] for entry in job.plan} == set(stages.STAGES)


def test_the_probed_stream_matches_the_fixture(completed, manifest):
    cfg, stream_id, _job = completed
    conn = open_db(cfg)
    row = conn.execute("SELECT * FROM streams WHERE id = ?", (stream_id,)).fetchone()
    conn.close()

    assert row["duration_s"] == pytest.approx(manifest["duration_s"], abs=0.05)
    assert row["resolution"] == manifest["resolution"]
    assert bool(row["is_vfr"]) is manifest["expected_is_vfr"]
    assert row["proxy_path"]


def test_markers_and_candidates_reach_the_database(completed, manifest):
    cfg, stream_id, _job = completed
    conn = open_db(cfg)
    markers = conn.execute(
        "SELECT COUNT(*) FROM events WHERE stream_id = ? AND source = 'marker'", (stream_id,)
    ).fetchone()[0]
    candidates = conn.execute(
        "SELECT COUNT(*) FROM candidates WHERE stream_id = ? AND is_current = 1", (stream_id,)
    ).fetchone()[0]
    conn.close()

    assert markers == len(manifest["markers"])
    assert candidates >= cfg.get("score.peak.min_candidates")


def test_the_run_is_idempotent(completed):
    """C7. A second job over a finished stream must skip everything, not
    re-encode a proxy."""
    cfg, stream_id, _job = completed
    registry = jobs.JobRegistry()
    again = wait_for(registry.start(cfg, stream_id))

    assert again.state == jobs.DONE, again.error
    assert again.stages_run == []
    assert all(not entry["will_run"] for entry in again.plan)


# --------------------------------------------------------------------------
# failure
# --------------------------------------------------------------------------


def test_a_failing_stage_leaves_the_job_failed(registered):
    cfg, stream_id = registered
    conn = open_db(cfg)
    conn.execute(
        "UPDATE streams SET master_path = ? WHERE id = ?",
        (str(cfg.data_root / "gone.mkv"), stream_id),
    )
    conn.close()

    job = wait_for(jobs.JobRegistry().start(cfg, stream_id))
    assert job.state == jobs.FAILED
    assert "probe" in job.error
    assert job.finished_at is not None


def test_a_failure_does_not_wedge_the_stream(registered):
    """The registry must let you fix the problem and try again."""
    cfg, stream_id = registered
    conn = open_db(cfg)
    conn.execute(
        "UPDATE streams SET master_path = ? WHERE id = ?",
        (str(cfg.data_root / "gone.mkv"), stream_id),
    )
    conn.close()

    registry = jobs.JobRegistry()
    assert wait_for(registry.start(cfg, stream_id)).state == jobs.FAILED
    assert registry.start(cfg, stream_id) is not None      # not refused


# --------------------------------------------------------------------------
# one writer at a time
# --------------------------------------------------------------------------


@pytest.fixture
def blocking_run(monkeypatch):
    """Replace the job body with one that waits, so mutual exclusion can be
    tested without encoding a proxy."""
    release = threading.Event()

    def hold(job, _cfg):
        release.wait(30)
        job.finish(jobs.DONE)

    monkeypatch.setattr(jobs, "_run_job", hold)
    yield release
    release.set()


def test_a_second_job_for_the_same_stream_is_refused(registered, blocking_run):
    """`reclaim_crashed` treats any `running` row as dead, so a second run
    started mid-encode would declare the first one's stage dead and re-run it
    on top of the file it is still writing."""
    cfg, stream_id = registered
    registry = jobs.JobRegistry()
    first = registry.start(cfg, stream_id)

    with pytest.raises(jobs.JobBusy, match="already running"):
        registry.start(cfg, stream_id)

    blocking_run.set()
    wait_for(first)
    assert registry.start(cfg, stream_id) is not None


def test_a_live_stage_in_another_process_blocks_a_start(registered):
    """A terminal `clipforge run` on the same stream, typically."""
    cfg, stream_id = registered
    conn = open_db(cfg)
    conn.execute(
        "INSERT OR REPLACE INTO pipeline_stages "
        "(stream_id, stage, status, heartbeat_at, host, pid) "
        "VALUES (?, 'proxy', 'running', datetime('now'), 'other-pc', 424242)",
        (stream_id,),
    )
    conn.close()

    with pytest.raises(jobs.JobBusy, match="single writer"):
        jobs.JobRegistry().start(cfg, stream_id)


def test_a_stale_running_row_does_not_block(registered, blocking_run):
    """A crash leaves `running` behind forever. Treating that as live would
    wedge the stream permanently — which is what `reclaim_crashed` undoes."""
    cfg, stream_id = registered
    conn = open_db(cfg)
    conn.execute(
        "INSERT OR REPLACE INTO pipeline_stages "
        "(stream_id, stage, status, heartbeat_at, host, pid) "
        "VALUES (?, 'proxy', 'running', datetime('now', '-1 hour'), 'other-pc', 424242)",
        (stream_id,),
    )
    conn.close()

    assert jobs.JobRegistry().start(cfg, stream_id) is not None


def test_our_own_stale_row_does_not_block(registered, blocking_run):
    """A `running` row left by this process with no live job is a leftover, and
    `reclaim_crashed` is the thing that clears it."""
    import os
    import socket

    cfg, stream_id = registered
    conn = open_db(cfg)
    conn.execute(
        "INSERT OR REPLACE INTO pipeline_stages "
        "(stream_id, stage, status, heartbeat_at, host, pid) "
        "VALUES (?, 'proxy', 'running', datetime('now'), ?, ?)",
        (stream_id, socket.gethostname(), os.getpid()),
    )
    conn.close()

    assert jobs.JobRegistry().start(cfg, stream_id) is not None


# --------------------------------------------------------------------------
# the log cursor
# --------------------------------------------------------------------------


def test_the_cursor_returns_only_unseen_lines():
    """The browser polls; re-sending the whole log every 700 ms would grow
    without bound over a multi-hour encode."""
    job = jobs.Job(id="j", stream_id="s")
    job.append("one")
    job.append("two")

    first = job.snapshot()
    assert first["lines"] == ["one", "two"]
    assert first["next_cursor"] == 2

    job.append("three")
    later = job.snapshot(since=first["next_cursor"])
    assert later["lines"] == ["three"]
    assert later["next_cursor"] == 3
    assert later["dropped"] == 0


def test_a_client_that_falls_behind_is_told_lines_were_dropped(monkeypatch):
    """The cursor is an absolute index, so trimming cannot silently make an old
    cursor point at the wrong line."""
    monkeypatch.setattr(jobs, "MAX_LINES", 3)
    job = jobs.Job(id="j", stream_id="s")
    for n in range(6):
        job.append(str(n))

    snapshot = job.snapshot(since=0)
    assert snapshot["lines"] == ["3", "4", "5"]
    assert snapshot["dropped"] == 3
    assert snapshot["next_cursor"] == 6


def test_a_cursor_past_the_end_returns_nothing():
    job = jobs.Job(id="j", stream_id="s")
    job.append("one")
    assert job.snapshot(since=99)["lines"] == []


# --------------------------------------------------------------------------
# the registry
# --------------------------------------------------------------------------


def test_jobs_are_retrievable_by_id_and_by_stream(registered, blocking_run):
    cfg, stream_id = registered
    registry = jobs.JobRegistry()
    job = registry.start(cfg, stream_id)

    assert registry.get(job.id) is job
    assert registry.for_stream(stream_id) is job
    assert registry.get("nope") is None
    assert registry.for_stream("nope") is None
