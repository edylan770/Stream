"""Runner behaviour: idempotency, staleness, crash recovery (C7, §5.1, A10).

The runner is exercised against fake stages rather than real ones. Real stages
need ffmpeg and minutes of encoding; the decisions being tested here are about
bookkeeping, and bookkeeping is easier to corner with a stage that writes one
byte on command and fails on demand.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from clipforge import config, db
from clipforge.pipeline import runner, stages
from clipforge.pipeline.atomic import atomic_output

STREAM_ID = "2026-08-14_test"


# --------------------------------------------------------------------------
# a miniature pipeline
# --------------------------------------------------------------------------


def make_stage_module(name, *, run=None, outputs=None, params=None, verify=None):
    module = types.ModuleType(f"tests._fake_{name}")
    module.run = run or (lambda ctx: None)
    if outputs is not None:
        module.outputs = outputs
    if params is not None:
        module.params = params
    if verify is not None:
        module.verify = verify
    sys.modules[module.__name__] = module
    return module.__name__


def writes_file(filename, payload=b"x"):
    """A stage that produces one artifact through the A10 path."""

    def run(ctx):
        with atomic_output(ctx.paths.dir / filename) as tmp:
            tmp.write_bytes(payload)

    def outputs(ctx):
        return [ctx.paths.dir / filename]

    return run, outputs


@pytest.fixture
def registry(monkeypatch):
    def install(specs):
        monkeypatch.setattr(stages, "STAGE_LIST", specs)
        monkeypatch.setattr(stages, "STAGES", {s.name: s for s in specs})
        monkeypatch.setattr(
            stages, "SPEC_ORDER", {s.name: i for i, s in enumerate(specs)}
        )
    return install


@pytest.fixture
def env(tmp_path, registry):
    """A database, a registered stream, and a two-stage pipeline."""
    cfg = config.load(overrides=[f"paths.data_root={tmp_path.as_posix()}"])
    conn = db.open_db(cfg.db_path)
    conn.execute(
        "INSERT INTO streams (id, date, master_path, marker_time_base) "
        "VALUES (?, '2026-08-14', 'D:/masters/x.mkv', 'vod')",
        (STREAM_ID,),
    )

    alpha_run, alpha_outputs = writes_file("alpha.bin")
    beta_run, beta_outputs = writes_file("beta.bin")
    state = {"alpha_params": {"v": 1}, "beta_fails": False, "alpha_calls": 0}

    def alpha(ctx):
        state["alpha_calls"] += 1
        alpha_run(ctx)

    def beta(ctx):
        if state["beta_fails"]:
            raise RuntimeError("beta exploded")
        beta_run(ctx)

    specs = [
        stages.StageSpec(
            name="register_stream", summary="", phase=1,
            external=True, external_hint="clipforge register",
        ),
        stages.StageSpec(
            name="alpha", summary="first", phase=1, requires=("register_stream",),
            module=make_stage_module(
                "alpha", run=alpha, outputs=alpha_outputs,
                params=lambda ctx: state["alpha_params"],
            ),
        ),
        stages.StageSpec(
            name="beta", summary="second", phase=1, requires=("alpha",),
            module=make_stage_module("beta", run=beta, outputs=beta_outputs),
        ),
        stages.StageSpec(name="gamma", summary="unbuilt", phase=3, requires=("beta",)),
    ]
    registry(specs)

    def make(**kwargs):
        return runner.Runner(cfg=cfg, conn=conn, stream_id=STREAM_ID,
                             log=lambda *_: None, **kwargs)

    yield types.SimpleNamespace(cfg=cfg, conn=conn, make=make, state=state, tmp=tmp_path)
    conn.close()


def actions(decisions):
    return {d.stage: d.action for d in decisions}


def run_all(engine):
    engine.reclaim_crashed()
    return engine.execute(engine.plan())


# --------------------------------------------------------------------------
# the three conditions §5.1 does not state
# --------------------------------------------------------------------------


def test_register_blocks_everything_until_it_is_done(env):
    engine = env.make()
    decisions = actions(engine.plan())
    assert decisions["register_stream"] == runner.BLOCKED
    assert decisions["alpha"] == runner.BLOCKED
    assert decisions["beta"] == runner.BLOCKED


def test_first_run_runs_everything_implemented(env):
    engine = env.make()
    engine.mark_external_done("register_stream")
    decisions = actions(engine.plan())
    assert decisions["alpha"] == runner.RUN
    assert decisions["beta"] == runner.RUN
    assert decisions["gamma"] == runner.DEFERRED


def test_second_run_is_all_skips(env):
    engine = env.make()
    engine.mark_external_done("register_stream")
    run_all(engine)

    decisions = actions(env.make().plan())
    assert decisions["alpha"] == runner.SKIP
    assert decisions["beta"] == runner.SKIP
    assert env.state["alpha_calls"] == 1


def test_deleted_output_forces_a_rerun(env):
    """A `done` row says a stage once succeeded, not that its artifact exists.
    §5.1's rule alone would skip forever after `rm -rf data/`."""
    engine = env.make()
    engine.mark_external_done("register_stream")
    run_all(engine)

    (env.tmp / "streams" / STREAM_ID / "alpha.bin").unlink()
    decision = next(d for d in env.make().plan() if d.stage == "alpha")
    assert decision.action == runner.RUN
    assert "missing output" in decision.reason


def test_changed_parameters_force_a_rerun(env):
    engine = env.make()
    engine.mark_external_done("register_stream")
    run_all(engine)

    env.state["alpha_params"] = {"v": 2}
    decision = next(d for d in env.make().plan() if d.stage == "alpha")
    assert decision.action == runner.RUN
    assert decision.reason == "parameters changed"


def test_failed_verification_forces_a_rerun(env, registry, monkeypatch):
    """A stage whose product is database rows cannot be checked by looking for
    files, so it supplies its own verify."""
    verdict = {"ok": True}
    specs = list(stages.STAGE_LIST)
    specs[1] = stages.StageSpec(
        name="alpha", summary="", phase=1, requires=("register_stream",),
        module=make_stage_module(
            "alpha_verify",
            verify=lambda ctx: (verdict["ok"], "rows are gone"),
        ),
    )
    registry(specs)

    engine = env.make()
    engine.mark_external_done("register_stream")
    run_all(engine)
    assert actions(env.make().plan())["alpha"] == runner.SKIP

    verdict["ok"] = False
    decision = next(d for d in env.make().plan() if d.stage == "alpha")
    assert decision.action == runner.RUN
    assert decision.reason == "rows are gone"


# --------------------------------------------------------------------------
# staleness cascade — the failure §5.1's rule cannot see
# --------------------------------------------------------------------------


def test_upstream_rerun_makes_downstream_stale(env):
    """probe re-runs, proxy was built from the old track map. Under §5.1's
    literal rule proxy stays done forever and serves a stale artifact."""
    engine = env.make()
    engine.mark_external_done("register_stream")
    run_all(engine)

    forced = env.make(force={"alpha"})
    decisions = {d.stage: d for d in forced.plan()}
    assert decisions["alpha"].action == runner.RUN
    assert decisions["beta"].action == runner.RUN
    assert decisions["beta"].reason == "alpha is re-running"


def test_staleness_survives_same_second_completions(env):
    """The reason completed_seq exists.

    These stages take microseconds, so `finished_at` — whole seconds via
    datetime('now') — is identical for the dependency and its dependent. A
    timestamp comparison reports "not stale" for a dependency that genuinely
    re-ran. The sequence counter does not.
    """
    engine = env.make()
    engine.mark_external_done("register_stream")
    run_all(engine)

    forced = env.make(force={"alpha"})
    forced.execute([d for d in forced.plan() if d.stage == "alpha"])

    alpha_row = forced.row("alpha")
    beta_row = forced.row("beta")
    assert alpha_row["finished_at"] == beta_row["finished_at"]  # same second
    assert alpha_row["completed_seq"] > beta_row["completed_seq"]

    decision = next(d for d in env.make().plan() if d.stage == "beta")
    assert decision.action == runner.RUN
    assert decision.reason == "alpha completed more recently"


def test_no_cascade_leaves_downstream_alone(env):
    engine = env.make()
    engine.mark_external_done("register_stream")
    run_all(engine)

    forced = env.make(force={"alpha"}, cascade=False)
    decisions = actions(forced.plan())
    assert decisions["alpha"] == runner.RUN
    assert decisions["beta"] == runner.SKIP


def test_completed_seq_is_monotonic(env):
    engine = env.make()
    engine.mark_external_done("register_stream")
    run_all(engine)
    seqs = [engine.row(s)["completed_seq"] for s in ("register_stream", "alpha", "beta")]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == 3


# --------------------------------------------------------------------------
# failure and recovery
# --------------------------------------------------------------------------


def test_failure_is_recorded_and_stops_the_run(env):
    env.state["beta_fails"] = True
    engine = env.make()
    engine.mark_external_done("register_stream")
    results = run_all(engine)

    assert [r.status for r in results] == ["done", "failed"]
    row = engine.row("beta")
    assert row["status"] == "failed"
    assert "beta exploded" in row["error"]


def test_a_failed_stage_reruns_next_time(env):
    env.state["beta_fails"] = True
    engine = env.make()
    engine.mark_external_done("register_stream")
    run_all(engine)

    env.state["beta_fails"] = False
    decision = next(d for d in env.make().plan() if d.stage == "beta")
    assert decision.action == runner.RUN
    assert "last status failed" in decision.reason


def test_attempt_count_increments(env):
    env.state["beta_fails"] = True
    engine = env.make()
    engine.mark_external_done("register_stream")
    run_all(engine)
    run_all(env.make())
    assert engine.row("beta")["attempt_count"] == 2


def test_a_stage_that_writes_nothing_is_not_marked_done(env, registry):
    """Otherwise the next run trusts a `done` row with no artifact behind it."""
    specs = list(stages.STAGE_LIST)
    specs[1] = stages.StageSpec(
        name="alpha", summary="", phase=1, requires=("register_stream",),
        module=make_stage_module(
            "alpha_lazy",
            run=lambda ctx: None,
            outputs=lambda ctx: [ctx.paths.dir / "never_written.bin"],
        ),
    )
    registry(specs)

    engine = env.make()
    engine.mark_external_done("register_stream")
    results = run_all(engine)
    assert results[0].status == "failed"
    assert engine.row("alpha")["status"] == "failed"
    assert "did not verify" in engine.row("alpha")["error"]


def test_crashed_running_row_is_reclaimed(env):
    """A killed process cannot clean up after itself, so `running` outlives it.
    Without reclaim, one kill wedges the stream permanently."""
    engine = env.make()
    engine.mark_external_done("register_stream")
    env.conn.execute(
        "INSERT INTO pipeline_stages (stream_id, stage, status, started_at, "
        "heartbeat_at, attempt_count) VALUES (?, 'alpha', 'running', "
        "datetime('now','-1 hour'), datetime('now','-1 hour'), 1)",
        (STREAM_ID,),
    )

    assert engine.reclaim_crashed() == ["alpha"]
    row = engine.row("alpha")
    assert row["status"] == "failed"
    assert row["attempt_count"] == 2
    assert actions(engine.plan())["alpha"] == runner.RUN


def test_fresh_heartbeat_warns_about_a_concurrent_run(env):
    messages = []
    engine = runner.Runner(
        cfg=env.cfg, conn=env.conn, stream_id=STREAM_ID, log=messages.append
    )
    env.conn.execute(
        "INSERT INTO pipeline_stages (stream_id, stage, status, started_at, "
        "heartbeat_at, attempt_count, host, pid) VALUES (?, 'alpha', 'running', "
        "datetime('now'), datetime('now'), 1, 'otherbox', 4242)",
        (STREAM_ID,),
    )
    engine.reclaim_crashed()
    assert any("single writer" in m for m in messages)
    assert any("otherbox" in m for m in messages)


def test_killed_stage_leaves_no_partial_artifact(env, registry):
    """A10 end to end: the file at the final path is never a half-written one."""
    specs = list(stages.STAGE_LIST)

    def explode(ctx):
        with atomic_output(ctx.paths.dir / "alpha.bin") as tmp:
            tmp.write_bytes(b"half an encode")
            raise RuntimeError("killed")

    specs[1] = stages.StageSpec(
        name="alpha", summary="", phase=1, requires=("register_stream",),
        module=make_stage_module(
            "alpha_killed", run=explode,
            outputs=lambda ctx: [ctx.paths.dir / "alpha.bin"],
        ),
    )
    registry(specs)

    engine = env.make()
    engine.mark_external_done("register_stream")
    run_all(engine)

    stream_dir = env.tmp / "streams" / STREAM_ID
    assert not (stream_dir / "alpha.bin").exists()
    assert not list(stream_dir.glob(".tmp-*"))


# --------------------------------------------------------------------------
# selection and instrumentation
# --------------------------------------------------------------------------


def test_force_all_reruns_everything(env):
    engine = env.make()
    engine.mark_external_done("register_stream")
    run_all(engine)

    forced = env.make(force={"alpha", "beta"})
    decisions = actions(forced.plan())
    assert decisions["alpha"] == decisions["beta"] == runner.RUN


def test_partial_plan_reports_unmet_dependencies(env):
    """--only beta with alpha never run should say so, not quietly run alpha."""
    engine = env.make()
    engine.mark_external_done("register_stream")
    decisions = {d.stage: d for d in engine.plan(["beta"])}
    assert decisions["beta"].action == runner.BLOCKED
    assert "alpha" in decisions["beta"].reason


def test_stage_duration_is_recorded_only_for_stages_that_ran(env):
    """§14. Logging skips as zero-second durations would drown the bottleneck
    analysis the metric exists for."""
    engine = env.make()
    engine.mark_external_done("register_stream")
    run_all(engine)
    run_all(env.make())  # second run: all skips

    rows = env.conn.execute(
        "SELECT value, meta FROM tool_metrics WHERE metric = 'stage_duration_s'"
    ).fetchall()
    assert len(rows) == 2
    assert {json.loads(r["meta"])["stage"] for r in rows} == {"alpha", "beta"}
    assert all(json.loads(r["meta"])["status"] == "done" for r in rows)


def test_failed_stages_are_recorded_with_their_status(env):
    env.state["beta_fails"] = True
    engine = env.make()
    engine.mark_external_done("register_stream")
    run_all(engine)

    metas = [
        json.loads(r["meta"])
        for r in env.conn.execute(
            "SELECT meta FROM tool_metrics WHERE metric = 'stage_duration_s'"
        )
    ]
    assert {m["stage"]: m["status"] for m in metas} == {"alpha": "done", "beta": "failed"}


def test_context_reads_the_stream_row_fresh(env):
    """probe mutates streams; the stages after it must see what it wrote, not
    the empty row that was there when the run started."""
    engine = env.make()
    assert engine.ctx.stream["duration_s"] is None
    env.conn.execute(
        "UPDATE streams SET duration_s = 682.5 WHERE id = ?", (STREAM_ID,)
    )
    assert engine.ctx.stream["duration_s"] == 682.5
