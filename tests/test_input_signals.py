"""§4.4's activity log into signals, and the ways a gap can lie.

Every log below is written through `capture/contract.py`'s own `input_record`,
so the shape under test is the shape the daemon writes. If the contract changes,
these break rather than drifting quietly past it — which is the entire reason
commit 16 put the contract in one module.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from clipforge import config, db, paths, signals
from clipforge.capture import contract
from clipforge.extract import input_signals
from clipforge.pipeline import runner, stages
from clipforge.pipeline.context import StageContext

#: A plausible anchor, in the epoch milliseconds A8 requires.
ANCHOR_MS = 1_755_123_456_789


def write_log(path, records):
    """A §4.4 log, written through the daemon's own record builder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with contract.JsonlSink(path) as sink:
        for record in records:
            sink.write(record)
    return path


def sample(offset_s, *, keys=0.0, clicks=0.0, velocity=0.0, anchor=ANCHOR_MS):
    return contract.input_record(
        anchor + int(offset_s * 1000), keys_per_s=keys,
        mouse_vel_px_s=velocity, clicks_per_s=clicks,
    )


def steady(start_s, end_s, *, hz=contract.INPUT_HZ, **kwargs):
    """A continuous run of records, at the logger's own rate."""
    step = 1.0 / hz
    n = int(round((end_s - start_s) * hz))
    return [sample(start_s + i * step, **kwargs) for i in range(n)]


@pytest.fixture(scope="module")
def cfg():
    return config.load()


# --------------------------------------------------------------------------
# parsing and the epoch -> VOD conversion (§4.1)
# --------------------------------------------------------------------------


def test_the_anchor_is_the_whole_conversion(tmp_path):
    """§4.1: one subtraction per record, no drift, no alignment hunting."""
    log = write_log(tmp_path / "input.jsonl", [
        sample(0.0, keys=1.0), sample(5.0, keys=2.0), sample(9.9, keys=3.0),
    ])
    parsed = input_signals.parse(log, record_start_epoch_ms=ANCHOR_MS, duration_s=10.0)

    assert parsed.times == pytest.approx([0.0, 5.0, 9.9])
    assert parsed.rate == pytest.approx([1.0, 2.0, 3.0])


def test_input_rate_is_keys_plus_clicks(tmp_path):
    """§5.4.1 names ONE signal — "keys+clicks per second" — not two."""
    log = write_log(tmp_path / "input.jsonl", [sample(1.0, keys=4.0, clicks=2.5)])
    parsed = input_signals.parse(log, record_start_epoch_ms=ANCHOR_MS, duration_s=10.0)

    assert parsed.rate == pytest.approx([6.5])


def test_a_daily_log_is_sliced_to_this_recording(tmp_path):
    """§4.5 forbids the capture daemons from depending on OBS to learn where it
    is writing, so the log is per DAY. A day with three streams has one file
    covering all three, and a sample from another one is not a sample of this.
    """
    log = write_log(tmp_path / "input-2026-08-17.jsonl", [
        sample(-600.0, keys=9.0),      # an earlier recording
        sample(2.0, keys=1.0),         # this one
        sample(4.0, keys=2.0),
        sample(3600.0, keys=9.0),      # a later one
    ])
    parsed = input_signals.parse(log, record_start_epoch_ms=ANCHOR_MS, duration_s=10.0)

    assert parsed.times == pytest.approx([2.0, 4.0])
    assert parsed.before_start == 1
    assert parsed.after_end == 1


def test_out_of_range_records_are_dropped_not_clamped(tmp_path):
    """Clamping would invent activity at second zero of every stream that shares
    a log with another — the rule `marker_events` already follows."""
    log = write_log(tmp_path / "input.jsonl", [sample(-30.0, keys=9.0)])
    parsed = input_signals.parse(log, record_start_epoch_ms=ANCHOR_MS, duration_s=10.0)

    assert parsed.kept == 0
    assert parsed.before_start == 1


def test_a_malformed_line_is_reported_rather_than_skipped(tmp_path):
    path = tmp_path / "input.jsonl"
    path.write_text('{"epoch_ms": "soon", "keys_per_s": 1}\n', encoding="utf-8")
    parsed = input_signals.parse(path, record_start_epoch_ms=ANCHOR_MS, duration_s=10.0)

    assert parsed.kept == 0
    assert parsed.malformed


def test_a_non_numeric_rate_is_malformed_not_zero(tmp_path):
    path = tmp_path / "input.jsonl"
    path.write_text(
        json.dumps({"epoch_ms": ANCHOR_MS + 1000, "keys_per_s": None,
                    "clicks_per_s": 0, "mouse_vel_px_s": 0}) + "\n",
        encoding="utf-8",
    )
    parsed = input_signals.parse(path, record_start_epoch_ms=ANCHOR_MS, duration_s=10.0)

    assert parsed.kept == 0
    assert parsed.malformed


# --------------------------------------------------------------------------
# THE gap rule
# --------------------------------------------------------------------------


def series_for(records, *, duration_s, signal_hz=10.0):
    parsed = input_signals.ParsedInput()
    for record in records:
        t = (record["epoch_ms"] - ANCHOR_MS) / 1000.0
        parsed.times.append(t)
        parsed.rate.append(record["keys_per_s"] + record["clicks_per_s"])
        parsed.velocity.append(record["mouse_vel_px_s"])
    return input_signals.to_series(
        parsed, signal_hz=signal_hz, duration_s=duration_s, source="test",
    )


def test_a_logger_gap_is_absent_not_idle(cfg):
    """THE trap this stage exists to avoid.

    §6.4's AFK penalty fires on *no input AND no speech*. If a stretch where the
    logger was down read as a zero input rate, a quiet passage of a stream would
    be penalised for something that was never observed.
    """
    records = steady(0.0, 20.0, keys=5.0) + steady(80.0, 100.0, keys=5.0)
    out = series_for(records, duration_s=100.0)

    times = out[input_signals.COVERAGE_KIND].times()
    hole = (times >= 25.0) & (times <= 75.0)

    assert np.all(out[input_signals.COVERAGE_KIND].values[hole] == 0.0)
    assert np.isnan(out[input_signals.RATE_KIND].values[hole]).all()
    assert not np.any(out[input_signals.RATE_KIND].values[hole] == 0.0)


def test_coverage_is_never_nan_because_it_is_always_known(cfg):
    """"The logger was not running here" is an observation about the logger, and
    it is available for every sample."""
    out = series_for(steady(0.0, 5.0, keys=1.0), duration_s=60.0)
    assert np.isfinite(out[input_signals.COVERAGE_KIND].values).all()


def test_a_real_zero_rate_is_kept_as_zero(cfg):
    """The other half: the logger running and recording no activity IS a
    measurement, and must not be confused with its absence."""
    out = series_for(steady(0.0, 10.0, keys=0.0, clicks=0.0), duration_s=10.0)

    rate = out[input_signals.RATE_KIND].values
    observed = out[input_signals.COVERAGE_KIND].values > 0.5
    assert observed.any()
    assert np.all(rate[observed] == 0.0)
    assert np.isfinite(rate[observed]).all()


def test_samples_in_one_bucket_are_averaged_not_summed(cfg):
    """Two 10 Hz records inside one 100 ms bucket describe the same tenth of a
    second twice, not twice as much input."""
    records = [sample(1.0, keys=4.0), sample(1.04, keys=6.0)]
    out = series_for(records, duration_s=10.0)

    rate = out[input_signals.RATE_KIND]
    assert rate.values[rate.index_at(1.05)] == pytest.approx(5.0)


def test_mouse_velocity_passes_through_untouched(cfg):
    out = series_for([sample(1.0, velocity=1840.5)], duration_s=10.0)
    velocity = out[input_signals.VELOCITY_KIND]

    assert velocity.values[velocity.index_at(1.05)] == pytest.approx(1840.5)
    assert velocity.params["unit"] == "px/s"


def test_every_series_lands_on_the_configured_grid(cfg):
    signal_hz = float(cfg.get("extract.signal_hz"))
    out = series_for(steady(0.0, 5.0, keys=1.0), duration_s=30.0, signal_hz=signal_hz)

    for kind in input_signals.KINDS:
        assert out[kind].sample_rate_hz == signal_hz
        assert len(out[kind]) == int(30.0 * signal_hz)


def test_an_empty_log_still_produces_a_full_length_mask(cfg):
    """A logger that never started is a stream fully covered by "unknown", not a
    stream with no series at all — `input_stillness` needs the mask either way.
    """
    out = series_for([], duration_s=30.0)

    assert np.all(out[input_signals.COVERAGE_KIND].values == 0.0)
    assert np.isnan(out[input_signals.RATE_KIND].values).all()


# --------------------------------------------------------------------------
# the stage in the pipeline
# --------------------------------------------------------------------------


def build_stream(tmp_path, fixture_dir, manifest, *, log=None, anchor=True):
    """A registered stream, optionally with an input log attached."""
    from clipforge.ingest import register

    cfg = config.load(overrides=[f"paths.data_root={(tmp_path / 'data').as_posix()}"])
    db.open_db(cfg.db_path).close()
    conn = db.open_db(cfg.db_path, migrate_to_latest=False)

    spec = register.RegisterSpec(
        master=fixture_dir / manifest["video"],
        input=str(log) if log else None,
        anchor=None if anchor else "",
    )
    result = register.register_stream(cfg, conn, spec)
    engine = runner.Runner(cfg=cfg, conn=conn, stream_id=result.stream_id,
                           log=lambda *_: None)
    engine.execute(engine.plan(["register_stream", "probe"]))
    ctx = StageContext(cfg=cfg, conn=conn, stream_id=result.stream_id,
                       log=lambda *_: None)
    return cfg, conn, ctx


def test_the_stage_defers_when_there_is_no_log(tmp_path, fixture_dir, manifest):
    """The state of every stream that exists today. §4.4's logger has never run
    on real footage, so this must be a deferral with a reason rather than a
    failure — `execute` stops at the first failure and `score` would never run.
    """
    cfg, conn, ctx = build_stream(tmp_path, fixture_dir, manifest)
    try:
        ok, why = stages.get("input_signals").available(ctx)
        assert not ok
        assert "input" in why.lower()
    finally:
        conn.close()


def test_the_stage_defers_when_there_is_no_anchor(tmp_path, fixture_dir, manifest,
                                                  monkeypatch):
    """§4.1: the records carry epoch milliseconds and the anchor is the only
    thing that converts them. Without it every sample would land somewhere
    else entirely, so the honest answer is to not produce the signal."""
    log = write_log(tmp_path / "capture" / "input-2026-08-17.jsonl",
                    steady(0.0, 5.0, keys=1.0))
    cfg, conn, ctx = build_stream(tmp_path, fixture_dir, manifest, log=log)
    try:
        conn.execute(
            "UPDATE streams SET record_start_epoch_ms = NULL WHERE id = ?",
            (ctx.stream_id,),
        )
        conn.commit()
        ok, why = stages.get("input_signals").available(ctx)
        assert not ok
        assert "anchor" in why.lower()
    finally:
        conn.close()


def test_the_stage_writes_all_three_series(tmp_path, fixture_dir, manifest):
    log = write_log(tmp_path / "capture" / "input-2026-08-17.jsonl",
                    steady(0.0, 30.0, keys=3.0, clicks=1.0, velocity=500.0))
    cfg, conn, ctx = build_stream(tmp_path, fixture_dir, manifest, log=log)
    try:
        ok, why = stages.get("input_signals").available(ctx)
        assert ok, why

        input_signals.run(ctx)
        present = set(signals.kinds(conn, ctx.stream_id))
        assert set(input_signals.KINDS) <= present

        verified, reason = input_signals.verify(ctx)
        assert verified, reason

        rate = signals.load(conn, ctx.stream_id, input_signals.RATE_KIND)
        observed = rate.values[np.isfinite(rate.values)]
        assert observed.size
        assert float(np.median(observed)) == pytest.approx(4.0)
    finally:
        conn.close()


def test_registration_finds_the_daemons_own_filename(tmp_path, fixture_dir, manifest):
    """`input_logger` writes `input-YYYY-MM-DD.jsonl`, not `input.jsonl`. Looking
    only for the bare name is a gap the operator hits on their first real
    stream."""
    from clipforge.ingest import register

    beside = fixture_dir / "input-2026-08-17.jsonl"
    write_log(beside, steady(0.0, 2.0, keys=1.0))
    try:
        cfg = config.load(
            overrides=[f"paths.data_root={(tmp_path / 'data').as_posix()}"])
        db.open_db(cfg.db_path).close()
        conn = db.open_db(cfg.db_path, migrate_to_latest=False)
        try:
            pre = register.preflight(cfg, conn, register.RegisterSpec(
                master=fixture_dir / manifest["video"]))
            assert pre.input_path == beside
        finally:
            conn.close()
    finally:
        beside.unlink()


def test_the_log_is_copied_into_the_stream(tmp_path, fixture_dir, manifest):
    log = write_log(tmp_path / "capture" / "input-2026-08-17.jsonl",
                    steady(0.0, 2.0, keys=1.0))
    cfg, conn, ctx = build_stream(tmp_path, fixture_dir, manifest, log=log)
    try:
        copied = paths.StreamPaths(cfg.data_root, ctx.stream_id).input_jsonl
        assert copied.is_file()
        assert copied.read_bytes() == log.read_bytes()

        sources = json.loads(
            (paths.StreamPaths(cfg.data_root, ctx.stream_id).sources_json)
            .read_text(encoding="utf-8"))
        assert sources["input_source"] == str(log)
    finally:
        conn.close()


# --------------------------------------------------------------------------
# the graph — the regression that matters
# --------------------------------------------------------------------------


def test_score_does_not_depend_on_input_signals():
    """A DEFERRED stage never enters the runner's `satisfied` set, so anything
    requiring it is BLOCKED. No stream in existence has an input log, so making
    `score` depend on this one would take candidate production to zero on every
    stream — the same silent shape as commit 31's `0.0 * NaN`.

    It is also why `score.requires` has never included `whisperx`.
    """
    assert "input_signals" not in stages.get("score").requires
    assert "input_signals" not in stages.ancestors("score")


def test_the_default_plan_still_reaches_score_without_a_log(tmp_path, fixture_dir,
                                                            manifest):
    """The same claim, end to end rather than by inspection."""
    cfg, conn, ctx = build_stream(tmp_path, fixture_dir, manifest)
    try:
        engine = runner.Runner(cfg=cfg, conn=conn, stream_id=ctx.stream_id,
                               log=lambda *_: None)
        plan = {d.stage: d for d in engine.plan()}

        assert plan["input_signals"].action == "deferred"
        assert plan["score"].action not in ("blocked",), plan["score"].reason

        results = {r.stage: r for r in engine.execute(engine.plan())}
        assert results["score"].status == "done", results["score"].error
        assert conn.execute(
            "SELECT COUNT(*) FROM candidates WHERE stream_id = ?", (ctx.stream_id,)
        ).fetchone()[0] > 0
    finally:
        conn.close()


def test_input_signals_needs_the_duration_it_slices_by():
    """It cuts a daily log down to one recording, which cannot be done without
    knowing how long the recording is."""
    assert "probe" in stages.get("input_signals").requires


def test_the_stage_is_registered_as_implemented():
    spec = stages.get("input_signals")
    assert spec.implemented
    assert spec.module == "clipforge.extract.input_signals"


# --------------------------------------------------------------------------
# what the derived layer does with it
# --------------------------------------------------------------------------


def test_stillness_becomes_computable_once_a_log_exists(cfg):
    """`input_stillness` has been inert since commit 32 for want of these two
    series. Nothing about it changes — it simply has inputs now."""
    from clipforge.score import derived, grid

    duration = 60.0
    out = series_for(
        steady(0.0, 20.0, keys=8.0) + steady(20.0, duration, keys=0.0),
        duration_s=duration,
    )
    timeline = grid.build(duration, float(cfg.get("score.score_grid_hz")))
    raw = {kind: grid.resample(out[kind], timeline) for kind in input_signals.KINDS}

    assert not derived.missing_inputs(raw).get("input_stillness")
    values, _units, _events = derived.compute(raw, timeline, cfg)
    assert (values["input_stillness"] > 0).any(), "stillness after activity"
