"""§5.1 stage 11 — the OBS log parser, and the limits of what it can claim.

**These tests do not prove the patterns are right, and no test here can.** No
real OBS log has ever been seen by this code, so what is asserted is the
*arithmetic*, the *refusals* and the *structure*: elapsed time converts to
recording-relative seconds, a scene already up when recording began runs from
t=0, an ambiguous log is refused rather than guessed, and no wall clock is
consulted anywhere.

Whether `Switched to scene 'X'` is what OBS actually writes is settled by
`clipforge scene-events --check` against a real log, not by anything in here.

`tests/fixtures/obs_logs/` is parametrized: dropping a real log in there is the
whole integration step. A log with a `.expected.json` beside it gets exact
assertions; one without still has to parse without crashing.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from clipforge import config, db
from clipforge.extract import obs_log, scene_events
from clipforge.pipeline import stages
from clipforge.pipeline.context import StageContext

LOG_DIR = Path(__file__).parent / "fixtures" / "obs_logs"


def _logs() -> list[Path]:
    return sorted(LOG_DIR.glob("*.txt"))


def _expected(log: Path) -> dict | None:
    sidecar = log.with_suffix(".expected.json")
    if not sidecar.is_file():
        return None
    return json.loads(sidecar.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def patterns():
    return obs_log.Patterns.from_config(
        config.load().get("extract.scene_events.patterns") or {})


# --------------------------------------------------------------------------
# the pattern set — a bad edit must fail loudly and early
# --------------------------------------------------------------------------


def test_the_shipped_patterns_compile():
    """They are UNVALIDATED, not unparseable. A typo in the YAML is a different
    failure from a regex that does not match OBS, and only one of them is
    detectable here."""
    compiled = obs_log.Patterns.from_config(
        config.load().get("extract.scene_events.patterns") or {})
    assert compiled.elapsed and compiled.scene_switch


def test_an_unknown_pattern_key_is_refused_by_name():
    """A key nobody reads is a typo, and a typo that is ignored is a pattern the
    operator believes they fixed."""
    with pytest.raises(obs_log.ObsLogError, match="unknown"):
        obs_log.Patterns.from_config({"scene_swtich": "x"})


def test_a_pattern_that_captures_nothing_is_refused():
    """THE worst failure mode: it matches every line, finds no scene name, and
    writes zero events with no error anywhere."""
    raw = dict(config.load().get("extract.scene_events.patterns"))
    raw["scene_switch"] = "Switched to scene"          # no (?P<scene>...)
    with pytest.raises(obs_log.ObsLogError, match="must capture"):
        obs_log.Patterns.from_config(raw)


def test_a_missing_required_pattern_is_refused():
    raw = dict(config.load().get("extract.scene_events.patterns"))
    del raw["elapsed"]
    with pytest.raises(obs_log.ObsLogError, match="elapsed"):
        obs_log.Patterns.from_config(raw)


def test_recording_path_is_the_only_optional_one():
    """It ships empty on purpose: guessing it would silently select the wrong
    recording, which is worse than refusing to select at all."""
    raw = dict(config.load().get("extract.scene_events.patterns"))
    raw["recording_path"] = ""
    assert obs_log.Patterns.from_config(raw).recording_path is None


def test_a_bad_regex_is_refused_with_the_key_named():
    raw = dict(config.load().get("extract.scene_events.patterns"))
    raw["scene_switch"] = "(?P<scene>["
    with pytest.raises(obs_log.ObsLogError, match="scene_switch"):
        obs_log.Patterns.from_config(raw)


# --------------------------------------------------------------------------
# the clock — the part that IS certain
# --------------------------------------------------------------------------


@pytest.mark.parametrize("line, seconds", [
    ("00:00:00.000: x", 0.0),
    ("00:00:30.000: x", 30.0),
    ("00:01:00.500: x", 60.5),
    ("01:02:03.456: x", 3723.456),
    ("10:00:00.000: x", 36000.0),
])
def test_elapsed_is_read_from_the_line_prefix(patterns, line, seconds):
    assert obs_log.elapsed_seconds(line, patterns) == pytest.approx(seconds)


def test_a_line_without_a_prefix_has_no_elapsed(patterns):
    """Continuation lines and banners exist; they must not be read as t=0."""
    assert obs_log.elapsed_seconds("  not a timestamped line", patterns) is None
    assert obs_log.elapsed_seconds("", patterns) is None


def test_recording_relative_time_is_a_subtraction(patterns):
    rec = obs_log.Recording(start_elapsed=300.0, stop_elapsed=360.0)
    assert obs_log.to_recording_seconds(312.5, rec) == pytest.approx(12.5)
    # Before the recording started is negative, and that is correct rather than
    # clamped: `scene_spans` decides what to do with it, not the clock.
    assert obs_log.to_recording_seconds(290.0, rec) == pytest.approx(-10.0)


def test_no_wall_clock_is_reachable_from_the_parser():
    """STRUCTURAL, not a habit — the same tactic the capture layer's import test
    uses.

    Reading the log's filename, or its `Current Date/Time:` line, would drag in
    a timezone and a DST discontinuity and be wrong by an hour twice a year with
    nothing downstream able to notice (§4.1, A8). The guarantee is that the
    module cannot do it: it imports nothing that can tell the time.
    """
    source = (Path(scene_events.__file__).parent / "obs_log.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"datetime", "time", "calendar", "zoneinfo"}, (
        f"obs_log.py imports a clock: {sorted(imported)}")


# --------------------------------------------------------------------------
# every log in the fixture directory
# --------------------------------------------------------------------------


def test_there_is_at_least_one_log_to_parse():
    assert _logs(), f"no .txt files in {LOG_DIR}"


@pytest.mark.parametrize("log", _logs(), ids=lambda p: p.stem)
def test_every_log_parses_without_crashing(log, patterns):
    """The generic contract, so a REAL log dropped into this directory is
    covered the moment it lands, with no test changes."""
    parsed = obs_log.parse(obs_log.read(log), patterns)
    assert parsed.lines > 0
    assert parsed.looks_parsed


@pytest.mark.parametrize("log", _logs(), ids=lambda p: p.stem)
def test_the_elapsed_pattern_claims_most_lines(log, patterns):
    """The load-bearing pattern. If it fails, every other number is meaningless,
    which is why the diagnostic reports it first and separately."""
    parsed = obs_log.parse(obs_log.read(log), patterns)
    timestamped = parsed.lines - parsed.without_timestamp
    assert timestamped / parsed.lines > 0.5


@pytest.mark.parametrize("log", [p for p in _logs() if _expected(p)],
                         ids=lambda p: p.stem)
def test_a_log_with_ground_truth_matches_it(log, patterns):
    """Asserted against the `.expected.json` beside the log, never against
    numbers typed into this file — the rule every numeric test here follows."""
    want = _expected(log)
    parsed = obs_log.parse(obs_log.read(log), patterns)

    assert len(parsed.recordings) == want["recordings"]

    if want["select_recording"] == "ambiguous":
        with pytest.raises(obs_log.Ambiguous):
            obs_log.select_recording(parsed, "some-master.mkv")
        recording = parsed.recordings[want["recording_index_for_spans"]]
    else:
        recording = obs_log.select_recording(parsed, "some-master.mkv")

    assert recording.start_elapsed == pytest.approx(want["recording"]["start_elapsed"])
    assert recording.stop_elapsed == pytest.approx(want["recording"]["stop_elapsed"])

    spans = obs_log.scene_spans(parsed, recording, want["duration_s"])
    assert len(spans) == len(want["spans"])
    for got, expected in zip(spans, want["spans"], strict=True):
        assert got.t == pytest.approx(expected["t"])
        assert got.t_end == pytest.approx(expected["t_end"])
        assert got.scene == expected["scene"]
        assert got.previous == expected["previous"]


# --------------------------------------------------------------------------
# choosing a recording — the refusals
# --------------------------------------------------------------------------


def test_one_recording_wins_with_no_clock_and_no_path(patterns):
    parsed = obs_log.parse(
        "00:05:00.000: ==== Recording Start ====\n"
        "00:06:00.000: ==== Recording Stop ====\n", patterns)
    assert obs_log.select_recording(parsed, "anything.mkv").start_elapsed == 300.0


def test_two_recordings_and_no_way_to_tell_is_refused(patterns):
    """A wrong pick offsets every event by a constant, and §4.1 is explicit that
    nothing downstream can detect a constant offset."""
    parsed = obs_log.parse(
        "00:05:00.000: ==== Recording Start ====\n"
        "00:06:00.000: ==== Recording Stop ====\n"
        "00:09:00.000: ==== Recording Start ====\n"
        "00:10:00.000: ==== Recording Stop ====\n", patterns)
    with pytest.raises(obs_log.Ambiguous, match="constant"):
        obs_log.select_recording(parsed, "master.mkv")


def test_a_logged_output_path_picks_the_right_recording_exactly():
    """The clock-free way out of the ambiguity above, once a real log shows how
    OBS writes the path."""
    raw = dict(config.load().get("extract.scene_events.patterns"))
    raw["recording_path"] = r"Recording to (?P<path>.+)$"
    compiled = obs_log.Patterns.from_config(raw)

    parsed = obs_log.parse(
        "00:05:00.000: Recording to D:/rec/first.mkv\n"
        "00:05:00.100: ==== Recording Start ====\n"
        "00:06:00.000: ==== Recording Stop ====\n"
        "00:09:00.000: Recording to D:/rec/second.mkv\n"
        "00:09:00.100: ==== Recording Start ====\n"
        "00:10:00.000: ==== Recording Stop ====\n", compiled)

    assert len(parsed.recordings) == 2
    chosen = obs_log.select_recording(parsed, "second.mkv")
    assert chosen.start_elapsed == pytest.approx(540.1)
    # Matched on the filename, so a master that has since moved still matches.
    assert obs_log.select_recording(parsed, "E:/elsewhere/first.mkv").start_elapsed \
        == pytest.approx(300.1)


def test_a_log_with_no_recording_at_all_is_refused(patterns):
    parsed = obs_log.parse("00:00:01.000: OBS 31.0.2\n", patterns)
    with pytest.raises(obs_log.ObsLogError, match="no recording start"):
        obs_log.select_recording(parsed, "master.mkv")


# --------------------------------------------------------------------------
# spans
# --------------------------------------------------------------------------


def test_the_scene_already_up_when_recording_began_runs_from_zero(patterns):
    """OBS logs a switch when it happens, not when recording starts. Dropping
    the switch that precedes the recording would leave the opening of every
    stream with no scene at all."""
    parsed = obs_log.parse(
        "00:00:10.000: Switched to scene 'Starting Soon'\n"
        "00:05:00.000: ==== Recording Start ====\n"
        "00:05:20.000: Switched to scene 'Gameplay'\n"
        "00:06:00.000: ==== Recording Stop ====\n", patterns)
    spans = obs_log.scene_spans(parsed, obs_log.select_recording(parsed, None), 60.0)
    assert [s.scene for s in spans] == ["Starting Soon", "Gameplay"]
    assert spans[0].t == 0.0


def test_a_switch_after_the_recording_stopped_is_not_included(patterns):
    parsed = obs_log.parse(
        "00:05:00.000: ==== Recording Start ====\n"
        "00:05:20.000: Switched to scene 'Gameplay'\n"
        "00:06:00.000: ==== Recording Stop ====\n"
        "00:07:00.000: Switched to scene 'Ending'\n", patterns)
    spans = obs_log.scene_spans(parsed, obs_log.select_recording(parsed, None), 60.0)
    assert [s.scene for s in spans] == ["Gameplay"]


def test_the_last_span_runs_to_the_end_of_the_recording(patterns):
    parsed = obs_log.parse(
        "00:05:00.000: ==== Recording Start ====\n"
        "00:05:20.000: Switched to scene 'Gameplay'\n"
        "00:06:00.000: ==== Recording Stop ====\n", patterns)
    spans = obs_log.scene_spans(parsed, obs_log.select_recording(parsed, None), 60.0)
    assert spans[-1].t_end == pytest.approx(60.0)


def test_spans_never_exceed_the_recording_duration(patterns):
    """The stream's own duration wins over the log's idea of the stop time: a
    span past the end of the footage is an event referencing frames that do not
    exist."""
    parsed = obs_log.parse(
        "00:05:00.000: ==== Recording Start ====\n"
        "00:05:10.000: Switched to scene 'Gameplay'\n"
        "00:59:00.000: ==== Recording Stop ====\n", patterns)
    spans = obs_log.scene_spans(parsed, obs_log.select_recording(parsed, None), 30.0)
    assert all(s.t_end <= 30.0 for s in spans)


def test_a_log_with_no_switches_produces_no_spans(patterns):
    """A session with one scene up throughout is legitimate and is not an error."""
    parsed = obs_log.parse(
        "00:05:00.000: ==== Recording Start ====\n"
        "00:06:00.000: ==== Recording Stop ====\n", patterns)
    assert obs_log.scene_spans(parsed, obs_log.select_recording(parsed, None), 60.0) == []


def test_an_unstopped_recording_falls_back_to_the_stream_duration(patterns):
    """OBS being killed mid-recording is ordinary, and the footage still exists."""
    parsed = obs_log.parse(
        "00:05:00.000: ==== Recording Start ====\n"
        "00:05:10.000: Switched to scene 'Gameplay'\n", patterns)
    recording = obs_log.select_recording(parsed, None)
    assert recording.stop_elapsed is None
    spans = obs_log.scene_spans(parsed, recording, 45.0)
    assert spans[-1].t_end == pytest.approx(45.0)


# --------------------------------------------------------------------------
# the stage
# --------------------------------------------------------------------------


def test_the_stage_is_registered():
    spec = stages.get("scene_events")
    assert spec.implemented
    assert spec.module == "clipforge.extract.scene_events"
    # `probe` as well as registration: spans are clipped to the duration.
    assert set(spec.requires) == {"register_stream", "probe"}


@pytest.fixture
def stream_ctx(tmp_path):
    cfg = config.load(overrides=[f"paths.data_root={(tmp_path / 'data').as_posix()}"])
    conn = db.open_db(cfg.db_path)
    conn.execute(
        "INSERT INTO streams (id, date, title, master_path, duration_s) "
        "VALUES ('sc', '2026-08-18', 'Scenes', 'D:/rec/master.mkv', 600.0)")
    from clipforge import paths
    paths.StreamPaths(cfg.data_root, "sc").ensure()
    ctx = StageContext(cfg=cfg, conn=conn, stream_id="sc", log=lambda _m: None)
    yield ctx
    conn.close()


def test_it_defers_when_there_is_no_log(stream_ctx):
    """Deferred, not failed — the trap `whisperx` documented. A failure here
    would stop `execute` and `score` would never produce candidates."""
    ok, why = scene_events.available(stream_ctx)
    assert not ok
    assert "no OBS log" in why
    # And it says how to attach one, plus why nothing is auto-discovered.
    assert "--obs-log" in why


def test_it_defers_rather_than_guessing_an_ambiguous_log(stream_ctx):
    src = LOG_DIR / "synthetic-UNVALIDATED.txt"
    scene_events.log_path(stream_ctx).write_text(
        src.read_text(encoding="utf-8"), encoding="utf-8")
    ok, why = scene_events.available(stream_ctx)
    assert not ok
    assert "2 recordings" in why


def test_it_writes_one_row_per_scene_span(stream_ctx):
    src = LOG_DIR / "synthetic-one-recording-UNVALIDATED.txt"
    want = _expected(src)
    scene_events.log_path(stream_ctx).write_text(
        src.read_text(encoding="utf-8"), encoding="utf-8")
    stream_ctx.conn.execute("UPDATE streams SET duration_s = ? WHERE id = 'sc'",
                            (want["duration_s"],))

    assert scene_events.available(stream_ctx)[0]
    scene_events.run(stream_ctx)

    rows = stream_ctx.conn.execute(
        "SELECT t, t_end, source, kind, meta FROM events "
        "WHERE stream_id = 'sc' ORDER BY t").fetchall()
    assert len(rows) == len(want["spans"])
    for row, expected in zip(rows, want["spans"], strict=True):
        assert row["t"] == pytest.approx(expected["t"])
        # SPANS, not points: §9.3's consumer wants how long a scene was up, and
        # `menu_screen` settled this shape in commit 36.
        assert row["t_end"] == pytest.approx(expected["t_end"])
        assert row["source"] == scene_events.SOURCE
        assert row["kind"] == scene_events.KIND
        assert json.loads(row["meta"])["scene"] == expected["scene"]


def test_running_twice_does_not_duplicate_rows(stream_ctx):
    """C7. The stage replaces its own rows rather than appending."""
    src = LOG_DIR / "synthetic-one-recording-UNVALIDATED.txt"
    scene_events.log_path(stream_ctx).write_text(
        src.read_text(encoding="utf-8"), encoding="utf-8")
    stream_ctx.conn.execute("UPDATE streams SET duration_s = 60.0 WHERE id = 'sc'")

    scene_events.run(stream_ctx)
    first = stream_ctx.conn.execute(
        "SELECT COUNT(*) FROM events WHERE stream_id = 'sc'").fetchone()[0]
    scene_events.run(stream_ctx)
    second = stream_ctx.conn.execute(
        "SELECT COUNT(*) FROM events WHERE stream_id = 'sc'").fetchone()[0]
    assert first == second


def test_scene_change_carries_no_profile_weight():
    """§16 rejects it as a scorer: "Low value. Retained as a chapter-boundary
    tie-breaker only." A weight here would be a scoring signal built on a parser
    that has never seen a real log."""
    cfg = config.load()
    for profile in cfg.profiles:
        assert scene_events.KIND not in profile.weights


# --------------------------------------------------------------------------
# registration: how a log gets attached at all
# --------------------------------------------------------------------------


def test_register_accepts_an_explicit_obs_log(tmp_path, fixture_dir, manifest):
    r"""`--obs-log`, the same shape `--input` already has.

    Explicit because there is deliberately no search of
    %APPDATA%\obs-studio\logs: choosing the right log out of a directory needs
    a wall clock, which `obs_log.py` refuses to use, and choosing wrong shifts
    every event by a constant.
    """
    from clipforge.ingest import register

    log = tmp_path / "somewhere" / "2026-08-18 09-14-02.txt"
    log.parent.mkdir(parents=True)
    log.write_text((LOG_DIR / "synthetic-one-recording-UNVALIDATED.txt")
                   .read_text(encoding="utf-8"), encoding="utf-8")

    cfg = config.load(overrides=[f"paths.data_root={(tmp_path / 'data').as_posix()}"])
    db.open_db(cfg.db_path).close()
    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    try:
        pre = register.preflight(cfg, conn, register.RegisterSpec(
            master=fixture_dir / manifest["video"], obs_log=str(log)))
        assert pre.obs_log_path == log

        result = register.register_stream(cfg, conn, register.RegisterSpec(
            master=fixture_dir / manifest["video"], obs_log=str(log)), pre)
        from clipforge import paths as _paths
        stream_paths = _paths.StreamPaths(cfg.data_root, result.stream_id)
        copied = stream_paths.raw_dir / "obs-log.txt"
        assert copied.is_file()
        assert copied.read_bytes() == log.read_bytes()

        sources = json.loads(stream_paths.sources_json.read_text(encoding="utf-8"))
        assert sources["obs_log_source"] == str(log)
    finally:
        conn.close()


def test_a_stray_text_file_beside_the_master_is_not_taken_as_a_log(
        tmp_path, fixture_dir, manifest):
    """The beside-the-master fallback looks for `obs-log.txt` and nothing else.

    A `*.txt` glob there would claim any stray notes file, and a wrong OBS log
    is a silent constant offset rather than a visible error.
    """
    from clipforge.ingest import register

    stray = fixture_dir / "notes.txt"
    stray.write_text("not an OBS log", encoding="utf-8")
    try:
        cfg = config.load(
            overrides=[f"paths.data_root={(tmp_path / 'data').as_posix()}"])
        db.open_db(cfg.db_path).close()
        conn = db.open_db(cfg.db_path, migrate_to_latest=False)
        try:
            pre = register.preflight(cfg, conn, register.RegisterSpec(
                master=fixture_dir / manifest["video"]))
            assert pre.obs_log_path is None
        finally:
            conn.close()
    finally:
        stray.unlink()
