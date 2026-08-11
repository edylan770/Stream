"""marker_events: §4.1's conversion, and what the events row is allowed to know."""

from __future__ import annotations

import json

import pytest

from clipforge import cli, config, db
from clipforge.extract import markers
from clipforge.pipeline import runner

ANCHOR = 1_755_123_456_789


def write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


def press(seconds, kind="marker_definite", anchor=ANCHOR):
    return {"epoch_ms": anchor + int(seconds * 1000), "kind": kind}


# --------------------------------------------------------------------------
# §4.1: the entire sync solution is one subtraction
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seconds", [0.0, 1.5, 120.0, 14400.0])
def test_epoch_converts_to_vod_seconds(seconds):
    assert markers.to_vod_seconds(ANCHOR + int(seconds * 1000), ANCHOR) == pytest.approx(seconds)


def test_conversion_is_exact_at_millisecond_resolution():
    assert markers.to_vod_seconds(ANCHOR + 1, ANCHOR) == pytest.approx(0.001)


def test_a_press_before_record_start_is_negative():
    """The daemon runs before OBS does; the arithmetic must not hide it."""
    assert markers.to_vod_seconds(ANCHOR - 5000, ANCHOR) == pytest.approx(-5.0)


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------


def test_parses_both_hotkeys(tmp_path):
    path = write_jsonl(tmp_path / "m.jsonl", [
        press(10.0, "marker_maybe"), press(20.0, "marker_definite"),
    ])
    result = markers.parse(
        path, record_start_epoch_ms=ANCHOR, duration_s=60.0, retro_offset_s=20.0
    )
    assert [e["kind"] for e in result.events] == ["marker_maybe", "marker_definite"]
    assert [e["t"] for e in result.events] == pytest.approx([10.0, 20.0])


def test_events_are_sorted_by_time(tmp_path):
    """The daemon appends in press order, but a crash-and-resume could not."""
    path = write_jsonl(tmp_path / "m.jsonl", [press(30.0), press(10.0), press(20.0)])
    result = markers.parse(path, record_start_epoch_ms=ANCHOR, duration_s=60.0, retro_offset_s=20.0)
    assert [e["t"] for e in result.events] == pytest.approx([10.0, 20.0, 30.0])


def test_the_stored_time_is_the_raw_press_not_the_offset_anchor(tmp_path):
    """§4.3 anchors the window at t-20, but that offset is a §17 tunable.

    Writing the shifted value here would freeze it into every parsed stream and
    make retuning it cost a re-parse of the whole catalogue — the same trap as
    baselining RMS at extract time (C3).
    """
    path = write_jsonl(tmp_path / "m.jsonl", [press(35.0)])
    result = markers.parse(path, record_start_epoch_ms=ANCHOR, duration_s=60.0, retro_offset_s=20.0)
    assert result.events[0]["t"] == pytest.approx(35.0)     # not 15.0
    # ...but the offset in force at parse time is recorded, for provenance.
    assert result.events[0]["meta"]["retro_offset_s_at_parse"] == 20.0


def test_presses_before_record_start_are_dropped_not_clamped(tmp_path):
    """Clamping to zero would invent a marker at the start of every stream
    where the daemon was running before OBS."""
    path = write_jsonl(tmp_path / "m.jsonl", [
        {"epoch_ms": ANCHOR - 3000, "kind": "marker_definite"}, press(10.0),
    ])
    result = markers.parse(path, record_start_epoch_ms=ANCHOR, duration_s=60.0, retro_offset_s=20.0)
    assert len(result.events) == 1
    assert result.skipped_before_start == 1


def test_presses_after_the_recording_are_dropped(tmp_path):
    path = write_jsonl(tmp_path / "m.jsonl", [press(10.0), press(999.0)])
    result = markers.parse(path, record_start_epoch_ms=ANCHOR, duration_s=60.0, retro_offset_s=20.0)
    assert len(result.events) == 1
    assert result.skipped_after_end == 1


def test_unknown_kinds_are_counted_not_imported(tmp_path):
    """A typo in the daemon should be reported, not silently scored."""
    path = write_jsonl(tmp_path / "m.jsonl", [
        press(10.0), {"epoch_ms": ANCHOR, "kind": "marker_mabye"},
    ])
    result = markers.parse(path, record_start_epoch_ms=ANCHOR, duration_s=60.0, retro_offset_s=20.0)
    assert len(result.events) == 1
    assert result.unknown_kinds == {"marker_mabye": 1}


def test_malformed_lines_are_collected(tmp_path):
    (tmp_path / "m.jsonl").write_text(
        json.dumps(press(10.0)) + "\nnot json at all\n", encoding="utf-8"
    )
    result = markers.parse(
        tmp_path / "m.jsonl", record_start_epoch_ms=ANCHOR, duration_s=60.0, retro_offset_s=20.0
    )
    assert len(result.events) == 1
    assert len(result.malformed) == 1
    assert "line 2" in result.malformed[0]


def test_non_integer_epoch_is_malformed(tmp_path):
    """A8 requires integer epoch milliseconds."""
    path = write_jsonl(tmp_path / "m.jsonl", [{"epoch_ms": "1755123456789", "kind": "marker_maybe"}])
    result = markers.parse(path, record_start_epoch_ms=ANCHOR, duration_s=60.0, retro_offset_s=20.0)
    assert result.events == []
    assert "epoch_ms" in result.malformed[0]


def test_blank_lines_are_ignored(tmp_path):
    (tmp_path / "m.jsonl").write_text(
        f"\n{json.dumps(press(10.0))}\n\n", encoding="utf-8"
    )
    result = markers.parse(
        tmp_path / "m.jsonl", record_start_epoch_ms=ANCHOR, duration_s=60.0, retro_offset_s=20.0
    )
    assert len(result.events) == 1
    assert result.malformed == []


# --------------------------------------------------------------------------
# the stage, against the fixture's real markers
# --------------------------------------------------------------------------


@pytest.fixture
def parsed(tmp_path, fixture_dir, manifest):
    import shutil
    from clipforge import paths

    cfg = config.load(overrides=[f"paths.data_root={(tmp_path / 'data').as_posix()}"])
    conn = db.open_db(cfg.db_path)
    conn.execute(
        "INSERT INTO streams (id, date, master_path, marker_time_base, "
        "record_start_epoch_ms, duration_s) VALUES ('fx', '2026-08-14', ?, 'epoch', ?, ?)",
        (str(fixture_dir / manifest["video"]),
         json.loads((fixture_dir / manifest["anchor"]).read_text())["record_start_epoch_ms"],
         manifest["duration_s"]),
    )
    sp = paths.StreamPaths(cfg.data_root, "fx").ensure()
    shutil.copy2(fixture_dir / manifest["markers_file"], sp.markers_jsonl)

    engine = runner.Runner(cfg=cfg, conn=conn, stream_id="fx", log=lambda *_: None)
    engine.mark_external_done("register_stream")
    engine.execute(engine.plan(["register_stream", "marker_events"]))
    yield cfg, conn, engine
    conn.close()


def rows(conn):
    return conn.execute(
        "SELECT * FROM events WHERE stream_id='fx' AND source='marker' ORDER BY t"
    ).fetchall()


def test_stage_writes_every_press(parsed, manifest):
    cfg, conn, engine = parsed
    assert len(rows(conn)) == len(manifest["markers"])


def test_press_times_round_trip_through_the_anchor(parsed, manifest):
    """The fixture wrote epoch ms from a known anchor; parsing must recover the
    VOD times the manifest declares."""
    cfg, conn, engine = parsed
    for row, expected in zip(rows(conn), manifest["markers"], strict=True):
        assert row["t"] == pytest.approx(expected["t_press"], abs=0.001)
        assert row["kind"] == expected["kind"]


def test_stage_is_idempotent_and_does_not_duplicate(parsed, manifest):
    """`events` has an autoincrement id and no natural key, so re-running would
    otherwise double every marker."""
    cfg, conn, engine = parsed
    before = len(rows(conn))
    forced = runner.Runner(cfg=cfg, conn=conn, stream_id="fx",
                           force={"marker_events"}, log=lambda *_: None)
    forced.execute(forced.plan(["register_stream", "marker_events"]))
    assert len(rows(conn)) == before


def test_epoch_markers_without_an_anchor_are_refused(parsed):
    cfg, conn, engine = parsed
    conn.execute("UPDATE streams SET record_start_epoch_ms = NULL WHERE id='fx'")
    forced = runner.Runner(cfg=cfg, conn=conn, stream_id="fx",
                           force={"marker_events"}, log=lambda *_: None)
    results = forced.execute(forced.plan(["register_stream", "marker_events"]))
    assert results[0].status == "failed"
    assert "record_start_epoch_ms" in results[0].error


def test_a_stream_with_no_markers_is_not_an_error(tmp_path, fixture_dir, manifest):
    """Most Phase 1 streams have none, and a session where the operator never
    pressed anything is perfectly normal."""
    cfg = config.load(overrides=[f"paths.data_root={(tmp_path / 'data').as_posix()}"])
    conn = db.open_db(cfg.db_path)
    conn.execute(
        "INSERT INTO streams (id, date, master_path, marker_time_base, duration_s) "
        "VALUES ('bare', '2026-08-14', ?, 'vod', 60.0)",
        (str(fixture_dir / manifest["video"]),),
    )
    engine = runner.Runner(cfg=cfg, conn=conn, stream_id="bare", log=lambda *_: None)
    engine.mark_external_done("register_stream")
    results = engine.execute(engine.plan(["register_stream", "marker_events"]))
    assert results[0].status == "done"
    conn.close()


# --------------------------------------------------------------------------
# synth-markers
# --------------------------------------------------------------------------


def test_synth_markers_writes_parseable_presses(parsed, capsys):
    """The point: exercise the real parser, not a test-only shortcut."""
    cfg, conn, engine = parsed
    conn.close()

    root = cfg.data_root.as_posix()
    assert cli.main([
        "synth-markers", "fx", "--at", "12.5", "--kind", "maybe",
        "--delay", "7", "--replace", "--set", f"paths.data_root={root}",
    ]) == 0

    conn2 = db.open_db(cfg.db_path, migrate_to_latest=False)
    forced = runner.Runner(cfg=cfg, conn=conn2, stream_id="fx",
                           force={"marker_events"}, log=lambda *_: None)
    forced.execute(forced.plan(["register_stream", "marker_events"]))

    found = conn2.execute(
        "SELECT t, kind FROM events WHERE stream_id='fx' AND source='marker'"
    ).fetchall()
    conn2.close()
    assert len(found) == 1
    # The press lands after the moment, as a real reaction does (§4.3).
    assert found[0]["t"] == pytest.approx(19.5, abs=0.01)
    assert found[0]["kind"] == "marker_maybe"


def test_synth_markers_appends_by_default(parsed):
    cfg, conn, engine = parsed
    before = len(rows(conn))
    conn.close()

    root = cfg.data_root.as_posix()
    cli.main(["synth-markers", "fx", "--at", "5", "--set", f"paths.data_root={root}"])

    conn2 = db.open_db(cfg.db_path, migrate_to_latest=False)
    forced = runner.Runner(cfg=cfg, conn=conn2, stream_id="fx",
                           force={"marker_events"}, log=lambda *_: None)
    forced.execute(forced.plan(["register_stream", "marker_events"]))
    after = conn2.execute(
        "SELECT count(*) FROM events WHERE stream_id='fx' AND source='marker'"
    ).fetchone()[0]
    conn2.close()
    assert after == before + 1


def test_synth_markers_creates_an_anchor_when_there_is_none(tmp_path, fixture_dir, manifest):
    """Arbitrary footage registers with no anchor, which is correct — but §4.3
    markers are epoch-based, so testing them needs one."""
    cfg = config.load(overrides=[f"paths.data_root={(tmp_path / 'data').as_posix()}"])
    conn = db.open_db(cfg.db_path)
    conn.execute(
        "INSERT INTO streams (id, date, master_path, marker_time_base, duration_s) "
        "VALUES ('bare', '2026-08-14', 'D:/m.mkv', 'vod', 600.0)"
    )
    conn.close()

    root = cfg.data_root.as_posix()
    assert cli.main(["synth-markers", "bare", "--at", "30", "--set", f"paths.data_root={root}"]) == 0

    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    row = conn.execute("SELECT * FROM streams WHERE id='bare'").fetchone()
    conn.close()
    assert row["record_start_epoch_ms"] is not None
    assert row["marker_time_base"] == "epoch"
