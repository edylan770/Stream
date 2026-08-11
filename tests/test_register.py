"""`clipforge register` — the external stage."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime

import pytest

from clipforge import cli, config, db, paths
from clipforge.ingest import register


@pytest.fixture
def workspace(tmp_path):
    """A database, a fake master, and the capture files beside it."""
    data_root = tmp_path / "data"
    source = tmp_path / "recordings"
    source.mkdir()
    master = source / "2026-08-14 Marvel Rivals.mkv"
    master.write_bytes(b"not really a video" * 100)

    cfg = config.load(overrides=[f"paths.data_root={data_root.as_posix()}"])
    db.open_db(cfg.db_path).close()

    return type(
        "Workspace", (), {
            "tmp": tmp_path, "data_root": data_root, "source": source,
            "master": master, "cfg": cfg,
            "argv": ["register", "--master", str(master),
                     "--set", f"paths.data_root={data_root.as_posix()}"],
        },
    )()


def run_cli(argv) -> int:
    return cli.main(argv)


def open_db(cfg):
    return db.open_db(cfg.db_path, migrate_to_latest=False)


def write_anchor(directory, epoch_ms=1_755_123_456_789, schema=1):
    payload = {"schema": schema, "record_start_epoch_ms": epoch_ms,
               "source": "obs_websocket", "written_at_epoch_ms": epoch_ms}
    path = directory / "anchor.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_markers(directory):
    path = directory / "markers.jsonl"
    path.write_text(
        '{"epoch_ms": 1755123474789, "kind": "marker_maybe"}\n', encoding="utf-8"
    )
    return path


# --------------------------------------------------------------------------
# the basics
# --------------------------------------------------------------------------


def test_registers_a_stream(workspace, capsys):
    assert run_cli(workspace.argv) == 0
    conn = open_db(workspace.cfg)
    row = conn.execute("SELECT * FROM streams").fetchone()
    assert row["master_path"] == str(workspace.master)
    assert row["marker_time_base"] == "vod"
    assert row["record_start_epoch_ms"] is None
    conn.close()


def test_stream_id_follows_the_spec_format(workspace):
    run_cli(workspace.argv + ["--title", "Marvel Rivals"])
    conn = open_db(workspace.cfg)
    stream_id = conn.execute("SELECT id FROM streams").fetchone()["id"]
    conn.close()
    assert stream_id.endswith("_marvel_rivals")
    # §3.2's example format: <ISO date>_<slug>
    datetime.strptime(stream_id.split("_")[0], "%Y-%m-%d")


def test_date_comes_from_the_master_not_today(workspace):
    """A stream recorded at 23:00 and processed at 09:00 belongs to the night
    it happened; streams.date is what cross-stream queries group by."""
    old = time.time() - 90 * 86400
    os.utime(workspace.master, (old, old))
    expected = datetime.fromtimestamp(old).date().isoformat()

    run_cli(workspace.argv)
    conn = open_db(workspace.cfg)
    row = conn.execute("SELECT date, id FROM streams").fetchone()
    conn.close()
    assert row["date"] == expected
    assert row["id"].startswith(expected)


def test_explicit_date_and_id_win(workspace):
    run_cli(workspace.argv + ["--id", "custom_id", "--date", "2020-01-01"])
    conn = open_db(workspace.cfg)
    row = conn.execute("SELECT id, date FROM streams").fetchone()
    conn.close()
    assert (row["id"], row["date"]) == ("custom_id", "2020-01-01")


def test_games_are_stored_as_a_json_array(workspace):
    run_cli(workspace.argv + ["--games", "Marvel Rivals, Valorant"])
    conn = open_db(workspace.cfg)
    games = conn.execute("SELECT games FROM streams").fetchone()["games"]
    conn.close()
    assert json.loads(games) == ["Marvel Rivals", "Valorant"]


def test_missing_master_fails_cleanly(workspace, capsys):
    argv = [a if a != str(workspace.master) else str(workspace.tmp / "nope.mkv")
            for a in workspace.argv]
    assert run_cli(argv) == 1
    assert "master not found" in capsys.readouterr().out


def test_register_marks_the_stage_done(workspace):
    """`run` treats register_stream as a real completed stage, so the staleness
    cascade can key off it."""
    run_cli(workspace.argv)
    conn = open_db(workspace.cfg)
    row = conn.execute(
        "SELECT status, completed_seq FROM pipeline_stages WHERE stage='register_stream'"
    ).fetchone()
    conn.close()
    assert row["status"] == "done"
    assert row["completed_seq"] is not None


# --------------------------------------------------------------------------
# the anchor — §4.1's entire sync solution
# --------------------------------------------------------------------------


def test_anchor_beside_the_master_is_found(workspace):
    write_anchor(workspace.source)
    run_cli(workspace.argv)
    conn = open_db(workspace.cfg)
    row = conn.execute("SELECT record_start_epoch_ms, marker_time_base FROM streams").fetchone()
    conn.close()
    assert row["record_start_epoch_ms"] == 1_755_123_456_789
    assert row["marker_time_base"] == "epoch"


def test_explicit_anchor_path_wins(workspace):
    elsewhere = workspace.tmp / "elsewhere"
    elsewhere.mkdir()
    anchor = write_anchor(elsewhere, epoch_ms=1_800_000_000_000)
    run_cli(workspace.argv + ["--anchor", str(anchor)])
    conn = open_db(workspace.cfg)
    value = conn.execute("SELECT record_start_epoch_ms FROM streams").fetchone()[0]
    conn.close()
    assert value == 1_800_000_000_000


def test_unknown_anchor_schema_is_refused(workspace, capsys):
    write_anchor(workspace.source, schema=99)
    assert run_cli(workspace.argv) == 1
    assert "schema" in capsys.readouterr().out


def test_seconds_where_milliseconds_belong_is_caught(workspace, capsys):
    """A8 requires epoch milliseconds. Seconds would put every marker decades
    before the recording, and nothing downstream could detect it."""
    write_anchor(workspace.source, epoch_ms=1_755_123_456)
    assert run_cli(workspace.argv) == 1
    assert "MILLISECONDS" in capsys.readouterr().out


def test_non_integer_anchor_is_refused(workspace, capsys):
    (workspace.source / "anchor.json").write_text(
        json.dumps({"schema": 1, "record_start_epoch_ms": "1755123456789"}), encoding="utf-8"
    )
    assert run_cli(workspace.argv) == 1
    assert "integer" in capsys.readouterr().out


def test_markers_without_an_anchor_warns(workspace, capsys):
    """Epoch markers with no anchor cannot be converted (§4.1)."""
    write_markers(workspace.source)
    assert run_cli(workspace.argv) == 0
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "anchor.json is not" in out


def test_fixture_anchor_is_accepted(fixture_dir, tmp_path):
    """The contract the fixture writes is the contract register reads."""
    assert register.read_anchor(fixture_dir / "anchor.json") > 0


# --------------------------------------------------------------------------
# capture files land in raw/
# --------------------------------------------------------------------------


def test_capture_files_are_copied_into_the_stream(workspace):
    """The master stays put; these are kilobytes, irreplaceable, and often in a
    scratch directory that gets cleaned."""
    write_anchor(workspace.source)
    write_markers(workspace.source)
    run_cli(workspace.argv)

    conn = open_db(workspace.cfg)
    stream_id = conn.execute("SELECT id FROM streams").fetchone()["id"]
    conn.close()

    sp = paths.StreamPaths(workspace.data_root, stream_id)
    assert (sp.raw_dir / "anchor.json").is_file()
    assert sp.markers_jsonl.is_file()
    # ...and the master was NOT copied.
    assert not any(p.suffix == ".mkv" for p in sp.dir.rglob("*"))


def test_sources_json_records_provenance(workspace):
    write_anchor(workspace.source)
    run_cli(workspace.argv)

    conn = open_db(workspace.cfg)
    stream_id = conn.execute("SELECT id FROM streams").fetchone()["id"]
    conn.close()

    sources = json.loads(
        paths.StreamPaths(workspace.data_root, stream_id).sources_json.read_text()
    )
    assert sources["master"]["path"] == str(workspace.master)
    assert sources["master"]["size"] == workspace.master.stat().st_size
    assert sources["anchor_source"].endswith("anchor.json")


def test_stream_directories_follow_the_spec_layout(workspace):
    run_cli(workspace.argv)
    conn = open_db(workspace.cfg)
    stream_id = conn.execute("SELECT id FROM streams").fetchone()["id"]
    conn.close()
    sp = paths.StreamPaths(workspace.data_root, stream_id)
    for name in paths.STREAM_SUBDIRS:
        assert (sp.dir / name).is_dir()


# --------------------------------------------------------------------------
# re-registration
# --------------------------------------------------------------------------


def test_reregistering_refuses_without_force(workspace, capsys):
    run_cli(workspace.argv + ["--id", "s1"])
    assert run_cli(workspace.argv + ["--id", "s1"]) == 1
    assert "--force" in capsys.readouterr().out


def test_force_repointing_the_master_reruns_dependents(workspace):
    """Changing the master invalidates everything built from it, so the
    register stage's completed_seq bumps and the cascade does the rest."""
    run_cli(workspace.argv + ["--id", "s1"])
    conn = open_db(workspace.cfg)
    before = conn.execute(
        "SELECT completed_seq FROM pipeline_stages WHERE stage='register_stream'"
    ).fetchone()[0]
    conn.close()

    other = workspace.source / "different.mkv"
    other.write_bytes(b"a different recording")
    assert run_cli(["register", "--master", str(other), "--id", "s1", "--force",
                    "--set", f"paths.data_root={workspace.data_root.as_posix()}"]) == 0

    conn = open_db(workspace.cfg)
    after = conn.execute(
        "SELECT completed_seq FROM pipeline_stages WHERE stage='register_stream'"
    ).fetchone()[0]
    master = conn.execute("SELECT master_path FROM streams").fetchone()[0]
    conn.close()
    assert after > before
    assert master == str(other)


def test_metadata_only_change_does_not_rerun_anything(workspace, capsys):
    """Fixing a typo in the title must not trigger a four-hour re-encode."""
    run_cli(workspace.argv + ["--id", "s1", "--title", "Marvel Rivalz"])
    conn = open_db(workspace.cfg)
    before = conn.execute(
        "SELECT completed_seq FROM pipeline_stages WHERE stage='register_stream'"
    ).fetchone()[0]
    conn.close()

    assert run_cli(workspace.argv + ["--id", "s1", "--force", "--title", "Marvel Rivals"]) == 0
    assert "metadata only" in capsys.readouterr().out

    conn = open_db(workspace.cfg)
    row = conn.execute("SELECT completed_seq FROM pipeline_stages "
                       "WHERE stage='register_stream'").fetchone()
    title = conn.execute("SELECT title FROM streams").fetchone()[0]
    conn.close()
    assert row[0] == before          # nothing downstream disturbed
    assert title == "Marvel Rivals"  # but the fix landed


def test_same_day_collisions_get_a_suffix(workspace):
    run_cli(workspace.argv + ["--title", "Marvel"])
    second = workspace.source / "second.mkv"
    second.write_bytes(b"another one")
    os.utime(second, (workspace.master.stat().st_mtime,) * 2)
    run_cli(["register", "--master", str(second), "--title", "Marvel",
             "--set", f"paths.data_root={workspace.data_root.as_posix()}"])

    conn = open_db(workspace.cfg)
    ids = sorted(r["id"] for r in conn.execute("SELECT id FROM streams"))
    conn.close()
    assert len(ids) == 2
    assert ids[1].endswith("_2")
