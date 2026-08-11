"""`clipforge run` exit codes and reporting.

Exit codes matter here: unattended overnight processing (§1.3) is the whole
point of the pipeline, and anything wrapping it in a script reads the code
rather than the text.
"""

from __future__ import annotations

import pytest

from clipforge import cli, config, db
from clipforge.pipeline import runner


@pytest.fixture
def registered(tmp_path, fixture_dir, manifest):
    data_root = (tmp_path / "data").as_posix()
    cfg = config.load(overrides=[f"paths.data_root={data_root}"])
    conn = db.open_db(cfg.db_path)
    conn.execute(
        "INSERT INTO streams (id, date, master_path, marker_time_base) "
        "VALUES ('fx', '2026-08-14', ?, 'epoch')",
        (str(fixture_dir / manifest["video"]),),
    )
    conn.close()
    return data_root


def test_unregistered_stream_cannot_proceed(registered, capsys):
    """register_stream is the one blockage the operator can act on."""
    assert cli.main(["run", "fx", "--set", f"paths.data_root={registered}"]) == 1
    out = capsys.readouterr().out
    assert "cannot proceed" in out
    assert "clipforge register" in out


def test_up_to_date_pipeline_exits_zero(registered, capsys, tmp_path):
    """Stages blocked on phases that are not built yet are the normal state of
    a Phase 1 pipeline. Reporting failure for that makes the exit code useless.
    """
    cfg = config.load(overrides=[f"paths.data_root={registered}"])
    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    engine = runner.Runner(cfg=cfg, conn=conn, stream_id="fx", log=lambda *_: None)
    engine.mark_external_done("register_stream")
    engine.execute(engine.plan(["register_stream", "probe", "proxy"]))
    conn.close()

    assert cli.main(["run", "fx", "--set", f"paths.data_root={registered}"]) == 0
    assert "everything up to date" in capsys.readouterr().out


def test_unknown_stream_names_the_known_ones(registered, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "nope", "--set", f"paths.data_root={registered}"])
    assert exc.value.code != 0
    assert "fx" in str(exc.value)


def test_dry_run_changes_nothing(registered, capsys):
    cfg = config.load(overrides=[f"paths.data_root={registered}"])
    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    engine = runner.Runner(cfg=cfg, conn=conn, stream_id="fx", log=lambda *_: None)
    engine.mark_external_done("register_stream")
    conn.close()

    assert cli.main(
        ["run", "fx", "--dry-run", "--set", f"paths.data_root={registered}"]
    ) == 0
    out = capsys.readouterr().out
    assert "would run" in out

    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    row = conn.execute(
        "SELECT status FROM pipeline_stages WHERE stage='probe'"
    ).fetchone()
    conn.close()
    assert row is None  # nothing was executed or even recorded
