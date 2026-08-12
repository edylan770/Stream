"""The app shell's HTTP surface: browse, register, plan, run, poll — and the
guard in front of all of it.

Every number comes from `manifest.json`, from the config object, or from the
stage registry. Nothing here asserts a literal.
"""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from clipforge import config, db
from clipforge.pipeline import stages
from clipforge.review import app as review_app
from clipforge.review import browse, guard


def make_client(cfg, **kwargs):
    """A client that looks like the bundled UI: loopback Host, header present."""
    headers = {"X-ClipForge": "1"}
    headers.update(kwargs.pop("headers", {}))
    return TestClient(
        review_app.create_app(cfg), base_url="http://127.0.0.1", headers=headers, **kwargs
    )


@pytest.fixture
def cfg(tmp_path):
    """An empty database — the state a fresh install is actually in."""
    built = config.load(overrides=[f"paths.data_root={(tmp_path / 'data').as_posix()}"])
    db.open_db(built.db_path).close()
    return built


@pytest.fixture
def client(cfg):
    with make_client(cfg) as test_client:
        yield test_client


@pytest.fixture
def master(fixture_dir, manifest):
    return fixture_dir / manifest["video"]


def wait_for_job(client, job_id, timeout_s: float = 600.0) -> dict:
    deadline = time.monotonic() + timeout_s
    snapshot = client.get(f"/api/jobs/{job_id}").json()
    while snapshot["state"] == "running" and time.monotonic() < deadline:
        time.sleep(0.1)
        snapshot = client.get(f"/api/jobs/{job_id}").json()
    assert snapshot["state"] != "running", f"job did not finish within {timeout_s}s"
    return snapshot


# --------------------------------------------------------------------------
# the guard (§2.2 says nothing about this; a run trigger on loopback needs it)
# --------------------------------------------------------------------------


def test_a_foreign_host_is_refused(cfg):
    """DNS rebinding makes an attacker's page same-origin, so the name they had
    to ask for is the only thing left that separates them."""
    with TestClient(review_app.create_app(cfg), base_url="http://evil.example") as c:
        response = c.get("/api/streams")
    assert response.status_code == 403
    assert "rebinding" in response.text


def test_a_cross_origin_request_is_refused(client):
    response = client.get("/api/streams", headers={"Origin": "http://evil.example"})
    assert response.status_code == 403
    assert "cross-origin" in response.text


def test_our_own_origin_is_allowed(client):
    assert client.get(
        "/api/streams", headers={"Origin": "http://127.0.0.1:8765"}
    ).status_code == 200


def test_a_post_without_the_header_is_refused(cfg, master):
    """The vector: a cross-origin simple POST needs no preflight, and
    `request.json()` parses a text/plain body regardless of its content type —
    so the run would happen even though the reply cannot be read."""
    with TestClient(review_app.create_app(cfg), base_url="http://127.0.0.1") as c:
        response = c.post("/api/register", json={"master": str(master)})
    assert response.status_code == 403
    assert guard.HEADER in response.text


def test_a_get_without_the_header_still_works(cfg):
    """A cross-origin page can issue one but cannot read the reply, and making
    the header universal would break every plain `curl` against a read route."""
    with TestClient(review_app.create_app(cfg), base_url="http://127.0.0.1") as c:
        assert c.get("/api/streams").status_code == 200


def test_the_header_requirement_can_be_turned_off(tmp_path, master):
    """`review.require_header: false` is the escape hatch for anything other
    than the bundled UI."""
    relaxed = config.load(overrides=[
        f"paths.data_root={(tmp_path / 'data').as_posix()}",
        "review.require_header=false",
    ])
    db.open_db(relaxed.db_path).close()
    with TestClient(review_app.create_app(relaxed), base_url="http://127.0.0.1") as c:
        assert c.post("/api/register", json={"master": str(master)}).status_code == 200


def test_an_extra_host_can_be_allowed(tmp_path):
    allowed = config.load(overrides=[
        f"paths.data_root={(tmp_path / 'data').as_posix()}",
        "review.allow_hosts=[studio-pc]",
    ])
    db.open_db(allowed.db_path).close()
    with TestClient(review_app.create_app(allowed), base_url="http://studio-pc") as c:
        assert c.get("/api/streams").status_code == 200


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "127.0.0.1:8765", "[::1]:80"])
def test_loopback_names_are_all_accepted(cfg, host):
    assert guard.check(cfg, "GET", {"host": host}) is None


# --------------------------------------------------------------------------
# browsing for a recording
# --------------------------------------------------------------------------


def test_browse_lists_the_fixture_directory(client, fixture_dir, manifest):
    listing = client.get("/api/browse", params={"path": str(fixture_dir)}).json()
    names = {entry["name"] for entry in listing["entries"]}
    assert manifest["video"] in names
    assert listing["error"] is None
    assert listing["roots"]


def test_browse_flags_the_recording_as_media(client, fixture_dir, manifest):
    listing = client.get("/api/browse", params={"path": str(fixture_dir)}).json()
    video = next(e for e in listing["entries"] if e["name"] == manifest["video"])
    assert video["is_media"] is True
    assert video["is_dir"] is False
    assert video["size"] > 0


def test_browse_reports_the_capture_files_beside_the_master(client, fixture_dir, manifest):
    """§4.1's anchor is the entire sync solution, and `find_capture_file` looks
    for it beside the master — so it is a property of the directory."""
    listing = client.get("/api/browse", params={"path": str(fixture_dir)}).json()
    assert listing["has_anchor"] is (fixture_dir / manifest["anchor"]).is_file()
    assert listing["has_markers"] is (fixture_dir / manifest["markers_file"]).is_file()


def test_browsing_a_file_lists_its_parent(client, master):
    """Pasting the master's own path should land somewhere useful."""
    listing = client.get("/api/browse", params={"path": str(master)}).json()
    assert listing["path"] == str(master.parent)


def test_a_missing_path_reports_an_error_rather_than_failing(client, tmp_path):
    """Browsing to a drive that is no longer plugged in is ordinary."""
    response = client.get("/api/browse", params={"path": str(tmp_path / "nope")})
    assert response.status_code == 200
    assert response.json()["error"]
    assert response.json()["entries"] == []


def test_browse_without_a_path_starts_somewhere_real(client):
    listing = client.get("/api/browse").json()
    assert listing["path"] == str(browse.default_start())


def test_browse_caps_a_large_directory(tmp_path):
    """A recording drive holds thousands of files, sometimes over a network."""
    crowded = tmp_path / "crowded"
    crowded.mkdir()
    limited = config.load(overrides=[
        f"paths.data_root={(tmp_path / 'data').as_posix()}",
        "review.browse.max_entries=5",
    ])
    db.open_db(limited.db_path).close()
    cap = int(limited.get("review.browse.max_entries"))
    for n in range(cap * 2):
        (crowded / f"clip{n:03d}.mkv").write_bytes(b"x")

    with make_client(limited) as c:
        listing = c.get("/api/browse", params={"path": str(crowded)}).json()
    assert len(listing["entries"]) == cap
    assert listing["truncated"] is True


def test_directories_sort_above_files(client, tmp_path):
    (tmp_path / "zzz_dir").mkdir()
    (tmp_path / "aaa.mkv").write_bytes(b"x")
    listing = client.get("/api/browse", params={"path": str(tmp_path)}).json()
    kinds = [entry["is_dir"] for entry in listing["entries"]]
    assert kinds == sorted(kinds, reverse=True)


# --------------------------------------------------------------------------
# preflight and register
# --------------------------------------------------------------------------


def test_preflight_predicts_what_register_produces(client, master):
    """Compared against what registration actually does, not against a literal."""
    predicted = client.post("/api/register/preflight", json={"master": str(master)}).json()
    actual = client.post("/api/register", json={"master": str(master)}).json()

    assert predicted["stream_id"] == actual["stream_id"]
    assert predicted["date"] == actual["date"]
    assert predicted["marker_time_base"] == actual["marker_time_base"]


def test_preflight_reads_the_anchor_before_anything_is_written(
    client, master, fixture_dir, manifest, cfg
):
    """§4.1: one number per stream, and every marker is converted through it."""
    anchor = json.loads((fixture_dir / manifest["anchor"]).read_text())
    predicted = client.post("/api/register/preflight", json={"master": str(master)}).json()

    assert predicted["anchor_ms"] == anchor["record_start_epoch_ms"]
    assert predicted["marker_time_base"] == "epoch"

    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    written = conn.execute("SELECT COUNT(*) FROM streams").fetchone()[0]
    conn.close()
    assert written == 0, "preflight must not write anything"


def test_preflight_warns_when_the_anchor_is_missing(client, tmp_path):
    """Without it §4.3's epoch presses are read as VOD seconds, and nothing
    downstream can detect that."""
    lonely = tmp_path / "recordings"
    lonely.mkdir()
    recording = lonely / "session.mkv"
    recording.write_bytes(b"x" * 64)
    (lonely / "markers.jsonl").write_text('{"epoch_ms": 1755123474789, "kind": "marker_maybe"}\n')

    predicted = client.post(
        "/api/register/preflight", json={"master": str(recording)}
    ).json()
    assert predicted["anchor_ms"] is None
    assert predicted["marker_time_base"] == "vod"
    assert predicted["warnings"]


def test_register_creates_the_stream(client, master, cfg):
    result = client.post(
        "/api/register", json={"master": str(master), "title": "Fixture night"}
    ).json()

    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    row = conn.execute("SELECT * FROM streams WHERE id = ?", (result["stream_id"],)).fetchone()
    conn.close()
    assert row["title"] == "Fixture night"
    assert row["master_path"] == str(master)
    assert result["created"] is True


def test_registering_twice_is_a_conflict_not_a_failure(client, master):
    """Re-registration invalidates every artifact built from the old master, so
    it stays a deliberate `--force` at a terminal."""
    first = client.post("/api/register", json={"master": str(master)}).json()
    again = client.post(
        "/api/register", json={"master": str(master), "id": first["stream_id"]}
    )
    assert again.status_code == 409
    assert "--force" in again.json()["detail"]


def test_registering_a_file_that_is_not_there_is_a_400(client, tmp_path):
    response = client.post("/api/register", json={"master": str(tmp_path / "gone.mkv")})
    assert response.status_code == 400
    assert "master not found" in response.json()["detail"]


def test_registering_with_no_master_is_a_400(client):
    assert client.post("/api/register", json={}).status_code == 400


# --------------------------------------------------------------------------
# the plan
# --------------------------------------------------------------------------


def test_the_plan_covers_every_stage_with_a_reason(client, master):
    stream_id = client.post("/api/register", json={"master": str(master)}).json()["stream_id"]
    plan = client.get(f"/api/streams/{stream_id}/plan").json()

    assert {entry["stage"] for entry in plan["stages"]} == set(stages.STAGES)
    assert all(entry["reason"] for entry in plan["stages"])
    assert plan["job"] is None


def test_the_plan_would_run_what_phase_1_has_built(client, master):
    """Computed from the registry, so Phase 2 updates this rather than passing
    while asserting a Phase 1 pipeline."""
    stream_id = client.post("/api/register", json={"master": str(master)}).json()["stream_id"]
    plan = client.get(f"/api/streams/{stream_id}/plan").json()

    expected = {
        spec.name for spec in stages.iter_specs()
        if spec.module is not None and spec.phase == 1
    }
    assert set(plan["will_run"]) == expected


def test_the_plan_for_an_unknown_stream_is_404(client):
    assert client.get("/api/streams/nope/plan").status_code == 404


# --------------------------------------------------------------------------
# running it
# --------------------------------------------------------------------------


def test_a_run_completes_and_matches_the_fixture(client, master, cfg, manifest):
    stream_id = client.post("/api/register", json={"master": str(master)}).json()["stream_id"]
    started = client.post(f"/api/streams/{stream_id}/run").json()
    finished = wait_for_job(client, started["id"])

    assert finished["state"] == "done", finished["error"]
    assert {entry["stage"] for entry in finished["stages_run"]} == {
        spec.name for spec in stages.iter_specs()
        if spec.module is not None and spec.phase == 1
    }

    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    markers = conn.execute(
        "SELECT COUNT(*) FROM events WHERE stream_id = ? AND source = 'marker'", (stream_id,)
    ).fetchone()[0]
    candidates = conn.execute(
        "SELECT COUNT(*) FROM candidates WHERE stream_id = ? AND is_current = 1", (stream_id,)
    ).fetchone()[0]
    duration = conn.execute(
        "SELECT duration_s FROM streams WHERE id = ?", (stream_id,)
    ).fetchone()[0]
    conn.close()

    assert markers == len(manifest["markers"])
    assert candidates >= cfg.get("score.peak.min_candidates")
    assert duration == pytest.approx(manifest["duration_s"], abs=0.05)


def test_the_log_streams_in_over_successive_polls(client, master):
    """The browser polls with an absolute cursor rather than re-reading the
    whole log every 700 ms."""
    stream_id = client.post("/api/register", json={"master": str(master)}).json()["stream_id"]
    job_id = client.post(f"/api/streams/{stream_id}/run").json()["id"]

    collected = []
    cursor = 0
    while True:
        snapshot = client.get(f"/api/jobs/{job_id}", params={"since": cursor}).json()
        collected.extend(snapshot["lines"])
        assert snapshot["dropped"] == 0
        cursor = snapshot["next_cursor"]
        if snapshot["state"] != "running":
            break
        time.sleep(0.1)

    assert collected
    assert len(collected) == cursor
    assert any("score" in line for line in collected)


def test_a_second_run_while_one_is_live_is_a_conflict(client, master):
    """Starting a second run mid-encode would mark the first one's stage dead
    and re-run it on top of the file it is still writing."""
    stream_id = client.post("/api/register", json={"master": str(master)}).json()["stream_id"]
    job_id = client.post(f"/api/streams/{stream_id}/run").json()["id"]

    second = client.post(f"/api/streams/{stream_id}/run")
    if second.status_code == 409:
        assert "already running" in second.json()["detail"]
    else:
        # The fixture is small enough that the first job can finish first; then
        # a second run is legitimate and must skip everything (C7).
        assert second.status_code == 200

    finished = wait_for_job(client, job_id)
    assert finished["state"] == "done", finished["error"]


def test_the_plan_reports_a_finished_job(client, master):
    stream_id = client.post("/api/register", json={"master": str(master)}).json()["stream_id"]
    job_id = client.post(f"/api/streams/{stream_id}/run").json()["id"]
    wait_for_job(client, job_id)

    plan = client.get(f"/api/streams/{stream_id}/plan").json()
    assert plan["job"] == job_id
    assert plan["job_live"] is False
    assert plan["will_run"] == []
    score = next(e for e in plan["stages"] if e["stage"] == "score")
    assert score["status"] == "done"
    assert score["finished_at"]


def test_running_an_unknown_stream_is_404(client):
    assert client.post("/api/streams/nope/run").status_code == 404


def test_polling_an_unknown_job_is_404(client):
    assert client.get("/api/jobs/nope").status_code == 404
