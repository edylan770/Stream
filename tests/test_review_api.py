"""The review server: payload shape, ratings, and the §7.5 instrumentation."""

from __future__ import annotations

import json
import shutil

import pytest
from fastapi.testclient import TestClient

from clipforge import config, db, paths
from clipforge.pipeline import runner
from clipforge.review import app as review_app
from clipforge.review import queries
from tests.fixtures.make_fixture import FixtureSpec, generate, load_manifest


@pytest.fixture(scope="session")
def processed(tmp_path_factory):
    """A fully processed stream: proxy, signals, markers, candidates."""
    directory = generate(FixtureSpec(name="long", duration_s=600.0))
    manifest = load_manifest(directory)

    root = tmp_path_factory.mktemp("review")
    cfg = config.load(overrides=[f"paths.data_root={(root / 'data').as_posix()}"])
    conn = db.open_db(cfg.db_path)
    anchor = json.loads((directory / manifest["anchor"]).read_text())["record_start_epoch_ms"]
    conn.execute(
        "INSERT INTO streams (id, date, title, master_path, marker_time_base, "
        "record_start_epoch_ms) VALUES ('fx', '2026-08-14', 'Fixture', ?, 'epoch', ?)",
        (str(directory / manifest["video"]), anchor),
    )
    sp = paths.StreamPaths(cfg.data_root, "fx").ensure()
    shutil.copy2(directory / manifest["markers_file"], sp.markers_jsonl)

    engine = runner.Runner(cfg=cfg, conn=conn, stream_id="fx", log=lambda *_: None)
    engine.mark_external_done("register_stream")
    engine.execute(engine.plan())
    conn.close()
    return cfg, manifest


@pytest.fixture
def client(processed):
    cfg, _ = processed
    # TestClient defaults to `Host: testserver`, which guard.py refuses — a
    # request arriving under a name this server does not answer to is how DNS
    # rebinding reaches a local server. The header goes on every request for the
    # same reason app.js sends it.
    with TestClient(
        review_app.create_app(cfg),
        base_url="http://127.0.0.1",
        headers={"X-ClipForge": "1"},
    ) as test_client:
        yield test_client


@pytest.fixture
def conn(processed):
    cfg, _ = processed
    connection = db.open_db(cfg.db_path, migrate_to_latest=False)
    yield connection
    connection.close()


# --------------------------------------------------------------------------
# the page and its assets are self-contained
# --------------------------------------------------------------------------


def test_index_serves(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "ClipForge" in response.text


def test_no_external_resources(client):
    """§2.2 says avoid heavy frameworks; a dedicated streaming PC may have no
    internet at all, so nothing may be fetched from a CDN."""
    page = client.get("/").text
    for asset in ("/static/app.css", "/static/app.js"):
        assert asset in page
        assert client.get(asset).status_code == 200
    assert "http://" not in page.replace("http://www.w3.org", "")
    assert "https://" not in page


# --------------------------------------------------------------------------
# the candidate payload
# --------------------------------------------------------------------------


def test_candidates_arrive_in_one_payload(client):
    """C4's four seconds per candidate cannot afford a round trip per keypress,
    so everything the screen needs is loaded once."""
    data = client.get("/api/streams/fx/candidates").json()
    assert data["candidates"]
    first = data["candidates"][0]
    for key in ("t_start", "t_end", "t_peak", "score", "contributions",
                "sparkline", "marker_anchored", "rating"):
        assert key in first


def test_candidates_are_ranked(client):
    scores = [c["score"] for c in client.get("/api/streams/fx/candidates").json()["candidates"]]
    assert scores == sorted(scores, reverse=True)


def test_only_current_candidates_are_served(client, conn, processed):
    """A superseded generation must not appear in review."""
    cfg, _ = processed
    conn.execute("UPDATE candidates SET is_current = 0 WHERE stream_id='fx'")
    try:
        assert client.get("/api/streams/fx/candidates").json()["candidates"] == []
    finally:
        conn.execute("UPDATE candidates SET is_current = 1 WHERE stream_id='fx'")


def test_contributions_and_context_are_separated(client):
    """Signal contributions drive the bars; the dB context is prose beneath."""
    first = client.get("/api/streams/fx/candidates").json()["candidates"][0]
    assert "mic_rms" in first["contributions"]
    assert all(not k.startswith("_") for k in first["contributions"])
    assert "total_raw" in first["context"]
    assert "total_smoothed" in first["context"]


def test_sparkline_is_drawn_from_stored_signals(client):
    """No ffmpeg, no files — §7.2's preview assets are Phase 3."""
    first = client.get("/api/streams/fx/candidates").json()["candidates"][0]
    assert first["sparkline_kind"] == "mic_rms"
    assert 2 < len(first["sparkline"]) <= queries.SPARKLINE_POINTS
    low, high = first["sparkline_range"]
    assert low < high


def test_marker_anchored_candidates_are_flagged(client):
    """§7.4's fourth section: the operator marked these deliberately, so they
    are the safety net when the weights rank them low."""
    candidates = client.get("/api/streams/fx/candidates").json()["candidates"]
    assert any(c["marker_anchored"] for c in candidates)
    flagged = next(c for c in candidates if c["marker_anchored"])
    assert flagged["markers"] or flagged["contributions"].get("marker_definite")


def test_stream_warnings_reach_the_screen(processed, tmp_path):
    """The §4.2 contamination notice belongs in front of the operator."""
    cfg, _ = processed
    connection = db.open_db(cfg.db_path, migrate_to_latest=False)
    connection.execute(
        "UPDATE streams SET audio_track_map = ? WHERE id='fx'",
        (json.dumps({"roles": {"mic": 0}, "warnings": ["single track, mic is contaminated"]}),),
    )
    detail = queries.stream_detail(connection, "fx")
    connection.close()
    assert detail["warnings"] == ["single track, mic is contaminated"]


def test_unknown_stream_is_404(client):
    assert client.get("/api/streams/nope/candidates").status_code == 404


# --------------------------------------------------------------------------
# rating (§7.3, §7.5)
# --------------------------------------------------------------------------


def test_rating_is_saved_with_review_ms(client, conn):
    candidate = client.get("/api/streams/fx/candidates").json()["candidates"][0]
    response = client.post(
        f"/api/candidates/{candidate['id']}/rating",
        json={"rating": 2, "review_ms": 3400},
    )
    assert response.status_code == 200

    row = conn.execute(
        "SELECT rating, review_ms, rating_source FROM ratings WHERE candidate_id = ?",
        (candidate["id"],),
    ).fetchone()
    assert (row["rating"], row["review_ms"], row["rating_source"]) == (2, 3400, "operator")


def test_rating_twice_updates_rather_than_failing(client, conn):
    """Changing your mind is normal; ratings has candidate_id as its primary
    key, so an insert would otherwise conflict."""
    candidate = client.get("/api/streams/fx/candidates").json()["candidates"][1]
    client.post(f"/api/candidates/{candidate['id']}/rating", json={"rating": 0})
    client.post(f"/api/candidates/{candidate['id']}/rating", json={"rating": 2})

    rows = conn.execute(
        "SELECT rating FROM ratings WHERE candidate_id = ?", (candidate["id"],)
    ).fetchall()
    assert len(rows) == 1 and rows[0]["rating"] == 2


def test_an_operator_rating_overrides_an_inherited_one(client, conn):
    """§14's tuning input counts operator ratings only, so a human keypress has
    to clear the inherited flag."""
    listing = client.get("/api/streams/fx/candidates").json()["candidates"]
    candidate, source = listing[2], listing[3]
    conn.execute(
        "INSERT OR REPLACE INTO ratings (candidate_id, rating, rating_source, inherited_from) "
        "VALUES (?, 1, 'inherited', ?)",
        (candidate["id"], source["id"]),
    )
    client.post(f"/api/candidates/{candidate['id']}/rating", json={"rating": 2})

    row = conn.execute(
        "SELECT rating, rating_source, inherited_from FROM ratings WHERE candidate_id = ?",
        (candidate["id"],),
    ).fetchone()
    assert (row["rating"], row["rating_source"], row["inherited_from"]) == (2, "operator", None)


@pytest.mark.parametrize("bad", [-1, 3, 99])
def test_invalid_ratings_are_rejected(client, bad):
    """§7.3 defines exactly three: 0 skip, 1 maybe, 2 clip it."""
    candidate = client.get("/api/streams/fx/candidates").json()["candidates"][0]
    assert client.post(
        f"/api/candidates/{candidate['id']}/rating", json={"rating": bad}
    ).status_code == 400


def test_ratings_show_up_on_the_next_load(client):
    candidate = client.get("/api/streams/fx/candidates").json()["candidates"][0]
    client.post(f"/api/candidates/{candidate['id']}/rating", json={"rating": 1})
    reloaded = client.get("/api/streams/fx/candidates").json()["candidates"]
    assert next(c for c in reloaded if c["id"] == candidate["id"])["rating"] == 1


# --------------------------------------------------------------------------
# instrumentation (§7.5, §14)
# --------------------------------------------------------------------------


def test_session_duration_is_recorded(client, conn):
    client.post("/api/streams/fx/session", json={"duration_s": 412.0, "reviewed": 6})
    row = conn.execute(
        "SELECT value, meta FROM tool_metrics WHERE metric='review_session_duration_s' "
        "ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert row["value"] == 412.0
    assert json.loads(row["meta"])["reviewed"] == 6


def test_metrics_report_the_median_not_the_mean(conn):
    """Leave the tab open over lunch and one candidate reads forty minutes; a
    mean would be swamped and §7.1's target unmeasurable."""
    rows = conn.execute(
        "SELECT id FROM candidates WHERE stream_id='fx' AND is_current=1 LIMIT 4"
    ).fetchall()
    for row, ms in zip(rows, [2000, 2500, 3000, 2_400_000], strict=False):
        conn.execute(
            "INSERT OR REPLACE INTO ratings (candidate_id, rating, review_ms, rating_source) "
            "VALUES (?, 2, ?, 'operator')",
            (row["id"], ms),
        )

    metrics = queries.review_metrics(conn, "fx")
    assert metrics["median_review_ms"] < 5_000
    assert metrics["mean_review_ms"] > 500_000      # the outlier is still recorded


def test_metrics_include_the_approval_rate(conn):
    """§14: 'approved / total candidates — is the threshold correct?'"""
    metrics = queries.review_metrics(conn, "fx")
    assert 0.0 <= metrics["approval_rate"] <= 1.0
    assert metrics["candidates"] > 0


# --------------------------------------------------------------------------
# media
# --------------------------------------------------------------------------


def test_proxy_is_served_with_ranges(client):
    response = client.get("/media/fx/proxy", headers={"Range": "bytes=0-1023"})
    assert response.status_code == 206
    assert len(response.content) == 1024
    assert response.headers["content-range"].startswith("bytes 0-1023/")


def test_proxy_seek_returns_different_bytes(client):
    """Proves the seek is real rather than always returning the file head."""
    head = client.get("/media/fx/proxy", headers={"Range": "bytes=0-255"}).content
    middle = client.get("/media/fx/proxy", headers={"Range": "bytes=500000-500255"}).content
    assert len(middle) == 256
    assert head != middle


def test_proxy_for_a_stream_without_one_is_404(client, conn):
    conn.execute(
        "INSERT INTO streams (id, date, master_path, marker_time_base) "
        "VALUES ('noproxy', '2026-08-14', 'D:/m.mkv', 'vod')"
    )
    try:
        response = client.get("/media/noproxy/proxy")
        assert response.status_code == 404
        assert "clipforge run" in response.json()["detail"]
    finally:
        conn.execute("DELETE FROM streams WHERE id='noproxy'")


# --------------------------------------------------------------------------
# stream list
# --------------------------------------------------------------------------


def test_stream_list_reports_readiness(client):
    streams = client.get("/api/streams").json()["streams"]
    fx = next(s for s in streams if s["id"] == "fx")
    assert fx["has_proxy"] is True
    assert fx["candidates"] > 0
    assert fx["rated"] >= 0
    assert fx["stages_done"] > 0
