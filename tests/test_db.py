"""Schema, migrations, and path handling."""

from __future__ import annotations

import sqlite3

import pytest

from clipforge import db, paths

# Every table §3.2 defines, plus export_items (the documented addition).
EXPECTED_TABLES = {
    "streams", "pipeline_stages", "signal_series", "events",
    "segments", "segment_embeddings",
    "candidates", "ratings",
    "exports", "export_items", "performance",
    "digests", "ideas", "idea_evidence",
    "clusters", "cluster_members", "ngrams", "open_loops",
    "tool_metrics",
}


@pytest.fixture
def conn(tmp_path):
    connection = db.open_db(tmp_path / "test.db")
    yield connection
    connection.close()


def _stream(conn, stream_id="2026-08-14_marvel", **kwargs):
    fields = {
        "id": stream_id,
        "date": "2026-08-14",
        "master_path": "D:/masters/x.mkv",
        "marker_time_base": "vod",
        **kwargs,
    }
    columns = ", ".join(fields)
    placeholders = ", ".join("?" * len(fields))
    conn.execute(f"INSERT INTO streams ({columns}) VALUES ({placeholders})", tuple(fields.values()))
    return stream_id


def _candidate(conn, stream_id, *, t_peak=100.0, generation=1, **kwargs):
    fields = {
        "stream_id": stream_id,
        "generation": generation,
        "profile": "naive",
        "t_start": t_peak - 5,
        "t_end": t_peak + 5,
        "t_peak": t_peak,
        "score_entertainment": 1.0,
        "score_gameplay": 0.0,
        "score_combined": 1.0,
        "feature_vector": "{}",
        "feature_schema_version": 1,
        "config_version": "naive@abc12345",
        **kwargs,
    }
    columns = ", ".join(fields)
    placeholders = ", ".join("?" * len(fields))
    cur = conn.execute(
        f"INSERT INTO candidates ({columns}) VALUES ({placeholders})", tuple(fields.values())
    )
    return cur.lastrowid


# --------------------------------------------------------------------------
# migrations
# --------------------------------------------------------------------------


def test_migrations_are_numbered_contiguously():
    versions = [version for version, _ in db.migration_files()]
    assert versions == list(range(1, len(versions) + 1))


def test_fresh_database_is_at_the_latest_version(conn):
    assert db.user_version(conn) == db.latest_version()


def test_every_spec_table_exists(conn):
    assert set(db.table_names(conn)) == EXPECTED_TABLES


def test_migrate_is_idempotent(tmp_path):
    path = tmp_path / "twice.db"
    first = db.connect(path)
    assert db.migrate(first) == [1]
    assert db.migrate(first) == []      # nothing pending the second time
    first.close()

    second = db.open_db(path)           # and reopening changes nothing
    assert db.migrate(second) == []
    assert db.user_version(second) == db.latest_version()
    second.close()


def test_open_without_migrating_rejects_a_stale_database(tmp_path):
    path = tmp_path / "stale.db"
    conn = db.connect(path)
    conn.close()
    with pytest.raises(db.MigrationError, match="schema version 0"):
        db.open_db(path, migrate_to_latest=False)


# --------------------------------------------------------------------------
# pragmas the design depends on
# --------------------------------------------------------------------------


def test_wal_is_enabled(conn):
    """A5: required for review-UI reads during batch writes."""
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


def test_foreign_keys_are_enforced(conn):
    """Off by default in SQLite, so this is worth asserting rather than assuming."""
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO events (stream_id, t, source, kind) VALUES ('ghost', 1.0, 'marker', 'x')"
        )


def test_cascade_delete_reaches_candidates_and_ratings(conn):
    stream_id = _stream(conn)
    candidate_id = _candidate(conn, stream_id)
    conn.execute("INSERT INTO ratings (candidate_id, rating) VALUES (?, 2)", (candidate_id,))
    conn.execute("DELETE FROM streams WHERE id = ?", (stream_id,))
    assert conn.execute("SELECT count(*) FROM candidates").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM ratings").fetchone()[0] == 0


# --------------------------------------------------------------------------
# the deviations from §3.2 that exist for a reason
# --------------------------------------------------------------------------


def test_stream_registers_without_a_record_start_anchor(conn):
    """Arbitrary test footage has no OBS anchor; §3.2 declared this NOT NULL."""
    _stream(conn, "no_anchor", record_start_epoch_ms=None, marker_time_base="vod")
    row = conn.execute("SELECT * FROM streams WHERE id = 'no_anchor'").fetchone()
    assert row["record_start_epoch_ms"] is None
    assert row["marker_time_base"] == "vod"


def test_marker_time_base_is_constrained(conn):
    with pytest.raises(sqlite3.IntegrityError):
        _stream(conn, "bad_base", marker_time_base="whenever")


def test_exact_rational_fps_survives_a_round_trip(conn):
    """59.94 is 60000/1001. A float column cannot express that exactly."""
    _stream(conn, "ntsc", fps=59.94, fps_num=60000, fps_den=1001)
    row = conn.execute("SELECT fps_num, fps_den FROM streams WHERE id='ntsc'").fetchone()
    assert (row["fps_num"], row["fps_den"]) == (60000, 1001)


def test_candidate_generations_coexist(conn):
    """Re-scoring must not disturb what came before (§6.1 vs §3.2)."""
    stream_id = _stream(conn)
    first = _candidate(conn, stream_id, t_peak=100.0, generation=1)
    conn.execute("INSERT INTO ratings (candidate_id, rating) VALUES (?, 2)", (first,))

    conn.execute("UPDATE candidates SET is_current = 0 WHERE stream_id = ?", (stream_id,))
    second = _candidate(conn, stream_id, t_peak=100.0, generation=2)
    conn.execute(
        "INSERT INTO ratings (candidate_id, rating, rating_source, inherited_from) "
        "VALUES (?, 2, 'inherited', ?)",
        (second, first),
    )

    assert conn.execute("SELECT count(*) FROM candidates").fetchone()[0] == 2
    # The operator's original judgment is still there and still marked as theirs.
    original = conn.execute(
        "SELECT rating, rating_source FROM ratings WHERE candidate_id = ?", (first,)
    ).fetchone()
    assert (original["rating"], original["rating_source"]) == (2, "operator")

    current = conn.execute(
        "SELECT id FROM candidates WHERE stream_id = ? AND is_current = 1", (stream_id,)
    ).fetchall()
    assert [row["id"] for row in current] == [second]


def test_duplicate_peak_within_a_generation_is_rejected(conn):
    stream_id = _stream(conn)
    _candidate(conn, stream_id, t_peak=100.0, generation=1)
    with pytest.raises(sqlite3.IntegrityError):
        _candidate(conn, stream_id, t_peak=100.0, generation=1)


def test_inherited_ratings_are_distinguishable_from_operator_ratings(conn):
    """§14's weight-tuning input must not double-count a carried-forward rating."""
    stream_id = _stream(conn)
    a = _candidate(conn, stream_id, t_peak=10.0, generation=1)
    b = _candidate(conn, stream_id, t_peak=20.0, generation=1)
    conn.execute("INSERT INTO ratings (candidate_id, rating) VALUES (?, 2)", (a,))
    conn.execute(
        "INSERT INTO ratings (candidate_id, rating, rating_source) VALUES (?, 2, 'inherited')",
        (b,),
    )
    count = conn.execute(
        "SELECT count(*) FROM ratings WHERE rating_source = 'operator'"
    ).fetchone()[0]
    assert count == 1


def test_rating_values_are_constrained(conn):
    stream_id = _stream(conn)
    candidate_id = _candidate(conn, stream_id)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO ratings (candidate_id, rating) VALUES (?, 5)", (candidate_id,))


def test_candidate_window_must_contain_its_peak(conn):
    stream_id = _stream(conn)
    with pytest.raises(sqlite3.IntegrityError):
        _candidate(conn, stream_id, t_peak=100.0, t_start=200.0, t_end=300.0)


def test_export_items_describe_a_multi_clip_timeline(conn):
    """§3.2's single exports.candidate_id cannot express a 40-clip FCPXML."""
    stream_id = _stream(conn)
    ids = [_candidate(conn, stream_id, t_peak=float(t)) for t in (10, 50, 90)]
    cur = conn.execute(
        "INSERT INTO exports (stream_id, kind, path, item_count) VALUES (?, 'fcpxml', ?, ?)",
        (stream_id, "streams/x/exports/timeline.fcpxml", len(ids)),
    )
    export_id = cur.lastrowid
    conn.executemany(
        "INSERT INTO export_items (export_id, seq, candidate_id, t_start, t_end, role) "
        "VALUES (?, ?, ?, ?, ?, 'clip')",
        [(export_id, seq, cid, 0.0, 1.0) for seq, cid in enumerate(ids)],
    )
    assert conn.execute(
        "SELECT count(*) FROM export_items WHERE export_id = ?", (export_id,)
    ).fetchone()[0] == 3


def test_export_kind_is_constrained(conn):
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO exports (kind, path) VALUES ('interpretive-dance', 'x')")


def test_signal_series_records_how_it_was_computed(conn):
    stream_id = _stream(conn)
    conn.execute(
        "INSERT INTO signal_series (stream_id, kind, sample_rate_hz, n_samples, data, params) "
        "VALUES (?, 'mic_rms', 10.0, 3, ?, ?)",
        (stream_id, b"\x00" * 12, '{"hop_s": 0.1}'),
    )
    row = conn.execute("SELECT params FROM signal_series").fetchone()
    assert '"hop_s"' in row["params"]


def test_pipeline_stage_status_is_constrained(conn):
    stream_id = _stream(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO pipeline_stages (stream_id, stage, status) VALUES (?, 'proxy', 'vibing')",
            (stream_id,),
        )


def test_transaction_rolls_back_on_error(conn):
    stream_id = _stream(conn)
    with pytest.raises(sqlite3.IntegrityError):
        with db.transaction(conn):
            _candidate(conn, stream_id, t_peak=1.0)
            _candidate(conn, stream_id, t_peak=1.0)  # duplicate
    assert conn.execute("SELECT count(*) FROM candidates").fetchone()[0] == 0


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------


def test_stream_paths_are_posix_in_the_database(tmp_path):
    """A DB written on Windows has to resolve elsewhere."""
    sp = paths.StreamPaths(tmp_path, "2026-08-14_marvel")
    assert sp.rel_proxy == "streams/2026-08-14_marvel/proxy/proxy.mp4"
    assert "\\" not in sp.rel_audio("mic")


def test_relative_and_absolute_round_trip(tmp_path):
    sp = paths.StreamPaths(tmp_path, "s1")
    assert sp.absolute(sp.rel_proxy) == sp.proxy
    assert sp.proxy.is_absolute()
    assert sp.proxy.is_relative_to(tmp_path)


def test_relocating_the_data_root_moves_every_artifact(tmp_path):
    """The point of storing relative paths: a new drive letter costs nothing."""
    stored = paths.StreamPaths(tmp_path / "old", "s1").rel_proxy
    relocated = paths.StreamPaths(tmp_path / "new", "s1")
    assert relocated.absolute(stored) == tmp_path / "new" / "streams" / "s1" / "proxy" / "proxy.mp4"


def test_ensure_creates_the_spec_layout(tmp_path):
    sp = paths.StreamPaths(tmp_path, "s1").ensure()
    for name in paths.STREAM_SUBDIRS:
        assert (sp.dir / name).is_dir()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Marvel Rivals", "marvel_rivals"),
        ("  Valorant!!  ", "valorant"),
        ("Roblox / co-op", "roblox_co_op"),
        ("!!!", ""),
    ],
)
def test_slugify(text, expected):
    assert paths.slugify(text) == expected


def test_stream_id_format_matches_the_spec_example():
    assert paths.make_stream_id("2026-08-14", "Marvel") == "2026-08-14_marvel"


def test_stream_id_collision_gets_a_suffix():
    existing = {"2026-08-14_marvel"}
    second = paths.make_stream_id("2026-08-14", "Marvel", existing)
    assert second == "2026-08-14_marvel_2"
    existing.add(second)
    assert paths.make_stream_id("2026-08-14", "Marvel", existing) == "2026-08-14_marvel_3"


def test_stream_id_falls_back_when_the_label_is_unusable():
    assert paths.make_stream_id("2026-08-14", "!!!") == "2026-08-14_stream"
