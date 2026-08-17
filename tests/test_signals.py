"""The signal_series domain type and its persistence.

A6 stores these as BLOBs rather than rows. That is the right call and it means
the round trip has to be exactly right, because a subtly wrong array is much
harder to notice than a missing one.
"""

from __future__ import annotations

import numpy as np
import pytest

from clipforge import db, signals


@pytest.fixture
def conn(tmp_path):
    connection = db.open_db(tmp_path / "sig.db")
    connection.execute(
        "INSERT INTO streams (id, date, master_path, marker_time_base) "
        "VALUES ('s1', '2026-08-14', 'D:/m.mkv', 'vod')"
    )
    yield connection
    connection.close()


def make(values, kind="mic_rms", rate=10.0, t0=0.1, params=None):
    return signals.Series(
        kind=kind, values=np.asarray(values, dtype=np.float32),
        sample_rate_hz=rate, t0=t0, params=params or {},
    )


# --------------------------------------------------------------------------
# timestamps
# --------------------------------------------------------------------------


def test_t0_offsets_every_timestamp():
    """t0 exists so a frame-based signal can say its samples measure a window
    centred half a frame after the window starts."""
    series = make([1, 2, 3, 4], t0=0.1)
    assert series.time_of(0) == pytest.approx(0.1)
    assert series.time_of(3) == pytest.approx(0.4)
    assert series.times().tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4])


def test_duration_spans_first_to_last_sample():
    assert make([1, 2, 3], rate=10.0).duration_s == pytest.approx(0.2)
    assert make([1]).duration_s == 0.0
    assert make([]).duration_s == 0.0


def test_index_at_finds_the_nearest_sample():
    series = make([0, 1, 2, 3], t0=0.1)
    assert series.index_at(0.1) == 0
    assert series.index_at(0.24) == 1
    assert series.index_at(0.26) == 2


def test_index_at_clamps_rather_than_raising():
    """Scoring asks for values near stream edges constantly."""
    series = make([0, 1, 2], t0=0.1)
    assert series.index_at(-99.0) == 0
    assert series.index_at(99.0) == 2


def test_index_at_on_an_empty_series_is_an_error():
    with pytest.raises(IndexError, match="empty"):
        make([]).index_at(1.0)


def test_slice_selects_by_time():
    series = make([0, 1, 2, 3, 4], t0=0.0, rate=10.0)
    assert series.slice(0.1, 0.3).tolist() == [1, 2, 3]
    assert series.slice(0.0, 0.0).tolist() == [0]
    assert series.slice(9.0, 10.0).tolist() == []


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------


def test_round_trip_is_exact(conn):
    values = np.linspace(-80.0, 0.0, 500, dtype=np.float32)
    signals.store(conn, "s1", make(values, params={"frame_s": 0.2}))

    loaded = signals.load(conn, "s1", "mic_rms")
    assert loaded is not None
    np.testing.assert_array_equal(loaded.values, values)
    assert loaded.sample_rate_hz == 10.0
    assert loaded.t0 == pytest.approx(0.1)
    assert loaded.params == {"frame_s": 0.2}


def test_dtype_is_recorded_explicitly(conn):
    """§3.2 defaults the column to 'float32', which does not say byte order,
    and tobytes() is native-endian. The database outlives the machine."""
    signals.store(conn, "s1", make([1.0, 2.0]))
    stored = conn.execute("SELECT dtype FROM signal_series").fetchone()[0]
    assert stored == "<f4"


def test_loaded_values_are_writable(conn):
    """np.frombuffer wraps the sqlite bytes read-only. Scoring smooths and
    resamples in place, so this would fail a long way from its cause."""
    signals.store(conn, "s1", make([1.0, 2.0, 3.0]))
    loaded = signals.load(conn, "s1", "mic_rms")
    loaded.values[0] = 42.0  # must not raise
    assert loaded.values[0] == 42.0


def test_empty_series_round_trips(conn):
    signals.store(conn, "s1", make([]))
    loaded = signals.load(conn, "s1", "mic_rms")
    assert loaded is not None and len(loaded) == 0


def test_storing_twice_replaces_rather_than_duplicating(conn):
    signals.store(conn, "s1", make([1.0]))
    signals.store(conn, "s1", make([1.0, 2.0, 3.0]))
    assert conn.execute("SELECT count(*) FROM signal_series").fetchone()[0] == 1
    assert len(signals.load(conn, "s1", "mic_rms")) == 3


def test_missing_series_is_none(conn):
    assert signals.load(conn, "s1", "nope") is None


def test_load_all_and_kinds(conn):
    signals.store(conn, "s1", make([1.0], kind="mic_rms"))
    signals.store(conn, "s1", make([2.0], kind="game_rms"))
    assert signals.kinds(conn, "s1") == ["game_rms", "mic_rms"]
    assert set(signals.load_all(conn, "s1")) == {"game_rms", "mic_rms"}


def test_a_corrupt_blob_is_detected(conn):
    """n_samples and the blob length are stored separately; if they disagree,
    something truncated the data and silently returning a short array would be
    worse than failing."""
    signals.store(conn, "s1", make([1.0, 2.0, 3.0]))
    conn.execute("UPDATE signal_series SET n_samples = 99")
    with pytest.raises(ValueError, match="declares 99"):
        signals.load(conn, "s1", "mic_rms")


def test_deleting_the_stream_removes_its_signals(conn):
    signals.store(conn, "s1", make([1.0]))
    conn.execute("DELETE FROM streams WHERE id = 's1'")
    assert conn.execute("SELECT count(*) FROM signal_series").fetchone()[0] == 0


# --------------------------------------------------------------------------
# summary stats
# --------------------------------------------------------------------------


def test_summarize_reports_the_distribution():
    stats = signals.summarize(make(np.arange(101, dtype=np.float32)))
    assert stats["n"] == 101
    assert stats["min"] == 0.0
    assert stats["max"] == 100.0
    assert stats["median"] == pytest.approx(50.0)


def test_summarize_handles_an_empty_series():
    assert signals.summarize(make([])) == {"n": 0, "observed": 0}


def test_summarize_separates_samples_from_observations():
    """`n` and `observed` came apart the moment a signal could have gaps. A
    reader given only `n` would take a median over a fifth of a pitch track for
    a median over all of it."""
    values = np.array([100.0, np.nan, np.nan, 200.0], dtype=np.float32)
    stats = signals.summarize(make(values))
    assert stats["n"] == 4
    assert stats["observed"] == 2
    assert stats["median"] == pytest.approx(150.0)
    assert stats["min"] == 100.0 and stats["max"] == 200.0


def test_summarize_of_an_all_absent_series_reports_no_observations():
    """What a party.wav with nobody in Discord produces. A normal result, and
    the caller has to be able to tell it apart from a full one without
    KeyError-ing on the percentiles."""
    stats = signals.summarize(make(np.full(10, np.nan, dtype=np.float32)))
    assert stats == {"n": 10, "observed": 0}
