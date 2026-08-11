"""Atomic artifact writes (A10)."""

from __future__ import annotations

import os

import pytest

from clipforge.pipeline import atomic


def test_temp_name_preserves_the_extension(tmp_path):
    """ffmpeg infers the muxer from the output extension. `proxy.mp4.tmp` is
    the obvious temp name and ffmpeg refuses to write it."""
    tmp = atomic.temp_path_for(tmp_path / "proxy.mp4", pid=1234)
    assert tmp.suffix == ".mp4"
    assert tmp.name == ".tmp-1234-proxy.mp4"


def test_temp_sits_beside_its_destination(tmp_path):
    """os.replace is only atomic within a filesystem."""
    final = tmp_path / "nested" / "proxy.mp4"
    assert atomic.temp_path_for(final).parent == final.parent


def test_success_moves_the_file_into_place(tmp_path):
    final = tmp_path / "out" / "proxy.mp4"
    with atomic.atomic_output(final) as tmp:
        tmp.write_bytes(b"encoded")
        assert not final.exists()  # nothing visible until the block completes
    assert final.read_bytes() == b"encoded"
    assert not list(final.parent.glob(".tmp-*"))


def test_failure_leaves_nothing_behind(tmp_path):
    final = tmp_path / "proxy.mp4"
    with pytest.raises(RuntimeError):
        with atomic.atomic_output(final) as tmp:
            tmp.write_bytes(b"half an encode")
            raise RuntimeError("killed mid-encode")
    assert not final.exists()
    assert not list(tmp_path.glob(".tmp-*"))


def test_failure_preserves_the_previous_artifact(tmp_path):
    """A crashed re-encode must not destroy the proxy that already worked."""
    final = tmp_path / "proxy.mp4"
    final.write_bytes(b"the good proxy")
    with pytest.raises(RuntimeError):
        with atomic.atomic_output(final) as tmp:
            tmp.write_bytes(b"garbage")
            raise RuntimeError("boom")
    assert final.read_bytes() == b"the good proxy"


def test_a_stage_that_writes_nothing_is_an_error(tmp_path):
    """Otherwise it would be recorded as done with no artifact."""
    with pytest.raises(FileNotFoundError, match="without writing"):
        with atomic.atomic_output(tmp_path / "proxy.mp4"):
            pass


def test_multi_output_is_all_or_nothing(tmp_path):
    """audio_split writes mic/game/party as one unit; two of three in place
    would satisfy an existence check while being wrong."""
    finals = [tmp_path / f"{name}.wav" for name in ("mic", "game", "party")]
    with pytest.raises(FileNotFoundError, match="all declared outputs"):
        with atomic.atomic_outputs(finals) as temps:
            temps[0].write_bytes(b"mic")
            temps[1].write_bytes(b"game")
            # party never written
    assert not any(f.exists() for f in finals)
    assert not list(tmp_path.glob(".tmp-*"))


def test_multi_output_success(tmp_path):
    finals = [tmp_path / f"{name}.wav" for name in ("mic", "game")]
    with atomic.atomic_outputs(finals) as temps:
        for tmp, name in zip(temps, (b"mic", b"game"), strict=True):
            tmp.write_bytes(name)
    assert [f.read_bytes() for f in finals] == [b"mic", b"game"]


# --------------------------------------------------------------------------
# orphan sweep
# --------------------------------------------------------------------------


def test_sweep_removes_debris_from_a_dead_run(tmp_path):
    dead = tmp_path / "proxy" / ".tmp-99999-proxy.mp4"
    dead.parent.mkdir(parents=True)
    dead.write_bytes(b"interrupted")
    removed = atomic.sweep_orphans(tmp_path)
    assert removed == [dead]
    assert not dead.exists()


def test_sweep_spares_our_own_temps(tmp_path):
    """A live encode's temp file must survive the sweep that runs alongside it."""
    mine = atomic.temp_path_for(tmp_path / "proxy.mp4")
    mine.write_bytes(b"in progress")
    assert atomic.sweep_orphans(tmp_path) == []
    assert mine.exists()


def test_sweep_ignores_real_artifacts(tmp_path):
    keeper = tmp_path / "proxy.mp4"
    keeper.write_bytes(b"finished")
    atomic.sweep_orphans(tmp_path)
    assert keeper.exists()


def test_sweep_handles_a_missing_directory(tmp_path):
    assert atomic.sweep_orphans(tmp_path / "never-created") == []


def test_sweep_recurses(tmp_path):
    for sub in ("proxy", "audio", "exports"):
        path = tmp_path / sub / f".tmp-4242-{sub}.bin"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"x")
    assert len(atomic.sweep_orphans(tmp_path)) == 3


def test_replace_retry_surfaces_a_helpful_message(tmp_path, monkeypatch):
    """On Windows the rename fails while a player or the review UI holds the
    destination open. That is a normal thing for the operator to be doing."""
    monkeypatch.setattr(atomic, "REPLACE_ATTEMPTS", 2)
    monkeypatch.setattr(atomic, "REPLACE_BACKOFF_S", 0.0)

    def always_busy(src, dst):
        raise PermissionError(32, "The process cannot access the file")

    monkeypatch.setattr(os, "replace", always_busy)
    source, destination = tmp_path / "a", tmp_path / "b"
    source.write_bytes(b"x")

    with pytest.raises(atomic.AtomicReplaceError, match="holding the destination open"):
        atomic.replace_with_retry(source, destination)


def test_replace_succeeds_after_a_transient_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(atomic, "REPLACE_BACKOFF_S", 0.0)
    real = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError(32, "busy")
        real(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    source, destination = tmp_path / "a", tmp_path / "b"
    source.write_bytes(b"payload")
    atomic.replace_with_retry(source, destination)
    assert destination.read_bytes() == b"payload"
    assert calls["n"] == 2
