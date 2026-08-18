"""§7.2's preview assets, and the three things §7.2 gets wrong.

Every numeric assertion reads the config object, the fixture manifest, or the
files the stage actually wrote. No expected byte count or encode time appears
here: those are recorded in `spec/GUESSES.md` and the module docstring, where a
number that moves is information rather than a failure.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from clipforge import config, db, ffmpeg, paths
from clipforge.pipeline import runner, stages
from clipforge.pipeline.context import StageContext
from clipforge.review import previews
from tests.fixtures.make_fixture import FixtureSpec, generate, load_manifest


# --------------------------------------------------------------------------
# naming: the part that makes §6.1's "re-scoring is free" true
# --------------------------------------------------------------------------


def test_asset_names_come_from_the_window_not_the_candidate_id():
    """THE reason this stage does not follow §7.2's own command.

    §7.2 writes `previews/{candidate_id}.webm`. Candidates here are append-only
    generations, so a re-score with different weights mints entirely new ids —
    and §6.1's promise that re-scoring is "free and infinitely repeatable" would
    silently come to include a full re-encode of every preview in the library.
    """
    assert previews.clip_name(162.0) == previews.clip_name(162.0)
    assert previews.strip_name(1.0, 9.0) == previews.strip_name(1.0, 9.0)
    # Different moments get different names...
    assert previews.clip_name(162.0) != previews.clip_name(163.0)
    # ...and the same moment gets one name regardless of float noise, because a
    # peak that survives a round trip through SQLite REAL must not mint a second
    # file for the same instant.
    assert previews.clip_name(162.0) == previews.clip_name(162.0 + 1e-9)


def test_a_nudged_window_reuses_the_clip_and_not_the_strip():
    """The clip is centred on the peak, so §7.3's boundary nudges do not touch it.

    This is what makes the content-addressing worth having rather than merely
    tidy: the expensive asset is the one keyed on the value a nudge cannot move.
    """
    peak, start, end = 100.0, 96.0, 104.0
    assert previews.clip_name(peak) == previews.clip_name(peak)
    assert previews.strip_name(start, end) != previews.strip_name(start - 0.5, end)


# --------------------------------------------------------------------------
# the 2 s window: C2 says shift, not shorten
# --------------------------------------------------------------------------


@pytest.mark.parametrize("peak", [0.0, 0.3, 0.999, 1.0])
def test_a_peak_at_the_head_still_gets_a_full_length_clip(peak):
    """§7.2's bare `t_peak - 1` asks ffmpeg to seek to a negative timestamp."""
    want = 2.0
    start, duration = previews.clip_window(peak, want, 600.0)
    assert start >= 0.0
    assert duration == pytest.approx(want)


@pytest.mark.parametrize("peak", [599.9, 600.0])
def test_a_peak_at_the_tail_slides_back_rather_than_truncating(peak):
    """C2: expand rather than contract. A short preview is a contracted one."""
    span, want = 600.0, 2.0
    start, duration = previews.clip_window(peak, want, span)
    assert duration == pytest.approx(want)
    assert start + duration <= span + 1e-9


def test_a_recording_shorter_than_the_clip_is_the_only_shortening_case():
    """There is no footage to slide onto, so this one genuinely cannot be 2 s."""
    start, duration = previews.clip_window(0.5, 2.0, 1.2)
    assert (start, duration) == (0.0, 1.2)


# --------------------------------------------------------------------------
# the thumb strip
# --------------------------------------------------------------------------


def test_strip_plan_spreads_the_configured_frame_count_across_the_window():
    frames = 5
    step, columns = previews.strip_plan(window_s=54.8, fps=30.0, frames=frames)
    assert columns == frames
    # Every selected index has to fall inside the window, or `tile` waits at EOF
    # for a frame that never arrives.
    total = round(54.8 * 30.0)
    assert step * (columns - 1) < total


def test_a_window_shorter_than_the_frame_count_narrows_the_strip():
    """§7.3's nudges are not clamped to `min_window_s`, so this is reachable."""
    step, columns = previews.strip_plan(window_s=0.1, fps=30.0, frames=5)
    assert step == 1
    assert columns == 3  # 0.1 s at 30 fps is three frames, and that is the strip


def test_strip_plan_never_returns_a_zero_step():
    """A zero step is `mod(n, 0)` inside a filter expression."""
    for window in (0.0, 0.001, 0.03, 1.0):
        step, columns = previews.strip_plan(window, 30.0, 5)
        assert step >= 1 and columns >= 1


# --------------------------------------------------------------------------
# the commands
# --------------------------------------------------------------------------


def test_clip_command_puts_ss_before_i(tmp_path):
    """A1, which §7.2 is emphatic about. `ffmpeg.run` enforces it too."""
    cfg = config.load()
    argv = previews.clip_command(
        "ffmpeg", tmp_path / "proxy.mp4", tmp_path / "out.webm",
        start_s=10.0, duration_s=2.0, cfg=cfg,
    )
    assert argv.index("-ss") < argv.index("-i")


def test_clip_command_reads_every_value_from_config(tmp_path):
    """§17: no tunable is inline. Asserted against the config, not a literal."""
    cfg = config.load()
    argv = previews.clip_command(
        "ffmpeg", tmp_path / "proxy.mp4", tmp_path / "out.webm",
        start_s=0.0, duration_s=2.0, cfg=cfg,
    )
    joined = " ".join(argv)
    assert str(cfg.get("previews.clip.codec")) in joined
    assert f"scale={int(cfg.get('previews.clip.width'))}:-2" in joined
    assert str(int(cfg.get("previews.clip.crf"))) in argv
    assert str(int(cfg.get("previews.clip.cpu_used"))) in argv
    assert str(cfg.get("previews.clip.deadline")) in argv
    # §7.2's own flag: the loop is silent, and `space` plays the proxy.
    assert "-an" in argv


def test_libvpx_only_flags_are_omitted_for_other_codecs(tmp_path):
    """`-cpu-used` is a libvpx option; emitting it for x264 fails at open."""
    cfg = config.load(overrides=["previews.clip.codec=libx264"])
    argv = previews.clip_command(
        "ffmpeg", tmp_path / "proxy.mp4", tmp_path / "out.mp4",
        start_s=0.0, duration_s=2.0, cfg=cfg,
    )
    assert "-cpu-used" not in argv
    assert "-deadline" not in argv


def test_strip_command_escapes_the_comma_inside_mod(tmp_path):
    """ffmpeg splits filter arguments on commas before the expression parser
    ever sees the string, so an unescaped `mod(n,5)` is two arguments."""
    cfg = config.load()
    argv = previews.strip_command(
        "ffmpeg", tmp_path / "proxy.mp4", tmp_path / "out.jpg",
        t_start=0.0, window_s=10.0, fps=30.0, cfg=cfg,
    )
    graph = argv[argv.index("-vf") + 1]
    assert "mod(n\\," in graph
    assert argv.index("-ss") < argv.index("-i")


# --------------------------------------------------------------------------
# the stage, end to end on the fixture
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def previewed(tmp_path_factory):
    """The fixture stream taken all the way through `previews`.

    Module-scoped: the whole point of this stage is that it is expensive, and
    running the pipeline once per test would take longer than the stage does.
    Tests that mutate the directory put it back.
    """
    directory = generate(FixtureSpec(name="long", duration_s=600.0))
    manifest = load_manifest(directory)

    root = tmp_path_factory.mktemp("previews")
    cfg = config.load(overrides=[f"paths.data_root={(root / 'data').as_posix()}"])
    conn = db.open_db(cfg.db_path)
    anchor = json.loads(
        (directory / manifest["anchor"]).read_text())["record_start_epoch_ms"]
    conn.execute(
        "INSERT INTO streams (id, date, title, master_path, marker_time_base, "
        "record_start_epoch_ms) VALUES ('fx', '2026-08-14', 'Fixture', ?, 'epoch', ?)",
        (str(directory / manifest["video"]), anchor),
    )
    stream_paths = paths.StreamPaths(cfg.data_root, "fx").ensure()
    shutil.copy2(directory / manifest["markers_file"], stream_paths.markers_jsonl)

    engine = runner.Runner(cfg=cfg, conn=conn, stream_id="fx", log=lambda *_: None)
    engine.mark_external_done("register_stream")
    engine.execute(engine.plan())
    conn.close()
    return cfg, "fx", stream_paths


def _probe(cfg, path, fields):
    argv = [ffmpeg.require("ffprobe", cfg.ffprobe_override), "-v", "error",
            "-select_streams", "v:0", "-show_entries", f"stream={fields}",
            "-of", "json", str(path)]
    output = subprocess.run(argv, capture_output=True, text=True, check=True).stdout
    return json.loads(output)["streams"][0]


def test_the_stage_is_registered_and_reaches_the_default_plan():
    spec = stages.get("previews")
    assert spec.implemented
    assert spec.module == "clipforge.review.previews"
    # Both are required and neither is optional: the clip comes from the proxy
    # and the windows come from scoring.
    assert set(spec.requires) == {"score", "proxy"}
    assert "previews" in stages.default_plan()


def test_every_current_candidate_gets_both_assets(previewed):
    cfg, stream_id, stream_paths = previewed
    conn = db.open_db(cfg.db_path)
    try:
        rows = conn.execute(
            "SELECT preview_path, thumbstrip_path, waveform_path FROM candidates "
            "WHERE stream_id = ? AND is_current = 1", (stream_id,)
        ).fetchall()
    finally:
        conn.close()

    assert rows
    for row in rows:
        assert row["preview_path"] and row["thumbstrip_path"]
        assert stream_paths.absolute(row["preview_path"]).is_file()
        assert stream_paths.absolute(row["thumbstrip_path"]).is_file()
        # §7.2's waveform PNG is deliberately not written -- the column stays
        # because §3.2 declares it. See the module docstring.
        assert row["waveform_path"] is None


def test_the_clip_is_the_configured_size_and_codec(previewed):
    """Read off the file, not off the command that was meant to write it."""
    cfg, stream_id, stream_paths = previewed
    conn = db.open_db(cfg.db_path)
    try:
        row = conn.execute(
            "SELECT preview_path FROM candidates WHERE stream_id = ? AND is_current = 1 "
            "AND preview_path IS NOT NULL LIMIT 1", (stream_id,)
        ).fetchone()
    finally:
        conn.close()
    assert row is not None

    stream = _probe(cfg, stream_paths.absolute(row["preview_path"]),
                    "codec_name,width,height")
    assert int(stream["width"]) == int(cfg.get("previews.clip.width"))
    # The height follows from `-2` rather than a second configured number, and
    # libvpx requires it even.
    assert int(stream["height"]) % 2 == 0
    assert stream["codec_name"].startswith("vp")


def test_the_strip_is_frames_times_width_wide(previewed):
    """§7.2: "5 frames evenly spaced across window, JPEG, 160px wide"."""
    cfg, stream_id, stream_paths = previewed
    conn = db.open_db(cfg.db_path)
    try:
        row = conn.execute(
            "SELECT thumbstrip_path FROM candidates WHERE stream_id = ? "
            "AND is_current = 1 AND thumbstrip_path IS NOT NULL "
            "ORDER BY (t_end - t_start) DESC LIMIT 1", (stream_id,)
        ).fetchone()
    finally:
        conn.close()
    assert row is not None

    stream = _probe(cfg, stream_paths.absolute(row["thumbstrip_path"]), "width,height")
    assert int(stream["width"]) == (int(cfg.get("previews.thumbstrip.frames"))
                                   * int(cfg.get("previews.thumbstrip.width")))


def _rerun(cfg, stream_id, wanted):
    conn = db.open_db(cfg.db_path)
    try:
        engine = runner.Runner(cfg=cfg, conn=conn, stream_id=stream_id,
                               log=lambda *_: None, force=set(wanted))
        engine.execute(engine.plan(list(wanted)))
    finally:
        conn.close()


def test_a_forced_rerun_writes_nothing(previewed):
    """C7, and the payoff of naming assets by window rather than candidate id."""
    cfg, stream_id, stream_paths = previewed
    before = {p.name: p.stat().st_mtime_ns for p in stream_paths.previews_dir.iterdir()}
    assert before

    _rerun(cfg, stream_id, ["previews"])

    after = {p.name: p.stat().st_mtime_ns for p in stream_paths.previews_dir.iterdir()}
    assert after == before, "a forced re-run re-encoded assets it could have reused"


def test_an_identical_rescore_reuses_every_asset(previewed):
    """§6.1: re-scoring is "free and infinitely repeatable".

    Under §7.2's own `{candidate_id}.webm` naming this would re-encode
    everything, because a new generation means entirely new candidate ids.
    """
    cfg, stream_id, stream_paths = previewed
    before = {p.name: p.stat().st_mtime_ns for p in stream_paths.previews_dir.iterdir()}

    _rerun(cfg, stream_id, ["score", "previews"])

    after = {p.name: p.stat().st_mtime_ns for p in stream_paths.previews_dir.iterdir()}
    for name, mtime in before.items():
        assert after.get(name) == mtime, f"{name} was re-encoded by an identical re-score"


def test_verify_fails_when_an_asset_is_deleted(previewed):
    """A stage whose product is files has to notice them going missing, or it
    looks permanently done — the same reason `score` overrides `verify`."""
    cfg, stream_id, stream_paths = previewed
    conn = db.open_db(cfg.db_path)
    try:
        ctx = StageContext(cfg=cfg, conn=conn, stream_id=stream_id, log=lambda _m: None)
        assert previews.verify(ctx)[0]

        victim = next(p for p in stream_paths.previews_dir.iterdir()
                      if p.suffix == ".webm")
        kept = victim.read_bytes()
        victim.unlink()
        try:
            ok, why = previews.verify(ctx)
            assert not ok
            assert "missing" in why
        finally:
            victim.write_bytes(kept)
    finally:
        conn.close()


def test_prune_removes_only_unreferenced_assets(previewed):
    cfg, stream_id, stream_paths = previewed
    orphan = stream_paths.previews_dir / "p999999999.webm"
    orphan.write_bytes(b"not referenced by any candidate")
    referenced = sorted(p.name for p in stream_paths.previews_dir.iterdir()
                        if p.name != orphan.name)

    conn = db.open_db(cfg.db_path)
    try:
        removed = previews.prune(conn, stream_paths)
    finally:
        conn.close()

    assert [p.name for p in removed] == [orphan.name]
    assert not orphan.exists()
    assert sorted(p.name for p in stream_paths.previews_dir.iterdir()) == referenced


def test_the_stage_defers_rather_than_fails_when_disabled(previewed):
    """Stages defer, they do not fail — and `score` must stay reachable."""
    cfg, stream_id, _ = previewed
    disabled = config.load(overrides=[
        f"paths.data_root={cfg.data_root.as_posix()}",
        "previews.enabled=false",
    ])
    conn = db.open_db(cfg.db_path)
    try:
        ctx = StageContext(cfg=disabled, conn=conn, stream_id=stream_id,
                           log=lambda _m: None)
        ok, why = previews.available(ctx)
        assert not ok
        assert "previews.enabled" in why
    finally:
        conn.close()


