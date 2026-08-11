"""The fixture is the measuring instrument. Calibrate it before trusting it.

Everything downstream asserts against `manifest.json`. If the manifest and the
media disagree, every later test is measuring the wrong thing while passing.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from clipforge import ffmpeg
from tests.fixtures.make_fixture import (
    BAND_LIMIT_HZ,
    FixtureSpec,
    build_pattern,
    dbfs_to_rms,
    periodic_band_limited_noise,
    rms_to_dbfs,
    scale_to_rms,
)

#: The authored signal is lossless FLAC; the only transform between it and a
#: 16 kHz mono WAV is resampling of band-limited content. Half a dB is
#: generous for that and still tight enough to catch a real error.
DB_TOLERANCE = 0.5


def probe_json(path, *args) -> dict:
    binary = ffmpeg.require("ffprobe")
    proc = ffmpeg.run([binary, "-v", "error", "-of", "json", *args, str(path)])
    return json.loads(proc.stdout)


# --------------------------------------------------------------------------
# signal-generation primitives
# --------------------------------------------------------------------------


def test_noise_is_band_limited():
    """Content above 8 kHz would be destroyed by the 16 kHz extraction in §5.3,
    silently invalidating every dB figure in the manifest."""
    sr = 48_000
    signal = periodic_band_limited_noise(sr, sr, BAND_LIMIT_HZ, seed=1)
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(signal.size, 1.0 / sr)

    above = spectrum[freqs > BAND_LIMIT_HZ * 1.05]
    below = spectrum[freqs < BAND_LIMIT_HZ * 0.95]
    assert above.max() < below.max() * 1e-9


def test_noise_is_periodic_so_tiling_leaves_no_seam():
    """A seam click is broadband and would survive the band limit as an impulse
    the RMS frames would see."""
    sr = 48_000
    signal = periodic_band_limited_noise(sr, sr, BAND_LIMIT_HZ, seed=2)
    doubled = np.concatenate([signal, signal])
    # The step across the wrap point must be no larger than a typical step
    # inside the segment.
    interior = np.abs(np.diff(signal)).max()
    seam = abs(doubled[sr] - doubled[sr - 1])
    assert seam <= interior


@pytest.mark.parametrize("dbfs", [-6.0, -12.0, -20.0, -45.0, -50.0])
def test_scale_to_rms_hits_the_target_exactly(dbfs):
    signal = periodic_band_limited_noise(8192, 48_000, BAND_LIMIT_HZ, seed=3)
    scaled = scale_to_rms(signal, dbfs_to_rms(dbfs))
    measured = rms_to_dbfs(float(np.sqrt(np.mean(np.square(scaled)))))
    assert measured == pytest.approx(dbfs, abs=1e-9)


def test_dbfs_round_trip():
    for dbfs in (-6.0, -35.0, -60.0):
        assert rms_to_dbfs(dbfs_to_rms(dbfs)) == pytest.approx(dbfs)


# --------------------------------------------------------------------------
# the event pattern
# --------------------------------------------------------------------------


def test_pattern_repeats_rather_than_stretches():
    """A 600 s fixture must have ten times the events, not the same three
    spread thinner — scoring needs peaks to find."""
    short, long = FixtureSpec(duration_s=60.0), FixtureSpec(duration_s=600.0)
    build_pattern(short)
    build_pattern(long)
    assert len(long.regions) == 10 * len(short.regions)
    assert len(long.markers) == 10 * len(short.markers)


def test_markers_are_pressed_after_their_events():
    """§4.3: realization plus reaction is 5-15 s. A marker pressed before the
    event it refers to would make the retro-offset untestable."""
    spec = FixtureSpec(duration_s=120.0)
    build_pattern(spec)
    assert spec.markers
    for marker in spec.markers:
        assert marker["t_press"] > marker["intended_event_end_t"]
        assert 5.0 <= marker["reaction_delay_s"] <= 15.0


def test_regions_never_exceed_the_requested_duration():
    spec = FixtureSpec(duration_s=45.0)
    build_pattern(spec)
    assert all(r.t_end <= spec.duration_s for r in spec.regions)
    assert all(m["t_press"] <= spec.duration_s for m in spec.markers)


# --------------------------------------------------------------------------
# the generated media
# --------------------------------------------------------------------------


def test_container_is_matroska(fixture_dir, manifest):
    """§4.2/A7: MKV is what real ingest sees. An MP4 fixture would not exercise
    the container the pipeline is built for."""
    info = probe_json(fixture_dir / manifest["video"], "-show_format")
    assert "matroska" in info["format"]["format_name"]
    assert float(info["format"]["duration"]) == pytest.approx(manifest["duration_s"], abs=0.2)


def test_four_audio_tracks_in_spec_order(fixture_dir, manifest):
    info = probe_json(fixture_dir / manifest["video"], "-show_streams")
    audio = [s for s in info["streams"] if s["codec_type"] == "audio"]
    assert len(audio) == manifest["expected_audio_track_count"] == 4

    for name, index in manifest["expected_track_map"].items():
        assert audio[index]["tags"]["title"] == manifest["expected_track_titles"][name]


def test_audio_is_lossless(fixture_dir, manifest):
    """AAC would move the RMS and make the manifest's figures approximate."""
    info = probe_json(fixture_dir / manifest["video"], "-show_streams")
    audio = [s for s in info["streams"] if s["codec_type"] == "audio"]
    assert {s["codec_name"] for s in audio} == {"flac"}


def test_video_is_cfr_at_the_declared_rate(fixture_dir, manifest):
    info = probe_json(fixture_dir / manifest["video"], "-show_streams")
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert video["r_frame_rate"] == video["avg_frame_rate"] == f"{manifest['fps']}/1"
    assert f"{video['width']}x{video['height']}" == manifest["resolution"]


def test_extracted_audio_matches_the_manifest(fixture_dir, manifest, tmp_path):
    """The whole point of the fixture.

    Extract each track the way §5.3 will — 16 kHz mono PCM — and confirm the
    measured RMS per region still matches what the manifest promised. If this
    drifts, every later assertion about mic_rms is measuring something other
    than what it claims to.
    """
    import soundfile as sf

    binary = ffmpeg.require("ffmpeg")
    video = fixture_dir / manifest["video"]
    tracks = {}
    for name, index in manifest["expected_track_map"].items():
        out = tmp_path / f"{name}.wav"
        ffmpeg.run([
            binary, "-hide_banner", "-nostdin", "-y",
            "-i", str(video),
            "-map", f"0:a:{index}", "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le", str(out),
        ])
        data, sr = sf.read(out)
        assert sr == 16000
        tracks[name] = data

    for region in manifest["regions"]:
        track = region["source_track"]
        data = tracks[track]
        start, end = int(region["t_start"] * 16000), int(region["t_end"] * 16000)
        # Trim the raised-cosine edges, which are not part of the level.
        pad = int(0.02 * 16000)
        window = data[start + pad : end - pad]
        measured = rms_to_dbfs(float(np.sqrt(np.mean(np.square(window)))))
        assert measured == pytest.approx(
            region["measured_dbfs"][track], abs=DB_TOLERANCE
        ), f"{region['label']} on {track}"


def test_the_contamination_case_is_actually_indistinguishable(manifest):
    """§4.2's claim, made concrete.

    On the mixed track the game explosion and the operator's loudest speech sit
    within a fraction of a dB of each other. A single-track recording therefore
    cannot tell "something exploded" from "he shouted" — which is exactly why
    §4.2 calls the separation non-negotiable and unrecoverable after the fact.
    """
    by_label = {r["label"].split("#")[0]: r for r in manifest["regions"]}
    explosion = by_label["game_explosion"]["measured_dbfs"]
    speech = by_label["speech_peak"]["measured_dbfs"]

    # On their own tracks they are trivially separable...
    assert explosion["game"] == pytest.approx(-6.0, abs=0.1)
    assert speech["mic"] == pytest.approx(-6.0, abs=0.1)
    # ...and the mic is near its floor during the explosion.
    assert explosion["mic"] < manifest["floors_dbfs"]["mic"] + 3.0

    # But mixed down they are the same event as far as RMS is concerned.
    assert abs(explosion["mixed"] - speech["mixed"]) < 1.0


# --------------------------------------------------------------------------
# capture-side files
# --------------------------------------------------------------------------


def test_anchor_matches_the_documented_contract(fixture_dir, manifest):
    """This file's shape is what Phase 0's obs_anchor.py must produce."""
    anchor = json.loads((fixture_dir / manifest["anchor"]).read_text(encoding="utf-8"))
    assert anchor["schema"] == 1
    assert isinstance(anchor["record_start_epoch_ms"], int)
    assert anchor["record_start_epoch_ms"] > 1_600_000_000_000  # plausible epoch ms


def test_markers_are_epoch_ms_only(fixture_dir, manifest):
    """A8: epoch milliseconds everywhere in capture. No VOD times, no formatted
    strings — conversion happens once, at ingest, via the anchor."""
    lines = (fixture_dir / manifest["markers_file"]).read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(manifest["markers"])
    for line in lines:
        record = json.loads(line)
        assert set(record) == {"epoch_ms", "kind"}
        assert isinstance(record["epoch_ms"], int)
        assert record["kind"] in ("marker_maybe", "marker_definite")


def test_marker_epoch_converts_back_to_the_intended_press_time(fixture_dir, manifest):
    """§4.1's conversion, verified end to end on known values."""
    anchor = json.loads((fixture_dir / manifest["anchor"]).read_text(encoding="utf-8"))
    start_ms = anchor["record_start_epoch_ms"]
    lines = (fixture_dir / manifest["markers_file"]).read_text(encoding="utf-8").splitlines()

    for line, expected in zip(lines, manifest["markers"], strict=True):
        record = json.loads(line)
        vod_s = (record["epoch_ms"] - start_ms) / 1000.0
        assert vod_s == pytest.approx(expected["t_press"], abs=0.001)


# --------------------------------------------------------------------------
# caching
# --------------------------------------------------------------------------


def test_regeneration_is_skipped_when_the_spec_is_unchanged(fixture_dir, manifest):
    from tests.fixtures.make_fixture import generate

    before = (fixture_dir / manifest["video"]).stat().st_mtime_ns
    generate(FixtureSpec())
    assert (fixture_dir / manifest["video"]).stat().st_mtime_ns == before


def test_spec_hash_changes_with_any_parameter():
    base = FixtureSpec()
    assert FixtureSpec().hash() == base.hash()
    assert FixtureSpec(duration_s=61.0).hash() != base.hash()
    assert FixtureSpec(fps="60").hash() != base.hash()
    assert FixtureSpec(track_titles=False).hash() != base.hash()


# --------------------------------------------------------------------------
# A1 enforcement lives with ffmpeg, but the fixture is the first real user
# --------------------------------------------------------------------------


def test_seek_before_input_is_accepted():
    ffmpeg.validate_seek_placement(["ffmpeg", "-ss", "10", "-i", "a.mkv", "out.mp4"])


def test_seek_after_input_is_rejected():
    with pytest.raises(ffmpeg.SeekPlacementError, match="A1"):
        ffmpeg.validate_seek_placement(["ffmpeg", "-i", "a.mkv", "-ss", "10", "out.mp4"])


def test_multi_input_seeks_are_all_accepted():
    """`-ss 20` sits after the first -i but before the input it applies to."""
    ffmpeg.validate_seek_placement(
        ["ffmpeg", "-ss", "10", "-i", "a.mkv", "-ss", "20", "-i", "b.mkv", "out.mp4"]
    )


def test_run_refuses_to_execute_an_a1_violation():
    with pytest.raises(ffmpeg.SeekPlacementError):
        ffmpeg.run(["ffmpeg", "-i", "a.mkv", "-ss", "1", "out.mp4"])


def test_run_raises_with_context_on_failure(needs_ffmpeg):
    binary = ffmpeg.require("ffprobe")
    with pytest.raises(ffmpeg.FFmpegError) as exc:
        ffmpeg.run([binary, "-v", "error", "definitely-not-a-file.mkv"])
    assert exc.value.returncode != 0
    assert "definitely-not-a-file" in str(exc.value)
