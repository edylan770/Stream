"""audio_features: RMS framing, dB conversion, and the fixture's real levels."""

from __future__ import annotations

import math

import numpy as np
import pytest
import soundfile as sf

from clipforge import config, db, signals
from clipforge.extract import features
from clipforge.pipeline import runner, stages

#: Frame RMS over band-limited noise varies frame to frame; the median over a
#: region converges much tighter than any single frame.
DB_TOLERANCE = 0.25


def framing(rate=16000, signal_hz=10.0, frame_s=0.2):
    return features.framing_for(rate, signal_hz=signal_hz, frame_s=frame_s)


def write_wav(path, values, rate=16000):
    sf.write(path, np.asarray(values, dtype=np.float32), rate, subtype="FLOAT")
    return path


# --------------------------------------------------------------------------
# framing
# --------------------------------------------------------------------------


def test_hop_is_derived_from_signal_hz():
    """One fact, not two. There was briefly a separate hop_s in config, which
    could disagree with signal_hz and silently shift every timestamp."""
    assert framing(16000, 10.0).hop == 1600
    assert framing(48000, 10.0).hop == 4800
    assert framing(16000, 20.0).hop == 800


def test_frame_length_comes_from_frame_s():
    assert framing(16000, frame_s=0.2).frame == 3200
    assert framing(16000, frame_s=0.128).frame == 2048  # librosa's default


def test_t0_is_half_a_frame():
    """No centering: frame i covers [i*hop, i*hop+frame), so its centre is half
    a frame later, and t0 is where that gets recorded."""
    assert framing(16000, frame_s=0.2).t0 == pytest.approx(0.1)
    assert framing(16000, frame_s=0.128).t0 == pytest.approx(0.064)


def test_frame_count_never_runs_past_the_end():
    f = framing(16000, 10.0, 0.2)
    assert f.n_frames(0) == 0
    assert f.n_frames(f.frame - 1) == 0
    assert f.n_frames(f.frame) == 1
    assert f.n_frames(f.frame + f.hop) == 2
    # The last frame must fit entirely inside the signal.
    n = f.n_frames(160000)
    assert (n - 1) * f.hop + f.frame <= 160000
    assert n * f.hop + f.frame > 160000


def test_impossible_framing_is_rejected():
    with pytest.raises(features.FeatureError, match="signal_hz"):
        features.framing_for(100, signal_hz=1000.0, frame_s=0.2)
    with pytest.raises(features.FeatureError, match="frame_s"):
        features.framing_for(16000, signal_hz=10.0, frame_s=0.0)


# --------------------------------------------------------------------------
# RMS and dB
# --------------------------------------------------------------------------


@pytest.mark.parametrize("amplitude", [1.0, 0.5, 0.1, 0.01])
def test_constant_signal_gives_exactly_its_amplitude(amplitude):
    f = framing()
    values = np.full(f.frame * 4, amplitude, dtype=np.float32)
    rms = features.frame_rms(values, f)
    assert rms.size > 0
    np.testing.assert_allclose(rms, amplitude, rtol=1e-6)


def test_sine_gives_amplitude_over_root_two():
    """The textbook value, and a check that framing is not off by a factor."""
    f = framing()
    t = np.arange(f.frame * 8) / f.sample_rate
    values = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    rms = features.frame_rms(values, f)
    np.testing.assert_allclose(rms, 0.5 / math.sqrt(2), rtol=0.02)


def test_short_signal_yields_no_frames():
    f = framing()
    assert features.frame_rms(np.zeros(f.frame - 1, dtype=np.float32), f).size == 0


@pytest.mark.parametrize("dbfs", [0.0, -6.0, -20.0, -60.0])
def test_db_conversion_round_trips(dbfs):
    amplitude = 10.0 ** (dbfs / 20.0)
    result = features.to_db(np.array([amplitude]), db_floor=-80.0)
    assert result[0] == pytest.approx(dbfs, abs=1e-9)


def test_digital_silence_floors_instead_of_returning_negative_infinity():
    """-inf propagates through every mean, z-score and plot downstream."""
    result = features.to_db(np.zeros(5), db_floor=-80.0)
    assert np.all(np.isfinite(result))
    assert np.all(result == -80.0)


def test_values_below_the_floor_are_clamped():
    tiny = 10.0 ** (-120.0 / 20.0)
    assert features.to_db(np.array([tiny]), db_floor=-80.0)[0] == -80.0


# --------------------------------------------------------------------------
# streaming
# --------------------------------------------------------------------------


def test_streaming_matches_a_single_pass(tmp_path, monkeypatch):
    """Overlap-save correctness: frames spanning a read-block boundary must be
    computed once and correctly. Forcing many small blocks is the only way to
    exercise it, since a 60 s fixture fits in one block otherwise.
    """
    rng = np.random.default_rng(7)
    values = rng.standard_normal(16000 * 5).astype(np.float32) * 0.1
    path = write_wav(tmp_path / "noise.wav", values)

    monkeypatch.setattr(features, "READ_BLOCK_SAMPLES", 1 << 20)
    whole = features.rms_series(
        path, "mic_rms", signal_hz=10.0, frame_s=0.2, db_floor=-80.0
    )

    # A block size that is deliberately not a multiple of the hop, so frames
    # land across boundaries at every possible offset.
    monkeypatch.setattr(features, "READ_BLOCK_SAMPLES", 997)
    chunked = features.rms_series(
        path, "mic_rms", signal_hz=10.0, frame_s=0.2, db_floor=-80.0
    )

    assert len(whole) == len(chunked)
    np.testing.assert_allclose(whole.values, chunked.values, atol=1e-4)


def test_sample_count_matches_the_framing(tmp_path):
    values = np.full(16000 * 3, 0.25, dtype=np.float32)
    path = write_wav(tmp_path / "flat.wav", values)
    series = features.rms_series(
        path, "mic_rms", signal_hz=10.0, frame_s=0.2, db_floor=-80.0
    )
    assert len(series) == framing().n_frames(values.size)
    assert series.t0 == pytest.approx(0.1)
    np.testing.assert_allclose(series.values, 20 * math.log10(0.25), atol=1e-4)


def test_params_record_how_the_series_was_made(tmp_path):
    """§3.2 gained this column so re-extracting at a different hop is visible
    rather than a silent overwrite of incomparable data."""
    path = write_wav(tmp_path / "x.wav", np.zeros(16000, dtype=np.float32))
    series = features.rms_series(
        path, "mic_rms", signal_hz=10.0, frame_s=0.2, db_floor=-80.0, role="mic"
    )
    assert series.params["hop_samples"] == 1600
    assert series.params["frame_samples"] == 3200
    assert series.params["source_sample_rate"] == 16000
    assert series.params["centered"] is False
    assert series.params["role"] == "mic"


def test_stereo_input_is_rejected(tmp_path):
    """audio_split writes mono; anything else means the wrong file."""
    path = tmp_path / "stereo.wav"
    sf.write(path, np.zeros((16000, 2), dtype=np.float32), 16000, subtype="FLOAT")
    with pytest.raises(features.FeatureError, match="channels"):
        features.rms_series(path, "mic_rms", signal_hz=10.0, frame_s=0.2, db_floor=-80.0)


# --------------------------------------------------------------------------
# against the fixture
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def extracted(tmp_path_factory, fixture_dir, manifest):
    root = tmp_path_factory.mktemp("features")
    cfg = config.load(overrides=[f"paths.data_root={(root / 'data').as_posix()}"])
    conn = db.open_db(cfg.db_path)
    conn.execute(
        "INSERT INTO streams (id, date, master_path, marker_time_base) "
        "VALUES ('fx', '2026-08-14', ?, 'epoch')",
        (str(fixture_dir / manifest["video"]),),
    )
    engine = runner.Runner(cfg=cfg, conn=conn, stream_id="fx", log=lambda *_: None)
    engine.mark_external_done("register_stream")
    results = engine.execute(engine.plan())
    yield cfg, conn, engine, results
    conn.close()


def test_stage_succeeds(extracted):
    cfg, conn, engine, results = extracted
    assert all(r.status == "done" for r in results), [r.error for r in results]


def test_one_series_per_extracted_track(extracted):
    """C6: log everything from day one. Only mic_rms is weighted in Phase 1.

    Derived from config rather than written down, because what the stage
    produces is now two things: an RMS series for every extracted track, and a
    pitch series for every track `extract.f0.roles` asks for — and §5.4.1
    defines no pitch signal for the game track, which has no voice.
    """
    cfg, conn, engine, _ = extracted
    expected = {f"{role}_rms" for role in cfg.get("ingest.audio.extract_roles")}
    if cfg.get("extract.f0.enabled"):
        expected |= {
            f"{role}_f0" for role in cfg.get("extract.f0.roles")
            if role in features.F0_FOR_ROLE
        }
    assert set(signals.kinds(conn, "fx")) == expected
    assert "game_f0" not in expected


def test_series_are_stored_at_the_configured_rate(extracted, manifest):
    cfg, conn, engine, _ = extracted
    expected_hz = cfg.get("extract.signal_hz")
    for kind in signals.kinds(conn, "fx"):
        series = signals.load(conn, "fx", kind)
        assert series.sample_rate_hz == expected_hz
        # Coverage to within one frame of the stream's duration.
        covered = series.t0 + series.duration_s
        assert covered == pytest.approx(manifest["duration_s"], abs=cfg.get("extract.rms.frame_s"))


def test_region_levels_match_the_manifest(extracted, manifest):
    """The point of the whole chain: what the pipeline measures is what was
    authored, per region, to a quarter of a dB."""
    cfg, conn, engine, _ = extracted
    series = {k: signals.load(conn, "fx", k) for k in signals.kinds(conn, "fx")}

    for region in manifest["regions"]:
        kind = features.SIGNAL_FOR_ROLE[region["source_track"]]
        # Trim a frame from each edge so no sample straddles the boundary.
        pad = cfg.get("extract.rms.frame_s")
        chunk = series[kind].slice(region["t_start"] + pad, region["t_end"] - pad)
        assert chunk.size > 0, region["label"]
        measured = float(np.median(chunk.astype(np.float64)))
        assert measured == pytest.approx(
            region["measured_dbfs"][region["source_track"]], abs=DB_TOLERANCE
        ), region["label"]


def test_floors_match_the_declared_noise_floors(extracted, manifest):
    """The median of each series is its floor, since the quiet stretches
    dominate. Derived from the manifest, including the headroom scaling."""
    cfg, conn, engine, _ = extracted
    scale_db = 20 * math.log10(manifest["headroom_scale"])

    for role, declared in manifest["floors_dbfs"].items():
        series = signals.load(conn, "fx", features.SIGNAL_FOR_ROLE[role])
        median = float(np.median(series.values.astype(np.float64)))
        assert median == pytest.approx(declared + scale_db, abs=DB_TOLERANCE)


def test_contamination_case_in_signal_form(extracted, manifest):
    """§4.2, now visible in the signals themselves.

    During the explosion the game track is loud and the mic sits at its floor;
    the two loud events are the same level on their own tracks. Everything that
    makes them separable was thrown away on the mixed track.
    """
    cfg, conn, engine, _ = extracted
    by_label = {r["label"].split("#")[0]: r for r in manifest["regions"]}
    explosion, speech = by_label["game_explosion"], by_label["speech_peak"]

    mic = signals.load(conn, "fx", "mic_rms")
    game = signals.load(conn, "fx", "game_rms")

    def level(series, region):
        pad = cfg.get("extract.rms.frame_s")
        chunk = series.slice(region["t_start"] + pad, region["t_end"] - pad)
        return float(np.median(chunk.astype(np.float64)))

    assert level(game, explosion) - level(mic, explosion) > 20
    assert level(game, explosion) == pytest.approx(level(mic, speech), abs=0.5)


def test_stage_is_idempotent(extracted):
    cfg, conn, engine, _ = extracted
    decision = next(d for d in engine.plan() if d.stage == "audio_features")
    assert decision.action == "skip"


def test_verify_catches_deleted_rows(extracted):
    """Like probe, the product is rows; file existence cannot check it."""
    cfg, conn, engine, _ = extracted
    conn.execute("DELETE FROM signal_series WHERE stream_id='fx' AND kind='game_rms'")
    try:
        ok, why = stages.get("audio_features").verify(engine.ctx)
        assert not ok
        assert "game_rms" in why
    finally:
        engine.execute(engine.plan())
    assert "game_rms" in signals.kinds(conn, "fx")


def test_reextracting_audio_makes_features_stale(extracted):
    cfg, conn, engine, _ = extracted
    forced = runner.Runner(cfg=cfg, conn=conn, stream_id="fx",
                           force={"audio_split"}, log=lambda *_: None)
    decisions = {d.stage: d for d in forced.plan()}
    assert decisions["audio_features"].action == "run"
    assert "audio_split" in decisions["audio_features"].reason
    # ...but the proxy is not downstream of audio_split and must be left alone.
    assert decisions["proxy"].action == "skip"


def test_changing_the_frame_length_reruns_extraction(extracted):
    cfg, conn, engine, _ = extracted
    retuned = config.load(overrides=[
        f"paths.data_root={cfg.data_root.as_posix()}",
        "extract.rms.frame_s=0.4",
    ])
    engine2 = runner.Runner(cfg=retuned, conn=conn, stream_id="fx", log=lambda *_: None)
    decisions = {d.stage: d for d in engine2.plan()}
    assert decisions["audio_features"].action == "run"
    assert decisions["audio_features"].reason == "parameters changed"
    assert decisions["audio_split"].action == "skip"


def test_signals_command_reports_the_series(extracted, capsys):
    from clipforge import cli

    assert cli.main(
        ["signals", "fx", "--set", f"paths.data_root={extracted[0].data_root.as_posix()}"]
    ) == 0
    out = capsys.readouterr().out
    assert "mic_rms" in out and "game_rms" in out


def test_signals_command_survives_a_signal_with_no_observations(extracted, capsys):
    """A pitch track over a party.wav with nobody in Discord is legitimately
    empty of observations. The command used to print the header row and then
    raise KeyError on the percentiles it had no samples for."""
    from clipforge import cli

    cfg, conn, _engine, _ = extracted
    signals.store(conn, "fx", signals.Series(
        kind="party_f0", values=np.full(50, np.nan), sample_rate_hz=10.0,
        t0=0.05, params={"unit": "Hz"},
    ))
    conn.commit()

    assert cli.main(
        ["signals", "fx", "--set", f"paths.data_root={cfg.data_root.as_posix()}"]
    ) == 0
    out = capsys.readouterr().out
    assert "no observations" in out


def test_signals_command_labels_pitch_in_hertz_not_decibels(extracted, capsys):
    """Every Phase 1 signal was dBFS, so the unit was a literal in the format
    string. A pitch printed as "166.2 dB" is a diagnostic stating something
    false."""
    from clipforge import cli

    cfg, conn, _engine, _ = extracted
    signals.store(conn, "fx", signals.Series(
        kind="mic_f0", values=np.full(50, 180.0), sample_rate_hz=10.0,
        t0=0.05, params={"unit": "Hz"},
    ))
    conn.commit()

    assert cli.main([
        "signals", "fx", "--kind", "mic_f0",
        "--set", f"paths.data_root={cfg.data_root.as_posix()}",
    ]) == 0
    line = next(ln for ln in capsys.readouterr().out.splitlines() if "mic_f0" in ln)
    assert line.rstrip().endswith("Hz")


def test_signals_command_reads_a_timestamp(extracted, manifest, capsys):
    """Manual spot-checking against the manifest is the point of --at."""
    from clipforge import cli

    speech = next(r for r in manifest["regions"] if r["label"].startswith("speech_peak"))
    midpoint = (speech["t_start"] + speech["t_end"]) / 2
    assert cli.main([
        "signals", "fx", "--at", str(midpoint), "--kind", "mic_rms",
        "--set", f"paths.data_root={extracted[0].data_root.as_posix()}",
    ]) == 0
    out = capsys.readouterr().out
    reported = float(out.split("mic_rms")[1].split("dB")[0].strip())
    assert reported == pytest.approx(speech["measured_dbfs"]["mic"], abs=1.0)
