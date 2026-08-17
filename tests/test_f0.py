"""§5.4.1's mic_f0 / party_f0.

Three kinds of assertion here, and only the first is about pyin:

* **Arithmetic that must hold** — the analysis hop divides the storage period,
  unvoiced is NaN, chunking cannot change a value.
* **Ground truth by construction** — a synthesised tone at a known frequency, and
  the speech fixture's authored utterance windows read from its manifest. Nothing
  below writes down a pitch that the fixture did not author.
* **The reason the design is shaped the way it is** — pyin REFUSES §5.4's own
  100 ms hop. That is asserted rather than described, so a librosa release that
  changes it is a failing test rather than a silent change of meaning.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import soundfile as sf

from clipforge import config, db, signals
from clipforge.extract import features
from clipforge.ingest import register
from clipforge.pipeline import runner
from clipforge.pipeline.context import StageContext


@pytest.fixture(scope="module")
def cfg():
    return config.load()


@pytest.fixture(scope="module")
def settings(cfg):
    return features.F0Settings.from_config(cfg)


def write_wav(path, values, rate=16000):
    sf.write(path, np.asarray(values, dtype=np.float32), rate, subtype="FLOAT")
    return path


def harmonic_tone(f0_hz: float, seconds: float, rate: int = 16000,
                  harmonics: int = 8) -> np.ndarray:
    """A pitched sound whose fundamental is known because we built it.

    Harmonic-rich rather than a sine: YIN measures periodicity, and a pure sine
    is the one waveform where an octave error costs nothing.
    """
    t = np.arange(int(seconds * rate)) / rate
    wave = sum(np.sin(2 * np.pi * f0_hz * k * t) / k for k in range(1, harmonics + 1))
    return (0.4 * wave / np.max(np.abs(wave))).astype(np.float32)


# --------------------------------------------------------------------------
# the constraint that shapes the whole design
# --------------------------------------------------------------------------


def test_pyin_refuses_the_signal_hop(cfg, tmp_path):
    """§5.4 puts every continuous signal at 10 Hz. pyin cannot be run there.

    Its HMM transition window is sized from the hop, and at 100 ms it is wider
    than the entire 65-400 Hz pitch grid. This is why pitch is tracked at
    `analysis_hop_s` and aggregated afterwards, and it is librosa raising rather
    than a judgement call.
    """
    librosa = pytest.importorskip("librosa")
    rate = int(cfg.get("ingest.audio.sample_rate_hz"))
    hop = int(round(rate / float(cfg.get("extract.signal_hz"))))

    with pytest.raises(Exception) as raised:
        librosa.pyin(
            harmonic_tone(150.0, 2.0, rate), sr=rate,
            fmin=float(cfg.get("extract.f0.fmin_hz")),
            fmax=float(cfg.get("extract.f0.fmax_hz")),
            frame_length=2048, hop_length=hop, center=False,
        )
    assert "size" in str(raised.value).lower()


def test_analysis_hop_must_divide_the_storage_period(settings, cfg):
    signal_hz = float(cfg.get("extract.signal_hz"))
    count = settings.frames_per_bucket(signal_hz)
    assert count >= 1
    # The shipped pair must be exact, not merely close.
    assert math.isclose(count * settings.analysis_hop_s, 1.0 / signal_hz, rel_tol=1e-12)


def test_a_hop_that_does_not_divide_is_refused(settings, cfg):
    """Otherwise each stored sample aggregates a different number of frames and
    the series' declared sample rate stops describing its own contents."""
    bad = dataclasses.replace(settings, analysis_hop_s=0.03)
    with pytest.raises(features.FeatureError, match="does not divide"):
        bad.frames_per_bucket(float(cfg.get("extract.signal_hz")))


# --------------------------------------------------------------------------
# ground truth by construction
# --------------------------------------------------------------------------


@pytest.mark.parametrize("f0_hz", [90.0, 150.0, 260.0])
def test_a_tone_recovers_the_frequency_it_was_built_with(tmp_path, settings, f0_hz):
    path = write_wav(tmp_path / f"tone{f0_hz:.0f}.wav", harmonic_tone(f0_hz, 4.0))
    series = features.f0_series(path, "mic_f0", signal_hz=10.0, settings=settings)

    voiced = series.values[np.isfinite(series.values)]
    assert voiced.size > 0.9 * len(series), "a continuous tone is voiced throughout"
    # 3% of the authored frequency. pyin quantises to `resolution` semitones and
    # a semitone is ~6%, so this is inside one bin either way.
    assert float(np.median(voiced)) == pytest.approx(f0_hz, rel=0.03)


def test_content_below_fmin_is_pinned_to_fmin_not_reported_absent(
    tmp_path, settings, cfg
):
    """MEASURED, and the opposite of what one would assume.

    A tone an octave below `fmin_hz` does not come back as "no pitch". Every
    frame comes back as exactly `fmin_hz`, confidently voiced — the search grid
    has no way to express "lower than this", so it pins. Rumble under a mic
    therefore produces a rock-steady pitch track rather than an empty one, which
    reads downstream as the *calmest* stretch of the stream rather than the
    noisiest. `rail_fraction` exists to count it and the stage warns.
    """
    fmin = float(cfg.get("extract.f0.fmin_hz"))
    path = write_wav(tmp_path / "rumble.wav", harmonic_tone(0.5 * fmin, 3.0))
    series = features.f0_series(path, "mic_f0", signal_hz=10.0, settings=settings)

    voiced = series.values[np.isfinite(series.values)]
    assert voiced.size > 0
    assert float(np.median(voiced)) == pytest.approx(fmin, rel=0.02)

    at_min, at_max = features.rail_fraction(series, settings)
    assert at_min > features.RAIL_WARN_FRACTION, "this is what the warning is for"
    assert at_max == 0.0


def test_rail_fraction_is_zero_for_pitch_in_the_middle(tmp_path, settings, cfg):
    middle = math.sqrt(float(cfg.get("extract.f0.fmin_hz"))
                       * float(cfg.get("extract.f0.fmax_hz")))
    path = write_wav(tmp_path / "middle.wav", harmonic_tone(middle, 3.0))
    series = features.f0_series(path, "mic_f0", signal_hz=10.0, settings=settings)

    at_min, at_max = features.rail_fraction(series, settings)
    assert at_min == 0.0
    assert at_max == 0.0


def test_silence_is_nan_and_never_zero(tmp_path, settings):
    """Zero is a measurement — a pitch of 0 Hz — and it would drag every mean,
    z-score and rolling variance downstream toward it. Absence is NaN."""
    path = write_wav(tmp_path / "silence.wav", np.zeros(16000 * 3, dtype=np.float32))
    series = features.f0_series(path, "mic_f0", signal_hz=10.0, settings=settings)

    assert len(series) > 0
    assert not np.any(series.values == 0.0)
    assert features.voiced_fraction(series) == 0.0


# --------------------------------------------------------------------------
# chunking
# --------------------------------------------------------------------------


def test_chunking_is_exact_on_real_audio(tmp_path, speech_fixture_dir, speech_manifest,
                                         settings):
    """pyin's observation matrix is 3.6 GB over a 4-hour stream, so chunking is
    forced. MEASURED on the speech fixture at three chunk lengths: every bucket
    voiced under both settings agreed to 0.0000 Hz — bit-identical, not merely
    close. Chunking changes only which frames the Viterbi had as context, and
    `overlap_s` of discarded real audio is what keeps that from mattering.
    """
    cfg, conn, ctx = stage_ctx(tmp_path, speech_fixture_dir, speech_manifest)
    try:
        path = ctx.paths.audio("mic")
        whole = features.f0_series(path, "mic_f0", signal_hz=10.0,
                                   settings=settings).values
        chunked = features.f0_series(
            path, "mic_f0", signal_hz=10.0,
            settings=dataclasses.replace(settings, chunk_s=17.0),
        ).values
    finally:
        conn.close()

    assert whole.size == chunked.size
    both = np.isfinite(whole) & np.isfinite(chunked)
    assert both.sum() > 100
    np.testing.assert_allclose(whole[both], chunked[both], rtol=1e-9)


def test_a_chunk_boundary_can_move_a_sample_by_at_most_one_bin(tmp_path, settings):
    """The honest limit of the guarantee above.

    On a synthetic step — a tone, a second of silence, a different tone — a chunk
    boundary landing in the transition CAN decode one bucket differently, because
    the two passes genuinely saw different context. Measured: 1 sample in 63, by
    3.5%, which is inside pyin's own `resolution` of a semitone. So the promise
    is "within one pitch bin", and it is tested at the boundary rather than
    asserted away.
    """
    signal = np.concatenate([
        harmonic_tone(120.0, 3.0),
        np.zeros(16000, dtype=np.float32),
        harmonic_tone(240.0, 3.0),
    ])
    path = write_wav(tmp_path / "two_tones.wav", signal)

    whole = features.f0_series(path, "mic_f0", signal_hz=10.0, settings=settings).values
    chunked = features.f0_series(
        path, "mic_f0", signal_hz=10.0,
        settings=dataclasses.replace(settings, chunk_s=1.7),
    ).values

    both = np.isfinite(whole) & np.isfinite(chunked)
    assert both.sum() > 0
    semitones = np.abs(12 * np.log2(chunked[both] / whole[both]))
    assert float(np.max(semitones)) <= 1.0
    # And the overwhelming majority are exact, not merely inside the bin.
    assert float(np.mean(semitones < 1e-9)) > 0.95


def test_chunking_does_not_lose_voiced_samples(tmp_path, settings):
    path = write_wav(tmp_path / "tone.wav", harmonic_tone(180.0, 6.0))
    whole = features.f0_series(path, "mic_f0", signal_hz=10.0, settings=settings)
    chunked = features.f0_series(
        path, "mic_f0", signal_hz=10.0,
        settings=dataclasses.replace(settings, chunk_s=1.3),
    )
    assert features.voiced_fraction(chunked) >= features.voiced_fraction(whole) - 0.02


# --------------------------------------------------------------------------
# the series contract
# --------------------------------------------------------------------------


def test_a_bucket_ignores_its_unvoiced_frames_rather_than_being_poisoned():
    """C2, and a bug that shipped for one commit.

    A 100 ms bucket holds five analysis frames and speech is voiced in bursts,
    so a bucket with one unvoiced frame is the normal case, not the exception.
    Its median must be the median of what WAS observed.

    The regression this pins: `np.median` silently IGNORES a masked array's
    mask — it warns "'partition' will ignore the 'mask'" and then takes the
    median of the raw data, NaN included, so every mixed bucket came back
    unvoiced. The suite passed anyway because the hit-rate assertions were
    loose enough to accept the lower number.
    """
    collected = np.array([
        [100.0, 200.0, np.nan, np.nan, np.nan],   # two observations
        [np.nan] * 5,                             # none at all
        [150.0, 150.0, 150.0, 150.0, 150.0],      # all five
    ])
    out = features._bucket_median(collected)

    assert out[0] == pytest.approx(150.0), "the median of the observed pair"
    assert np.isnan(out[1]), "no observations is an absence, not a value"
    assert out[2] == pytest.approx(150.0)


def test_bucket_median_raises_no_warnings():
    """`warnings.catch_warnings` mutates a PROCESS-WIDE filter and is documented
    as not thread-safe, and `review/jobs` runs stages on background threads. So
    the all-absent case must not raise a warning that has to be silenced."""
    import warnings as warnings_module

    with warnings_module.catch_warnings():
        warnings_module.simplefilter("error")
        features._bucket_median(np.full((3, 5), np.nan))
        features._downsample_mean(np.full(50, np.nan), 100.0, 10.0)


def test_downsample_keeps_partial_buckets_and_drops_empty_ones():
    values = np.array([1.0, np.nan, 3.0, np.nan] * 5)
    out = features._downsample_mean(values, 100.0, 50.0)   # 2 samples per bucket
    assert out[0] == pytest.approx(1.0), "the mean of what was observed"

    assert np.isnan(features._downsample_mean(np.full(10, np.nan), 100.0, 50.0)).all()


def test_series_records_both_rates(tmp_path, settings, cfg):
    """A reader has to be able to tell what produced the number without knowing
    which extraction module wrote it."""
    path = write_wav(tmp_path / "tone.wav", harmonic_tone(150.0, 2.0))
    series = features.f0_series(path, "mic_f0", signal_hz=10.0, settings=settings,
                                role="mic")

    assert series.sample_rate_hz == 10.0
    assert series.t0 == pytest.approx(0.05)          # the bucket's centre
    assert series.params["unit"] == "Hz"
    assert series.params["unvoiced"] == "nan"
    assert series.params["analysis_rate_hz"] == settings.analysis_rate_hz
    assert series.params["analysis_hop_s"] == settings.analysis_hop_s
    assert series.params["frames_per_sample"] == settings.frames_per_bucket(10.0)
    assert series.params["aggregate"] == "median"


def test_nan_survives_the_blob_round_trip(tmp_path, settings):
    """A6 stores signals as float32 BLOBs. NaN has to come back as NaN, or the
    absence of pitch turns into whatever float32 makes of it."""
    cfg = config.load(overrides=[f"paths.data_root={(tmp_path / 'data').as_posix()}"])
    conn = db.open_db(cfg.db_path)
    conn.execute(
        "INSERT INTO streams (id, date, title, master_path, marker_time_base) "
        "VALUES ('s', '2026-01-01', 't', 'm.mkv', 'vod')"
    )
    path = write_wav(tmp_path / "mixed.wav", np.concatenate([
        harmonic_tone(150.0, 2.0), np.zeros(16000 * 2, dtype=np.float32),
    ]))
    original = features.f0_series(path, "mic_f0", signal_hz=10.0, settings=settings)
    signals.store(conn, "s", original)
    loaded = signals.load(conn, "s", "mic_f0")
    conn.close()

    assert np.isnan(loaded.values).any(), "the silent half must still be absent"
    np.testing.assert_array_equal(
        np.isnan(original.values), np.isnan(loaded.values)
    )


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


def stage_ctx(tmp_path, fixture_dir, manifest, overrides=()):
    cfg = config.load(overrides=[
        f"paths.data_root={(tmp_path / 'data').as_posix()}", *overrides,
    ])
    db.open_db(cfg.db_path).close()
    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    result = register.register_stream(
        cfg, conn, register.RegisterSpec(master=fixture_dir / manifest["video"]),
    )
    engine = runner.Runner(cfg=cfg, conn=conn, stream_id=result.stream_id,
                           log=lambda *_: None)
    engine.execute(engine.plan(["register_stream", "probe", "audio_split"]))
    return cfg, conn, StageContext(cfg=cfg, conn=conn, stream_id=result.stream_id,
                                   log=lambda *_: None)


def test_roles_are_configurable_and_game_is_never_one(tmp_path, fixture_dir, manifest):
    """§5.4.1 declares no game_f0, and pitch is the most expensive thing in
    extraction — so which roles get one has to be a real knob."""
    cfg, conn, ctx = stage_ctx(tmp_path, fixture_dir, manifest)
    try:
        assert set(features._f0_roles(ctx)) <= {"mic", "party"}
        assert "game" not in features._f0_roles(ctx)
    finally:
        conn.close()

    cfg, conn, ctx = stage_ctx(tmp_path / "b", fixture_dir, manifest,
                               ["extract.f0.roles=[mic]"])
    try:
        assert set(features._f0_roles(ctx)) == {"mic"}
    finally:
        conn.close()

    cfg, conn, ctx = stage_ctx(tmp_path / "c", fixture_dir, manifest,
                               ["extract.f0.enabled=false"])
    try:
        assert features._f0_roles(ctx) == {}
    finally:
        conn.close()


def test_disabled_pitch_still_produces_rms(tmp_path, fixture_dir, manifest):
    """`audio_features` must not become all-or-nothing: RMS is Phase 1 and pitch
    is Phase 3, and a machine that cannot afford one still needs the other."""
    cfg, conn, ctx = stage_ctx(tmp_path, fixture_dir, manifest,
                               ["extract.f0.enabled=false"])
    try:
        features.run(ctx)
        present = set(signals.kinds(conn, ctx.stream_id))
        assert "mic_rms" in present
        assert "mic_f0" not in present
        ok, why = features.verify(ctx)
        assert ok, why
    finally:
        conn.close()


# --------------------------------------------------------------------------
# the fixtures, which is where the numbers come from
# --------------------------------------------------------------------------


def test_band_limited_noise_has_almost_no_pitch(tmp_path, fixture_dir, manifest,
                                                settings):
    """The noise fixture is deliberately unpitched. A tracker that finds pitch in
    it is inventing one, and everything downstream would believe it."""
    cfg, conn, ctx = stage_ctx(tmp_path, fixture_dir, manifest)
    try:
        series = features.f0_series(
            ctx.paths.audio("mic"), "mic_f0", signal_hz=10.0, settings=settings,
        )
        assert features.voiced_fraction(series) < 0.20
    finally:
        conn.close()


@pytest.fixture(scope="module")
def speech_pitch(tmp_path_factory, speech_fixture_dir, speech_manifest, settings):
    """Both tracks of the speech fixture, tracked once and shared."""
    tmp_path = tmp_path_factory.mktemp("speech_pitch")
    cfg, conn, ctx = stage_ctx(tmp_path, speech_fixture_dir, speech_manifest)
    try:
        out = {
            role: features.f0_series(
                ctx.paths.audio(role), f"{role}_f0", signal_hz=10.0,
                settings=settings, role=role,
            )
            for role in ("mic", "party")
        }
    finally:
        conn.close()
    return out


def authored_mask(series, manifest, track) -> np.ndarray:
    times = series.times()
    inside = np.zeros(times.size, dtype=bool)
    for utterance in manifest["utterances"]:
        if utterance["track"] == track:
            inside |= ((times >= utterance["t_start"])
                       & (times <= utterance["t_end"]))
    return inside


@pytest.mark.parametrize("track", ["mic", "party"])
def test_pitch_lands_where_the_fixture_authored_speech(speech_pitch, speech_manifest,
                                                       track):
    """The assertion is the CONTRAST, not an absolute rate.

    Roughly a fifth of speech frames are genuinely unvoiced — every s, f, t and
    k — so demanding 100% would be demanding a wrong answer. What must hold is
    that pitch appears where a person was talking and not where the fixture
    authored silence.
    """
    series = speech_pitch[track]
    voiced = np.isfinite(series.values)
    speech = authored_mask(series, speech_manifest, track)

    during_speech = float(np.mean(voiced[speech]))
    during_silence = float(np.mean(voiced[~speech]))

    assert during_speech > 0.5
    assert during_silence < 0.1
    assert during_speech > 5 * max(during_silence, 1e-3)


def test_the_two_speakers_are_told_apart_by_pitch(speech_pitch, speech_manifest, cfg):
    """The fixture puts a different voice on each track (`speaker_tracks`), so
    their medians must differ — and both must be inside the configured range,
    which is the check that catches an octave error."""
    fmin = float(cfg.get("extract.f0.fmin_hz"))
    fmax = float(cfg.get("extract.f0.fmax_hz"))

    medians = {}
    for track, series in speech_pitch.items():
        speech = authored_mask(series, speech_manifest, track)
        voiced = series.values[speech & np.isfinite(series.values)]
        assert voiced.size > 0
        medians[track] = float(np.median(voiced))
        assert fmin <= medians[track] <= fmax

    assert speech_manifest["speaker_tracks"] == {"A": "mic", "B": "party"}
    # A semitone apart at the very least, or "which voice is this" is not a
    # question pitch can answer on this material.
    assert abs(12 * math.log2(medians["mic"] / medians["party"])) > 1.0


def test_the_authored_silence_is_silent_to_the_tracker(speech_pitch, speech_manifest):
    """A4's twenty seconds of nothing, which Phase 2 used to prove VAD was on.
    The same window proves the pitch tracker is not hallucinating either."""
    longest = max(speech_manifest["silence_windows"], key=lambda w: w[1] - w[0])
    for series in speech_pitch.values():
        times = series.times()
        inside = (times >= longest[0]) & (times <= longest[1])
        assert float(np.mean(np.isfinite(series.values[inside]))) < 0.1


def test_the_stage_writes_pitch_for_every_configured_role(
    tmp_path, speech_fixture_dir, speech_manifest
):
    cfg, conn, ctx = stage_ctx(tmp_path, speech_fixture_dir, speech_manifest)
    try:
        features.run(ctx)
        present = set(signals.kinds(conn, ctx.stream_id))
        assert {"mic_f0", "party_f0"} <= present

        ok, why = features.verify(ctx)
        assert ok, why

        for kind in ("mic_f0", "party_f0"):
            series = signals.load(conn, ctx.stream_id, kind)
            assert series.sample_rate_hz == float(cfg.get("extract.signal_hz"))
            assert len(series) > 0
            assert np.isnan(series.values).any(), "silence has to survive as absence"
    finally:
        conn.close()
