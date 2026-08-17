"""§5.5's laughter detection.

Nothing here writes down a frequency, a level or a score. Which regions count as
"in band" is decided by comparing the fixture manifest's authored `mod_hz`
against `extract.laughter.band_hz` from config; how early an event may fire comes
from `score_window_s`; what counts as detected comes from the configured
threshold. Move the band in config and every expectation below moves with it.

**What these tests can and cannot show.** The fixture authors sinusoidal
amplitude modulation, so this validates the MECHANISM — that the detector
responds to envelope periodicity inside the band and not outside it, and not to
loudness. It cannot show that real laughter looks like this. §5.5's claim that
the mechanism finds laughter needs a hand-labelled clip, and that is recorded in
spec/GUESSES.md rather than implied by a green test.
"""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pytest
import soundfile as sf

from clipforge import config, db, signals
from clipforge.extract import features
from clipforge.ingest import register
from clipforge.pipeline import runner
from clipforge.pipeline.context import StageContext
from clipforge.score import derived, grid


@pytest.fixture(scope="module")
def cfg():
    return config.load()


@pytest.fixture(scope="module")
def settings(cfg):
    return features.LaughterSettings.from_config(cfg)


def band_of(cfg) -> tuple[float, float]:
    band = cfg.get("extract.laughter.band_hz")
    return float(band[0]), float(band[1])


def in_band(cfg, mod_hz: float, depth: float) -> bool:
    """The fixture's own definition of a positive, derived from config."""
    low, high = band_of(cfg)
    return depth > 0.0 and low <= mod_hz <= high


def write_wav(path, values, rate=16000):
    sf.write(path, np.asarray(values, dtype=np.float32), rate, subtype="FLOAT")
    return path


def modulated_noise(seconds, mod_hz, depth, *, rate=16000, level_db=-18.0, seed=11):
    """Band-limited noise with a sine on its envelope — the fixture's recipe."""
    n = int(seconds * rate)
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    spectrum[np.fft.rfftfreq(n, 1 / rate) > 6000] = 0.0
    signal = np.fft.irfft(spectrum, n)
    if mod_hz > 0 and depth > 0:
        t = np.arange(n) / rate
        signal = signal * (1.0 + depth * np.sin(2 * np.pi * mod_hz * t))
    rms = float(np.sqrt(np.mean(signal**2)))
    return signal * (10 ** (level_db / 20.0)) / (rms or 1.0)


# --------------------------------------------------------------------------
# the Nyquist constraint that shapes the whole design
# --------------------------------------------------------------------------


def test_the_stored_signal_grid_cannot_carry_this_band(cfg, settings):
    """THE finding. §5.4 stores every continuous signal at 10 Hz, whose Nyquist
    is 5 Hz — below this band's own top edge. Reusing `mic_rms` would produce a
    number that looks like a laughter score and is an artefact."""
    signal_hz = float(cfg.get("extract.signal_hz"))
    assert band_of(cfg)[1] >= signal_hz / 2, (
        "if the band ever fits inside the stored grid, this whole design is "
        "unnecessary and should be simplified rather than left as folklore"
    )
    with pytest.raises(features.FeatureError, match="Nyquist"):
        settings.check_nyquist(signal_hz, "the stored signal grid")


def test_the_filter_cannot_even_be_designed_at_the_stored_rate(cfg):
    """Not merely inaccurate: scipy refuses. Asserted so that a future scipy
    which silently accepted it would fail here rather than ship an artefact."""
    from scipy.signal import butter

    signal_hz = float(cfg.get("extract.signal_hz"))
    low, high = band_of(cfg)
    with pytest.raises(ValueError):
        butter(4, [low / (signal_hz / 2), high / (signal_hz / 2)], btype="band")


def test_the_configured_envelope_rate_clears_the_band(cfg, settings):
    settings.check_nyquist(settings.envelope_hz, "the laughter envelope")
    assert settings.envelope_hz > 2 * band_of(cfg)[1]


def test_the_envelope_rate_must_divide_into_the_storage_grid(cfg, settings, tmp_path):
    """Otherwise the buckets aggregate unevenly and the stored series' declared
    rate stops describing its own contents."""
    signal_hz = float(cfg.get("extract.signal_hz"))
    ratio = settings.envelope_hz / signal_hz
    assert math.isclose(ratio, round(ratio))

    path = write_wav(tmp_path / "x.wav", modulated_noise(6.0, 5.0, 0.8))
    with pytest.raises(features.FeatureError, match="whole multiple"):
        features.laughter_series(
            path, "mic_laughter", signal_hz=signal_hz * 1.5, settings=settings,
        )


# --------------------------------------------------------------------------
# the mechanism, on synthetic input
# --------------------------------------------------------------------------


def test_modulation_in_the_band_outscores_modulation_outside_it(tmp_path, cfg,
                                                                settings):
    """At one level, so nothing here can be explained by loudness."""
    low, high = band_of(cfg)
    centre = math.sqrt(low * high)

    scores = {}
    for label, mod_hz in (("in", centre), ("below", low / 3), ("above", high * 2)):
        path = write_wav(tmp_path / f"{label}.wav", modulated_noise(8.0, mod_hz, 0.8))
        series = features.laughter_series(
            path, "mic_laughter", signal_hz=10.0, settings=settings)
        values = series.values[np.isfinite(series.values)]
        scores[label] = float(np.median(values))

    assert scores["in"] > scores["below"]
    assert scores["in"] > scores["above"]


def test_an_unmodulated_carrier_does_not_reach_the_threshold(tmp_path, cfg,
                                                             settings):
    """The share is scale-free, so noise wandering into the band by chance is
    the failure mode this has to survive."""
    path = write_wav(tmp_path / "flat.wav", modulated_noise(8.0, 0.0, 0.0))
    series = features.laughter_series(
        path, "mic_laughter", signal_hz=10.0, settings=settings)
    values = series.values[np.isfinite(series.values)]

    assert float(np.max(values)) < float(cfg.get("score.derived.laughter.threshold"))


def test_the_depth_gate_is_what_rejects_quiet_noise(tmp_path, settings):
    """A share alone cannot tell a real laugh from a noise floor that happens to
    fluctuate in band — both are ~1.0. Depth is the half that can."""
    envelope, _framing, _n = features.stream_frame_rms(
        write_wav(tmp_path / "flat.wav", modulated_noise(8.0, 0.0, 0.0)),
        signal_hz=settings.envelope_hz, frame_s=settings.frame_s,
    )
    _share, depth = features.band_share(envelope, settings)
    assert float(np.median(depth[np.isfinite(depth)])) < settings.min_depth

    envelope, _framing, _n = features.stream_frame_rms(
        write_wav(tmp_path / "mod.wav", modulated_noise(8.0, 5.5, 0.8)),
        signal_hz=settings.envelope_hz, frame_s=settings.frame_s,
    )
    _share, depth = features.band_share(envelope, settings)
    assert float(np.median(depth[np.isfinite(depth)])) > settings.min_depth


def test_the_score_stays_inside_the_range_it_claims(tmp_path, settings):
    """A "share" above 1 is not a share.

    MEASURED on the noise fixture, whose regions start and stop abruptly: the
    band-passed component rings at a step and can locally exceed the detrended
    total it is a fraction of — 3.8 in that case. Nothing downstream broke,
    because the duration gate discards a transient, but a signal documented as
    0..1 has to be 0..1 or the threshold beside it stops meaning what it says.
    """
    stepped = np.concatenate([
        np.zeros(16000, dtype=np.float32),
        modulated_noise(2.0, 5.5, 0.9),
        np.zeros(16000, dtype=np.float32),
        modulated_noise(2.0, 0.0, 0.0),
    ])
    series = features.laughter_series(
        write_wav(tmp_path / "steps.wav", stepped), "mic_laughter",
        signal_hz=10.0, settings=settings,
    )
    values = series.values[np.isfinite(series.values)]

    assert values.size
    assert float(np.max(values)) <= 1.0
    assert float(np.min(values)) >= 0.0


def test_silence_is_absent_not_zero(tmp_path, settings):
    """Commit 31's convention. Zero would be a measurement — "no periodicity
    here" — where the truth is that there was nothing to measure."""
    path = write_wav(tmp_path / "silent.wav", np.zeros(16000 * 6, dtype=np.float32))
    series = features.laughter_series(
        path, "mic_laughter", signal_hz=10.0, settings=settings)

    assert len(series) > 0
    assert np.isnan(series.values).all()


def test_a_clip_too_short_to_hold_a_cycle_is_absent(tmp_path, settings):
    path = write_wav(tmp_path / "blip.wav", modulated_noise(0.2, 5.5, 0.8))
    series = features.laughter_series(
        path, "mic_laughter", signal_hz=10.0, settings=settings)
    assert len(series) == 0 or np.isnan(series.values).all()


def test_the_series_records_how_it_was_made(tmp_path, settings):
    path = write_wav(tmp_path / "x.wav", modulated_noise(8.0, 5.5, 0.8))
    series = features.laughter_series(
        path, "mic_laughter", signal_hz=10.0, settings=settings, role="mic")

    assert series.params["band_hz"] == list(settings.band_hz)
    assert series.params["envelope_hz"] == settings.envelope_hz
    assert series.params["unit"] == ""          # a ratio has none
    assert series.params["role"] == "mic"


# --------------------------------------------------------------------------
# against the fixture's authored modulation
# --------------------------------------------------------------------------


def stage_ctx(tmp_path, fixture_dir, manifest):
    cfg = config.load(overrides=[f"paths.data_root={(tmp_path / 'data').as_posix()}"])
    db.open_db(cfg.db_path).close()
    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    result = register.register_stream(
        cfg, conn, register.RegisterSpec(master=fixture_dir / manifest["video"]))
    engine = runner.Runner(cfg=cfg, conn=conn, stream_id=result.stream_id,
                           log=lambda *_: None)
    engine.execute(engine.plan(["register_stream", "probe", "audio_split"]))
    return cfg, conn, StageContext(cfg=cfg, conn=conn, stream_id=result.stream_id,
                                   log=lambda *_: None)


@pytest.fixture(scope="module")
def laughter_scores(tmp_path_factory, laughter_fixture_dir, laughter_manifest,
                    settings):
    """Both tracks of the laughter fixture, scored once and shared."""
    cfg, conn, ctx = stage_ctx(
        tmp_path_factory.mktemp("laughter"), laughter_fixture_dir, laughter_manifest)
    try:
        out = {
            role: features.laughter_series(
                ctx.paths.audio(role), f"{role}_laughter", signal_hz=10.0,
                settings=settings, role=role)
            for role in ("mic", "party")
        }
    finally:
        conn.close()
    return cfg, out


def region_values(series, region, settle_s):
    """Samples strictly inside a region, past the smoothing window's settling."""
    times = series.times()
    inside = ((times >= region["t_start"] + settle_s)
              & (times <= region["t_end"] - settle_s))
    values = series.values[inside]
    return values[np.isfinite(values)]


def test_the_fixture_authors_one_level_so_loudness_cannot_explain_anything(
    laughter_manifest
):
    """The premise every ranking below rests on, asserted rather than assumed."""
    levels = {m["dbfs"] for m in laughter_manifest["modulation"]}
    assert len(levels) == 1

    measured = [
        r["measured_dbfs"][r["source_track"]] for r in laughter_manifest["regions"]
    ]
    assert max(measured) - min(measured) < 0.5, (
        "modulation must not change a region's measured level, or the detector "
        "could be responding to loudness"
    )


def test_every_in_band_region_outscores_every_other_region(laughter_scores,
                                                           laughter_manifest,
                                                           settings):
    """A ranking, so no score threshold is written down here."""
    cfg, scores = laughter_scores
    settle = settings.score_window_s

    positives, negatives = [], []
    for region in laughter_manifest["modulation"]:
        values = region_values(scores[region["track"]], region, settle)
        assert values.size, f"no samples inside {region['label']}"
        target = positives if in_band(cfg, region["mod_hz"], region["mod_depth"]) \
            else negatives
        target.append((region["label"], float(np.median(values))))

    assert positives and negatives
    worst_positive = min(score for _label, score in positives)
    best_negative = max(score for _label, score in negatives)
    assert worst_positive > best_negative, (
        f"in-band {sorted(positives, key=lambda x: x[1])[:2]} did not clear "
        f"out-of-band {sorted(negatives, key=lambda x: -x[1])[:2]}"
    )


def test_the_configured_threshold_sits_inside_that_gap(laughter_scores,
                                                       laughter_manifest, settings):
    """The threshold is only meaningful if it separates the two groups — which
    is a property of the number in config, so it belongs in a test."""
    cfg, scores = laughter_scores
    threshold = float(cfg.get("score.derived.laughter.threshold"))
    settle = settings.score_window_s

    for region in laughter_manifest["modulation"]:
        values = region_values(scores[region["track"]], region, settle)
        positive = in_band(cfg, region["mod_hz"], region["mod_depth"])
        if positive:
            assert float(np.min(values)) > threshold, region["label"]
        else:
            assert float(np.max(values)) < threshold, region["label"]


def test_deeper_modulation_scores_at_least_as_high_as_shallower(laughter_scores,
                                                               laughter_manifest,
                                                               settings):
    """Pairs at the same rate and different depths, found in the manifest."""
    cfg, scores = laughter_scores
    settle = settings.score_window_s

    by_rate: dict[tuple[str, float], list[tuple[float, float]]] = {}
    for region in laughter_manifest["modulation"]:
        if not in_band(cfg, region["mod_hz"], region["mod_depth"]):
            continue
        values = region_values(scores[region["track"]], region, settle)
        key = (region["track"], region["mod_hz"])
        by_rate.setdefault(key, []).append(
            (region["mod_depth"], float(np.median(values)))
        )

    compared = 0
    for pairs in by_rate.values():
        pairs.sort()
        for (_shallow_d, shallow), (_deep_d, deep) in zip(pairs, pairs[1:]):
            assert deep >= shallow - 1e-6
            compared += 1
    assert compared, "the fixture should author two depths at one rate"


# --------------------------------------------------------------------------
# events (§5.4.2), thresholded at score time
# --------------------------------------------------------------------------


def laugh_events(cfg, scores, duration_s):
    timeline = grid.build(duration_s, float(cfg.get("score.score_grid_hz")))
    raw = {
        f"{role}_laughter": grid.resample(series, timeline)
        for role, series in scores.items()
    }
    _values, _units, events = derived.compute(raw, timeline, cfg)
    return events


def test_every_in_band_region_produces_an_event(laughter_scores, laughter_manifest,
                                                settings):
    """An event may fire up to half the score window EARLY: the window is
    centred, so the score at t reflects the audio around t, and the response to
    a region therefore begins before the region does. That tolerance comes from
    `score_window_s` rather than from a number typed here."""
    cfg, scores = laughter_scores
    events = laugh_events(cfg, scores, float(laughter_manifest["duration_s"]))
    lead = settings.score_window_s / 2.0
    kinds = {"mic": "laugh_operator", "party": "laugh_party"}

    for region in laughter_manifest["modulation"]:
        if not in_band(cfg, region["mod_hz"], region["mod_depth"]):
            continue
        fired = events.get(kinds[region["track"]], [])
        assert any(region["t_start"] - lead <= t <= region["t_end"] for t in fired), (
            f"{region['label']} authored {region['mod_hz']} Hz produced no event; "
            f"{kinds[region['track']]} fired at {fired}"
        )


def test_no_out_of_band_region_produces_an_event(laughter_scores, laughter_manifest,
                                                 settings):
    cfg, scores = laughter_scores
    events = laugh_events(cfg, scores, float(laughter_manifest["duration_s"]))
    lead = settings.score_window_s / 2.0
    kinds = {"mic": "laugh_operator", "party": "laugh_party"}

    for region in laughter_manifest["modulation"]:
        if in_band(cfg, region["mod_hz"], region["mod_depth"]):
            continue
        fired = events.get(kinds[region["track"]], [])
        assert not any(region["t_start"] - lead <= t <= region["t_end"]
                       for t in fired), f"{region['label']} fired at {fired}"


def test_one_event_per_region_not_one_per_sample(laughter_scores,
                                                 laughter_manifest):
    """`score.events.combine` is `sum`, so a six-second laugh emitting sixty
    kernels would outweigh every marker in the stream."""
    cfg, scores = laughter_scores
    events = laugh_events(cfg, scores, float(laughter_manifest["duration_s"]))

    for track, kind in (("mic", "laugh_operator"), ("party", "laugh_party")):
        authored = sum(
            1 for m in laughter_manifest["modulation"]
            if m["track"] == track and in_band(cfg, m["mod_hz"], m["mod_depth"])
        )
        assert len(events.get(kind, [])) <= authored


@pytest.fixture(scope="module")
def speech_laughter(tmp_path_factory, speech_fixture_dir, speech_manifest, settings):
    cfg, conn, ctx = stage_ctx(
        tmp_path_factory.mktemp("speech_laugh"), speech_fixture_dir, speech_manifest)
    try:
        out = {
            role: features.laughter_series(
                ctx.paths.audio(role), f"{role}_laughter", signal_hz=10.0,
                settings=settings, role=role)
            for role in ("mic", "party")
        }
    finally:
        conn.close()
    return cfg, out


def test_ordinary_speech_produces_no_laugh_events(speech_laughter, speech_manifest):
    """THE regression this signal needed.

    A syllable rate of 4-6 Hz is INSIDE §5.5's band, and the speech fixture
    authors no laughter at all. Earlier cuts produced ten laugh events on it —
    mostly utterance offsets, which look periodic for about one smoothing window.
    The share metric, the depth gate and `min_duration_s` each removed some of
    them; this asserts the result rather than the reasoning.
    """
    cfg, scores = speech_laughter
    events = laugh_events(cfg, scores, float(speech_manifest["duration_s"]))

    assert not events.get("laugh_operator"), events
    assert not events.get("laugh_party"), events


def test_speech_still_scores_below_the_threshold_on_the_whole(speech_laughter, cfg):
    """The continuous score does cross on a minority of speech samples — that is
    what `min_duration_s` exists for — but it must not be the typical value."""
    _cfg, scores = speech_laughter
    threshold = float(cfg.get("score.derived.laughter.threshold"))

    for series in scores.values():
        values = series.values[np.isfinite(series.values)]
        assert float(np.median(values)) < threshold


# --------------------------------------------------------------------------
# the stage, and the schema
# --------------------------------------------------------------------------


def test_the_stage_stores_a_score_per_configured_role(tmp_path, laughter_fixture_dir,
                                                      laughter_manifest):
    cfg, conn, ctx = stage_ctx(tmp_path, laughter_fixture_dir, laughter_manifest)
    try:
        features.run(ctx)
        present = set(signals.kinds(conn, ctx.stream_id))
        assert set(features.LAUGHTER_FOR_ROLE.values()) <= present

        ok, why = features.verify(ctx)
        assert ok, why

        for kind in features.LAUGHTER_FOR_ROLE.values():
            series = signals.load(conn, ctx.stream_id, kind)
            assert series.sample_rate_hz == float(cfg.get("extract.signal_hz"))
    finally:
        conn.close()


def test_game_never_gets_a_laughter_track():
    """§5.5 names the mic and the party track. A laugh track is not what an
    explosion is."""
    assert "game" not in features.LAUGHTER_FOR_ROLE


def test_roles_and_enabled_are_honoured(tmp_path, laughter_fixture_dir,
                                        laughter_manifest):
    cfg, conn, ctx = stage_ctx(tmp_path, laughter_fixture_dir, laughter_manifest)
    try:
        assert set(features._laughter_roles(ctx)) == {"mic", "party"}
    finally:
        conn.close()

    for overrides, expected in (
        (["extract.laughter.roles=[party]"], {"party"}),
        (["extract.laughter.enabled=false"], set()),
    ):
        cfg2 = config.load(overrides=[
            f"paths.data_root={(tmp_path / 'data').as_posix()}", *overrides])
        ctx2 = StageContext(cfg=cfg2, conn=db.open_db(cfg2.db_path, migrate_to_latest=False),
                            stream_id=ctx.stream_id, log=lambda *_: None)
        try:
            assert set(features._laughter_roles(ctx2)) == expected
        finally:
            ctx2.conn.close()


def test_the_feature_schema_declares_both_scores(cfg):
    """A9's vector has to hold them, and `feature_schema.yaml`'s own rule is
    append-and-bump rather than renumber."""
    schema = cfg.feature_schema
    assert schema.version >= 2
    for kind in features.LAUGHTER_FOR_ROLE.values():
        assert kind in schema.signals
    # The events they become are declared too, and were before this commit.
    for kind in ("laugh_operator", "laugh_party"):
        assert kind in schema.signals


def test_the_events_are_declared_as_derivable():
    names = {d.name for d in derived.DERIVATIONS if d.emits_events}
    assert {"laugh_operator", "laugh_party"} <= names


def test_a_stream_without_laughter_scores_derives_no_laugh_events(cfg):
    """The `missing_inputs` contract: absent input means absent signal, never a
    silent zero."""
    timeline = grid.build(30.0, float(cfg.get("score.score_grid_hz")))
    _v, _u, events = derived.compute(
        {"mic_rms": np.full(timeline.size, -40.0)}, timeline, cfg)

    assert not events.get("laugh_operator")
    assert not events.get("laugh_party")
    missing = derived.missing_inputs({"mic_rms": np.zeros(10)})
    assert missing["laugh_operator"] == ["mic_laughter"]
    assert missing["laugh_party"] == ["party_laughter"]
