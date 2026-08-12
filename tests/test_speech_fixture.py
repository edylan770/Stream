"""The speech fixture — Phase 2's measuring instrument.

Two layers. Most of what this fixture *does* is placement, level control and
manifest shape, which is ordinary code and gets tested on a fake renderer with
no model. Only the last group needs Piper.

Everything reads from `manifest.json` or the spec object. The point of a fixture
whose ground truth is known by construction is defeated the moment a test
hardcodes a number the fixture also knows.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.fixtures import speech
from tests.fixtures.make_fixture import (
    BAND_LIMIT_HZ,
    SPEECH_DURATION_S,
    SPEECH_SCRIPT,
    VOCABULARY_TERMS,
    FixtureSpec,
    build_speech_pattern,
    speech_windows,
)


@pytest.fixture
def placed():
    """The script laid out with the fake renderer — no model needed."""
    spec = FixtureSpec(name="t", kind="speech", duration_s=SPEECH_DURATION_S)
    rendered: dict = {}
    build_speech_pattern(spec, speech.ToneRenderer(), rendered)
    return spec, rendered


def linear_ratio(a_db: float, b_db: float) -> float:
    """§5.8 compares energies. dB are logarithms; this is the conversion."""
    return 10 ** ((a_db - b_db) / 10.0)


# --------------------------------------------------------------------------
# placement
# --------------------------------------------------------------------------


def test_every_scripted_line_is_placed(placed):
    spec, _ = placed
    assert len(spec.utterances) == len(SPEECH_SCRIPT)


def test_lines_start_where_the_script_says(placed):
    spec, _ = placed
    for region, (_speaker, _track, start, _dbfs, _text) in zip(
        sorted(spec.utterances, key=lambda r: r.label), SPEECH_SCRIPT, strict=True
    ):
        assert region.t_start == pytest.approx(start, abs=1e-3)


def test_ends_are_measured_not_declared(placed):
    """A line's true duration is whatever the synthesiser produced. Declaring it
    would make the manifest describe the intention rather than the file."""
    spec, rendered = placed
    for region in spec.utterances:
        expected = rendered[region.label].size / spec.audio_sr
        assert region.t_end - region.t_start == pytest.approx(expected, abs=1e-3)


def test_speakers_are_on_the_tracks_4_2_assigns(placed):
    """A on mic (§4.2 track 2), B on party (track 4). §5.8 transcribes them
    separately, so they cannot share a track."""
    spec, _ = placed
    tracks = {region.speaker: region.track for region in spec.utterances}
    assert tracks == {"A": "mic", "B": "party"}


def test_nothing_runs_past_the_end_of_the_fixture(placed):
    spec, _ = placed
    assert max(r.t_end for r in spec.utterances) <= spec.duration_s


def test_a_script_that_overruns_is_refused():
    """Silently truncating a line would corrupt the ground truth it exists to
    provide."""
    spec = FixtureSpec(name="t", kind="speech", duration_s=10.0)
    with pytest.raises(ValueError, match="past the fixture"):
        build_speech_pattern(spec, speech.ToneRenderer(), {})


# --------------------------------------------------------------------------
# the windows Phase 2 needs, derived rather than declared
# --------------------------------------------------------------------------


def test_there_is_a_long_silence(placed):
    """A4: VAD must be on or Whisper hallucinates phantom text over silence, and
    that would poison the transcript and every downstream signal. This window is
    what turns the warning into an assertion."""
    spec, _ = placed
    windows = speech_windows(spec)
    assert windows["longest_silence_s"] > 15.0


def test_the_silence_really_contains_no_speech(placed):
    spec, _ = placed
    for start, end in speech_windows(spec)["silence_windows"]:
        for region in spec.utterances:
            assert region.t_end <= start or region.t_start >= end


def test_the_two_speakers_overlap_somewhere(placed):
    """§5.8's `both` is its only ambiguous branch, and §5.4.1 calls everyone
    talking at once a signal in its own right."""
    spec, _ = placed
    overlaps = speech_windows(spec)["overlap_windows"]
    assert overlaps
    assert all(end > start for start, end in overlaps)


def test_overlap_windows_are_across_tracks_not_within_one(placed):
    """Two consecutive lines from the same speaker are not an overlap."""
    spec, _ = placed
    for start, end in speech_windows(spec)["overlap_windows"]:
        tracks = {
            r.track for r in spec.utterances if r.t_start < end and r.t_end > start
        }
        assert len(tracks) > 1


# --------------------------------------------------------------------------
# the script's content, which later commits assert against
# --------------------------------------------------------------------------


def test_the_vocabulary_terms_actually_appear(placed):
    """§5.7's seeding is tested A/B against these, so a term that never gets
    spoken would make that test vacuous."""
    spec, _ = placed
    spoken = " ".join(r.text for r in spec.utterances).lower()
    for term in VOCABULARY_TERMS:
        assert term.lower() in spoken, f"{term} is never said"


def test_a_phrase_repeats_enough_for_phrase_repeat(placed):
    """§5.6 flags a phrase repeated >=3x within 90 s as bit formation."""
    spec, _ = placed
    from tests.fixtures.make_fixture import REPEATED_PHRASE

    said = [r for r in spec.utterances if r.text == REPEATED_PHRASE]
    assert len(said) >= 3
    assert max(r.t_start for r in said) - min(r.t_start for r in said) <= 90.0


def test_passive_phrases_from_the_spec_list_are_present(placed):
    """§5.6's own examples, so phrase_detect has real material."""
    spec, _ = placed
    from tests.fixtures.make_fixture import PASSIVE_PHRASES

    spoken = " ".join(r.text for r in spec.utterances).lower()
    assert sum(phrase in spoken for phrase in PASSIVE_PHRASES) >= 3


# --------------------------------------------------------------------------
# conditioning — the part that would silently corrupt every measured level
# --------------------------------------------------------------------------


def test_band_limiting_removes_everything_above_the_cutoff():
    """NOT COSMETIC. §5.3 extracts at 16 kHz, whose Nyquist is 8 kHz, and the
    fixture's contract is that measured_dbfs equals what the pipeline extracts.
    Energy above the cutoff is lost in that downsample, so leaving it in makes
    every level in the manifest quietly wrong."""
    sr = 48_000
    t = np.arange(sr) / sr
    signal = (np.sin(2 * np.pi * 1_000 * t) + np.sin(2 * np.pi * 9_000 * t)).astype(np.float32)

    limited = speech.band_limit(signal, sr, BAND_LIMIT_HZ)
    spectrum = np.abs(np.fft.rfft(limited))
    freqs = np.fft.rfftfreq(limited.size, 1 / sr)

    assert spectrum[freqs > BAND_LIMIT_HZ].max() < spectrum[freqs < BAND_LIMIT_HZ].max() * 1e-6


def test_rendered_speech_is_band_limited(placed):
    """1e-4 rather than zero: the brick wall is exact, but the irfft/rfft round
    trip through float32 leaves about -105 dB of numerical noise above the
    cutoff. That is four orders of magnitude below anything the 16 kHz
    downsample would notice."""
    spec, rendered = placed
    audio = next(iter(rendered.values()))
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(audio.size, 1 / spec.audio_sr)
    above = spectrum[freqs > BAND_LIMIT_HZ * 1.05].sum()
    assert above / spectrum.sum() < 1e-4


def test_resampling_preserves_duration():
    signal = np.zeros(22_050, dtype=np.float32)
    out = speech.resample(signal, 22_050, 48_000)
    assert out.size == pytest.approx(48_000, rel=0.01)


def test_resampling_is_a_no_op_at_the_same_rate():
    signal = np.arange(100, dtype=np.float32)
    assert np.array_equal(speech.resample(signal, 48_000, 48_000), signal)


def test_rendered_audio_is_at_the_authoring_rate(placed):
    """Piper renders at 22.05 kHz; the fixture is authored at 48 kHz.

    Within a handful of samples, not exactly: the region's boundaries are
    rounded to four decimals for the manifest, which is a fifth of a millisecond
    of slack at 48 kHz."""
    spec, rendered = placed
    for region in spec.utterances:
        expected = int(round((region.t_end - region.t_start) * spec.audio_sr))
        assert abs(rendered[region.label].size - expected) <= 10


# --------------------------------------------------------------------------
# the manifest — what commits 19-21 read
# --------------------------------------------------------------------------


def test_a_noise_fixture_has_no_speech_keys(manifest):
    """The speech section must not leak into the fixtures that predate it."""
    assert "utterances" not in manifest


def test_the_manifest_describes_every_utterance(speech_manifest):
    assert len(speech_manifest["utterances"]) == len(SPEECH_SCRIPT)
    for entry in speech_manifest["utterances"]:
        assert entry["text"]
        assert entry["speaker"] in ("A", "B")
        assert entry["voice"]
        assert entry["t_end"] > entry["t_start"]


def test_the_manifest_carries_per_track_levels_for_each_line(speech_manifest):
    """This is the design: `measure_regions` already records every track's level
    across each region, so making utterances regions gives §5.8's ground truth
    for free."""
    for entry in speech_manifest["utterances"]:
        assert {"mic", "party", "game"} <= set(entry["measured_dbfs"])


def test_each_line_is_louder_on_its_own_speakers_track(speech_manifest):
    """Outside the overlap, the authored speaker's track dominates by far more
    than §5.8's ratio needs — so commit 20's rule has an unambiguous answer."""
    overlaps = speech_manifest["overlap_windows"]
    for entry in speech_manifest["utterances"]:
        overlapped = any(
            entry["t_start"] < end and entry["t_end"] > start for start, end in overlaps
        )
        if overlapped:
            continue
        mic = entry["measured_dbfs"]["mic"]
        party = entry["measured_dbfs"]["party"]
        ratio = linear_ratio(mic, party) if entry["speaker"] == "A" else linear_ratio(party, mic)
        assert ratio > 1.5, f"{entry['label']} does not separate: {entry['measured_dbfs']}"


def test_the_overlap_line_is_genuinely_ambiguous(speech_manifest):
    """And inside the overlap, neither track wins — which is the whole reason
    §5.8 has a `both` branch."""
    overlaps = speech_manifest["overlap_windows"]
    ambiguous = [
        entry for entry in speech_manifest["utterances"]
        if any(entry["t_start"] < end and entry["t_end"] > start for start, end in overlaps)
    ]
    assert ambiguous
    assert any(
        linear_ratio(e["measured_dbfs"]["mic"], e["measured_dbfs"]["party"]) < 1.5
        and linear_ratio(e["measured_dbfs"]["party"], e["measured_dbfs"]["mic"]) < 1.5
        for e in ambiguous
    )


def test_the_literal_spec_rule_gets_the_overlap_wrong(speech_manifest):
    """MEASURED. §5.8's pseudocode compares `mean_energy` values with a 1.5
    ratio, and our series are stored in dBFS — logarithms. Multiplying a
    negative dB value by 1.5 moves the threshold DOWN, so the test fires when it
    should not.

    On this fixture the overlapped line reads mic -23.57 / party -22.01: the
    party track is 1.4x the louder, and the literal rule calls it 'operator'.
    Commit 20 converts to linear power first; this test is why."""
    overlaps = speech_manifest["overlap_windows"]
    ambiguous = [
        e for e in speech_manifest["utterances"]
        if any(e["t_start"] < end and e["t_end"] > start for start, end in overlaps)
        and e["speaker"] == "B"
    ]
    assert ambiguous, "the fixture must contain an overlapped party line"

    entry = ambiguous[0]
    mic = entry["measured_dbfs"]["mic"]
    party = entry["measured_dbfs"]["party"]

    assert party > mic, "the party track should be the louder one here"
    assert mic > party * 1.5, "the naive dB comparison should (wrongly) fire"
    assert linear_ratio(mic, party) < 1.5, "the correct comparison should not"


def test_the_contamination_case_is_present(speech_manifest):
    """§4.2: loud game audio under a quiet mic is indistinguishable from
    shouting on a single-track recording."""
    contaminated = [
        e for e in speech_manifest["utterances"]
        if linear_ratio(e["measured_dbfs"]["game"], e["measured_dbfs"]["mic"]) > 1.5
    ]
    assert contaminated


def test_the_quiet_track_sits_at_its_declared_floor(speech_manifest):
    """If the floor crept up, VAD would have something to find during the
    silence and A4's test would pass for the wrong reason.

    `floors_dbfs` is DECLARED, i.e. pre-headroom, while `measured_dbfs` is what
    ended up in the file — so the comparison has to put the shared headroom
    scale back on. Comparing the two directly is off by ~9 dB and looks like a
    fixture bug when it is an arithmetic one."""
    floors = speech_manifest["floors_dbfs"]
    scale_db = 20 * np.log10(speech_manifest["headroom_scale"])
    expected_mic_floor = floors["mic"] + scale_db

    quietest = min(e["measured_dbfs"]["mic"] for e in speech_manifest["utterances"])
    assert quietest == pytest.approx(expected_mic_floor, abs=1.0)


def test_the_fixture_still_looks_like_a_recording(speech_manifest, manifest):
    """It has to run the whole Phase 1 pipeline, not just be transcribable."""
    assert speech_manifest["expected_track_map"] == manifest["expected_track_map"]
    assert speech_manifest["expected_audio_track_count"] == 4
    assert speech_manifest["markers"]


# --------------------------------------------------------------------------
# real synthesis
# --------------------------------------------------------------------------


def test_piper_is_deterministic(needs_piper):
    """VITS samples noise during inference, so the default settings give
    different audio every run. This fixture is a measuring instrument; it has to
    be identical on any machine that regenerates it."""
    renderer = speech.PiperRenderer()
    line = "Hawkeye just hit that shot from across the map."
    first_sr, first = renderer.render(line, speech.VOICES["A"])
    second_sr, second = renderer.render(line, speech.VOICES["A"])

    assert first_sr == second_sr
    assert np.array_equal(first, second)


def test_the_two_voices_are_different(needs_piper):
    renderer = speech.PiperRenderer()
    line = "Let's go."
    _, a = renderer.render(line, speech.VOICES["A"])
    _, b = renderer.render(line, speech.VOICES["B"])
    assert a.size != b.size or not np.array_equal(a, b)


def test_rendered_speech_is_not_silent(needs_piper):
    renderer = speech.PiperRenderer()
    _, audio = renderer.render("Oh my god, did you see that?", speech.VOICES["A"])
    assert float(np.max(np.abs(audio))) > 0.1


def test_a_missing_voice_says_how_to_fix_it(tmp_path):
    renderer = speech.PiperRenderer(voices_dir=tmp_path)
    with pytest.raises(speech.VoicesMissing, match="download-voices"):
        renderer.render("hello", speech.VOICES["A"])
