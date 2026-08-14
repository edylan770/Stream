"""§8.2/§8.3's loudness normalisation, and §8.2's export presets.

The claim this commit rests on is measurable, so it is measured rather than
asserted: `test_two_pass_beats_one_pass_on_real_windows` reruns the comparison
that decided the design, over windows shaped like the ones §6.3 produces, and
fails if one-pass ever comes out closer on average.
"""

from __future__ import annotations

import json
import re

import pytest

from clipforge import config, ffmpeg
from clipforge.render import clips, loudness, presets

#: A real pass-1 blob, captured from the fixture. Kept verbatim so the parser
#: is tested against ffmpeg's actual output rather than a tidied version.
PASS_ONE = """
[Parsed_loudnorm_0 @ 000001]
{
\t"input_i" : "-16.86",
\t"input_tp" : "-3.09",
\t"input_lra" : "6.30",
\t"input_thresh" : "-30.55",
\t"output_i" : "-14.73",
\t"output_tp" : "-1.50",
\t"output_lra" : "1.80",
\t"output_thresh" : "-25.93",
\t"normalization_type" : "dynamic",
\t"target_offset" : "0.73"
}
[out#0/null @ 000002] video:0KiB audio:6375KiB
"""


# --------------------------------------------------------------------------
# parsing pass 1
# --------------------------------------------------------------------------


def test_the_measurement_is_read_from_ffmpegs_own_output():
    measured = loudness.parse(PASS_ONE)

    assert measured is not None
    assert measured.i == pytest.approx(-16.86)
    assert measured.tp == pytest.approx(-3.09)
    assert measured.lra == pytest.approx(6.30)
    assert measured.thresh == pytest.approx(-30.55)
    assert measured.target_offset == pytest.approx(0.73)


def test_the_last_json_object_wins():
    """ffmpeg writes its own diagnostics around the blob, and a graph may hold
    more than one reporting filter."""
    earlier = PASS_ONE.replace("-16.86", "-99.00")
    assert loudness.parse(earlier + PASS_ONE).i == pytest.approx(-16.86)


@pytest.mark.parametrize("junk", ["", "no json here", "{", "{}", '{"input_i": "x"}'])
def test_unreadable_output_returns_none_rather_than_raising(junk):
    """An ffmpeg whose JSON cannot be read degrades to §8.3's single pass; it
    does not stop the render."""
    assert loudness.parse(junk) is None


# --------------------------------------------------------------------------
# the filter chain
# --------------------------------------------------------------------------


def test_the_sample_rate_is_pinned():
    """MEASURED: loudnorm resamples internally and hands the encoder 192 kHz
    unless something says otherwise, and AAC-LC tops out at 96 kHz."""
    cfg = config.load()

    chain = loudness.apply_filter(cfg, None)

    assert chain.endswith(f"aresample={cfg.get('render.loudness.sample_rate_hz')}")
    assert str(loudness.LOUDNORM_INTERNAL_HZ) not in chain


def test_one_pass_carries_only_the_targets():
    chain = loudness.apply_filter(config.load(), None)

    assert "measured_I" not in chain
    assert "linear" not in chain


def test_two_pass_carries_every_measured_value():
    """All five, or loudnorm silently falls back to single-pass behaviour."""
    cfg = config.load()
    measured = loudness.parse(PASS_ONE)

    chain = loudness.apply_filter(cfg, measured)

    for key in ("measured_I", "measured_TP", "measured_LRA",
                "measured_thresh", "offset"):
        assert key in chain
    assert "linear=true" in chain


def test_the_targets_come_from_config():
    cfg = config.load(overrides=["render.loudness.target_lufs=-16",
                                 "render.loudness.true_peak_db=-2"])

    chain = loudness.apply_filter(cfg, None)

    assert "I=-16" in chain and "TP=-2" in chain


def test_the_measure_filter_includes_the_edits_that_precede_it():
    """Pass 1 must see what pass 2 will normalise. §8.6's muting and §8.2's
    filler removal go in `before`; measuring un-muted audio and normalising
    muted audio would be wrong by however much the muting removed."""
    chain = loudness.measure_filter(config.load(), before="volume=0")

    assert chain.startswith("volume=0,")
    assert "print_format=json" in chain


def test_the_applied_chain_keeps_the_edits_too():
    chain = loudness.chain(config.load(), loudness.parse(PASS_ONE),
                           before="volume=0", source_lufs=-18.0)

    assert chain.startswith("volume=0,loudnorm=")


def test_disabling_normalisation_leaves_the_edits_alone():
    cfg = config.load(overrides=["render.loudness.enabled=false"])

    assert loudness.chain(cfg, None, before="volume=0") == "volume=0"
    assert loudness.chain(cfg, None) is None


# --------------------------------------------------------------------------
# the silence guard -- the finding this module exists for
# --------------------------------------------------------------------------


def test_a_clip_needing_too_much_gain_is_left_alone():
    """MEASURED: two-pass over the fixture's silent opening landed at -32.0
    LUFS against a -14 target, eighteen LU off, because pass 1's measurements
    are meaningless and pass 2 believes them."""
    cfg = config.load()
    target = float(cfg.get("render.loudness.target_lufs"))
    limit = float(cfg.get("render.loudness.max_gain_db"))

    allowed, why = loudness.should_normalise(cfg, target - limit - 3)

    assert not allowed
    assert "noise floor" in why
    assert loudness.chain(cfg, loudness.parse(PASS_ONE),
                          source_lufs=target - limit - 3) is None


def test_the_guard_is_a_gain_limit_so_it_tracks_the_target():
    """A fixed floor would quietly become stricter or looser when the target
    moves; the same source needs the same gain either way."""
    quiet = -30.0
    lenient = config.load(overrides=["render.loudness.target_lufs=-23"])
    strict = config.load(overrides=["render.loudness.target_lufs=-9"])

    assert loudness.should_normalise(lenient, quiet)[0] is True    # +7 dB
    assert loudness.should_normalise(strict, quiet)[0] is False    # +21 dB


def test_digital_silence_is_left_alone():
    allowed, why = loudness.should_normalise(config.load(), float("-inf"))

    assert not allowed and "silence" in why


def test_an_ordinary_clip_is_normalised():
    allowed, why = loudness.should_normalise(config.load(), -18.3)

    assert allowed and why == ""


def test_the_gate_reads_ebur128_not_loudnorms_own_number():
    """THE FINDING. Over the fixture's silent opening, standalone ebur128
    reports -36.2 LUFS and loudnorm's own input_i reports -17.75 -- a level no
    floor test could tell from quiet speech. A guard on the wrong number sails
    straight past the one case it exists to catch."""
    cfg = config.load()
    loudnorm_says = loudness.parse(PASS_ONE.replace("-16.86", "-17.75")).i
    ebur128_says = -36.2      # the same window, measured on its own

    assert loudness.should_normalise(cfg, loudnorm_says)[0] is True
    assert loudness.should_normalise(cfg, ebur128_says)[0] is False


def test_the_gain_limit_is_configurable():
    assert loudness.should_normalise(
        config.load(overrides=["render.loudness.max_gain_db=1"]),
        -16.86)[0] is False


def test_an_unmeasurable_source_still_normalises():
    """Nothing measurable went wrong, so carry on rather than skip."""
    assert loudness.should_normalise(config.load(), None)[0] is True


# --------------------------------------------------------------------------
# checking the finished file
# --------------------------------------------------------------------------


def test_a_clip_on_target_produces_no_warning():
    cfg = config.load()
    target = float(cfg.get("render.loudness.target_lufs"))

    assert loudness.off_target(cfg, target) == ""
    assert loudness.off_target(cfg, target + 0.5) == ""


def test_a_clip_off_target_says_by_how_much():
    cfg = config.load()
    target = float(cfg.get("render.loudness.target_lufs"))
    tolerance = float(cfg.get("render.loudness.verify_tolerance_lu"))

    warning = loudness.off_target(cfg, target - tolerance - 2.0)

    assert f"{tolerance + 2.0:.1f} LU" in warning


def test_a_silent_result_is_reported():
    assert "silent" in loudness.off_target(config.load(), float("-inf"))


def test_no_measurement_is_not_a_warning():
    assert loudness.off_target(config.load(), None) == ""


# --------------------------------------------------------------------------
# presets
# --------------------------------------------------------------------------


def test_the_default_preset_loads():
    cfg = config.load()

    preset = presets.load(cfg)

    assert preset.name == cfg.get("render.presets.default")


def test_every_shipped_preset_loads():
    cfg = config.load()
    block = cfg.get("render.presets")

    for name, body in block.items():
        if isinstance(body, dict):
            assert presets.load(cfg, name).name == name


def test_the_presets_are_near_identical_and_say_so():
    """They differ only by a duration cap today. If that ever stops being
    true, this test should be the thing that notices."""
    cfg = config.load()
    loaded = [presets.load(cfg, n) for n, b in cfg.get("render.presets").items()
              if isinstance(b, dict)]

    encode_settings = {json.dumps(p.settings, sort_keys=True) for p in loaded}
    caps = {p.max_duration_s for p in loaded}

    assert len(encode_settings) == 1, "presets have started to differ; update the docs"
    assert len(caps) == len(loaded), "the caps are the one real difference"


def test_an_unknown_preset_lists_the_ones_that_exist():
    with pytest.raises(presets.PresetError, match="shorts"):
        presets.load(config.load(), "vertical-supreme")


def test_a_preset_may_not_set_a_resolution():
    """§8.4 puts `output` in the crop template and the dst rectangles are
    expressed in that space. Two owners for one number is how they drift."""
    cfg = config.load(overrides=["render.presets.bad.output=[1080, 1920]"])

    with pytest.raises(presets.PresetError, match="crop template"):
        presets.load(cfg, "bad")


def test_a_preset_overrides_the_base_encode_settings():
    cfg = config.load(overrides=["render.presets.tiny.crf=40"])
    preset = presets.load(cfg, "tiny")

    assert presets.setting(cfg, preset, "crf", None) == 40
    # Anything the preset does not name still comes from render.video.
    assert presets.setting(cfg, preset, "codec", None) == cfg.get("render.video.codec")


def test_no_preset_falls_back_to_render_video():
    cfg = config.load()

    assert presets.setting(cfg, None, "crf", None) == cfg.get("render.video.crf")


def test_a_long_clip_warns_and_is_never_truncated():
    """C2: silently cutting an approved clip loses the moment it was approved
    for."""
    cfg = config.load()
    preset = presets.load(cfg, "reels")

    warning = presets.too_long(preset, preset.max_duration_s + 10)

    assert "NOT truncated" in warning
    assert presets.too_long(preset, preset.max_duration_s - 1) == ""


def test_a_preset_with_no_cap_never_warns():
    cfg = config.load(overrides=["render.presets.uncapped.crf=20"])

    assert presets.too_long(presets.load(cfg, "uncapped"), 10_000) == ""


# --------------------------------------------------------------------------
# the command
# --------------------------------------------------------------------------


def test_the_audio_chain_reaches_the_encode():
    argv = clips.build_command(
        "ffmpeg", "m.mkv", "o.mp4", cfg=config.load(),
        plan=clips.timeline.build(0.0, 10.0), graph="[0:v]null[vout]",
        audio_index=1, audio_chain="loudnorm=I=-14,aresample=48000",
    )

    assert argv[argv.index("-af") + 1] == "loudnorm=I=-14,aresample=48000"


def test_a_silent_render_has_no_audio_filter():
    argv = clips.build_command(
        "ffmpeg", "m.mkv", "o.mp4", cfg=config.load(),
        plan=clips.timeline.build(0.0, 10.0), graph="[0:v]null[vout]",
        audio_index=None, audio_chain="loudnorm=I=-14",
    )

    assert "-af" not in argv and "-an" in argv


def test_the_preset_supplies_the_encode_settings():
    cfg = config.load(overrides=["render.presets.tiny.crf=40",
                                 "render.presets.tiny.preset=ultrafast"])

    argv = clips.build_command(
        "ffmpeg", "m.mkv", "o.mp4", cfg=cfg,
        plan=clips.timeline.build(0.0, 10.0), graph="[0:v]null[vout]",
        audio_index=1, preset=presets.load(cfg, "tiny"),
    )

    assert argv[argv.index("-crf") + 1] == "40"
    assert argv[argv.index("-preset") + 1] == "ultrafast"


def test_the_audio_chain_is_empty_until_muting_exists():
    """The seam §8.6 and filler removal will fill. Empty is the honest answer
    today, and the test records that it is deliberate."""
    assert clips.audio_filters(config.load(), clips.timeline.build(0.0, 5.0)) == ""


# --------------------------------------------------------------------------
# the measurement that decided the design
# --------------------------------------------------------------------------


def _encode_and_measure(ffmpeg_bin, master, start, duration, audio_filter, out):
    """Encode the window, then measure the FILE on its own.

    Deliberately not `-af "<chain>,ebur128" -f null`, which is how this was
    first measured and is wrong: chaining changes what `ebur128` reports (see
    `render/loudness.py`). The only number that means anything is the one taken
    from the file the operator would actually post.
    """
    ffmpeg.run(
        [ffmpeg_bin, "-hide_banner", "-nostdin", "-y", "-ss", f"{start}",
         "-i", str(master), "-t", f"{duration}", "-map", "0:a:0",
         "-af", audio_filter, "-c:a", "aac", "-b:a", "192k", str(out)],
        check=False,
    )
    return loudness.integrated(ffmpeg_bin, out)


def test_two_pass_beats_one_pass_on_real_windows(
    speech_fixture_dir, speech_manifest, needs_ffmpeg, tmp_path
):
    """The measurement this whole module rests on, rerun on real files.

    Windows are jittered around the fixture's authored speech, because a 0.2 s
    shift was measured moving two-pass by 1.9 LU -- so a single window proves
    nothing. Both the mean and the WORST case must favour two-pass: the
    operator cares that no clip is badly wrong, not that the average is good.

    MEASURED over nine windows, encoded and measured standalone:

        one-pass            mean 1.03 LU   worst 3.00 LU
        two-pass + offset   mean 0.63 LU   worst 1.70 LU

    If that stops being true, §8.3's single pass is the right thing to ship and
    this should say so rather than the config quietly staying wrong.
    """
    cfg = config.load()
    master = speech_fixture_dir / speech_manifest["video"]
    ffmpeg_bin = ffmpeg.require("ffmpeg")
    target = float(cfg.get("render.loudness.target_lufs"))

    speech = [u for u in speech_manifest["utterances"]]
    windows = [
        (speech[0]["t_start"], 8.0),
        (speech[0]["t_start"] + 0.2, 8.0),
        (speech[1]["t_start"], 8.0),
        (speech_manifest["overlap_windows"][0][0] - 0.2, 8.5),
        (speech[-3]["t_start"], 10.0),
    ]

    one_errors, two_errors = [], []
    for index, (start, duration) in enumerate(windows):
        one = _encode_and_measure(
            ffmpeg_bin, master, start, duration,
            loudness.apply_filter(cfg, None), tmp_path / f"one{index}.m4a")
        measured = loudness.measure(
            ffmpeg_bin, master, cfg=cfg, clip_start=start, duration=duration,
            audio_index=0)
        assert measured is not None, "pass 1 produced no measurement"
        two = _encode_and_measure(
            ffmpeg_bin, master, start, duration,
            loudness.apply_filter(cfg, measured), tmp_path / f"two{index}.m4a")

        assert one is not None and two is not None
        one_errors.append(abs(one - target))
        two_errors.append(abs(two - target))

    mean_one = sum(one_errors) / len(one_errors)
    mean_two = sum(two_errors) / len(two_errors)
    assert mean_two < mean_one, (
        f"two-pass no longer wins on average: {mean_two:.2f} LU against "
        f"one-pass {mean_one:.2f} LU"
    )
    assert max(two_errors) <= max(one_errors), (
        f"two-pass worst case {max(two_errors):.2f} LU is no better than "
        f"one-pass {max(one_errors):.2f} LU"
    )


def test_the_configured_tolerance_covers_measured_variance():
    """`verify_tolerance_lu` exists to flag something unusual. Set below the
    variance two-pass actually shows, it would fire on ordinary clips and stop
    being read -- MEASURED worst case 1.70 LU over nine windows."""
    tolerance = float(config.load().get("render.loudness.verify_tolerance_lu"))

    assert tolerance >= 1.7


def test_the_silent_window_is_the_one_two_pass_gets_wrong(
    speech_fixture_dir, speech_manifest, needs_ffmpeg
):
    """The guard's justification, measured rather than asserted.

    Over the fixture's authored silence, a standalone ebur128 pass reports a
    level the guard rejects -- and loudnorm's own measurement of the same
    window does not, which is why the gate does not use it.
    """
    cfg = config.load()
    master = speech_fixture_dir / speech_manifest["video"]
    ffmpeg_bin = ffmpeg.require("ffmpeg")
    start, end = speech_manifest["silence_windows"][0]
    duration = max(end - start, 1.0)

    source_lufs = loudness.source_loudness(
        ffmpeg_bin, master, clip_start=start, duration=duration, audio_index=0)
    measured = loudness.measure(
        ffmpeg_bin, master, cfg=cfg, clip_start=start, duration=duration,
        audio_index=0)

    assert source_lufs is not None and measured is not None
    allowed, why = loudness.should_normalise(cfg, source_lufs)
    assert not allowed, f"a {source_lufs:.1f} LUFS window should not be normalised"
    assert why
    # And the number the gate deliberately does not use would have let it pass.
    assert loudness.should_normalise(cfg, measured.i)[0] is True, (
        "loudnorm's input_i no longer disagrees with ebur128 on silence; the "
        "gate could be simplified"
    )
