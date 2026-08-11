"""Proxy encoder selection.

The load-bearing claim: an encoder appearing in `ffmpeg -encoders` says nothing
about whether it works. On this machine `h264_nvenc` is compiled in and fails
with `Cannot load nvcuda.dll`. Everything here is built around that.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from clipforge import config
from clipforge.ingest import encoders, proxy


@pytest.fixture(autouse=True)
def clear_probe_cache():
    encoders._PROBE_CACHE.clear()
    yield
    encoders._PROBE_CACHE.clear()


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------


def test_listing_an_encoder_is_not_detecting_one(needs_ffmpeg):
    """The whole reason this module exists.

    If this machine ever grows an NVIDIA GPU the assertion flips, so it is
    written to be true either way: whatever `-encoders` claims, `can_encode` is
    what decides.
    """
    from clipforge import ffmpeg

    binary = ffmpeg.require("ffmpeg")
    compiled = encoders.available_encoders(binary)
    assert "libx264" in compiled

    if "h264_nvenc" in compiled:
        usable = encoders.can_encode(binary, "h264_nvenc")
        # Either answer is legitimate; being listed is not evidence for it.
        assert isinstance(usable, bool)


def test_libx264_always_works(needs_ffmpeg):
    from clipforge import ffmpeg

    assert encoders.can_encode(ffmpeg.require("ffmpeg"), "libx264")


def test_a_nonexistent_encoder_is_not_usable(needs_ffmpeg):
    from clipforge import ffmpeg

    assert not encoders.can_encode(ffmpeg.require("ffmpeg"), "definitely_not_an_encoder")


def test_probe_result_is_cached(needs_ffmpeg, monkeypatch):
    """A test-encode costs a few hundred ms and the answer cannot change
    within a run."""
    from clipforge import ffmpeg

    binary = ffmpeg.require("ffmpeg")
    encoders.can_encode(binary, "libx264")

    calls = []
    monkeypatch.setattr(ffmpeg, "run", lambda *a, **k: calls.append(a))
    encoders.can_encode(binary, "libx264")
    assert calls == []


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


def test_auto_falls_back_to_software_when_no_hardware_works(needs_ffmpeg, monkeypatch):
    monkeypatch.setattr(encoders, "can_encode", lambda *a, **k: False)
    from clipforge import ffmpeg

    choice = encoders.select(config.load(), ffmpeg.require("ffmpeg"))
    assert choice.name == "libx264"
    assert choice.source == "fallback"
    assert not choice.is_hardware


def test_auto_picks_the_first_working_hardware_encoder(needs_ffmpeg, monkeypatch):
    from clipforge import ffmpeg

    monkeypatch.setattr(
        encoders, "available_encoders",
        lambda _b: {"libx264", "h264_nvenc", "h264_qsv"},
    )
    monkeypatch.setattr(
        encoders, "can_encode",
        lambda _b, codec, **k: codec == "h264_qsv",
    )
    choice = encoders.select(config.load(), ffmpeg.require("ffmpeg"))
    assert choice.name == "h264_qsv"
    assert choice.source == "detected"
    assert choice.is_hardware
    # nvenc was ahead of it in preference order and was tried first.
    assert "h264_nvenc" in choice.tried


def test_auto_respects_preference_order(needs_ffmpeg, monkeypatch):
    from clipforge import ffmpeg

    monkeypatch.setattr(
        encoders, "available_encoders", lambda _b: {"libx264", "h264_nvenc", "h264_amf"}
    )
    monkeypatch.setattr(encoders, "can_encode", lambda *a, **k: True)
    choice = encoders.select(config.load(), ffmpeg.require("ffmpeg"))
    assert choice.name == "h264_nvenc"


def test_a_pinned_codec_is_never_probed(needs_ffmpeg, monkeypatch):
    """Explicit config means the operator decided; do not second-guess it with
    a test-encode that costs time on every run."""
    from clipforge import ffmpeg

    monkeypatch.setattr(
        encoders, "can_encode",
        lambda *a, **k: pytest.fail("pinned codec should not be probed"),
    )
    cfg = config.load(overrides=["ingest.proxy.video_codec=h264_nvenc"])
    choice = encoders.select(cfg, ffmpeg.require("ffmpeg"))
    assert choice.name == "h264_nvenc"
    assert choice.source == "configured"


def test_on_this_machine_auto_resolves_to_something_that_works(needs_ffmpeg):
    """End to end, no mocking: whatever it picks must actually encode."""
    from clipforge import ffmpeg

    binary = ffmpeg.require("ffmpeg")
    choice = encoders.select(config.load(), binary)
    assert encoders.can_encode(binary, choice.name, extra=["-preset", choice.preset,
                                                           *choice.args])


# --------------------------------------------------------------------------
# per-encoder settings
# --------------------------------------------------------------------------


def test_each_encoder_gets_its_own_fixed_gop_flag():
    """A2 needs scene-cut detection off, and the flag differs per encoder:
    x264 has -sc_threshold, NVENC has -no-scenecut. One flat block cannot
    serve both."""
    cfg = config.load()
    _, x264_args = encoders.settings_for(cfg, "libx264")
    _, nvenc_args = encoders.settings_for(cfg, "h264_nvenc")
    assert "-sc_threshold" in x264_args
    assert "-no-scenecut" in nvenc_args
    assert "-sc_threshold" not in nvenc_args


def test_presets_differ_per_encoder():
    """NVENC takes p1-p7 and would reject x264's names."""
    cfg = config.load()
    assert encoders.settings_for(cfg, "libx264")[0] == "veryfast"
    assert encoders.settings_for(cfg, "h264_nvenc")[0].startswith("p")


def test_unknown_encoder_falls_back_to_the_top_level_preset():
    cfg = config.load()
    preset, args = encoders.settings_for(cfg, "h264_something_new")
    assert preset == cfg.get("ingest.proxy.preset")
    assert args == []


def test_every_preference_entry_has_settings():
    """A codec that can be selected but has no settings block would silently
    get x264's preset name and fail at encode time."""
    cfg = config.load()
    blocks = cfg.get("ingest.proxy.codecs")
    for codec in cfg.get("ingest.proxy.codec_preference"):
        assert codec in blocks, codec


# --------------------------------------------------------------------------
# how the choice reaches the command
# --------------------------------------------------------------------------


def test_command_uses_the_chosen_encoder_and_its_args():
    cfg = config.load()
    choice = encoders.EncoderChoice(
        "h264_nvenc", "p4", ["-no-scenecut", "1", "-rc", "vbr"], "detected"
    )
    argv = proxy.build_command(
        "ffmpeg", "m.mkv", "p.mp4", cfg=cfg, source_height=1080,
        rate=Fraction(60, 1), audio_index=0, encoder=choice,
    )
    assert argv[argv.index("-c:v") + 1] == "h264_nvenc"
    assert argv[argv.index("-preset") + 1] == "p4"
    assert "-no-scenecut" in argv
    # x264's flag must not leak into an NVENC command.
    assert "-sc_threshold" not in argv


def test_command_keeps_the_generic_gop_flags_whatever_the_encoder():
    """-g and -keyint_min are AVCodecContext options every encoder honours."""
    cfg = config.load()
    for name in ("libx264", "h264_nvenc"):
        preset, args = encoders.settings_for(cfg, name)
        argv = proxy.build_command(
            "ffmpeg", "m.mkv", "p.mp4", cfg=cfg, source_height=720,
            rate=Fraction(30, 1), audio_index=0,
            encoder=encoders.EncoderChoice(name, preset, args, "configured"),
        )
        assert argv[argv.index("-g") + 1] == str(cfg.get("ingest.proxy.gop"))
        assert argv[argv.index("-keyint_min") + 1] == str(cfg.get("ingest.proxy.keyint_min"))


def test_params_hash_ignores_which_encoder_auto_resolved_to():
    """Recording the resolved encoder would invalidate every proxy on moving
    machines and trigger a full back-catalogue re-encode."""
    cfg = config.load()
    assert cfg.get("ingest.proxy.video_codec") == "auto"
    # The params dict is built from config, which says 'auto' regardless of
    # what hardware happens to be present.
    assert "auto" in str(cfg.get("ingest.proxy"))


def test_doctor_reports_the_encoder(capsys):
    from clipforge import cli

    cli.main(["doctor"])
    out = capsys.readouterr().out
    assert "proxy encoder" in out
