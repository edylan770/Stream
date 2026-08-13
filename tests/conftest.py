"""Shared fixtures.

The config layer deliberately reads a machine-local `local.yaml` and
`$CLIPFORGE_CONFIG`. That is correct at runtime and poison in tests: the suite
would pass or fail depending on what the developer happens to have on disk.
Every test therefore runs against the packaged defaults unless it says
otherwise.
"""

from __future__ import annotations

import pytest

from clipforge import config, ffmpeg
from tests.fixtures.make_fixture import (
    SPEECH_DURATION_S,
    FixtureSpec,
    generate,
    load_manifest,
)


def pytest_addoption(parser):
    parser.addoption(
        "--asr", action="store_true", default=False,
        help="run the tests that load a real Whisper model (minutes, and "
             "downloads ~500 MB the first time)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "asr: needs a real Whisper model; opt in with --asr"
    )


def pytest_collection_modifyitems(config, items):
    """Skip real-ASR tests unless asked for.

    A marker alone is not enough: once whisperx is installed a bare `pytest -q`
    would transcribe the fixture on every run, which is minutes. The gate has to
    be a positive opt-in, not a filter someone might forget.
    """
    if config.getoption("--asr"):
        return
    skip = pytest.mark.skip(reason="needs --asr (loads a real Whisper model)")
    for item in items:
        if "asr" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def hermetic_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LOCAL_FILE", tmp_path / "no-such-local.yaml")
    monkeypatch.delenv(config.ENV_CONFIG, raising=False)


@pytest.fixture(scope="session")
def has_ffmpeg() -> bool:
    return ffmpeg.find_binary("ffmpeg").ok and ffmpeg.find_binary("ffprobe").ok


@pytest.fixture(scope="session")
def needs_ffmpeg(has_ffmpeg):
    """Skip rather than fail on a machine that has not installed ffmpeg.

    The suite has to stay runnable on the build machine before setup and on CI,
    so anything shelling out to ffmpeg opts in explicitly.
    """
    if not has_ffmpeg:
        pytest.skip("ffmpeg/ffprobe not available")


@pytest.fixture(scope="session")
def fixture_dir(needs_ffmpeg):
    """The synthetic stream, generated once per session and cached on disk.

    Regeneration is skipped when the spec hash is unchanged, so this costs a
    few seconds the first time and nothing thereafter.
    """
    return generate(FixtureSpec())


@pytest.fixture(scope="session")
def manifest(fixture_dir) -> dict:
    """Ground truth. Tests assert against this, never against constants."""
    return load_manifest(fixture_dir)


@pytest.fixture(scope="session")
def ntsc_fixture_dir(needs_ffmpeg):
    """29.97 fps — the rate that actually exercises the FCPXML frame math.

    Every other fixture, and Testvid, runs at an integer rate where every frame
    boundary lands on a clean multiple and a rounding bug in the rational path
    passes silently. 30000/1001 is the one that does not.
    """
    return generate(FixtureSpec(name="ntsc", duration_s=30.0, fps="30000/1001"))


@pytest.fixture(scope="session")
def ntsc_manifest(ntsc_fixture_dir) -> dict:
    return load_manifest(ntsc_fixture_dir)


# --------------------------------------------------------------------------
# speech (Phase 2)
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def has_piper() -> bool:
    from tests.fixtures.speech import voices_available

    try:
        import piper  # noqa: F401
    except ImportError:
        return False
    return voices_available()


@pytest.fixture(scope="session")
def needs_piper(has_piper):
    """Skip rather than fail when the voice models are not downloaded.

    They are ~60 MB each and fetched over the network. A fresh clone with no
    network has to go green, so nothing here downloads implicitly — see
    `make_fixture.py --download-voices`.
    """
    if not has_piper:
        pytest.skip(
            "piper voices not available; "
            "run `python tests/fixtures/make_fixture.py --download-voices`"
        )


@pytest.fixture(scope="session")
def speech_fixture_dir(needs_ffmpeg, needs_piper):
    """Authored dialogue: speaker A on the mic track, B on the party track.

    Phase 2 has nothing else to work on — every other fixture is band-limited
    noise by design, because that is what makes a level detector testable and
    exactly what makes a transcriber untestable.
    """
    return generate(FixtureSpec(
        name="speech", kind="speech", duration_s=SPEECH_DURATION_S
    ))


@pytest.fixture(scope="session")
def speech_manifest(speech_fixture_dir) -> dict:
    """Ground truth by construction: exact text, exact placement, known speaker."""
    return load_manifest(speech_fixture_dir)
