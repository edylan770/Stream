"""The double-click launchers.

These exist because `ClipForge.cmd` failed in the worst available way: with LF
line endings cmd.exe split a `-m` out of a later line into a phantom command,
printed `'m' is not recognized`, and then **carried on and did the right thing**.
A launcher that half-works is worse than one that does not, and nothing about
the symptom points at line endings.

So the requirements are asserted here rather than trusted to `.gitattributes`,
which only governs what git writes on checkout.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

LAUNCHERS = ("ClipForge.cmd", "clipforge.ps1")


@pytest.mark.parametrize("name", LAUNCHERS)
def test_the_launcher_exists(name):
    assert (ROOT / name).is_file()


@pytest.mark.parametrize("name", LAUNCHERS)
def test_line_endings_are_crlf(name):
    """MEASURED BUG: an LF-only .cmd runs, misbehaves, and does not say why."""
    raw = (ROOT / name).read_bytes()
    assert b"\n" in raw
    assert raw.replace(b"\r\n", b"") .count(b"\n") == 0, (
        f"{name} has bare LF line endings; cmd.exe misparses those"
    )


@pytest.mark.parametrize("name", LAUNCHERS)
def test_the_launcher_is_ascii(name):
    """Batch files are read in the console's OEM codepage, not UTF-8, so a
    stray em-dash in a comment arrives as mojibake."""
    raw = (ROOT / name).read_bytes()
    try:
        raw.decode("ascii")
    except UnicodeDecodeError as exc:
        pytest.fail(f"{name} has a non-ASCII byte at offset {exc.start}")


def test_the_batch_file_has_no_angle_brackets():
    """cmd applies redirection before `rem` swallows a line, so a `<id>` in a
    comment is parsed as a redirect."""
    assert "<" not in (ROOT / "ClipForge.cmd").read_text(encoding="ascii")
    assert ">" not in (ROOT / "ClipForge.cmd").read_text(encoding="ascii")


@pytest.mark.parametrize("name", LAUNCHERS)
def test_the_launcher_locates_the_venv_relative_to_itself(name):
    """Both must work from any working directory, and from a desktop shortcut."""
    text = (ROOT / name).read_text(encoding="ascii")
    assert ".venv" in text
    # %~dp0 in cmd, $MyInvocation in PowerShell — never a bare relative path.
    assert "%~dp0" in text or "MyInvocation" in text


@pytest.mark.parametrize("name", LAUNCHERS)
def test_a_missing_venv_produces_an_instruction(name):
    """Not a traceback: this is the first thing a new machine hits."""
    text = (ROOT / name).read_text(encoding="ascii")
    assert "not set up" in text
    assert "py -3.12 -m venv .venv" in text


@pytest.mark.parametrize("name", LAUNCHERS)
def test_no_arguments_opens_the_app(name):
    """The double-click case."""
    assert "review" in (ROOT / name).read_text(encoding="ascii")


def test_gitattributes_pins_the_line_endings():
    """The test above checks the working tree; this checks that a fresh clone
    gets the same thing regardless of the user's core.autocrlf."""
    text = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "*.cmd text eol=crlf" in text
    assert "*.ps1 text eol=crlf" in text
