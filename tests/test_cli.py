"""CLI and preflight smoke tests."""

from __future__ import annotations

import pytest

from clipforge import cli, doctor, ffmpeg


def test_help_does_not_import_command_modules(capsys):
    """`--help` lists commands from the table without importing their modules."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for name in cli.COMMANDS:
        assert name in out


def test_bare_invocation_prints_help(capsys):
    assert cli.main([]) == 2
    assert "usage: clipforge" in capsys.readouterr().out


def test_unknown_command_is_rejected():
    with pytest.raises(SystemExit) as exc:
        cli.main(["definitely-not-a-command"])
    assert exc.value.code != 0


def test_peek_command_ignores_flags():
    assert cli._peek_command(["--version"]) is None
    assert cli._peek_command(["doctor"]) == "doctor"
    assert cli._peek_command(["--foo", "doctor"]) == "doctor"
    assert cli._peek_command(["nonsense"]) is None


def test_find_binary_reports_missing_cleanly():
    info = ffmpeg.find_binary("clipforge-no-such-binary")
    assert not info.ok
    assert info.source == "missing"
    assert info.path is None


def test_require_raises_actionable_message():
    with pytest.raises(ffmpeg.FFmpegNotFound) as exc:
        ffmpeg.require("clipforge-no-such-binary")
    assert "clipforge doctor" in str(exc.value)


def test_doctor_runs_and_reports(capsys):
    checks = doctor.run_checks()
    labels = [c.label for c in checks]
    assert "python" in labels
    assert "ffmpeg" in labels
    assert "ffprobe" in labels

    code = doctor.report(checks)
    out = capsys.readouterr().out
    assert "clipforge" in out
    # Exit code must track the required failures, whatever this machine has.
    assert code == (1 if any(not c.ok and c.required for c in checks) else 0)
