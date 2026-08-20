"""CLI and preflight smoke tests."""

from __future__ import annotations

import contextlib
import io
import re
from pathlib import Path

import pytest

from clipforge import cli, doctor, ffmpeg

PACKAGE = Path(__file__).resolve().parent.parent / "clipforge"

#: A backticked instruction to the operator: `clipforge db init`. Words and long
#: flags only -- anything with a dot or an ellipsis (`clipforge db ...`,
#: `clipforge rubric --write notes.md`) is prose around a placeholder rather than
#: a command line to check.
HINT = re.compile(r"`clipforge ([a-z][a-z0-9 -]*)`")

#: argparse's way of saying "everything you typed was understood, you just left
#: something out" -- which is what a hint written mid-sentence looks like:
#: `clipforge run` with no stream id, `clipforge scene-events --check` with no
#: path. Any OTHER parse failure means the hint names something that is not there.
INCOMPLETE = ("the following arguments are required", "expected one argument")


def _hints() -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    for path in sorted([*PACKAGE.rglob("*.py"), *PACKAGE.rglob("*.yaml")]):
        if "__pycache__" in path.parts:
            continue
        for match in HINT.finditer(path.read_text(encoding="utf-8")):
            found.append((match.group(1).strip(), path))
    return found


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


def test_every_hint_to_the_operator_names_a_real_command():
    """A message telling the operator to run something must name a real command.

    MEASURED BUG: `db.open_db`'s stale-schema refusal said ``Run `clipforge
    init-db` `` for three commits. There is no `init-db` -- argparse rejects it
    with `invalid choice` -- so the one message an operator meets before the app
    will start at all told them to type something that does not work.

    The hints are EXTRACTED rather than listed, so one written later is covered
    without anyone remembering to add it here. Renaming a command or a flag and
    leaving a message behind fails this test.
    """
    hints = _hints()
    assert len(hints) > 10, "the hint regex found almost nothing; it has stopped matching"

    for hint, source in hints:
        argv = hint.split()
        buf = io.StringIO()
        try:
            with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
                cli.build_parser(cli._peek_command(argv)).parse_args(argv)
        except SystemExit:
            error = buf.getvalue().strip()
            assert any(reason in error for reason in INCOMPLETE), (
                f"{source.relative_to(PACKAGE.parent)} tells the operator to run "
                f"`clipforge {hint}`, which the CLI rejects: {error}"
            )


def test_a_stale_schema_is_reported_without_a_traceback(tmp_path, capsys):
    """`clipforge review` on an un-migrated database: one line, exit 1.

    This is what a double-click of ClipForge.cmd hits after a migration lands,
    and the launcher pauses on a non-zero exit so the reason is what stays on
    screen. A traceback puts the useful sentence under six frames of importlib.
    """
    from clipforge import db

    path = tmp_path / "stale.db"
    db.connect(path).close()          # exists, schema version 0

    code = cli.main(["review", "--set", f"paths.db={path}", "--no-open"])

    assert code == 1
    err = capsys.readouterr().err
    assert "schema version 0" in err
    assert "clipforge db init" in err
    assert "Traceback" not in err
