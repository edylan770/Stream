"""ffmpeg / ffprobe discovery and subprocess wrappers.

A1 — `-ss` GOES BEFORE `-i`. ALWAYS.
--------------------------------------------------------------------------
`-ss` before `-i` is *input seeking*: ffmpeg jumps via the container index and
the seek costs milliseconds regardless of file size. After `-i` it is *output
seeking*: ffmpeg decodes from frame zero and discards, which on a 50 GB master
costs minutes. Appendix A1 calls this the single most common performance
mistake in this domain.

Accuracy footnote, because someone will eventually be tempted to "optimise"
this: input seeking is *also* frame-accurate here, because modern ffmpeg
decodes from the keyframe preceding the requested timestamp and discards the
lead-in. That holds only when we re-encode. With `-c copy` the cut snaps to the
nearest keyframe. Phase 1 never stream-copies, so we get fast *and* exact. If
you add a `-c copy` fast path later, you have silently broken every cut point.

Command construction rule: always build argument *lists*, never shell strings.
Windows paths contain spaces and the streaming PC's drive layout is unknown.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

#: Environment overrides, honoured before PATH lookup. Config-file values take
#: precedence over both (wired up in the config layer).
ENV_FFMPEG = "CLIPFORGE_FFMPEG"
ENV_FFPROBE = "CLIPFORGE_FFPROBE"


class FFmpegNotFound(RuntimeError):
    """Raised when a required binary cannot be located."""


class FFmpegError(RuntimeError):
    """A binary ran and failed. Carries enough context to diagnose it."""

    def __init__(self, argv: list[str], returncode: int, stderr: str):
        self.argv = argv
        self.returncode = returncode
        self.stderr = stderr
        tail = "\n".join(stderr.strip().splitlines()[-12:])
        super().__init__(
            f"{Path(argv[0]).name} exited {returncode}\n"
            f"  command: {' '.join(argv[1:])}\n"
            f"  stderr tail:\n{tail}"
        )


class SeekPlacementError(RuntimeError):
    """A1 violation: `-ss` used as an output option."""


@dataclass(frozen=True)
class BinaryInfo:
    """Where a binary lives, and what it reports itself to be."""

    name: str
    path: str | None
    version: str | None
    source: str  # 'config' | 'env' | 'path' | 'missing'

    @property
    def ok(self) -> bool:
        return self.path is not None and self.version is not None


def _version_of(path: str) -> str | None:
    """Return the first line of `<binary> -version`, or None if it will not run."""
    try:
        proc = subprocess.run(
            [path, "-version"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    first = (proc.stdout or proc.stderr).splitlines()
    if not first:
        return None
    match = re.match(r"^(ff(?:mpeg|probe) version \S+)", first[0])
    return match.group(1) if match else first[0].strip()


def find_binary(name: str, override: str | os.PathLike[str] | None = None) -> BinaryInfo:
    """Locate `ffmpeg` or `ffprobe`.

    Resolution order: explicit override (config) -> environment variable ->
    PATH. Nothing here may assume a machine-specific install location.
    """
    env_name = {"ffmpeg": ENV_FFMPEG, "ffprobe": ENV_FFPROBE}.get(name)

    candidates: list[tuple[str, str]] = []
    if override:
        candidates.append(("config", str(override)))
    if env_name and os.environ.get(env_name):
        candidates.append(("env", os.environ[env_name]))
    found_on_path = shutil.which(name)
    if found_on_path:
        candidates.append(("path", found_on_path))

    for source, candidate in candidates:
        # An override may name a directory containing the binary, or the
        # binary itself. Accept both; the streaming PC will likely use a
        # portable extract directory.
        resolved = candidate
        if Path(candidate).is_dir():
            on_disk = shutil.which(name, path=candidate)
            if not on_disk:
                continue
            resolved = on_disk
        version = _version_of(resolved)
        if version:
            return BinaryInfo(name=name, path=resolved, version=version, source=source)

    return BinaryInfo(name=name, path=None, version=None, source="missing")


def require(name: str, override: str | os.PathLike[str] | None = None) -> str:
    """Return the path to a working binary, or raise with an actionable message."""
    info = find_binary(name, override)
    if not info.ok:
        raise FFmpegNotFound(
            f"{name} not found. Install it (Windows: `winget install --id Gyan.FFmpeg`) "
            f"or set paths.{name} in clipforge/config/local.yaml. "
            f"Run `clipforge doctor` to check."
        )
    assert info.path is not None
    return info.path


def validate_seek_placement(argv: Sequence[str]) -> None:
    """Enforce A1 mechanically instead of by comment.

    `-ss` is an *input* option when it precedes the `-i` it applies to, and an
    *output* option otherwise. Input seeking jumps via the container index in
    milliseconds; output seeking decodes from frame zero and, on a 50 GB master,
    takes minutes. Appendix A1 calls this the most common performance mistake in
    the domain, so every command this module runs is checked rather than
    trusted.

    Multi-input commands are handled correctly: in
    `-ss 10 -i a.mkv -ss 20 -i b.mkv` both seeks precede an input and both are
    fast. The failure is a `-ss` with no `-i` after it.
    """
    tokens = list(argv)
    for index, token in enumerate(tokens):
        if token != "-ss":
            continue
        if "-i" not in tokens[index + 1 :]:
            raise SeekPlacementError(
                "A1 violation: -ss appears with no -i after it, which makes it an "
                "output option (decode from frame zero). Move it before the input "
                f"it applies to. Command: {' '.join(tokens[1:])}"
            )


def filter_file(path: str | os.PathLike[str]) -> tuple[str, str]:
    """A file referenced from inside a filter graph: `(value, working_dir)`.

    NOT the same problem as passing a path as an argument. Inside a filter
    description ffmpeg re-parses the string: `:` separates options, `,`
    separates filters, `'` quotes, `[` and `]` mark pads. A Windows path is
    therefore split at its drive letter — `-vf ass=C:/x/y.ass` makes ffmpeg
    read `/x/y.ass` as the `ass` filter's *second* option and fail with
    "Unable to parse original_size", naming neither the file nor the colon.

    MEASURED, on this machine's ffmpeg, over five awkward directory names:

    | form                                    | result |
    |---|---|
    | `ass=filename='C\\:/dir/f.ass'`          | works — except an apostrophe |
    | `ass=filename=C\\:/dir/f.ass` (unquoted) | fails to parse |
    | `'...o'\\''brien...'` (shell-style)      | fails |
    | `'...o\\'brien...'`                      | fails |
    | run from the directory, bare filename   | **works in every case** |

    There is no escaping that survives an apostrophe in a parent directory, and
    `C:\\Users\\O'Brien` is an ordinary Windows home directory. So the file is
    referenced by bare name and ffmpeg is run from beside it — nothing to
    escape, because the only name in the string is one this project chose.

    Inputs and outputs are unaffected: those are argv elements, not filter
    values, and stay absolute.
    """
    resolved = Path(path)
    return resolved.name, str(resolved.parent)


def run(
    argv: Sequence[str],
    *,
    timeout: float | None = None,
    check: bool = True,
    cwd: str | os.PathLike[str] | None = None,
    on_stderr_line: Callable[[str], None] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ffmpeg/ffprobe with A1 enforced and stderr captured.

    Arguments are always a list — never a shell string. Paths on the target
    machine are unknown and will contain spaces.

    `cwd` exists for `filter_file`: a filter graph that names a file has to name
    it relatively, because no escaping of an absolute Windows path survives the
    filter parser. Every other path stays absolute regardless.
    """
    validate_seek_placement(argv)
    argv = [str(a) for a in argv]
    working_dir = str(cwd) if cwd is not None else None

    if on_stderr_line is None:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, check=False,
            encoding="utf-8", errors="replace", cwd=working_dir,
        )
        if check and proc.returncode != 0:
            raise FFmpegError(argv, proc.returncode, proc.stderr or "")
        return proc

    # Streaming mode: used for long encodes so a heartbeat can be updated and
    # progress reported while ffmpeg works.
    collected: list[str] = []
    with subprocess.Popen(
        argv,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        cwd=working_dir,
    ) as popen:
        assert popen.stderr is not None
        for line in popen.stderr:
            collected.append(line)
            on_stderr_line(line.rstrip("\r\n"))
        returncode = popen.wait(timeout=timeout)

    stderr = "".join(collected)
    if check and returncode != 0:
        raise FFmpegError(argv, returncode, stderr)
    return subprocess.CompletedProcess(argv, returncode, "", stderr)
