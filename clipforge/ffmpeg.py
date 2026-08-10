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
from dataclasses import dataclass
from pathlib import Path

#: Environment overrides, honoured before PATH lookup. Config-file values take
#: precedence over both (wired up in the config layer).
ENV_FFMPEG = "CLIPFORGE_FFMPEG"
ENV_FFPROBE = "CLIPFORGE_FFPROBE"


class FFmpegNotFound(RuntimeError):
    """Raised when a required binary cannot be located."""


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
