"""Atomic artifact writes (Appendix A10).

> "Every extraction stage writes to a temp path and atomically renames on
> success. This is what makes resumability actually work rather than leaving
> half-written files that look complete."

Three details the spec does not mention but that decide whether it works:

**The temp name keeps the real extension.** `proxy.mp4.tmp` would be the
obvious choice and it breaks immediately: ffmpeg infers the muxer from the
output extension and refuses to write a file it cannot classify. Names are
`.tmp-<pid>-proxy.mp4`.

**The temp file sits beside its destination.** `os.replace` is only atomic
within a filesystem; a temp in the system temp directory would silently degrade
to a copy that can be interrupted halfway.

**The pid is in the name on purpose.** It lets the orphan sweep distinguish
debris left by a dead run from a live run's work in progress, so cleaning up
after a crash cannot destroy a concurrent encode.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

TMP_PREFIX = ".tmp-"
_TMP_PATTERN = re.compile(r"^\.tmp-(\d+)-")

#: Windows raises a sharing violation if anything has the destination open.
REPLACE_ATTEMPTS = 5
REPLACE_BACKOFF_S = 0.4


class AtomicReplaceError(OSError):
    """The rename could not complete, most likely because a file is held open."""


def temp_path_for(final: Path, pid: int | None = None) -> Path:
    return final.parent / f"{TMP_PREFIX}{pid or os.getpid()}-{final.name}"


def replace_with_retry(source: Path, destination: Path) -> None:
    """`os.replace` with backoff.

    On Windows the rename fails while any process holds the destination open —
    a media player left on the old proxy, or the review UI streaming the very
    file being regenerated. That is a normal thing for the operator to be doing
    and should not surface as a bare PermissionError from deep inside a stage.
    """
    last: OSError | None = None
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            os.replace(source, destination)
            return
        except OSError as exc:  # PermissionError on Windows, EBUSY elsewhere
            last = exc
            if attempt < REPLACE_ATTEMPTS - 1:
                time.sleep(REPLACE_BACKOFF_S * (attempt + 1))

    raise AtomicReplaceError(
        f"could not move {source.name} into place at {destination}. "
        f"Something is probably holding the destination open — a media player, "
        f"or the review UI streaming it. Close it and re-run. ({last})"
    ) from last


@contextmanager
def atomic_output(final: Path) -> Iterator[Path]:
    """Yield a temp path; move it to `final` only if the block succeeds.

    On any exception the temp file is removed and `final` is left untouched —
    so a killed stage leaves the previous artifact intact rather than a
    truncated file that looks finished.
    """
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = temp_path_for(final)
    tmp.unlink(missing_ok=True)
    try:
        yield tmp
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

    if not tmp.exists():
        raise FileNotFoundError(
            f"stage completed without writing its output: {tmp}"
        )
    replace_with_retry(tmp, final)


@contextmanager
def atomic_outputs(finals: list[Path]) -> Iterator[list[Path]]:
    """Same, for a stage that writes several files as one unit.

    `audio_split` produces mic/game/party together; leaving two of three in
    place after a failure would satisfy an existence check while being wrong.
    """
    temps = []
    for final in finals:
        final.parent.mkdir(parents=True, exist_ok=True)
        tmp = temp_path_for(final)
        tmp.unlink(missing_ok=True)
        temps.append(tmp)

    try:
        yield temps
    except BaseException:
        for tmp in temps:
            tmp.unlink(missing_ok=True)
        raise

    missing = [t for t in temps if not t.exists()]
    if missing:
        for tmp in temps:
            tmp.unlink(missing_ok=True)
        raise FileNotFoundError(
            "stage did not write all declared outputs: "
            + ", ".join(t.name for t in missing)
        )
    for tmp, final in zip(temps, finals, strict=True):
        replace_with_retry(tmp, final)


def sweep_orphans(root: Path, *, pid: int | None = None) -> list[Path]:
    """Delete temp files left by dead runs. Returns what was removed.

    Only temps whose embedded pid differs from ours are eligible: a concurrent
    run's in-progress encode must survive our cleanup, even though single-writer
    is the documented assumption.
    """
    if not root.exists():
        return []

    current = pid or os.getpid()
    removed: list[Path] = []
    for path in root.rglob(f"{TMP_PREFIX}*"):
        if not path.is_file():
            continue
        match = _TMP_PATTERN.match(path.name)
        if match and int(match.group(1)) == current:
            continue
        try:
            path.unlink()
            removed.append(path)
        except OSError:
            # Held open by whatever wrote it. Leave it; the next run tries again.
            continue
    return removed
