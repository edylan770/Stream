"""A server-side file browser, because the browser cannot supply a path.

Drag-and-drop hands a web page a `File` object and deliberately never a
filesystem path, so "drop your recording here" would mean *uploading* a 40-55 GB
master (§13.1) to localhost in order to discover where it already is. The
directory listing therefore happens on the server and the browser only ever
sends the path back.

Shaped for §7.3's "the operator should never need the mouse", which the spec
scopes to the review screen but which matters at least as much here: choosing a
path on a recording drive is the most mouse-dependent thing in the application.
So a listing is one flat ordered array the UI can walk with j/k/enter rather
than a tree, and any absolute path can be sent directly.

Nothing here restricts *where* you may look. A root jail would be security
theatre — masters live on whatever drive has room — and `guard.py` is the thing
that decides who may ask.
"""

from __future__ import annotations

import os
import string
from dataclasses import dataclass, field
from pathlib import Path

#: Reported per directory rather than per file: `register.find_capture_file`
#: looks for these beside the master, so the answer is a property of the folder.
CAPTURE_FILES = ("anchor.json", "markers.jsonl")


@dataclass(frozen=True)
class Entry:
    name: str
    path: str
    is_dir: bool
    size: int | None = None
    is_media: bool = False

    def to_json(self) -> dict:
        return {
            "name": self.name, "path": self.path, "is_dir": self.is_dir,
            "size": self.size, "is_media": self.is_media,
        }


@dataclass(frozen=True)
class Listing:
    path: str
    parent: str | None
    roots: list[str]
    entries: list[Entry] = field(default_factory=list)
    #: More entries exist than were returned (see `max_entries`).
    truncated: bool = False
    has_anchor: bool = False
    has_markers: bool = False
    error: str | None = None

    def to_json(self) -> dict:
        return {
            "path": self.path,
            "parent": self.parent,
            "roots": list(self.roots),
            "entries": [e.to_json() for e in self.entries],
            "truncated": self.truncated,
            "has_anchor": self.has_anchor,
            "has_markers": self.has_markers,
            "error": self.error,
        }


def roots() -> list[str]:
    """Drive or volume roots, so "this PC" is always one click away.

    `os.listdrives` is 3.12+ and `pyproject.toml` declares `requires-python
    >=3.11`, so the letter probe is a real fallback rather than dead code.
    """
    if os.name != "nt":
        return ["/"]

    lister = getattr(os, "listdrives", None)
    if lister is not None:
        try:
            return list(lister())
        except OSError:
            pass

    found = []
    for letter in string.ascii_uppercase:
        drive = f"{letter}:\\"
        # A mapped-but-disconnected network drive raises rather than returning
        # False, and one of those must not cost the whole listing.
        try:
            if os.path.exists(drive):
                found.append(drive)
        except OSError:
            continue
    return found


def default_start() -> Path:
    """Where the browser opens before anything has been chosen."""
    return Path.home()


def _is_media(path: Path, media_extensions) -> bool:
    return path.suffix.lower() in media_extensions


def list_dir(path, *, media_extensions=(), max_entries: int = 500) -> Listing:
    """One directory, ordered and capped.

    A path naming a *file* lists its parent, so pasting the master's own path
    lands somewhere useful instead of erroring.

    Errors come back in the `error` field rather than as exceptions: browsing to
    a drive that is no longer plugged in is ordinary, not exceptional, and the
    UI has to be able to draw it.
    """
    extensions = {str(e).lower() for e in media_extensions}
    available = roots()

    try:
        target = Path(path).expanduser()
        if not target.is_absolute():
            target = target.resolve()
    except (OSError, ValueError) as exc:
        return Listing(path=str(path), parent=None, roots=available, error=str(exc))

    if target.is_file():
        target = target.parent

    parent = str(target.parent) if target.parent != target else None

    if not target.is_dir():
        return Listing(
            path=str(target), parent=parent, roots=available,
            error=f"{target} is not a directory that exists right now",
        )

    try:
        raw = sorted(target.iterdir(), key=lambda p: p.name.lower())
    except OSError as exc:
        return Listing(
            path=str(target), parent=parent, roots=available,
            error=f"cannot read {target}: {exc}",
        )

    directories: list[Entry] = []
    media: list[Entry] = []
    others: list[Entry] = []

    for item in raw:
        # Per entry, not per listing: a single unreadable item in a drive root —
        # System Volume Information, a stale reparse point — must not take the
        # whole request down with it.
        try:
            if item.is_dir():
                directories.append(Entry(name=item.name, path=str(item), is_dir=True))
                continue
            is_media = _is_media(item, extensions)
            entry = Entry(
                name=item.name, path=str(item), is_dir=False,
                size=item.stat().st_size, is_media=is_media,
            )
        except OSError:
            continue
        (media if is_media else others).append(entry)

    ordered = directories + media + others
    truncated = len(ordered) > max_entries

    return Listing(
        path=str(target),
        parent=parent,
        roots=available,
        entries=ordered[:max_entries],
        truncated=truncated,
        has_anchor=(target / CAPTURE_FILES[0]).is_file(),
        has_markers=(target / CAPTURE_FILES[1]).is_file(),
    )
