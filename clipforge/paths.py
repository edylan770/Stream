"""Where a stream's artifacts live (spec §2.3 layout).

Two forms of every path:

  * **relative** — what goes in the database. The DB is the irreplaceable
    backup tier (§13.2) and it has to survive being carried to the streaming PC
    or landing on a different drive letter.
  * **absolute** — what gets handed to ffmpeg.

No symlinks. §2.3 suggests them for `raw/`, but on Windows they need Developer
Mode or an elevated prompt, and the payoff is a convenience this project does
not need: `streams.master_path` plus a `sources.json` sidecar records the same
thing with no privilege requirement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date as date_cls
from pathlib import Path, PurePosixPath

#: Subdirectories created under each stream (§2.3).
STREAM_SUBDIRS = ("raw", "proxy", "audio", "previews", "exports")

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def slugify(text: str, max_len: int = 40) -> str:
    """Lowercase, alphanumeric, underscore-separated."""
    slug = _SLUG_STRIP.sub("_", text.strip().lower()).strip("_")
    return slug[:max_len].rstrip("_")


def make_stream_id(
    date: str | date_cls,
    label: str,
    existing: set[str] | None = None,
    max_len: int = 40,
) -> str:
    """Build `<ISO date>_<slug>`, suffixed on collision.

    §3.2 gives the format by example ('2026-08-14_marvel') and says nothing
    about two streams on one day, which will happen the first time a session
    gets split in two.
    """
    iso = date.isoformat() if isinstance(date, date_cls) else str(date)
    slug = slugify(label, max_len) or "stream"
    base = f"{iso}_{slug}"
    if not existing or base not in existing:
        return base
    for n in range(2, 1000):
        candidate = f"{base}_{n}"
        if candidate not in existing:
            return candidate
    raise ValueError(f"cannot allocate a stream id for {base}")


@dataclass(frozen=True)
class StreamPaths:
    """Artifact locations for one stream."""

    data_root: Path
    stream_id: str

    # -- relative (for the database) --------------------------------------

    @property
    def rel_dir(self) -> PurePosixPath:
        # Forward slashes in the DB regardless of host OS, so a database
        # written on Windows resolves on anything else.
        return PurePosixPath("streams") / self.stream_id

    def rel(self, *parts: str) -> str:
        return str(self.rel_dir.joinpath(*parts))

    @property
    def rel_proxy(self) -> str:
        return self.rel("proxy", "proxy.mp4")

    def rel_audio(self, role: str) -> str:
        return self.rel("audio", f"{role}.wav")

    def rel_export(self, filename: str) -> str:
        return self.rel("exports", filename)

    def rel_preview(self, filename: str) -> str:
        return self.rel("previews", filename)

    # -- absolute (for I/O) ------------------------------------------------

    @property
    def dir(self) -> Path:
        return self.data_root / "streams" / self.stream_id

    def absolute(self, relative: str | PurePosixPath) -> Path:
        """Resolve a DB-stored relative path against this data root."""
        return self.data_root / Path(str(relative))

    @property
    def proxy(self) -> Path:
        return self.absolute(self.rel_proxy)

    def audio(self, role: str) -> Path:
        return self.absolute(self.rel_audio(role))

    @property
    def raw_dir(self) -> Path:
        return self.dir / "raw"

    @property
    def sources_json(self) -> Path:
        """Sidecar recording the master and capture files this stream came from."""
        return self.raw_dir / "sources.json"

    @property
    def markers_jsonl(self) -> Path:
        return self.raw_dir / "markers.jsonl"

    @property
    def exports_dir(self) -> Path:
        return self.dir / "exports"

    @property
    def previews_dir(self) -> Path:
        return self.dir / "previews"

    def ensure(self) -> StreamPaths:
        """Create the §2.3 subdirectory layout."""
        for name in STREAM_SUBDIRS:
            (self.dir / name).mkdir(parents=True, exist_ok=True)
        return self


def resolve(data_root: Path, relative: str | None) -> Path | None:
    """Resolve a nullable DB-stored relative path."""
    return None if not relative else data_root / Path(relative)
