"""Writing the timeline (§10.5).

> For YouTube assembly, **do not extract clip files.** Generate an FCPXML or EDL
> referencing master files with in/out points. [...] a 40-clip compilation
> "renders" in milliseconds because it is a text file.

The whole document is built out of `timebase.TimeBase`, so every `offset`,
`start` and `duration` in the file is an exact integer multiple of the format's
`frameDuration` by construction rather than by care. See timebase.py for why
that is the thing to get right.

Structure, which Resolve is fussier about than the DTD is:

    <fcpxml>
      <resources>
        <format .../>                 the frame grid
        <asset ...><media-rep/></asset>   the file on disk
      </resources>
      <library><event><project><sequence><spine>
        <asset-clip/> ...             one per moment, abutting
      </spine></sequence></project></event></library>

`asset-clip@offset` is where the clip sits on the timeline; `@start` is the
in-point in the source. Both are frames-as-rationals.
"""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from xml.dom import minidom

from clipforge.render.timebase import DOWN, UP, TimeBase

#: §5.2 gives the proxy a single stereo track; a master follows §4.2's layout.
DEFAULT_AUDIO_CHANNELS = 2

#: Resolve reads this to decide timecode display. NDF is correct for everything
#: this project produces: drop-frame is a *display* convention for 29.97 and
#: choosing it would change the timecodes the operator reads off the timeline
#: without changing a single frame boundary.
TC_FORMAT = "NDF"


class ExportError(RuntimeError):
    pass


@dataclass(frozen=True)
class Clip:
    """One moment, already placed on the frame grid."""

    #: Frame indices into the source.
    source_start: int
    source_end: int
    #: Frame index on the timeline.
    timeline_offset: int
    name: str

    @property
    def frames(self) -> int:
        return self.source_end - self.source_start


def plan_clips(moments, timebase: TimeBase, source_frames: int, label) -> list[Clip]:
    """Place moments onto the grid and lay them end to end.

    Starts floor and ends ceil so a moment never loses a frame to rounding, then
    everything is clamped inside the source: a window whose end was expanded
    past the last frame would otherwise reference footage that does not exist,
    which Resolve reports as a media error rather than as a rounding problem.
    """
    clips: list[Clip] = []
    offset = 0
    for index, moment in enumerate(moments):
        start = max(timebase.frame_at(moment.t_start, DOWN), 0)
        end = min(timebase.frame_at(moment.t_end, UP), source_frames)
        if end <= start:
            continue
        clips.append(Clip(
            source_start=start, source_end=end,
            timeline_offset=offset, name=label(index, moment),
        ))
        offset += end - start
    return clips


def _asset_uid(path: Path) -> str:
    """Stable across re-exports, so re-importing relinks rather than duplicating."""
    return hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:32].upper()


def build(
    *,
    source: Path,
    timebase: TimeBase,
    source_frames: int,
    clips: list[Clip],
    resolution: tuple[int, int],
    project_name: str,
    event_name: str = "ClipForge",
    audio_sources: int = 1,
    audio_channels: int = DEFAULT_AUDIO_CHANNELS,
    version: str = "1.10",
) -> ET.ElementTree:
    """Assemble the document."""
    width, height = resolution

    root = ET.Element("fcpxml", version=str(version))
    resources = ET.SubElement(root, "resources")

    ET.SubElement(resources, "format", {
        "id": "r1",
        "name": f"ClipForge {width}x{height} {float(timebase.rate):.3f}".rstrip("0").rstrip("."),
        "frameDuration": timebase.frame_duration_value,
        "width": str(width),
        "height": str(height),
        "colorSpace": "1-1-1",
    })

    asset = ET.SubElement(resources, "asset", {
        "id": "r2",
        "name": source.stem,
        "uid": _asset_uid(source),
        "start": "0s",
        "duration": timebase.value(source_frames),
        "hasVideo": "1",
        "hasAudio": "1" if audio_sources else "0",
        "audioSources": str(max(audio_sources, 1)),
        "audioChannels": str(audio_channels),
        "format": "r1",
    })
    # Resolve resolves media by this, not by the asset name. as_uri() is what
    # turns D:\Recordings\x.mkv into file:///D:/Recordings/x.mkv, escaping
    # included -- a hand-built file:// string gets spaces wrong.
    ET.SubElement(asset, "media-rep", {
        "kind": "original-media",
        "src": source.as_uri(),
    })

    library = ET.SubElement(root, "library")
    event = ET.SubElement(library, "event", name=event_name)
    project = ET.SubElement(event, "project", name=project_name)

    total = sum(clip.frames for clip in clips)
    sequence = ET.SubElement(project, "sequence", {
        "format": "r1",
        "duration": timebase.value(total),
        "tcStart": "0s",
        "tcFormat": TC_FORMAT,
        "audioLayout": "stereo",
        "audioRate": "48k",
    })
    spine = ET.SubElement(sequence, "spine")

    for clip in clips:
        ET.SubElement(spine, "asset-clip", {
            "ref": "r2",
            "name": clip.name,
            "offset": timebase.value(clip.timeline_offset),
            "start": timebase.value(clip.source_start),
            "duration": timebase.value(clip.frames),
            "format": "r1",
            "tcFormat": TC_FORMAT,
        })

    return ET.ElementTree(root)


def to_bytes(tree: ET.ElementTree) -> bytes:
    """Serialise, with the DOCTYPE Resolve's importer expects."""
    body = ET.tostring(tree.getroot(), encoding="unicode")
    pretty = minidom.parseString(body).documentElement.toprettyxml(indent="    ")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<!DOCTYPE fcpxml>\n"
        + pretty
    ).encode("utf-8")
