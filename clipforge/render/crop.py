"""§8.4 — vertical reframe from a static crop template.

> **Gaming has a structural advantage here.** Commercial tools perform dynamic
> subject tracking because talking-head footage moves. The operator's layout
> does not move — facecam is always in the same corner, gameplay fills the same
> rectangle. Therefore: **a static crop template per OBS scene.** No model, no
> tracking, deterministic, instant, identical every time.

That reasoning is right and it is the cheapest correct answer in the whole spec.
The example beside it is not checked, and three things have to be decided that
§8.4 does not mention.

**1. §8.4's own example distorts gameplay by 23%.**

    - name: gameplay
      src: [480, 140, 960, 800]     # aspect 1.200
      dst: [0, 810, 1080, 1110]     # aspect 0.973

A literal `crop` then `scale` to those numbers stretches the region vertically
by 1.233x — a circle becomes an ellipse. So `fit` is per-region config and the
default is not `stretch`:

* `fill` (default) — scale to cover `dst`, then centre-crop the overflow.
  Geometry stays true; a sliver of edge is lost.
* `contain` — scale to fit inside `dst` and pad the remainder.
* `stretch` — §8.4 taken literally, for anyone who wants it.

**2. Coordinates are declared against a resolution, and yours may differ.**
`source_resolution` says what the numbers were measured on. A 1440p canvas, or
the 640x360 test fixture, scales by `actual / declared`. **A different aspect
ratio is refused**: scaling x and y by different factors moves every region
somewhere the operator did not put it, and the result is a plausible-looking
frame that is quietly wrong.

**3. Nothing checks that the regions actually cover the output.** §8.4 never
says they must. Two failures are possible and they are not equally bad:

* **Overlap is refused.** One region is drawn over another, so part of what the
  template says to show cannot be seen. There is no layout for which that is
  the intent.
* **A gap is reported, not refused.** A band of `pad_color` across the frame is
  a legitimate design — and it is also exactly what a typo looks like. Deciding
  which would be guessing, so it is logged with its rectangle and left alone.

**Not `vstack`.** §8.4 says "crop + scale + vstack", which only works when the
regions are full-width and stacked in source order. `dst` gives arbitrary
rectangles, so the first region is `pad`ded out to the full output at its own
offset and the rest are `overlay`ed at theirs. That is general, it needs no
`color` source (which would be an infinite stream needing a duration bound),
and it makes the validator about geometry rather than about whether `vstack`
happens to apply.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from clipforge.render import RenderError

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

FILL = "fill"
CONTAIN = "contain"
STRETCH = "stretch"
FITS = (FILL, CONTAIN, STRETCH)

#: How far the source aspect may differ from the template's declared aspect
#: before the template is refused. 1% covers 1920x1080 against 1280x720 rounding
#: and nothing else -- 16:9 against 16:10 is 11% and must not pass.
ASPECT_TOLERANCE = 0.01

_HEX = re.compile(r"^#?([0-9a-fA-F]{6})$")


class CropError(RenderError):
    pass


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def aspect(self) -> float:
        return self.w / self.h if self.h else 0.0

    def intersect(self, other: Rect) -> Rect | None:
        x = max(self.x, other.x)
        y = max(self.y, other.y)
        right = min(self.right, other.right)
        bottom = min(self.bottom, other.bottom)
        if right <= x or bottom <= y:
            return None
        return Rect(x, y, right - x, bottom - y)

    def inside(self, w: int, h: int) -> bool:
        return (self.x >= 0 and self.y >= 0
                and self.right <= w and self.bottom <= h)

    def scaled(self, factor: float, bounds: tuple[int, int]) -> Rect:
        """Move onto a differently sized source.

        Rounds outward -- origin down, extent up -- so a rounding error grows the
        region rather than shaving a pixel off the edge of the facecam (C2).
        Then clamps back inside the frame, because growing past it is worse.

        `math.ceil`, not the `-int(-x)` trick: that is a floor for floats,
        because `int()` truncates toward zero. It read as a ceiling and was
        not one.
        """
        width, height = bounds
        x = max(math.floor(self.x * factor), 0)
        y = max(math.floor(self.y * factor), 0)
        w = min(math.ceil(self.w * factor), width - x)
        h = min(math.ceil(self.h * factor), height - y)
        return Rect(x, y, max(w, 1), max(h, 1))

    def cover_factor(self, other: Rect) -> float:
        """How much this rectangle must be enlarged to fill `other`."""
        return max(other.w / self.w, other.h / self.h)

    def as_list(self) -> list[int]:
        return [self.x, self.y, self.w, self.h]


def _rect(values, where: str) -> Rect:
    try:
        x, y, w, h = (int(v) for v in values)
    except (TypeError, ValueError) as exc:
        raise CropError(f"{where} must be [x, y, w, h], got {values!r}") from exc
    if w <= 0 or h <= 0:
        raise CropError(f"{where} has a non-positive size: {values!r}")
    return Rect(x, y, w, h)


def _pair(values, where: str) -> tuple[int, int]:
    try:
        width, height = (int(v) for v in values)
    except (TypeError, ValueError) as exc:
        raise CropError(f"{where} must be [width, height], got {values!r}") from exc
    if width <= 0 or height <= 0:
        raise CropError(f"{where} has a non-positive size: {values!r}")
    return width, height


# --------------------------------------------------------------------------
# templates
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Region:
    name: str
    src: Rect
    dst: Rect
    #: None means "use the template's default", which is `render.crop.fit`.
    fit: str | None = None

    def fit_or(self, default: str) -> str:
        return self.fit or default


@dataclass(frozen=True)
class Template:
    name: str
    source_resolution: tuple[int, int]
    output: tuple[int, int]
    regions: tuple[Region, ...]

    @property
    def output_rect(self) -> Rect:
        return Rect(0, 0, self.output[0], self.output[1])

    @property
    def source_aspect(self) -> float:
        return self.source_resolution[0] / self.source_resolution[1]


def parse_template(name: str, data: dict) -> Template:
    if not isinstance(data, dict):
        raise CropError(f"template {name!r} must be a mapping, got {type(data).__name__}")

    regions_raw = data.get("regions") or []
    if not regions_raw:
        raise CropError(f"template {name!r} has no regions")

    regions = []
    for index, entry in enumerate(regions_raw):
        if not isinstance(entry, dict):
            raise CropError(f"{name}.regions[{index}] must be a mapping")
        region_name = str(entry.get("name") or f"region{index}")
        fit = entry.get("fit")
        if fit is not None and str(fit) not in FITS:
            raise CropError(
                f"{name}.{region_name}.fit must be one of {FITS}, got {fit!r}"
            )
        regions.append(Region(
            name=region_name,
            src=_rect(entry.get("src"), f"{name}.{region_name}.src"),
            dst=_rect(entry.get("dst"), f"{name}.{region_name}.dst"),
            fit=None if fit is None else str(fit),
        ))

    return Template(
        name=name,
        source_resolution=_pair(
            data.get("source_resolution"), f"{name}.source_resolution"),
        output=_pair(data.get("output"), f"{name}.output"),
        regions=tuple(regions),
    )


def load_templates(cfg) -> dict[str, Template]:
    """Read the sidecar named by `render.crop.file`.

    Same shape as `phrases.yaml`: a file named from `clipforge.yaml`, resolved
    against the config directory, parsed by the module that owns it.
    """
    path = Path(cfg.get("render.crop.file", "crop_templates.yaml"))
    if not path.is_absolute():
        path = CONFIG_DIR / path
    if not path.is_file():
        raise CropError(f"crop templates not found: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    templates = data.get("templates") or {}
    if not templates:
        raise CropError(f"{path} declares no templates")
    return {str(name): parse_template(str(name), body)
            for name, body in templates.items()}


def get_template(cfg, name: str | None = None) -> Template:
    templates = load_templates(cfg)
    wanted = str(name or cfg.get("render.crop.template"))
    if wanted not in templates:
        raise CropError(
            f"no crop template {wanted!r}. Available: "
            f"{', '.join(sorted(templates))}. They live in "
            f"{cfg.get('render.crop.file', 'crop_templates.yaml')}."
        )
    return templates[wanted]


# --------------------------------------------------------------------------
# resolving against a real source
# --------------------------------------------------------------------------


def resolve(template: Template, width: int, height: int) -> Template:
    """Move a template onto the source it is actually being applied to.

    A matching resolution returns the template unchanged, so the common case
    involves no arithmetic at all and cannot drift.
    """
    declared_w, declared_h = template.source_resolution
    if (width, height) == (declared_w, declared_h):
        return template

    actual_aspect = width / height if height else 0.0
    if abs(actual_aspect - template.source_aspect) > ASPECT_TOLERANCE:
        raise CropError(
            f"template {template.name!r} was measured on "
            f"{declared_w}x{declared_h} ({template.source_aspect:.3f}) and this "
            f"source is {width}x{height} ({actual_aspect:.3f}). Scaling x and y "
            f"by different factors would move every region somewhere you did "
            f"not put it, so this is refused rather than guessed. Measure a "
            f"template on this canvas, or set source_resolution to match."
        )

    factor = width / declared_w
    return replace(
        template,
        source_resolution=(width, height),
        regions=tuple(
            replace(region, src=region.src.scaled(factor, (width, height)))
            for region in template.regions
        ),
    )


@dataclass(frozen=True)
class Check:
    """What validation found. Warnings are reported; errors raise."""

    warnings: tuple[str, ...] = ()
    #: Output area not covered by any region, in pixels.
    uncovered: int = 0
    #: Regions whose source rectangle is smaller than its destination.
    upscaled: tuple[str, ...] = ()


def validate(
    template: Template,
    width: int,
    height: int,
    *,
    upscale_warn_factor: float = 2.0,
) -> Check:
    """Everything §8.4 leaves unchecked. Raises on the unrecoverable ones.

    `upscale_warn_factor` exists because *some* upscaling is inherent: a 16:9
    master reframed to 9:16 always enlarges by about 1.78x, so warning on any
    enlargement at all would fire on every clip and mean nothing. The number
    worth hearing about is a region blown up far past that -- a 360x270 facecam
    filling 1080x810 is 3x, and that one is genuinely soft.
    """
    output = template.output_rect
    if output.w % 2 or output.h % 2:
        raise CropError(
            f"template {template.name!r} output {output.w}x{output.h} has an odd "
            f"dimension; libx264 with yuv420p rejects it outright"
        )

    warnings: list[str] = []
    upscaled: list[str] = []

    for region in template.regions:
        if not region.src.inside(width, height):
            raise CropError(
                f"{template.name}.{region.name}.src {region.src.as_list()} falls "
                f"outside the {width}x{height} source"
            )
        if not region.dst.inside(output.w, output.h):
            raise CropError(
                f"{template.name}.{region.name}.dst {region.dst.as_list()} falls "
                f"outside the {output.w}x{output.h} output"
            )
        if region.src.cover_factor(region.dst) > upscale_warn_factor:
            upscaled.append(
                f"{region.name} ({region.src.cover_factor(region.dst):.1f}x)")

    for index, region in enumerate(template.regions):
        for other in template.regions[index + 1:]:
            overlap = region.dst.intersect(other.dst)
            if overlap is not None:
                raise CropError(
                    f"{template.name}: {region.name} and {other.name} overlap on "
                    f"the output at {overlap.as_list()}. One would be drawn over "
                    f"the other, so part of what the template says to show could "
                    f"not be seen -- there is no layout where that is intended."
                )

    covered = sum(region.dst.area for region in template.regions)
    uncovered = max(output.area - covered, 0)
    if uncovered:
        share = uncovered / output.area * 100
        warnings.append(
            f"{share:.1f}% of the output is not covered by any region and will "
            f"be render.crop.pad_color. That is a legitimate letterbox and also "
            f"exactly what a typo looks like, so it is reported rather than "
            f"refused."
        )
    if upscaled:
        warnings.append(
            f"upscaling {', '.join(upscaled)} past "
            f"{upscale_warn_factor:g}x -- that region will be visibly soft"
        )

    return Check(warnings=tuple(warnings), uncovered=uncovered,
                 upscaled=tuple(upscaled))


# --------------------------------------------------------------------------
# the filter graph
# --------------------------------------------------------------------------


def pad_colour(colour: str) -> str:
    """`#RRGGBB` -> `0xRRGGBB`.

    ffmpeg accepts several colour spellings; `0x` is the one that cannot be
    confused with a comment, a filter separator, or a named colour.
    """
    text = str(colour).strip()
    match = _HEX.match(text)
    if match:
        return f"0x{match.group(1).upper()}"
    if re.fullmatch(r"[A-Za-z]+", text):
        return text.lower()          # a named ffmpeg colour, e.g. `black`
    raise CropError(f"pad colour must be #RRGGBB or an ffmpeg colour name, got {colour!r}")


def region_chain(region: Region, fit: str, pad: str) -> str:
    """crop -> scale, in whichever way `fit` asks for."""
    src, dst = region.src, region.dst
    crop = f"crop={src.w}:{src.h}:{src.x}:{src.y}"

    if fit == STRETCH:
        return f"{crop},scale={dst.w}:{dst.h}"
    if fit == FILL:
        # Cover the destination, then take the middle. `crop` centres by
        # default, which is what "no tracking jitter" means here.
        return (f"{crop},scale={dst.w}:{dst.h}:force_original_aspect_ratio=increase"
                f",crop={dst.w}:{dst.h}")
    if fit == CONTAIN:
        return (f"{crop},scale={dst.w}:{dst.h}:force_original_aspect_ratio=decrease"
                f",pad={dst.w}:{dst.h}:(ow-iw)/2:(oh-ih)/2:{pad}")
    raise CropError(f"unknown fit {fit!r}; expected one of {FITS}")


def filter_complex(
    template: Template,
    *,
    default_fit: str = FILL,
    pad_color: str = "#000000",
    captions: str | None = None,
    label: str = "vout",
    trim: str = "",
) -> str:
    """The whole video graph: regions, composition, then captions.

    `captions` is a *bare filename* -- ffmpeg must be run from its directory.
    See `ffmpeg.filter_file` for the measurement behind that.

    Captions come last, after every scale, because libass draws onto the frame
    it is given: burning them before the composition would scale the text with
    the region (§8.3's own burn-in has the same order).
    """
    pad = pad_colour(pad_color)
    output = template.output_rect
    steps: list[str] = []

    # §8.2's filler removal, when there is any: the cuts happen BEFORE the crop
    # so everything after works on one continuous clip, and so `ass` still
    # comes last (finding 7). `edits.trim_chain` builds this already labelled
    # `[0:v]` -> `[vcut]`, and returns "" when nothing was cut -- which keeps
    # the no-cut graph exactly as it was.
    entry = "[0:v]"
    if trim:
        steps.append(trim)
        entry = "[vcut]"

    count = len(template.regions)
    if count > 1:
        outputs = "".join(f"[r_s{index}]" for index in range(count))
        steps.append(f"{entry}split={count}{outputs}")

    for index, region in enumerate(template.regions):
        source = f"[r_s{index}]" if count > 1 else entry
        chain = region_chain(region, region.fit_or(default_fit), pad)
        steps.append(f"{source}{chain}[r{index}]")

    # The first region is padded out to the full frame at its own offset, which
    # makes the canvas without needing a `color` source -- that would be an
    # infinite stream and would need a duration bound to terminate.
    first = template.regions[0].dst
    steps.append(
        f"[r0]pad={output.w}:{output.h}:{first.x}:{first.y}:{pad}[c0]"
    )
    for index in range(1, count):
        dst = template.regions[index].dst
        steps.append(f"[c{index - 1}][r{index}]overlay={dst.x}:{dst.y}[c{index}]")

    last = f"[c{count - 1}]"
    if captions:
        steps.append(f"{last}ass={captions}[{label}]")
    else:
        steps.append(f"{last}null[{label}]")

    return ";".join(steps)


def describe(template: Template, check: Check) -> list[str]:
    """Human-readable geometry, for `--dry-run` and the run log."""
    width, height = template.source_resolution
    lines = [
        f"template  {template.name}",
        f"source    {width}x{height}",
        f"output    {template.output[0]}x{template.output[1]}",
    ]
    for region in template.regions:
        lines.append(
            f"  {region.name:<12} src {str(region.src.as_list()):<24} "
            f"dst {str(region.dst.as_list()):<24} "
            f"{'' if region.fit is None else region.fit}"
        )
    lines.extend(f"  WARNING  {warning}" for warning in check.warnings)
    return lines
