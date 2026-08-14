"""§8.2's export presets -- per-platform encode settings.

> | Export presets | Per-platform encode settings | Equal |

"Equal" is §8.2's own verdict on how much this buys, and the honest description
of what shipped is that **the three presets are nearly identical**. Shorts,
TikTok and Reels all take 1080x1920 and all normalise to about -14 LUFS, so the
only thing that genuinely differs is a maximum duration -- and that is a number
that changes when a platform feels like changing it. The values here are
guesses recorded as such in spec/GUESSES.md; check them against the platform's
current documentation before trusting one.

Rather than invent differences to justify three blocks, the config says they
are the same and leaves the mechanism in place for when one of them is not.

**A preset carries encode settings, not a resolution.** §8.4 puts `output` in
the crop template, and the template's `dst` rectangles are expressed in that
space -- a preset that also declared a size would give one number two sources
of truth. A platform that genuinely wanted a different frame gets a template.

**`max_duration_s` warns and never truncates.** Silently cutting an approved
clip loses the payoff, which is the part the operator approved it for (C2).
"""

from __future__ import annotations

from dataclasses import dataclass

from clipforge.render import RenderError

#: Keys a preset may override. Everything else about the encode comes from
#: `render.video`, so a preset stays a short list of differences rather than a
#: second copy of the settings.
OVERRIDABLE = (
    "codec", "crf", "preset", "pix_fmt",
    "audio_codec", "audio_bitrate", "container",
)


class PresetError(RenderError):
    pass


@dataclass(frozen=True)
class Preset:
    name: str
    settings: dict
    #: None means no cap. Warned about, never enforced by truncation.
    max_duration_s: float | None = None
    note: str = ""

    def get(self, key: str, fallback):
        return self.settings.get(key, fallback)


def load(cfg, name: str | None = None) -> Preset:
    """The named preset, or `render.presets.default`."""
    block = cfg.get("render.presets", {}) or {}
    wanted = str(name or block.get("default") or "")
    if not wanted:
        raise PresetError(
            "no preset named and render.presets.default is unset")

    available = {k: v for k, v in block.items() if isinstance(v, dict)}
    if wanted not in available:
        raise PresetError(
            f"no preset {wanted!r}. Available: {', '.join(sorted(available))}. "
            f"They live under `render.presets` in clipforge.yaml."
        )

    body = dict(available[wanted])
    cap = body.pop("max_duration_s", None)
    note = str(body.pop("note", ""))
    unknown = sorted(set(body) - set(OVERRIDABLE))
    if unknown:
        raise PresetError(
            f"preset {wanted!r} sets {unknown}, which a preset does not "
            f"control. Encode settings are {sorted(OVERRIDABLE)}; the output "
            f"resolution belongs to the crop template (§8.4)."
        )

    return Preset(
        name=wanted,
        settings=body,
        max_duration_s=None if cap is None else float(cap),
        note=note,
    )


def setting(cfg, preset: Preset | None, key: str, fallback=None):
    """A preset's value for `key`, falling back to `render.video.<key>`.

    One lookup order in one place, so a preset that forgets a key behaves the
    same as no preset at all rather than encoding with a Python default nobody
    wrote down.
    """
    base = cfg.get(f"render.video.{key}", fallback)
    return preset.get(key, base) if preset is not None else base


def too_long(preset: Preset | None, duration_s: float) -> str:
    """A warning when the clip outruns the preset's cap, or ''."""
    if preset is None or preset.max_duration_s is None:
        return ""
    if duration_s <= preset.max_duration_s:
        return ""
    return (
        f"the clip is {duration_s:.1f}s and {preset.name} is capped at "
        f"{preset.max_duration_s:g}s. NOT truncated -- cutting an approved clip "
        f"silently would lose the moment it was approved for. Trim the window "
        f"in review, or check the platform's current limit."
    )
