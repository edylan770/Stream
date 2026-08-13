"""§8.3 — ASS captions: word groups, an active-word highlight, speaker colour.

> **Format: ASS (Advanced SubStation Alpha).** Chosen because it supports inline
> color/style overrides mid-line, which is required for word-level highlighting
> and speaker coloring.

Four things here depart from §8.3, each for a reason worth keeping.

**1. A line runs to the NEXT word's start, not to its own end.** §8.3's
pseudocode is `start, end = active_word.start, active_word.end`. A word's `end`
is the end of its *audio*, and the next word starts later — so between every
pair of words the Dialogue line expires and the screen is blank. The caption
strobes at the word rate, several times a second. Emitting on the *boundaries*
between words instead keeps the group continuously on screen with only the
highlight moving, which is what §8.3 describes in prose and what the format is
for.

**2. Groups are built per source track, and each track gets its own MarginV.**
Two people talking at once produce two simultaneous word streams (see
`render/words.py` for why `speaker` cannot decide this). Interleaving them into
one line reads as gibberish — "Oh my / no way / god" — so each role's group
stacks on its own row. With one speaker the output is identical to what a
single-style renderer would produce.

**3. Line boundaries are quantised to centiseconds ONCE, as a shared array.**
ASS carries hundredths of a second. Rounding each line's start and end
independently makes adjacent lines overlap by up to 10 ms (two copies of the
same text drawn on top of each other, which reads as a bold flicker) or leave a
10 ms hole. Quantising the boundary list means consecutive lines share an exact
edge and neither can happen.

**4. `PlayResX`/`PlayResY` are the OUTPUT resolution.** libass scales every font
and margin by `PlayRes*`. Writing the master's 1920x1080 while the video is a
1080x1920 vertical crop makes every caption the wrong size, and §8.4's whole
point is that those two differ. Related, and recorded here so nobody
"optimises" it: `ass` must come *after* `scale` in the filter chain, which is
the order §8.3's burn-in command already uses.

A3 — **ASS colour is BGR.** `&HBBGGRR&` inline, `&HAABBGGRR` in the style table.
Config takes `#RRGGBB` and `ass_colour`/`style_colour` convert. Getting this
backwards is silent: every colour still parses, they are just wrong.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from clipforge.render.words import RenderWord, by_role

#: The style used when a word has no role — §4.2 found a single track, or the
#: segment came from `mixed`. Named in config under `render.captions.styles`.
DEFAULT_STYLE = "default"

#: `[V4+ Styles]` field order. Positional, and a renderer will silently
#: misinterpret every field after the first mistake.
STYLE_FORMAT = (
    "Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
    "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, "
    "Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, "
    "MarginV, Encoding"
)

EVENT_FORMAT = (
    "Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
)

_HEX = re.compile(r"^#?([0-9a-fA-F]{6})$")

#: Words ending in one of these end a caption group when
#: `break_on_sentence_end` is on. Trailing quotes and brackets are stripped
#: first, so `"insane!"` still counts.
SENTENCE_END = ".?!"
_TRAILING = "\"')]}»”’"


class AssError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# A3 — colour
# --------------------------------------------------------------------------


def _bgr(colour: str) -> str:
    match = _HEX.match(str(colour).strip())
    if not match:
        raise AssError(
            f"colour must be #RRGGBB, got {colour!r}. ASS itself is BGR; config "
            f"takes RGB and this converts (A3)."
        )
    digits = match.group(1).upper()
    red, green, blue = digits[0:2], digits[2:4], digits[4:6]
    return f"{blue}{green}{red}"


def ass_colour(colour: str) -> str:
    """`#RRGGBB` -> `&HBBGGRR&`, the inline `\\c` override form of A3/§8.3."""
    return f"&H{_bgr(colour)}&"


def override(colour: str) -> str:
    """`#RRGGBB` -> `{\\c&HBBGGRR&}`, the full inline override token of §8.3.

    Separate from `ass_colour` because A3 names the *value* form and §8.3 uses
    the *statement* form, and emitting one where the other is meant produces a
    line that renders the raw `&H...&` as visible text.
    """
    return "{\\c" + ass_colour(colour) + "}"


def style_colour(colour: str, alpha: int = 0) -> str:
    """`#RRGGBB` -> `&HAABBGGRR`, the style-table form.

    `alpha` is ASS's *transparency*: 0 is opaque, 255 invisible. That is the
    opposite of every other alpha in this project, which is why it is named and
    defaulted here rather than written out at each call site.
    """
    if not 0 <= int(alpha) <= 255:
        raise AssError(f"alpha must be 0-255 (0 = opaque), got {alpha}")
    return f"&H{int(alpha):02X}{_bgr(colour)}"


# --------------------------------------------------------------------------
# text
# --------------------------------------------------------------------------


def escape(text: str) -> str:
    """Make transcript text safe to put in a Dialogue line.

    `{` and `}` open and close an override block; an unbalanced one swallows
    the rest of the line. A literal newline ends the event. Both are escaped.

    A lone backslash immediately before `n`, `N` or `h` would also be read as an
    override (`\\N` is a hard line break). WhisperX does not emit backslashes in
    word text, so rather than mangle a character that never appears, the pair is
    separated with a zero-width space — the text still reads correctly and the
    escape cannot fire.
    """
    cleaned = text.replace("{", r"\{").replace("}", r"\}")
    cleaned = re.sub(r"[\r\n]+", " ", cleaned)
    return re.sub(r"\\(?=[nNh])", "\\\\​", cleaned)


def timestamp(seconds: float) -> str:
    """ASS's `H:MM:SS.cc`. Negative times clamp to zero rather than wrap."""
    total_cs = max(int(round(float(seconds) * 100)), 0)
    hours, rest = divmod(total_cs, 360000)
    minutes, rest = divmod(rest, 6000)
    secs, centis = divmod(rest, 100)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{centis:02d}"


# --------------------------------------------------------------------------
# grouping
# --------------------------------------------------------------------------


def _ends_sentence(text: str) -> bool:
    return text.rstrip(_TRAILING).endswith(tuple(SENTENCE_END))


def _rebalance(groups: list[list[RenderWord]], minimum: int) -> list[list[RenderWord]]:
    """Stop a run of words ending in a one-word group.

    §8.3 asks for 3-5 words on screen. Greedy chunking of 9 words at size 4
    leaves 4/4/1, and a single word alone on screen for a beat looks like a
    bug. Words move back from the previous group until both clear `minimum`;
    when there are not enough words for two groups, they become one.
    """
    if len(groups) < 2:
        return groups
    last, previous = groups[-1], groups[-2]
    if len(last) >= minimum:
        return groups
    if len(previous) + len(last) < minimum * 2:
        groups[-2] = previous + last
        groups.pop()
        return groups
    while len(last) < minimum and len(previous) > minimum:
        last.insert(0, previous.pop())
    return groups


def group_words(
    words: list[RenderWord],
    *,
    size: int = 4,
    minimum: int = 3,
    max_gap_s: float = 1.2,
    break_on_sentence_end: bool = True,
) -> list[list[RenderWord]]:
    """§8.3's `chunk_words`, with the breaks a transcript actually needs.

    A group also ends at a silence longer than `max_gap_s` — otherwise a word
    before a four-second pause shares the screen with the word after it, and the
    highlight sits on a word nobody is saying — and at a sentence boundary,
    because a group spanning a full stop reads as one sentence.
    """
    if size < 1:
        raise AssError(f"render.captions.group_size must be >= 1, got {size}")
    minimum = max(1, min(minimum, size))

    groups: list[list[RenderWord]] = []
    current: list[RenderWord] = []
    for word in words:
        if current:
            gap = word.start - current[-1].end
            if gap > max_gap_s or len(current) >= size or (
                break_on_sentence_end and _ends_sentence(current[-1].text)
            ):
                groups.append(current)
                current = []
        current.append(word)
    if current:
        groups.append(current)

    return _rebalance(groups, minimum)


# --------------------------------------------------------------------------
# styles
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Style:
    """One §8.3 speaker colour, as an ASS style row."""

    name: str
    colour: str
    outline_colour: str = "#000000"
    #: Distance from the edge named by `alignment`. This is what stacks two
    #: simultaneous speakers rather than letting them collide.
    margin_v: int = 220
    font: str = "Arial"
    font_size: int = 64
    bold: bool = True
    outline: float = 3.0
    shadow: float = 0.0
    #: Numpad layout. 2 is bottom-centre.
    alignment: int = 2
    margin_l: int = 60
    margin_r: int = 60
    border_style: int = 1

    def to_line(self) -> str:
        primary = style_colour(self.colour)
        return "Style: " + ",".join(str(field_value) for field_value in (
            self.name, self.font, self.font_size,
            primary,                                  # PrimaryColour
            primary,                                  # SecondaryColour (unused)
            style_colour(self.outline_colour),        # OutlineColour
            style_colour(self.outline_colour, 128),   # BackColour (shadow)
            -1 if self.bold else 0, 0, 0, 0,          # Bold Italic Underline Strike
            100, 100, 0, 0,                           # ScaleX ScaleY Spacing Angle
            self.border_style,
            _number(self.outline), _number(self.shadow),
            self.alignment, self.margin_l, self.margin_r, self.margin_v,
            1,                                        # Encoding: 1 = default
        ))


def _number(value: float) -> str:
    """ASS accepts either; keep integers integral so the file diffs cleanly."""
    as_float = float(value)
    return str(int(as_float)) if as_float.is_integer() else f"{as_float:g}"


# --------------------------------------------------------------------------
# the document
# --------------------------------------------------------------------------


@dataclass
class Script:
    """An ASS file being built. `render()` is the only output."""

    width: int
    height: int
    styles: dict[str, Style] = field(default_factory=dict)
    #: `(start_cs, dialogue_line)`. The key is kept beside the text so the file
    #: can be ordered by time numerically — sorting the rendered strings would
    #: work only while every clip is under ten hours, which is true and is
    #: exactly the kind of assumption that stops being true silently.
    events: list[tuple[int, str]] = field(default_factory=list)
    wrap_style: int = 0
    #: Counted while emitting, so the caller can report rather than guess.
    lines: int = 0
    skipped: int = 0

    def header(self) -> list[str]:
        return [
            "[Script Info]",
            "; Generated by ClipForge (spec §8.3). Do not edit by hand.",
            "ScriptType: v4.00+",
            # The output canvas, NOT the master's — see the module docstring.
            f"PlayResX: {self.width}",
            f"PlayResY: {self.height}",
            f"WrapStyle: {self.wrap_style}",
            # Without this, outline and shadow are in raw pixels and do not
            # follow PlayRes*, so the same config looks different per preset.
            "ScaledBorderAndShadow: yes",
            "YCbCr Matrix: None",
            "",
        ]

    def render(self) -> str:
        parts = self.header()
        parts.append("[V4+ Styles]")
        parts.append(f"Format: {STYLE_FORMAT}")
        parts.extend(style.to_line() for style in self.styles.values())
        parts.append("")
        parts.append("[Events]")
        parts.append(f"Format: {EVENT_FORMAT}")
        parts.extend(line for _, line in sorted(self.events, key=lambda e: e[0]))
        # ASS files are CRLF by convention and libass accepts either; a trailing
        # newline keeps the last event from being dropped by strict parsers.
        return "\n".join(parts) + "\n"


def _dialogue(start: float, end: float, style: str, text: str) -> str:
    return (
        f"Dialogue: 0,{timestamp(start)},{timestamp(end)},{style},,0,0,0,,{text}"
    )


def _group_text(group: list[RenderWord], active: int, base: str, highlight: str,
                uppercase: bool) -> str:
    """§8.3's generation step: the whole group, one word overridden.

    The style already carries `base` as its PrimaryColour, so only the active
    word needs an override — and it has to be closed again, or every following
    word inherits the highlight.
    """
    pieces = []
    for index, word in enumerate(group):
        text = escape(word.text.upper() if uppercase else word.text)
        if index == active:
            pieces.append(f"{override(highlight)}{text}{override(base)}")
        else:
            pieces.append(text)
    return " ".join(pieces)


def _boundaries(group: list[RenderWord], group_end: float) -> list[int]:
    """Line edges in centiseconds, shared between adjacent lines.

    One rounding for the whole group, so consecutive lines meet exactly instead
    of overlapping or leaving a hole (see the module docstring).
    """
    edges = [int(round(word.start * 100)) for word in group]
    edges.append(int(round(group_end * 100)))
    monotonic: list[int] = []
    for edge in edges:
        monotonic.append(edge if not monotonic else max(edge, monotonic[-1]))
    return monotonic


def emit_group(
    script: Script,
    group: list[RenderWord],
    style_name: str,
    *,
    group_end: float,
    base_colour: str,
    highlight_colour: str,
    min_line_s: float,
    uppercase: bool,
) -> None:
    """One Dialogue line per active-word state, per §8.3."""
    edges = _boundaries(group, group_end)
    min_cs = max(int(round(min_line_s * 100)), 0)

    for index in range(len(group)):
        start_cs, end_cs = edges[index], edges[index + 1]
        if end_cs - start_cs < min_cs:
            # Two words sharing a start — interpolated ones do this. The word
            # is still on screen as part of the group; it just never becomes
            # the highlighted one, which is better than a 4 ms flash.
            script.skipped += 1
            continue
        script.events.append((start_cs, _dialogue(
            start_cs / 100.0, end_cs / 100.0, style_name,
            _group_text(group, index, base_colour, highlight_colour, uppercase),
        )))
        script.lines += 1


# --------------------------------------------------------------------------
# top level
# --------------------------------------------------------------------------


def build(
    words: list[RenderWord] | tuple[RenderWord, ...],
    *,
    width: int,
    height: int,
    styles: dict[str, Style],
    highlight: str,
    duration: float,
    group_size: int = 4,
    min_group_size: int = 3,
    max_gap_s: float = 1.2,
    tail_hold_s: float = 0.35,
    min_line_s: float = 0.08,
    uppercase: bool = False,
    wrap_style: int = 0,
) -> Script:
    """The whole caption track for one clip.

    `duration` is the finished clip's length in output seconds — a caption is
    clamped to it so a group holding past the last frame cannot exist.
    """
    if DEFAULT_STYLE not in styles:
        raise AssError(
            f"styles must include {DEFAULT_STYLE!r}; it is what a word with no "
            f"track and no speaker falls back to"
        )

    script = Script(width=width, height=height, styles=styles,
                    wrap_style=wrap_style)

    for role, stream in sorted(by_role(list(words)).items()):
        style_name = role if role in styles else DEFAULT_STYLE
        style = styles[style_name]
        groups = group_words(
            stream, size=group_size, minimum=min_group_size,
            max_gap_s=max_gap_s,
        )
        for index, group in enumerate(groups):
            # Hold the group briefly past the last word so a sentence does not
            # vanish on its final syllable — but never into the next group, and
            # never past the end of the clip.
            limit = groups[index + 1][0].start if index + 1 < len(groups) else duration
            group_end = min(group[-1].end + tail_hold_s, limit, duration)
            if group_end <= group[0].start:
                script.skipped += len(group)
                continue
            emit_group(
                script, group, style_name,
                group_end=group_end,
                base_colour=style.colour,
                highlight_colour=highlight,
                min_line_s=min_line_s,
                uppercase=uppercase,
            )

    return script


def styles_from_config(cfg, *, roles: tuple[str, ...] = ()) -> dict[str, Style]:
    """Build the style table from `render.captions`.

    Per-style keys override the shared ones, so a config that only sets a colour
    per speaker still gets the same font, size and outline everywhere — which is
    the common case, and the one where two styles drifting apart would look like
    a bug rather than a choice.
    """
    def shared(key: str, default):
        return cfg.get(f"render.captions.{key}", default)

    declared = dict(shared("styles", {}) or {})
    for role in roles:
        declared.setdefault(role, {})
    declared.setdefault(DEFAULT_STYLE, {})

    styles: dict[str, Style] = {}
    for name, overrides in declared.items():
        settings = dict(overrides or {})
        styles[name] = Style(
            name=name,
            colour=settings.get("colour", shared("colour", "#FFFFFF")),
            outline_colour=settings.get(
                "outline_colour", shared("outline_colour", "#000000")),
            margin_v=int(settings.get("margin_v", shared("margin_v", 220))),
            font=str(settings.get("font", shared("font", "Arial"))),
            font_size=int(settings.get("font_size", shared("font_size", 64))),
            bold=bool(settings.get("bold", shared("bold", True))),
            outline=float(settings.get("outline", shared("outline", 3.0))),
            shadow=float(settings.get("shadow", shared("shadow", 0.0))),
            alignment=int(settings.get("alignment", shared("alignment", 2))),
            margin_l=int(settings.get("margin_l", shared("margin_l", 60))),
            margin_r=int(settings.get("margin_r", shared("margin_r", 60))),
            border_style=int(
                settings.get("border_style", shared("border_style", 1))),
        )
    return styles


def build_from_config(cfg, words, *, width: int, height: int, duration: float,
                      roles: tuple[str, ...] = ()) -> Script:
    """`build`, with every knob read from `render.captions`."""
    def setting(key: str, default):
        return cfg.get(f"render.captions.{key}", default)

    return build(
        words,
        width=width,
        height=height,
        styles=styles_from_config(cfg, roles=roles),
        highlight=str(setting("highlight", "#FFD400")),
        duration=duration,
        group_size=int(setting("group_size", 4)),
        min_group_size=int(setting("min_group_size", 3)),
        max_gap_s=float(setting("max_gap_s", 1.2)),
        tail_hold_s=float(setting("tail_hold_s", 0.35)),
        min_line_s=float(setting("min_line_s", 0.08)),
        uppercase=bool(setting("uppercase", False)),
        wrap_style=int(setting("wrap_style", 0)),
    )
