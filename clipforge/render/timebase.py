"""The frame grid.

HANDOFF calls this the hard part of §10.5, and it is:

> FCPXML times must be exact integer multiples of a rational `frameDuration`
> (29.97 is `1001/30000`, not `1/29.97`), or Resolve rejects the file or
> silently rounds.

Three rules follow, and all three are the kind that look like pedantry until a
timeline is a frame out.

**Everything is a `Fraction`.** 1/29.97 is 0.033367... and no float sum of those
lands back on a frame boundary. `Fraction(1001, 30000)` does, exactly, forever.

**Values are written unreduced, over the format's timescale.** `Fraction` would
happily reduce 30 frames at 29.97 to `1001/1000s`. That is the same instant and
exactly the sort of thing a naive parser mishandles; Final Cut itself emits
`3603600/30000s`, keeping the timescale as the denominator, so that is what gets
written.

**Starts floor and ends ceil, never round-to-nearest.** Rounding a boundary to
the closest frame can eat the first or last frame of a moment. C2 is explicit
that a false positive costs three seconds of review and a false negative costs a
clip, so boundaries expand onto the grid rather than contracting onto it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

#: How a real time is placed onto the frame grid.
DOWN = "down"      # floor — clip starts
UP = "up"          # ceil  — clip ends
NEAREST = "nearest"

#: How close to a frame boundary counts as being on it, measured in frames.
#:
#: MEASURED, not guessed. `candidates.t_start` is SQLite REAL, so a time that
#: was computed as an exact multiple of 1001/30000 comes back as the nearest
#: double, which is a hair off in either direction. Frame 250 at 29.97 is
#: 250250/30000 s exactly; the double is larger, so `ceil` returned 251 and a
#: frame-aligned window silently grew by a frame at each end.
#:
#: Double error on a time under a few hours is ~1e-12 s, which is ~3e-11 frames.
#: 1e-9 frames is comfortably above that and ~30 picoseconds below anything a
#: person could mean, so it separates float noise from a genuine sub-frame
#: boundary without ever swallowing one.
BOUNDARY_EPSILON = Fraction(1, 10**9)


class TimebaseError(ValueError):
    pass


@dataclass(frozen=True)
class TimeBase:
    """A constant frame rate, as the exact rational it actually is.

    `num`/`den` are `streams.fps_num`/`fps_den` — 30000/1001 for 29.97, 60/1 for
    sixty. Note the inversion: the *rate* is num/den, so the *frame duration* is
    den/num.
    """

    num: int
    den: int = 1

    def __post_init__(self) -> None:
        if self.num <= 0 or self.den <= 0:
            raise TimebaseError(
                f"frame rate must be positive, got {self.num}/{self.den}. "
                f"A stream with no fps_num has not been probed."
            )

    @classmethod
    def from_stream(cls, row) -> TimeBase:
        num, den = row["fps_num"], row["fps_den"]
        if num is None or den is None:
            raise TimebaseError(
                f"stream {row['id']!r} has no exact frame rate recorded. "
                f"Run `clipforge run {row['id']} --only probe`."
            )
        return cls(int(num), int(den))

    @property
    def rate(self) -> Fraction:
        """Frames per second."""
        return Fraction(self.num, self.den)

    @property
    def frame_duration(self) -> Fraction:
        """Seconds per frame — what FCPXML calls frameDuration."""
        return Fraction(self.den, self.num)

    # -- placing real times onto the grid ----------------------------------

    def frame_at(self, seconds: float, mode: str = NEAREST) -> int:
        """The frame index a time falls on.

        `seconds` comes from the database as a float, so it is converted
        exactly (`Fraction(float)` is lossless) and only then divided. Doing the
        division in float first is what introduces the drift this module exists
        to avoid.

        A time within `BOUNDARY_EPSILON` of a frame is *on* that frame for every
        mode. Without that, `float(exact_frame_time)` lands a double's-width
        above the true rational and `ceil` adds a frame that is not there.
        """
        exact = Fraction(seconds) / self.frame_duration
        nearest = round(exact)
        if abs(exact - nearest) <= BOUNDARY_EPSILON:
            return int(nearest)

        if mode == DOWN:
            return math.floor(exact)
        if mode == UP:
            return math.ceil(exact)
        if mode == NEAREST:
            return math.floor(exact + Fraction(1, 2))
        raise TimebaseError(f"unknown rounding mode {mode!r}")

    def seconds(self, frames: int) -> Fraction:
        """The exact time of a frame index."""
        return frames * self.frame_duration

    # -- writing them out --------------------------------------------------

    def value(self, frames: int) -> str:
        """An FCPXML time attribute for a whole number of frames.

        Deliberately not reduced: the denominator stays the format's timescale,
        which is what Final Cut emits and what every parser therefore expects.
        Zero is the literal `0s`, which is the one form the format spells out.
        """
        if frames == 0:
            return "0s"
        return f"{frames * self.den}/{self.num}s"

    @property
    def frame_duration_value(self) -> str:
        return self.value(1)

    def describe(self) -> str:
        rate = float(self.rate)
        exact = "" if self.den == 1 else f" ({self.num}/{self.den})"
        return f"{rate:.3f} fps{exact}".replace(".000 fps", " fps")


def parse_value(text: str) -> Fraction:
    """Read an FCPXML time attribute back. Used by the tests to verify output.

    Accepts both forms the format allows: `"1001/30000s"` and `"5s"`.
    """
    body = text.strip()
    if not body.endswith("s"):
        raise TimebaseError(f"not an FCPXML time value: {text!r}")
    body = body[:-1]
    if "/" in body:
        numerator, _, denominator = body.partition("/")
        return Fraction(int(numerator), int(denominator))
    return Fraction(int(body))
