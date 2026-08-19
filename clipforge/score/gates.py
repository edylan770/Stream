"""§6.4's negative signals — ONE gating helper, not per-signal ad hoc logic.

§6.4 is the most prescriptive section in the spec:

> "**This is a specific requirement and must be implemented exactly as
> described.** Negative signals must never fire when audio indicates something
> is happening. Audio is ground truth for 'is anything going on.'"

and it closes with a structural instruction rather than a numeric one:

> "**Generalized rule:** all negative signals are conditional on audio being
> flat. Implement as a shared gating helper, not per-signal ad hoc logic."

So there is exactly one helper here — `sustained` — and each negative is that
helper plus its own condition. A third negative (§6.4 names loading screens and
flat-prosody monologue) is a condition and a config row, not new machinery.

THREE THINGS §6.4 DOES NOT SAY, AND ONE IT DOES

**How large a penalty is.** The composite is a *signed* sum of weighted
z-scores, so a multiplicative penalty is wrong in a way nothing downstream could
detect: `0.5 × −1.8` is `−0.9`, which *raises* the composite everywhere it is
already negative — precisely where a menu or an AFK stretch sits. The penalty is
subtracted. Its magnitude (`score.negatives.*_penalty`) is arbitrary and is
recorded as such in `spec/GUESSES.md`.

**What "no input activity" means.** §4.4's logger writes keys, clicks *and*
mouse velocity, and a mouse moving with no clicks is input activity. Both must
be idle. That makes the penalty harder to fire, which is the direction C2 wants:
a false AFK penalty costs a clip, a missed one costs a candidate the operator
skips in three seconds.

**How long the ramp is.** §6.4 says "ramping in gradually (not a step
function)" and names no duration; §17's table has no row for one.
`score.negatives.ramp_s` is new and arbitrary — but not entirely free: §6.2
smooths the composite with sigma = 2 s *after* the penalty is applied, so a ramp
much shorter than that is smoothed into something indistinguishable from a step
and the requirement becomes decorative.

**What it does say, twice, is that a gap is not a zero.** Neither penalty is
computed at all unless its inputs exist. A stream with no input log has UNKNOWN
input, not idle input — reading absence as idleness is how the AFK penalty would
swallow a whole stream, and it is exactly what commit 35's `input_coverage`
exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from clipforge.score import derived, grid

#: §6.4's two negatives. Both names are already declared in
#: `config/feature_schema.yaml`, so the gate level reaches A9's vector under the
#: key the schema has always had for it.
MENU = "menu_screen"
AFK = "afk"

#: The tracks §6.4 calls "audio", i.e. the ones that can carry speech. Game
#: audio is deliberately not among them: menu music is not somebody talking, and
#: §6.4's whole point is that *speech* is the ground truth for "something is
#: happening".
VOICE_KINDS = ("mic_rms", "party_rms")

#: What the AFK gate needs before it will claim anything. `input_coverage` is
#: not optional — see the module docstring.
AFK_KINDS = ("input_rate", "mouse_velocity", "input_coverage")


@dataclass(frozen=True)
class Penalty:
    """One evaluated negative: its 0..1 gate level and what it costs."""

    name: str
    values: np.ndarray
    weight: float

    @property
    def fired(self) -> bool:
        return bool(np.any(self.values > 0.0))

    @property
    def share(self) -> float:
        """Fraction of the stream under this penalty at all."""
        if self.values.size == 0:
            return 0.0
        return float(np.mean(self.values > 0.0))


# --------------------------------------------------------------------------
# THE shared helper (§6.4's "not per-signal ad hoc logic")
# --------------------------------------------------------------------------


def run_length(active: np.ndarray) -> np.ndarray:
    """How many consecutive true samples end at each sample.

    §6.4 says "for >= 60 CONTINUOUS seconds" and "no speech detected for >= 8
    seconds". Both are run lengths, and both reset the instant the condition
    breaks — which is also §6.4's "ANY speech detected -> reset the timer to
    zero".
    """
    flags = np.asarray(active, dtype=bool)
    if flags.size == 0:
        return np.zeros(0, dtype=np.int64)
    index = np.arange(1, flags.size + 1, dtype=np.int64)
    last_break = np.maximum.accumulate(np.where(flags, 0, index))
    return np.where(flags, index - last_break, 0)


def sustained(active: np.ndarray, *, sustain_samples: int, ramp_samples: int) -> np.ndarray:
    """§6.4's shape, for every negative that will ever exist here.

    A condition that has held continuously for `sustain_samples`, then ramping
    0 → 1 over `ramp_samples`, and **removed** — not decayed — the instant the
    condition breaks. §6.4's words are "reset the timer to zero, remove
    penalty", and a penalty that faded out over a second would keep punishing a
    moment that had already started.

    `derived.ramp_up` is reused rather than reimplemented so this and
    `sudden_silence` cannot drift into ramping differently.
    """
    held = run_length(active) >= max(int(sustain_samples), 1)
    return derived.ramp_up(held, max(int(ramp_samples), 1))


# --------------------------------------------------------------------------
# the conditions
# --------------------------------------------------------------------------


def _speech_and_quiet(
    raw: dict[str, np.ndarray], inputs: derived.Inputs, baseline_samples: int
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """`(speech, energy_below_baseline, kinds_used)` over the voice tracks.

    `speech` is `derived.speech_gate` verbatim — the same test §5.4.1's
    `overlap_speech` uses. §6.4's "ANY speech detected" and §5.4.1's "VAD on
    both tracks" are the same question asked twice, and a second energy test
    here could disagree with the first with nothing to report it.

    `energy_below_baseline` is §6.4's THIRD menu condition and is not implied by
    the second: the speech gate fires above `baseline + vad.margin_db`, where
    this is the bare rolling baseline. Strictly stronger, and it is what §6.4
    means by "audio is ground truth for 'is anything going on'".

    **`<=`, where §6.4 writes `<`.** A stretch pinned at `extract.rms.db_floor`
    is exactly equal to its own rolling mean whenever the whole baseline window
    is pinned too — so the strict reading fails on digital silence, which is the
    most unambiguous "nothing is happening" audio there is. The same degenerate
    case `score.zscore_std_floor` exists for, one condition over.
    """
    used = [kind for kind in VOICE_KINDS if kind in raw]
    if not used:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=bool), []

    size = int(inputs.timeline.size)   # authoritative: every array is on this grid
    speech = np.zeros(size, dtype=bool)
    quiet = np.ones(size, dtype=bool)
    std_floor = float(inputs.cfg.get("score.zscore_std_floor"))

    for kind in used:
        rms = np.asarray(raw[kind], dtype=np.float64)
        speech |= derived.speech_gate(rms, inputs, baseline_samples)
        _z, baseline, _std = grid.rolling_zscore(rms, baseline_samples, std_floor)
        quiet &= rms <= baseline

    return speech, quiet, used


def speech_activity(
    raw: dict[str, np.ndarray], inputs: derived.Inputs, baseline_samples: int
) -> np.ndarray:
    """§6.4's "ANY speech detected", for callers outside this module.

    A thin public name over `_speech_and_quiet`, added so §9.3's chapter
    segmentation can ask the question without restating it. The union over
    `VOICE_KINDS` is exactly the hazard HANDOFF records: §6.4's "ANY speech" and
    §5.4.1's "VAD on both tracks" are the same question asked twice, a second
    energy test could disagree with the first with nothing to report it, and a
    test asserts the two agree sample for sample. A third copy in `digest/`
    would reopen that.
    """
    speech, _quiet, _used = _speech_and_quiet(raw, inputs, baseline_samples)
    return speech


def menu_active(intervals, timeline: np.ndarray) -> np.ndarray:
    """§6.4's `menu_screen_active`, from `events.t .. events.t_end`.

    §6.4 asks for a STATE and `feature_schema.yaml` declares `menu_screen` as an
    event — but `events.t_end` has been in the schema since `0001_init.sql`
    ("NULL for instantaneous"), so an interval is already expressible and
    nothing has to move. **Phase 7's vision must set `t_end`**: a row without it
    covers a single sample and can never satisfy an 8-second grace period.
    """
    active = np.zeros(timeline.size, dtype=bool)
    for start, end in intervals:
        active |= (timeline >= float(start)) & (timeline <= float(end))
    return active


def compute(
    raw: dict[str, np.ndarray], timeline: np.ndarray, cfg, *, menu_intervals=None
) -> tuple[list[Penalty], dict[str, str]]:
    """Evaluate every negative whose inputs exist.

    Returns `(penalties, reasons)`. A negative that could not be evaluated is
    absent from the first and explains itself in the second — the same shape and
    the same rule as `derived.missing_inputs`, and for the same reason: "the AFK
    penalty needs an input log" is the sentence that explains why §6.4 did
    nothing, where a silent zero would look like a measurement.

    Nothing here reads a profile. §6.5's two profiles produce two composites and
    subtract the same two arrays from both.
    """
    grid_hz = float(cfg.get("score.score_grid_hz"))
    inputs = derived.Inputs(timeline=timeline, grid_hz=grid_hz, raw=raw, cfg=cfg)
    baseline_samples = grid.window_samples(
        cfg.get("score.rolling_baseline_window_s"), grid_hz
    )

    def samples(seconds) -> int:
        return max(int(round(float(seconds) * grid_hz)), 1)

    def opt(key: str):
        return cfg.get(f"score.negatives.{key}")

    penalties: list[Penalty] = []
    reasons: dict[str, str] = {}

    speech, quiet, voice = _speech_and_quiet(raw, inputs, baseline_samples)
    if not voice:
        note = (
            f"no {' or '.join(VOICE_KINDS)} for this stream, and §6.4 gates every "
            f"negative on audio being flat — with no voice track there is nothing "
            f"to be flat"
        )
        return [], {MENU: note, AFK: note}

    ramp = samples(opt("ramp_s"))

    # -- menu / lobby ------------------------------------------------------
    if menu_intervals:
        eligible = menu_active(menu_intervals, timeline) & ~speech & quiet
        penalties.append(Penalty(
            name=MENU,
            values=sustained(eligible,
                             sustain_samples=samples(opt("menu_grace_period_s")),
                             ramp_samples=ramp),
            weight=float(opt("menu_penalty")),
        ))
    else:
        reasons[MENU] = (
            "no menu_screen events — §5.9's screen classification is Phase 7, so "
            "this gate is built and inert. Vision writes one events row per menu "
            "span, with t_end set"
        )

    # -- AFK ---------------------------------------------------------------
    absent = [kind for kind in AFK_KINDS if kind not in raw]
    if absent:
        reasons[AFK] = (
            f"needs {', '.join(absent)} — §4.4's input log has not been attached "
            f"to this stream, so its input is UNKNOWN rather than idle"
        )
    else:
        covered = np.asarray(raw["input_coverage"], dtype=np.float64) > 0.5
        rate = np.asarray(raw["input_rate"], dtype=np.float64)
        velocity = np.asarray(raw["mouse_velocity"], dtype=np.float64)

        # NaN compares False, which is the answer wanted here: an unrecorded
        # sample is not idle, it is unobserved. `covered` says the same thing
        # from the other side and both are kept — the mask is an observation
        # about the logger, the NaN is the absence of a rate.
        idle = (
            covered
            & (rate <= float(opt("afk_max_input_rate")))
            & (velocity <= float(opt("afk_max_mouse_velocity_px_s")))
        )
        penalties.append(Penalty(
            name=AFK,
            values=sustained(idle & ~speech,
                             sustain_samples=samples(opt("afk_threshold_s")),
                             ramp_samples=ramp),
            weight=float(opt("afk_penalty")),
        ))

    return penalties, reasons


# --------------------------------------------------------------------------
# applying them
# --------------------------------------------------------------------------


def apply(composite: np.ndarray, penalties: list[Penalty]) -> np.ndarray:
    """§6.2 step 6: after the weighted sum, before the smoothing.

    Subtraction, not scaling — see the module docstring. Written as its own step
    rather than folded into `composite_of`, even though addition commutes and
    the two are arithmetically identical, because §6.4's requirement is about
    the GATING and about this ordering; both should be legible in the code that
    performs them rather than inferable from a minus sign in a weight.
    """
    out = np.array(composite, dtype=np.float64, copy=True)
    for penalty in penalties:
        out -= penalty.weight * penalty.values
    return out


def as_tracks(penalties: list[Penalty]):
    """The penalties as `SignalTrack`s, so A9 and the review UI see them.

    Weight is NEGATIVE here and positive on `Penalty`, which is not an
    inconsistency: `Penalty.weight` is "what this costs" and a track's weight is
    "what it contributed". Appending these to the track list *after*
    `composite_of` has run keeps the breakdown's sum-of-contributions equal to
    the composite it is explaining.

    **Not z-scored, unlike every other continuous track.** A rolling z-score of
    a mostly-zero array does two wrong things at once: it makes the rare firing
    enormous, and it makes every *unpenalised* sample slightly positive — a
    negative signal that quietly rewards the rest of the stream. They enter on
    the same unit-peak scale the event kernels use, which is what makes §6.5's
    weights comparable. `score.zscore_std_floor_by_signal` therefore does not
    apply to them and should not.
    """
    from clipforge.score.features import SignalTrack

    return [
        SignalTrack(name=p.name, values=p.values, weight=-p.weight,
                    unit="", is_penalty=True)
        for p in penalties
    ]
