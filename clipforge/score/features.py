"""Feature vectors and contribution breakdowns (A9, §6.5).

A9 requires the full vector on every candidate from the first run, "even for
signals not currently weighted", because the vectors cannot be reconstructed
later and they are the input to all future tuning. Phase 1 computes three of the
thirty-one signals `config/feature_schema.yaml` declares; the other twenty-eight
are written as explicit nulls so a Phase 1 vector and a Phase 3 vector have the
same shape and can be compared directly.

The contribution breakdown is what makes weights tunable at all. §6.2 smooths
the composite *after* summing, so the smoothed value at `t_peak` is not the sum
of `weight × signal` — the breakdown records both totals rather than quietly
disagreeing with the score it is supposed to explain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

#: Keys the breakdown adds alongside the per-signal contributions. Underscore-
#: prefixed so they cannot collide with a signal name from feature_schema.yaml.
TOTAL_RAW = "_total_raw"
TOTAL_SMOOTHED = "_total_smoothed"
SUPPRESSED_BY = "_spacing_suppressed_by"


@dataclass
class SignalTrack:
    """One scored signal, on the common grid."""

    name: str
    values: np.ndarray          # z-score for continuous, kernel level for events
    weight: float
    is_event: bool = False
    baseline: np.ndarray | None = None   # rolling mean, continuous only
    raw: np.ndarray | None = None        # pre-z values, continuous only
    #: What `raw` and `baseline` are in. Every Phase 1 signal was dBFS, so this
    #: was a literal in the context key. `mic_f0` is hertz, and "_mic_f0_db"
    #: holding 166.2 is a label stating something false.
    unit: str = "dB"

    def at(self, index: int) -> float:
        return float(self.values[index])

    def contribution(self, index: int) -> float:
        """What this signal added to the composite here.

        `nan_to_num` rather than NaN so the breakdown adds up to the number it
        is explaining: `composite_of` treats an unobserved sample as adding
        nothing, and a breakdown that disagreed with the score beside it would
        be worse than no breakdown.
        """
        value = self.at(index)
        return self.weight * (value if math.isfinite(value) else 0.0)


@dataclass
class Breakdown:
    contributions: dict[str, float] = field(default_factory=dict)
    total_raw: float = 0.0
    total_smoothed: float = 0.0
    suppressed_by: float | None = None

    def to_json_dict(self) -> dict:
        out: dict[str, float | None] = dict(self.contributions)
        out[TOTAL_RAW] = round(self.total_raw, 6)
        out[TOTAL_SMOOTHED] = round(self.total_smoothed, 6)
        if self.suppressed_by is not None:
            out[SUPPRESSED_BY] = round(self.suppressed_by, 3)
        return out


def breakdown(
    tracks: list[SignalTrack], index: int, smoothed_value: float,
    suppressed_by: float | None = None,
) -> Breakdown:
    """Per-signal contribution at `t_peak`, sorted by magnitude.

    Sorted because the question being answered is always "what made this score
    what it did", and the answer is the first two or three entries.

    **Unweighted signals are excluded.** A9 requires them in the feature vector,
    so `build_tracks` loads them at weight 0 — but a row of `0.000` in the review
    UI's `?` panel is noise in the one place that explains the number beside it.
    The vector answers "what was observed"; this answers "what moved the score".
    """
    contributions = {
        track.name: round(track.contribution(index), 6)
        for track in tracks
        if track.weight
    }
    ordered = dict(sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True))
    return Breakdown(
        contributions=ordered,
        total_raw=float(sum(contributions.values())),
        total_smoothed=float(smoothed_value),
        suppressed_by=suppressed_by,
    )


def vector(schema, tracks: list[SignalTrack], index: int) -> dict[str, float | None]:
    """The full A9 vector: every declared signal, null where not computed.

    Values follow what `feature_schema.yaml` documents — the rolling z-score for
    continuous signals, the decayed kernel level for events — so the numbers
    mean the same thing in Phase 3 as they do now.

    Signals with gaps record **null**, not NaN and not zero. `json.dumps` writes
    a bare `NaN`, which is not valid JSON and which the review UI's `JSON.parse`
    rejects outright — so one unvoiced sample would have taken the whole review
    screen down. Null is also what `feature_schema.yaml` already means by "not
    computed", and "there was no pitch here" is exactly that.
    """
    out = schema.empty_vector()
    for track in tracks:
        if track.name in out:
            value = track.at(index)
            out[track.name] = round(value, 6) if math.isfinite(value) else None
    return out


def unweighted_context(tracks: list[SignalTrack], index: int) -> dict[str, float]:
    """Absolute levels behind the z-scores, for the review UI's `?` panel.

    "mic_rms z = 2.8" is only meaningful next to "which is −18 dB against a
    −34 dB baseline". §14's weight tuning needs the z-score; a human reading the
    breakdown needs the dB.

    Deliberately NOT merged into `feature_vector`: A9's whole value is that a
    Phase 1 vector and a Phase 3 vector have identical shape, so that field
    holds exactly the keys `feature_schema.yaml` declares and nothing else.
    This belongs with the human-facing breakdown.
    """
    context: dict[str, float | None] = {}
    for track in tracks:
        suffix = track.unit.lower()
        if track.raw is not None:
            context[f"_{track.name}_{suffix}"] = _finite(track.raw[index])
        if track.baseline is not None:
            context[f"_{track.name}_baseline_{suffix}"] = _finite(track.baseline[index])
    return context


def _finite(value) -> float | None:
    """Null rather than NaN — this dict is serialised to JSON and read by a
    browser, and `JSON.parse` refuses a bare NaN."""
    number = float(value)
    return round(number, 2) if math.isfinite(number) else None
