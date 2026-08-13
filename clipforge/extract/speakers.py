"""§5.8 — who said it, decided by arithmetic rather than by a model.

> Because mic and party are separate tracks, speaker identity is simply *which
> track has energy*. [...] This is more reliable than pyannote diarization and
> requires no Hugging Face token, no license acceptance, and no model.

That is right, and §16 formally rejects diarization on the same grounds. But the
pseudocode §5.8 gives is wrong for the data this project stores.

**THE dB BUG.** §5.8 reads:

    mic_e   = mean_energy(mic_rms, seg_start, seg_end)
    if mic_e > party_e * 1.5:  return "operator"

`mic_rms` and `party_rms` are stored in **dBFS** — logarithms — because §5.4.1
asks for dB and commit 10 stores what it was asked for. Multiplying a logarithm
by 1.5 does not scale an energy; on negative dB it moves the threshold *down*,
making the test easier to pass the quieter the other track is.

MEASURED on the speech fixture's overlapped line: mic −23.57 dBFS, party
−22.01 dBFS. The party track is 1.4x the louder. The literal rule evaluates
`−23.57 > −33.02` as true and returns "operator". Converting to linear power
first gives 0.0044 against 0.0063 — neither exceeding the other by the ratio —
so the answer is "both", which is what an overlap is.

So: dB comes back to linear power, the mean is taken *there*, and only then is
the ratio applied. `tests/test_speech_fixture.py` asserts all three facts about
that line, so the fixture has been guarding this since before the stage existed.

**Vocabulary.** §3.2 comments `segments.speaker` as `'operator','party','unknown'`
and §5.8's algorithm returns `'both'`. §8.3 then needs `both` ("alternate per
word by source track"), so `both` is stored and `unknown` means there was no
energy to judge by. There is no CHECK constraint; the comment is illustrative.
"""

from __future__ import annotations

import numpy as np

from clipforge import db, signals
from clipforge.pipeline.context import StageContext

OPERATOR = "operator"
PARTY = "party"
BOTH = "both"
UNKNOWN = "unknown"

MIC_KIND = "mic_rms"
PARTY_KIND = "party_rms"


class SpeakerError(RuntimeError):
    pass


def db_to_power(values: np.ndarray) -> np.ndarray:
    """dBFS -> linear power. The conversion §5.8's pseudocode is missing."""
    return np.power(10.0, np.asarray(values, dtype=np.float64) / 10.0)


def mean_power(series, t_start: float, t_end: float) -> float | None:
    """Mean linear power across a window, or None when there are no samples.

    Averaging in the linear domain, not the log one: the mean of two dB values
    is the geometric mean of the powers, which is not what "mean energy" means
    and understates any window containing a peak.
    """
    if series is None:
        return None
    window = series.slice(t_start, t_end)
    if window.size == 0:
        # A segment shorter than the sample spacing can fall between samples.
        # One sample is better than none; only an empty series gives up.
        if len(series) == 0:
            return None
        window = np.array([series.value_at((t_start + t_end) / 2.0)])
    return float(np.mean(db_to_power(window)))


def classify(mic_power: float | None, party_power: float | None, ratio: float) -> str:
    """§5.8's decision, on energies rather than logarithms."""
    if mic_power is None and party_power is None:
        return UNKNOWN
    if party_power is None:
        return OPERATOR
    if mic_power is None:
        return PARTY
    if mic_power > party_power * ratio:
        return OPERATOR
    if party_power > mic_power * ratio:
        return PARTY
    return BOTH


# --------------------------------------------------------------------------
# stage hooks
# --------------------------------------------------------------------------


def params(ctx: StageContext) -> dict:
    return {"ratio": ctx.cfg.get("extract.speaker.dominance_ratio")}


def outputs(ctx: StageContext) -> list:
    return []  # updates `segments.speaker`, writes no files


def available(ctx: StageContext) -> tuple[bool, str]:
    """Follows the transcript: labelling nothing is not worth a stage run."""
    from clipforge.extract import transcript

    return transcript.available(ctx)


def verify(ctx: StageContext) -> tuple[bool, str]:
    unlabelled = ctx.conn.execute(
        "SELECT COUNT(*) FROM segments WHERE stream_id = ? AND speaker IS NULL",
        (ctx.stream_id,),
    ).fetchone()[0]
    if unlabelled:
        return False, f"{unlabelled} segment(s) still have no speaker"
    total = ctx.conn.execute(
        "SELECT COUNT(*) FROM segments WHERE stream_id = ?", (ctx.stream_id,)
    ).fetchone()[0]
    if not total:
        return False, "no segments to label (has whisperx run?)"
    return True, ""


def run(ctx: StageContext) -> None:
    rows = ctx.conn.execute(
        "SELECT id, t_start, t_end, track FROM segments WHERE stream_id = ? ORDER BY seq",
        (ctx.stream_id,),
    ).fetchall()
    if not rows:
        raise SpeakerError(
            "no segments to label. whisperx has not run, or produced nothing."
        )

    mic = signals.load(ctx.conn, ctx.stream_id, MIC_KIND)
    party = signals.load(ctx.conn, ctx.stream_id, PARTY_KIND)
    if mic is None and party is None:
        raise SpeakerError(
            f"neither {MIC_KIND} nor {PARTY_KIND} is available. "
            f"audio_features has not run, or found no usable tracks."
        )
    if party is None:
        # A single-track recording. Everything is the operator by construction,
        # and §4.2 already warned at probe time that the mic is contaminated.
        ctx.log(
            f"    no {PARTY_KIND}: single-track recording, so every segment is "
            f"the operator (§4.2 warned about this at probe time)"
        )

    ratio = float(ctx.cfg.get("extract.speaker.dominance_ratio"))
    labels: list[tuple[str, int]] = []
    counts: dict[str, int] = {}

    for row in rows:
        speaker = classify(
            mean_power(mic, row["t_start"], row["t_end"]),
            mean_power(party, row["t_start"], row["t_end"]),
            ratio,
        )
        labels.append((speaker, row["id"]))
        counts[speaker] = counts.get(speaker, 0) + 1

    with db.transaction(ctx.conn):
        ctx.conn.executemany("UPDATE segments SET speaker = ? WHERE id = ?", labels)

    summary = "  ".join(
        f"{name}={counts.get(name, 0)}" for name in (OPERATOR, PARTY, BOTH, UNKNOWN)
        if counts.get(name)
    )
    ctx.log(f"    {len(labels)} segment(s): {summary}")
    for name, count in counts.items():
        ctx.metric(f"segments_speaker.{name}", float(count))

    if counts.get(UNKNOWN):
        ctx.log(
            f"    WARNING  {counts[UNKNOWN]} segment(s) had no energy to judge by. "
            f"§8.3's caption colouring will fall back to the default style."
        )
