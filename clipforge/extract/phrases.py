"""§5.1 stage 8 — what the transcript says, as signals and events (§5.6, §5.4.1).

Four products, one pass over `segments`:

* `phrase_excitement` events — §5.6's pattern list, matched on natural speech.
* `phrase_repeat` events — the same phrase three times inside 90 s, which §5.4.2
  calls "bit formation in real time".
* `speech_rate` — §5.4.1's words/sec in a 3 s sliding window, which "detects
  both excitement and dead air".
* `swear_density` — §5.6's "profanity count per 10 s window".

**Matching respects word boundaries.** §5.6 says only "match against the
transcript", and a substring search fires "no way" inside "I know ways around
it". Apostrophes are optional in the pattern, so "let's" and "lets" are one
entry rather than two.

**`speech_rate` counts aligned words only.** Words that forced alignment could
not place have no timestamps and cannot belong to a window. That biases the rate
*down* wherever alignment struggled — and §5.4.1 says this signal detects dead
air, so a bad alignment would masquerade as silence. The stage reports the
excluded share so the two can be told apart.
"""

from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from clipforge import db, signals
from clipforge.config import CONFIG_DIR
from clipforge.pipeline.context import StageContext

SOURCE = "phrase"
KIND_EXCITEMENT = "phrase_excitement"
KIND_REPEAT = "phrase_repeat"

SPEECH_RATE = "speech_rate"
SWEAR_DENSITY = "swear_density"

#: Everything except letters, digits and inner apostrophes. Used for the
#: normalisation §5.4.2 needs and does not define.
_STRIP = re.compile(r"[^\w\s']+")
_SPACES = re.compile(r"\s+")


class PhraseError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Phrases:
    excitement: list[str]
    profanity: list[str]
    stopwords: frozenset[str]
    tics: frozenset[str]


def load_phrases(cfg) -> Phrases:
    path = Path(cfg.get("extract.phrases.file", "phrases.yaml"))
    if not path.is_absolute():
        path = CONFIG_DIR / path
    if not path.is_file():
        raise PhraseError(f"phrase list not found: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    repeat = data.get("repeat") or {}
    return Phrases(
        excitement=[str(p) for p in (data.get("excitement") or [])],
        profanity=[str(p) for p in (data.get("profanity") or [])],
        stopwords=frozenset(str(w).lower() for w in (repeat.get("stopwords") or [])),
        tics=frozenset(normalise(str(t)) for t in (repeat.get("tics") or [])),
    )


def normalise(text: str) -> str:
    """The rule §5.4.2 needs and never states.

    "Let's go!", "lets go" and "Let's go." are one bit. Casefold, drop
    punctuation, collapse whitespace, and drop apostrophes so the contraction
    and its bare form agree.
    """
    return _SPACES.sub(" ", _STRIP.sub(" ", text.lower()).replace("'", "")).strip()


def compile_patterns(phrases: list[str]) -> list[tuple[str, re.Pattern]]:
    """Word-boundary regexes, with apostrophes optional.

    §5.6 says "match against the transcript" and nothing more. Substring
    matching makes "no way" fire inside "I know ways around it", and every one
    of those is a candidate the operator has to reject by hand.
    """
    compiled = []
    for phrase in phrases:
        words = normalise(phrase).split()
        if not words:
            continue
        # Each word may carry an apostrophe anywhere inside it in the source
        # text, so "lets" matches "let's" without a second list entry.
        parts = [r"\s*'?\s*".join(re.escape(ch) for ch in word) for word in words]
        compiled.append((normalise(phrase), re.compile(
            r"\b" + r"\W+".join(parts) + r"\b", re.IGNORECASE
        )))
    return compiled


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------


@dataclass
class Detected:
    events: list[dict]
    #: normalised n-gram -> how many times it fired a repeat event
    repeats: dict[str, int]


def find_excitement(rows, patterns) -> list[dict]:
    """One event per matched phrase per segment.

    The event lands at the segment's start, not at the word: §6.2 step 4 gives
    every event a decay kernel several seconds wide, so sub-second placement
    inside one utterance buys nothing and would imply a precision the match does
    not have.
    """
    out = []
    for row in rows:
        text = row["text"] or ""
        for phrase, pattern in patterns:
            if pattern.search(text):
                out.append({
                    "t": float(row["t_start"]),
                    "t_end": float(row["t_end"]),
                    "kind": KIND_EXCITEMENT,
                    "value": 1.0,
                    "meta": {
                        "phrase": phrase,
                        # §5.4.2 weights laugh_party above laugh_operator because
                        # someone ELSE reacting is external validation. The same
                        # argument applies here, and Phase 3 can split the weight
                        # without re-extracting.
                        "speaker": row["speaker"],
                        "seq": row["seq"],
                    },
                })
    return out


def ngrams(text: str, low: int, high: int) -> list[str]:
    words = normalise(text).split()
    return [
        " ".join(words[i:i + n])
        for n in range(low, high + 1)
        for i in range(len(words) - n + 1)
    ]


def is_filler(phrase: str, phrases: Phrases) -> bool:
    """§11.2's `is_baseline_tic`, standing in until there is a corpus.

    §11.2 computes the baseline from the first ~10 streams, which do not exist.
    A hand-written list cannot know which phrases *this* operator says
    constantly; ten streams of transcript can. Until then this is the honest
    approximation, and Phase 6 should replace it with the measurement.
    """
    if phrase in phrases.tics:
        return True
    words = phrase.split()
    return all(word in phrases.stopwords for word in words)


def find_repeats(rows, phrases: Phrases, *, window_s: float, minimum: int,
                 low: int, high: int) -> tuple[list[dict], dict[str, int]]:
    """§5.4.2: the same phrase >=3x inside 90 s.

    Fires on the third occurrence and each one after — that is when the bit
    becomes detectable, and firing on the first would mean reaching backwards in
    time to score a moment that had not happened yet.
    """
    seen: dict[str, deque] = {}
    events: list[dict] = []
    counts: dict[str, int] = {}

    for row in rows:
        when = float(row["t_start"])
        fired: dict[str, int] = {}

        for phrase in set(ngrams(row["text"] or "", low, high)):
            if is_filler(phrase, phrases):
                continue
            history = seen.setdefault(phrase, deque())
            history.append(when)
            while history and when - history[0] > window_s:
                history.popleft()
            if len(history) >= minimum:
                fired[phrase] = len(history)

        # MEASURED: a repeated four-word line fires every n-gram inside it, so
        # "Let's go, Hawkeye" produced `lets go`, `go hawkeye` AND
        # `lets go hawkeye` at the same instant. Event kernels combine with
        # `sum` for non-marker kinds (§6.2 step 4), so a longer catchphrase
        # would score higher purely for containing more n-grams — length as a
        # signal, which it is not. Keep only the maximal phrases: the longest
        # description of the bit, once.
        for phrase, count in maximal(fired).items():
            counts[phrase] = counts.get(phrase, 0) + 1
            events.append({
                "t": when,
                "t_end": float(row["t_end"]),
                "kind": KIND_REPEAT,
                "value": float(count),
                "meta": {
                    "phrase": phrase,
                    "count": count,
                    "window_s": window_s,
                    "speaker": row["speaker"],
                    "seq": row["seq"],
                },
            })
    return events, counts


def maximal(fired: dict[str, int]) -> dict[str, int]:
    """Drop any phrase wholly contained in another that fired at the same time."""
    keep = {}
    for phrase, count in fired.items():
        if not any(
            other != phrase and _contains(other, phrase) for other in fired
        ):
            keep[phrase] = count
    return keep


def _contains(haystack: str, needle: str) -> bool:
    """Word-sequence containment, not substring: "go" is not inside "going"."""
    big, small = haystack.split(), needle.split()
    return any(
        big[i:i + len(small)] == small for i in range(len(big) - len(small) + 1)
    )


# --------------------------------------------------------------------------
# continuous signals
# --------------------------------------------------------------------------


def aligned_words(rows) -> tuple[list[float], int, int]:
    """Word centre times, plus (aligned, total) counts."""
    centres: list[float] = []
    total = 0
    for row in rows:
        for word in json.loads(row["words"] or "[]"):
            total += 1
            if word.get("start") is None or word.get("end") is None:
                continue
            centres.append((float(word["start"]) + float(word["end"])) / 2.0)
    return sorted(centres), len(centres), total


def rate_series(times: list[float], *, duration_s: float, hz: float,
                window_s: float, kind: str) -> signals.Series:
    """Events per second over a centred sliding window, sampled at `hz`.

    Counting into a histogram at the grid rate and convolving with a boxcar,
    rather than a loop per sample: a 4-hour stream is 144k samples and the naive
    form is quadratic in the number of words.
    """
    n = max(int(round(duration_s * hz)), 1)
    counts = np.zeros(n, dtype=np.float64)
    if times:
        indices = np.clip((np.asarray(times) * hz).astype(int), 0, n - 1)
        np.add.at(counts, indices, 1.0)

    width = max(int(round(window_s * hz)), 1)
    kernel = np.ones(width, dtype=np.float64)
    summed = np.convolve(counts, kernel, mode="same")
    return signals.Series(
        kind=kind, values=summed / window_s, sample_rate_hz=hz,
        params={"window_s": window_s},
    )


def swear_times(rows, patterns) -> list[float]:
    """When profanity was said, from word timestamps where they exist."""
    out: list[float] = []
    for row in rows:
        words = json.loads(row["words"] or "[]")
        placed = [w for w in words if w.get("start") is not None]
        for word in placed:
            token = normalise(str(word.get("w", "")))
            if any(pattern.fullmatch(token) for _phrase, pattern in patterns):
                out.append(float(word["start"]))
        if not placed:
            # No word timestamps: fall back to the segment, so a stream whose
            # alignment failed still produces the signal rather than a flat line.
            for phrase, pattern in patterns:
                if pattern.search(row["text"] or ""):
                    out.append(float(row["t_start"]))
    return sorted(out)


# --------------------------------------------------------------------------
# stage hooks
# --------------------------------------------------------------------------


def params(ctx: StageContext) -> dict:
    phrases = load_phrases(ctx.cfg)
    return {
        "excitement": sorted(normalise(p) for p in phrases.excitement),
        "profanity": sorted(normalise(p) for p in phrases.profanity),
        "repeat": ctx.cfg.get("extract.phrases.repeat"),
        "speech_rate_window_s": ctx.cfg.get("extract.phrases.speech_rate_window_s"),
        "swear_window_s": ctx.cfg.get("extract.phrases.swear_window_s"),
        "signal_hz": ctx.cfg.get("extract.signal_hz"),
    }


def outputs(ctx: StageContext) -> list[Path]:
    return []  # events and signal_series rows


def available(ctx: StageContext) -> tuple[bool, str]:
    from clipforge.extract import transcript

    return transcript.available(ctx)


def verify(ctx: StageContext) -> tuple[bool, str]:
    present = set(signals.kinds(ctx.conn, ctx.stream_id))
    missing = sorted({SPEECH_RATE, SWEAR_DENSITY} - present)
    if missing:
        return False, f"no signal_series for {', '.join(missing)}"
    return True, ""


def _replace_events(ctx: StageContext, events: list[dict]) -> None:
    """Replace this stream's phrase events, following markers._replace_events."""
    ctx.conn.execute(
        "DELETE FROM events WHERE stream_id = ? AND source = ?", (ctx.stream_id, SOURCE)
    )
    ctx.conn.executemany(
        "INSERT INTO events (stream_id, t, t_end, source, kind, value, meta) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (ctx.stream_id, e["t"], e["t_end"], SOURCE, e["kind"], e["value"],
             json.dumps(e["meta"]))
            for e in events
        ],
    )


def run(ctx: StageContext) -> None:
    rows = ctx.conn.execute(
        "SELECT seq, t_start, t_end, text, speaker, words FROM segments "
        "WHERE stream_id = ? ORDER BY seq",
        (ctx.stream_id,),
    ).fetchall()
    if not rows:
        raise PhraseError(
            "no segments to search. whisperx has not run, or produced nothing."
        )

    cfg = ctx.cfg
    phrases = load_phrases(cfg)
    excitement_patterns = compile_patterns(phrases.excitement)
    profanity_patterns = compile_patterns(phrases.profanity)

    events = find_excitement(rows, excitement_patterns)
    repeat_events, repeat_counts = find_repeats(
        rows, phrases,
        window_s=float(cfg.get("extract.phrases.repeat.window_s")),
        minimum=int(cfg.get("extract.phrases.repeat.min_occurrences")),
        low=int(cfg.get("extract.phrases.repeat.min_words")),
        high=int(cfg.get("extract.phrases.repeat.max_words")),
    )
    events.extend(repeat_events)

    duration_s = float(ctx.stream["duration_s"] or rows[-1]["t_end"])
    hz = float(cfg.get("extract.signal_hz"))
    centres, aligned, total = aligned_words(rows)

    speech = rate_series(
        centres, duration_s=duration_s, hz=hz,
        window_s=float(cfg.get("extract.phrases.speech_rate_window_s")),
        kind=SPEECH_RATE,
    )
    swears = rate_series(
        swear_times(rows, profanity_patterns), duration_s=duration_s, hz=hz,
        window_s=float(cfg.get("extract.phrases.swear_window_s")),
        kind=SWEAR_DENSITY,
    )

    with db.transaction(ctx.conn):
        _replace_events(ctx, events)
        signals.store(ctx.conn, ctx.stream_id, speech)
        signals.store(ctx.conn, ctx.stream_id, swears)

    _report(ctx, events, repeat_counts, speech, swears, aligned, total)


def _report(ctx, events, repeat_counts, speech, swears, aligned, total) -> None:
    excitement = [e for e in events if e["kind"] == KIND_EXCITEMENT]
    repeats = [e for e in events if e["kind"] == KIND_REPEAT]

    ctx.log(f"    {len(excitement)} phrase_excitement, {len(repeats)} phrase_repeat")
    if repeat_counts:
        top = sorted(repeat_counts.items(), key=lambda kv: -kv[1])[:3]
        ctx.log("    repeated: " + ", ".join(f"{p!r} x{c}" for p, c in top))

    ctx.log(
        f"    {SPEECH_RATE}    peak {float(np.max(speech.values)):.2f} words/s, "
        f"{SWEAR_DENSITY} peak {float(np.max(swears.values)) * 10:.1f} per 10 s"
    )
    ctx.metric("phrase_events", float(len(events)))

    if total and aligned < total:
        share = 1.0 - aligned / total
        ctx.log(
            f"    {total - aligned} of {total} word(s) ({share:.0%}) had no timestamps "
            f"and could not count toward {SPEECH_RATE}"
        )
        if share > 0.2:
            # §5.4.1 calls speech_rate a detector of "both excitement and dead
            # air". A rate depressed by failed alignment reads as dead air.
            ctx.log(
                f"    WARNING  most words are unaligned, so {SPEECH_RATE} understates "
                f"the real rate and will look like silence."
            )
