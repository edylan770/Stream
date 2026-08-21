"""Seed the authored three-chapter stream (`transcript_fixture.json`).

§9.4's digest is a map-reduce **over chapters**, and until this existed nothing
in the repository had more than one. The speech fixture is 95 s and thirteen
utterances — HANDOFF already calls it "honestly one chapter" — and
`fixture_long` is band-limited noise with no transcript at all.

MEDIA-FREE, AND THAT IS THE POINT. `digest/chapters.py`'s silence input reads
`signal_series` through §6.4's speech gate and never touches audio, so this
writes the arrays directly. No ffmpeg, no Ollama, no network, no minutes of
encoding: it runs on a fresh clone in well under a second. `test_chapters.py`'s
own fixture still covers §9.3 against real measured audio at 95 s; the two are
complementary rather than a replacement.

THE SIGNAL IS BURSTS AND PAUSES, NOT A FLAT LEVEL, AND THAT IS FORCED.
`derived.speech_gate` is `rms > rolling_mean + vad.margin_db`, so a constant
"speech" level can never exceed its own rolling baseline and a flat stretch
reads as silence everywhere. Real speech alternates; the baseline settles
between the two and the gate separates them. `assert_gate_separates` below
checks that against the gate AS IT SHIPS rather than trusting the arithmetic —
the levels in the JSON were chosen to satisfy the existing gate, and the gate
was never touched to fit them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from clipforge import db, paths, signals

FIXTURE = Path(__file__).parent / "transcript_fixture.json"

#: Words per second, for giving an authored utterance a plausible duration.
#: Only has to keep segments inside their chapter and out of the silence gaps;
#: nothing asserts against it.
WORDS_PER_S = 2.6


def load() -> dict:
    """The authored ground truth. Tests read this; they never hardcode."""
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def levels(data: dict) -> np.ndarray:
    """The authored dBFS track: bursts and pauses, silence inside the gaps."""
    hz = float(data["signal_hz"])
    n = int(round(float(data["duration_s"]) * hz))
    t = np.arange(n, dtype=np.float64) / hz

    lv = data["levels_dbfs"]
    period = float(data["burst_s"]) + float(data["pause_s"])
    speaking = (t % period) < float(data["burst_s"])

    values = np.where(speaking, float(lv["speech"]), float(lv["pause"]))
    for gap in data["silence_gaps"]:
        inside = (t >= float(gap["t_start"])) & (t < float(gap["t_end"]))
        values[inside] = float(lv["silence"])
    return values


def seed(cfg, conn, data: dict | None = None) -> dict:
    """Insert the stream, both signal series and every segment.

    Deliberately does NOT run `phrases`: that needs a `StageContext` and a
    config the caller owns, and a fixture that quietly runs stages hides which
    code path a test is exercising. Callers that want `phrase_repeat` events
    call `phrases.run` themselves.
    """
    data = data or load()
    stream_id = str(data["stream_id"])

    conn.execute(
        "INSERT INTO streams (id, date, title, master_path, duration_s, marker_time_base) "
        "VALUES (?, ?, ?, ?, ?, 'vod')",
        (stream_id, data["date"], data["title"], "D:/authored.mkv",
         float(data["duration_s"])),
    )
    paths.StreamPaths(cfg.data_root, stream_id).ensure()

    values = levels(data)
    hz = float(data["signal_hz"])
    with db.transaction(conn):
        # Both voice tracks, because `gates.speech_activity` unions VOICE_KINDS
        # and a gap is only a gap when nobody is talking on either one.
        for kind in ("mic_rms", "party_rms"):
            signals.store(conn, stream_id, signals.Series(
                kind=kind, values=values, sample_rate_hz=hz,
                t0=0.5 / hz, params={"unit": "dB", "source": "authored"},
            ))

        tracks = data["tracks"]
        rows = []
        for u in data["utterances"]:
            start = float(u["t"])
            duration = max(1.0, len(str(u["text"]).split()) / WORDS_PER_S)
            rows.append((
                stream_id, int(u["seq"]), start, start + duration, u["text"],
                "operator" if u["track"] == "mic" else "party",
                tracks.get(u["track"]),
            ))
        conn.executemany(
            "INSERT INTO segments (stream_id, seq, t_start, t_end, text, speaker, track) "
            "VALUES (?,?,?,?,?,?,?)", rows)

    return data


def chapter_of(data: dict, t: float) -> int:
    """Which authored chapter a time falls in. -1 inside a silence gap."""
    for gap in data["silence_gaps"]:
        if float(gap["t_start"]) <= t < float(gap["t_end"]):
            return -1
    for chapter in data["chapters"]:
        if float(chapter["t_start"]) <= t < float(chapter["t_end"]):
            return int(chapter["index"])
    return -1


def assert_gate_separates(cfg, conn, data: dict) -> None:
    """The fixture's own precondition, checked rather than assumed.

    Every assertion about chapters rests on §6.4's gate reading the authored
    speech as speech and the authored gaps as silence. If a library change or a
    config edit ever breaks that, this says so in one line instead of leaving a
    chapter test failing for a reason that looks like a chapter bug.
    """
    from clipforge.score import derived, gates, grid

    hz = float(cfg.get("score.score_grid_hz"))
    timeline = grid.build(float(data["duration_s"]), hz)
    raw = {name: grid.resample(series, timeline)
           for name, series in signals.load_all(conn, str(data["stream_id"])).items()
           if name in gates.VOICE_KINDS}
    inputs = derived.Inputs(timeline=timeline, grid_hz=hz, raw=raw, cfg=cfg)
    baseline = grid.window_samples(cfg.get("score.rolling_baseline_window_s"), hz)
    speech = gates.speech_activity(raw, inputs, baseline)

    # MEASURED: the gate marks exactly `vad.hangover_s` worth of samples at the
    # HEAD of each gap and none after -- `speech_gate` holds the flag that far
    # past the last burst, deliberately, so one sentence does not become nine
    # utterances. GUESSES records the same overrun on the speech fixture. The
    # window is read from config rather than written down, so raising the
    # hangover moves this assertion instead of breaking it.
    hangover = float(cfg.get("score.derived.vad.hangover_s"))
    for gap in data["silence_gaps"]:
        start, end = float(gap["t_start"]), float(gap["t_end"])
        after = (timeline >= start + hangover) & (timeline < end)
        leaked = int(speech[after].sum())
        assert leaked == 0, (
            f"the gate hears speech {leaked} sample(s) into the authored gap at "
            f"{start}-{end}s, past the {hangover}s hangover that explains the head")
        assert (end - start) - hangover >= float(cfg.get("digest.chapters.min_silence_s")), (
            f"the authored gap at {start}-{end}s is not long enough to survive "
            f"the hangover and still clear digest.chapters.min_silence_s")

    for chapter in data["chapters"]:
        # Sampled away from the edges, where a gap's baseline still bleeds in.
        lo = float(chapter["t_start"]) + 120.0
        hi = float(chapter["t_end"]) - 120.0
        inside = (timeline >= lo) & (timeline < hi)
        share = float(speech[inside].mean())
        assert share > 0.5, (
            f"the gate hears speech in only {share:.1%} of authored chapter "
            f"{chapter['index']}, so the fixture's levels do not clear "
            f"score.derived.vad.margin_db")
