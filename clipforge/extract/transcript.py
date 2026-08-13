"""§5.1 stage 6 — the transcript, with word-level timestamps (§5.7, §5.8).

Everything in Phase 2 and beyond reads what this writes: speaker assignment
(§5.8), speech rate, phrase detection (§5.6), §6.3's word-boundary snapping, and
Phase 5's digests. §12.1 makes `segments.seq` the identifier every LLM call
returns instead of a timestamp, so it has to be stable and unique per stream.

**The transcriber is injectable.** whisperx is ~100 packages and 2-3 GB of
wheels, then ~3 GB of model, and none of that can be a precondition for running
the test suite. `run` takes any object with a `transcribe` method; the real one
is imported lazily so this module loads on a machine that has never seen torch.

**Two passes, merged, never combined.** §5.8 says to run separately on the mic
and party tracks and "merge segments by timestamp", which is underspecified the
moment both tracks have speech at once. Two people talking over each other are
two segments that overlap in time, not one segment — combining them would
interleave two sentences into a single unreadable `text`. So both rows survive
and `speaker_assign` labels them.

**Notes on §5.7, verified against whisperx 3.8.6 / faster-whisper 1.2 rather
than assumed:**

* `initial_prompt` and `hotwords` are NOT exclusive. `get_prompt` concatenates
  the hotwords and then the prompt into one `sot_prev` section — but truncates
  each to half the prompt budget, so putting a long vocabulary in both crowds
  itself out. Hence `vocabulary_mode` and a cap on the term count.
* An unrecognised key in `asr_options` raises `TypeError`, because whisperx
  splats the merged dict into `TranscriptionOptions`. Options are filtered
  against what the installed version actually accepts.
* §5.7's `vad_onset`/`vad_offset` apply to the silero path too, not just
  pyannote.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from clipforge import db
from clipforge.config import CONFIG_DIR
from clipforge.pipeline.context import StageContext

#: Suffix of the gitignored half of the vocabulary. §5.7 wants real people's
#: names in there, which should not need a commit.
LOCAL_VOCABULARY_SUFFIX = ".local.txt"

#: CTranslate2 accepts these on CPU. float16 is CUDA-only and raises there.
CPU_COMPUTE_TYPE = "int8"
CUDA_COMPUTE_TYPE = "float16"

#: Whisper has positional encodings for 448 prompt tokens and raises above it.
#:
#: MEASURED, and it is why `vocabulary_mode` defaults to `hotwords` rather than
#: §5.7's both. faster-whisper truncates hotwords to `max_length // 2 - 1` = 223
#: tokens AND the initial prompt to another 223, then prepends sot_prev and
#: appends the sot sequence -- 451 in total. So seeding both at any real length
#: overflows by construction, and the failure is
#: `RuntimeError: No position encodings are defined for positions >= 448`,
#: raised from inside CTranslate2 with nothing pointing at the vocabulary.
MAX_PROMPT_TOKENS = 448

#: Rough tokens-per-character for English BPE. Used to trim the term list before
#: the model sees it, so truncation is ours and logged rather than silent and
#: mid-word inside faster-whisper.
CHARS_PER_TOKEN = 4.0

#: Token budget per mechanism. `both` gets half each, with room for the sot
#: framing, because that is the case that actually breaks.
VOCABULARY_TOKEN_BUDGET = {"hotwords": 180, "prompt": 180, "both": 100}


class TranscriptError(RuntimeError):
    pass


@dataclass(frozen=True)
class Word:
    """One word. §3.2 names the keys `w`, `start`, `end`, `score`.

    `start`/`end` are None when forced alignment could not place the word.
    Dropping it would corrupt the text §5.6 matches against; inventing a
    timestamp would corrupt §6.3's snapping. Null is the honest third answer.
    """

    w: str
    start: float | None = None
    end: float | None = None
    score: float | None = None

    def to_json(self) -> dict:
        return {"w": self.w, "start": self.start, "end": self.end, "score": self.score}

    @property
    def aligned(self) -> bool:
        return self.start is not None and self.end is not None


@dataclass(frozen=True)
class Segment:
    t_start: float
    t_end: float
    text: str
    #: ffmpeg audio stream index this came from (§3.2 `segments.track`).
    track: int
    role: str = ""
    words: list[Word] = field(default_factory=list)


class Transcriber(Protocol):
    def transcribe(self, path: Path, *, role: str, track: int) -> list[Segment]: ...


# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------


def resolve_compute_type(device: str, configured: str | None) -> str:
    """§5.7 hardcodes float16, which CTranslate2 refuses on CPU.

    The spec assumes CUDA and is right to on the streaming PC. Everywhere else
    the literal value fails with an error that names neither the setting nor the
    reason, so `auto` picks per device.
    """
    if configured and str(configured).lower() != "auto":
        return str(configured)
    return CUDA_COMPUTE_TYPE if str(device).startswith("cuda") else CPU_COMPUTE_TYPE


def vocabulary_paths(cfg) -> tuple[Path, Path]:
    name = str(cfg.get("extract.whisperx.vocabulary_file", "vocabulary.txt"))
    shared = CONFIG_DIR / name
    local = shared.with_name(shared.stem + LOCAL_VOCABULARY_SUFFIX)
    return shared, local


def load_vocabulary(cfg) -> list[str]:
    """Terms from the tracked file plus the gitignored local one.

    Order-stable and deduplicated, because this feeds `params_hash`: a
    vocabulary that reshuffled between runs would invalidate every completed
    transcription for no reason.
    """
    terms: list[str] = []
    seen: set[str] = set()
    for path in vocabulary_paths(cfg):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            term = line.split("#", 1)[0].strip()
            if term and term.lower() not in seen:
                seen.add(term.lower())
                terms.append(term)

    cap = int(cfg.get("extract.whisperx.max_vocabulary_terms", 160))
    return terms[:cap]


def fit_terms(terms: list[str], mode: str) -> list[str]:
    """Trim the term list to what will fit in Whisper's prompt.

    Ours rather than faster-whisper's, which truncates the encoded tokens
    mid-list without saying so — and which, for `both`, truncates to a total
    that still exceeds the 448-position limit and raises from inside
    CTranslate2.
    """
    budget = VOCABULARY_TOKEN_BUDGET.get(mode, 180) * CHARS_PER_TOKEN
    kept: list[str] = []
    used = 0.0
    for term in terms:
        cost = len(term) + 2  # ", "
        if used + cost > budget:
            break
        kept.append(term)
        used += cost
    return kept


def vocabulary_options(cfg, terms: list[str], log=None) -> dict:
    """Turn the term list into whatever §5.7's two mechanisms want.

    §5.7 sets an `initial_prompt` AND a `hotwords` list. Verified against
    faster-whisper: they are not exclusive — `get_prompt` concatenates the
    hotwords and then the prompt into one `sot_prev` section. But see
    MAX_PROMPT_TOKENS for why using both is a trap.
    """
    mode = str(cfg.get("extract.whisperx.vocabulary_mode", "hotwords")).lower()
    if mode not in ("hotwords", "prompt", "both", "none"):
        raise TranscriptError(
            f"unknown vocabulary_mode {mode!r}; expected hotwords, prompt, both or none"
        )
    if not terms or mode == "none":
        return {}

    fitted = fit_terms(terms, mode)
    if log is not None and len(fitted) < len(terms):
        log(
            f"    vocabulary trimmed to {len(fitted)} of {len(terms)} terms to fit "
            f"Whisper's {MAX_PROMPT_TOKENS}-token prompt ({mode})"
        )

    joined = ", ".join(fitted)
    options: dict = {}
    if mode in ("hotwords", "both"):
        options["hotwords"] = joined
    if mode in ("prompt", "both"):
        options["initial_prompt"] = joined
    return options


# --------------------------------------------------------------------------
# the real transcriber
# --------------------------------------------------------------------------


@dataclass
class WhisperXTranscriber:
    """§5.7, against the API as it actually is in whisperx 3.8."""

    model_name: str
    device: str
    compute_type: str
    language: str | None
    batch_size: int
    vad_method: str
    vad_options: dict
    align: bool
    asr_options: dict = field(default_factory=dict)
    log: object = print

    def __post_init__(self) -> None:
        self._model = None
        self._align = None

    def _say(self, message: str) -> None:
        self.log(message)  # type: ignore[operator]

    def load(self):
        if self._model is not None:
            return self._model
        try:
            from whisperx.asr import load_model
        except ImportError as exc:
            raise TranscriptError(
                "whisperx is not installed. Install the ASR extra:\n"
                '    pip install -e ".[asr]"\n'
                "It is ~2-3 GB of wheels, which is why it is not a core dependency."
            ) from exc

        options = self._accepted_asr_options()
        self._say(
            f"    model {self.model_name} on {self.device} ({self.compute_type}), "
            f"vad={self.vad_method}"
        )
        if options:
            # Which seeding actually reached the model. §5.7 sets two mechanisms
            # and it is worth being able to see which one was applied.
            self._say(f"    seeding {', '.join(sorted(options))}")

        self._model = load_model(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
            language=self.language,
            vad_method=self.vad_method,
            vad_options=dict(self.vad_options),
            asr_options=options or None,
        )
        return self._model

    def _accepted_asr_options(self) -> dict:
        """Drop anything the installed faster-whisper does not know.

        whisperx splats the merged dict into `TranscriptionOptions`, so one
        unrecognised key is a TypeError from deep inside a library rather than a
        message about configuration.
        """
        try:
            from faster_whisper.transcribe import TranscriptionOptions

            allowed = set(TranscriptionOptions.__dataclass_fields__)
        except Exception:  # noqa: BLE001 — an unknown layout means pass it all through
            return dict(self.asr_options)

        kept = {k: v for k, v in self.asr_options.items() if k in allowed}
        for name in sorted(set(self.asr_options) - set(kept)):
            self._say(f"    WARNING  this whisperx does not accept asr option {name!r}")
        return kept

    def _aligner(self, language: str):
        if self._align is None:
            from whisperx.alignment import load_align_model

            self._align = load_align_model(language_code=language, device=self.device)
        return self._align

    def transcribe(self, path: Path, *, role: str, track: int) -> list[Segment]:
        import whisperx
        from whisperx.alignment import align as align_segments

        audio = whisperx.load_audio(str(path))
        result = self.load().transcribe(
            audio, batch_size=self.batch_size, language=self.language
        )
        raw = list(result.get("segments") or [])
        language = result.get("language") or self.language or "en"

        if self.align and raw:
            model, metadata = self._aligner(language)
            raw = list(
                align_segments(raw, model, metadata, audio, self.device).get("segments")
                or []
            )

        return [
            Segment(
                t_start=float(entry["start"]),
                t_end=float(entry["end"]),
                text=str(entry.get("text", "")).strip(),
                track=track,
                role=role,
                words=[
                    Word(
                        w=str(word.get("word", "")).strip(),
                        start=_optional_float(word.get("start")),
                        end=_optional_float(word.get("end")),
                        score=_optional_float(word.get("score")),
                    )
                    for word in (entry.get("words") or [])
                    if str(word.get("word", "")).strip()
                ],
            )
            for entry in raw
            if entry.get("start") is not None and entry.get("end") is not None
        ]


def _optional_float(value) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# merging
# --------------------------------------------------------------------------


def merge_passes(passes: list[list[Segment]]) -> list[Segment]:
    """§5.8's "merge segments by timestamp", with the ambiguity resolved.

    Both tracks' segments are kept even where they overlap: that is two people
    talking at once, which §5.4.1 counts as a signal in its own right, not one
    utterance to be reconciled. Sorted by start, then end, then track, so the
    order is total and `seq` is reproducible.
    """
    everything = [segment for group in passes for segment in group]
    return sorted(everything, key=lambda s: (s.t_start, s.t_end, s.track))


# --------------------------------------------------------------------------
# stage hooks
# --------------------------------------------------------------------------


def _track_map(ctx: StageContext) -> dict:
    raw = ctx.stream["audio_track_map"]
    return json.loads(raw) if raw else {}


def _roles(ctx: StageContext) -> dict[str, int]:
    from clipforge.extract import audio

    return audio.roles_to_extract(
        _track_map(ctx), ctx.cfg.get("extract.whisperx.roles")
    )


def params(ctx: StageContext) -> dict:
    """What would change the transcript.

    `device` and `compute_type` are deliberately absent. They do nudge the
    output, but invalidating a forty-minute transcription because the library
    moved from the streaming PC to a laptop is the mistake `master_identity`
    already avoids by excluding mtime and path.
    """
    terms = load_vocabulary(ctx.cfg)
    return {
        "model": ctx.cfg.get("extract.whisperx.model"),
        "language": ctx.cfg.get("extract.whisperx.language", None),
        "roles": sorted(_roles(ctx)),
        "vad_method": ctx.cfg.get("extract.whisperx.vad_method"),
        "vad_options": ctx.cfg.get("extract.whisperx.vad_options"),
        "align": ctx.cfg.get("extract.whisperx.align"),
        "vocabulary_mode": ctx.cfg.get("extract.whisperx.vocabulary_mode"),
        "vocabulary": hashlib.sha256("\n".join(terms).encode()).hexdigest()[:16],
    }


def outputs(ctx: StageContext) -> list[Path]:
    return []  # writes `segments` rows, not files


def available(ctx: StageContext) -> tuple[bool, str]:
    """Whether transcription can run here — see `StageSpec.available`.

    Off by default. §15 puts the transcript in Phase 2 and is emphatic about
    shipping Phase 1 first, and until `speaker_assign`, `speech_rate` and
    `phrase_detect` exist a transcript feeds nothing: it costs a multi-GB
    download and, on the wrong hardware, hours per stream, to produce rows
    nothing reads yet.

    Returning False here makes the stage DEFERRED, so a Phase 1 run skips it
    and still reaches `score`. Failing instead would take the whole pipeline
    down on any machine without the ASR extra.
    """
    if not bool(ctx.cfg.get("extract.whisperx.enabled", False)):
        return False, "disabled (set extract.whisperx.enabled to turn on Phase 2)"

    import importlib.util

    if importlib.util.find_spec("whisperx") is None:
        return False, 'whisperx not installed (pip install -e ".[asr]")'
    return True, ""


def verify(ctx: StageContext) -> tuple[bool, str]:
    """The product is rows, so file existence cannot check it."""
    count = ctx.conn.execute(
        "SELECT COUNT(*) FROM segments WHERE stream_id = ?", (ctx.stream_id,)
    ).fetchone()[0]
    if not count:
        return False, "no segments rows"
    return True, ""


def build_transcriber(ctx: StageContext) -> WhisperXTranscriber:
    cfg = ctx.cfg
    device = str(cfg.get("extract.whisperx.device"))
    terms = load_vocabulary(cfg)
    return WhisperXTranscriber(
        model_name=str(cfg.get("extract.whisperx.model")),
        device=device,
        compute_type=resolve_compute_type(
            device, cfg.get("extract.whisperx.compute_type", None)
        ),
        language=cfg.get("extract.whisperx.language", None),
        batch_size=int(cfg.get("extract.whisperx.batch_size", 16)),
        vad_method=str(cfg.get("extract.whisperx.vad_method")),
        vad_options=dict(cfg.get("extract.whisperx.vad_options") or {}),
        align=bool(cfg.get("extract.whisperx.align", True)),
        asr_options=vocabulary_options(cfg, terms, log=ctx.log),
        log=ctx.log,
    )


def store(ctx: StageContext, segments: list[Segment]) -> None:
    """Replace this stream's transcript in one transaction.

    Replace rather than append: `seq` is §12.1's LLM-facing identifier with a
    UNIQUE index behind it, and a re-run that added a second set would both
    collide and silently double the transcript.
    """
    with db.transaction(ctx.conn):
        ctx.conn.execute("DELETE FROM segments WHERE stream_id = ?", (ctx.stream_id,))
        ctx.conn.executemany(
            "INSERT INTO segments (stream_id, seq, t_start, t_end, text, speaker, "
            "track, words) VALUES (?, ?, ?, ?, ?, NULL, ?, ?)",
            [
                (
                    ctx.stream_id, seq, segment.t_start, segment.t_end, segment.text,
                    segment.track,
                    json.dumps([word.to_json() for word in segment.words]),
                )
                # seq is assigned AFTER the merge, never per pass.
                for seq, segment in enumerate(segments)
            ],
        )


def run(ctx: StageContext, transcriber: Transcriber | None = None) -> None:
    roles = _roles(ctx)
    if not roles:
        raise TranscriptError(
            "no speech tracks to transcribe. §5.8 needs mic (and ideally party); "
            "audio_split found neither. Check `clipforge status`."
        )

    engine = transcriber or build_transcriber(ctx)

    passes: list[list[Segment]] = []
    for role, track in roles.items():
        path = ctx.paths.audio(role)
        if not path.is_file():
            raise TranscriptError(
                f"{path.name} is missing. audio_split should have written it; "
                f"re-run with --force audio_split."
            )
        ctx.heartbeat()
        found = engine.transcribe(path, role=role, track=track)
        ctx.log(f"    {role:<6} {len(found)} segment(s)")
        passes.append(found)
        ctx.heartbeat()

    merged = merge_passes(passes)
    store(ctx, merged)
    _report(ctx, merged)


def _report(ctx: StageContext, segments: list[Segment]) -> None:
    words = [word for segment in segments for word in segment.words]
    unaligned = [word for word in words if not word.aligned]
    speech_s = sum(segment.t_end - segment.t_start for segment in segments)

    ctx.log(
        f"    {len(segments)} segment(s), {len(words)} word(s), "
        f"{speech_s:.1f}s of speech"
    )
    ctx.metric("transcript_segments", float(len(segments)))
    ctx.metric("transcript_words", float(len(words)))

    if unaligned:
        # A handful is normal. A large fraction means the alignment model is
        # wrong for this audio, and §6.3's snapping will have little to work on.
        share = len(unaligned) / len(words)
        ctx.log(
            f"    {len(unaligned)} word(s) ({share:.0%}) could not be aligned and "
            f"have no timestamps"
        )
        if share > 0.2:
            ctx.log(
                "    WARNING  most words are unaligned. Word-boundary snapping "
                "(§6.3) and speech rate will be unreliable."
            )

    if not segments:
        ctx.log(
            "    WARNING  nothing was transcribed. If the audio is not silent, "
            "check extract.whisperx.language and the VAD settings."
        )
