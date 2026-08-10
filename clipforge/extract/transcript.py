"""WhisperX transcription (spec §5.7). Phase 2 — not built.

Deliberately empty. Notes recorded here so the constraints are not rediscovered:

- VAD filtering must be on (A4). Without it Whisper hallucinates text over
  silence and music, poisoning the transcript and every downstream signal.
- Run separately on the mic and party tracks, then merge by timestamp (§5.8).
  No pyannote diarization — separate tracks make speaker ID deterministic.
- Seed `config/vocabulary.txt` (§5.7) or the transcript says "Hawk I" forever.
- Word-level alignment drives captions, cut-point snapping, and speech rate.

Wheel availability for torch/WhisperX on the target machine's interpreter is an
open question to resolve when this phase starts.
"""
