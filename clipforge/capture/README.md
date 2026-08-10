# Layer A — Capture (spec §4)

**Phase 0. Not built.** These scripts are deployed to the streaming PC and run
independently of this application (§2.1): if the input logger dies mid-stream,
one signal is lost, the stream is not.

Planned:

| Script | Purpose | Spec |
|---|---|---|
| `marker_daemon.py` | F1/F2 hotkeys → JSONL of `{epoch_ms, kind}` | §4.3 |
| `input_logger.py` | keyboard/mouse activity at 10 Hz aggregate → JSONL | §4.4 |
| `obs_anchor.py` | record-start wall-clock epoch ms → JSON | §4.1 |

Two rules these must obey when written:

- **Epoch milliseconds everywhere** (A8). Never local time, never formatted
  strings. Conversion to VOD-relative seconds happens at ingest, using the single
  `record_start_epoch_ms` anchor.
- **Never able to interrupt OBS** (§4.5).

Until they exist, `clipforge synth-markers` writes marker JSONL in exactly this
format so the ingest side is exercised on the real code path.
