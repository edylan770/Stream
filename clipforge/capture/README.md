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

---

## File contracts

The spec defines the *mechanism* for the anchor (§4.1) and the *shape* of a
marker line (§4.3), but never the files themselves — they did not need to exist
until something had to read them. Ingest reads them now, so the shapes below are
fixed. `tests/fixtures/make_fixture.py` writes exactly these, and the ingest
tests assert against them. **Change one and you must change all three.**

### `anchor.json` — written once, at record start

```json
{
  "schema": 1,
  "record_start_epoch_ms": 1755123456789,
  "source": "obs_websocket",
  "written_at_epoch_ms": 1755123456789
}
```

| Field | Meaning |
|---|---|
| `schema` | Contract version. Ingest refuses anything it does not know. |
| `record_start_epoch_ms` | **The** sync anchor (§4.1). Wall-clock epoch ms at the moment OBS began recording. |
| `source` | `obs_websocket` (preferred — the `RecordStateChanged` event), `hotkey_script` (the fallback in §4.1), or `synthetic` (fixtures). |
| `written_at_epoch_ms` | When the file was written. Differs from the anchor for the hotkey fallback, and the gap is the fallback's error bar. |

`vod_time_s = (event_epoch_ms - record_start_epoch_ms) / 1000.0`. One number per
stream, no drift, no manual alignment — this is the entire sync solution, so it
is worth getting the file right.

### `markers.jsonl` — appended, one line per keypress

```json
{"epoch_ms": 1755123474789, "kind": "marker_maybe"}
{"epoch_ms": 1755123491789, "kind": "marker_definite"}
```

Only those two keys. `kind` is `marker_maybe` (F1) or `marker_definite` (F2).
No VOD times, no local timestamps, no trailing metadata — the daemon must stay
dumb enough that it cannot fail in an interesting way.

Append and flush per line. A crash mid-stream should cost the presses that
never happened, not the ones already written.

### `input.jsonl` — Phase 3, shape fixed by §4.4

```json
{"epoch_ms": 1755123456700, "keys_per_s": 4.2, "mouse_vel_px_s": 1840.5, "clicks_per_s": 2.0}
```

10 Hz aggregate, not per-event.

---

Until the daemons exist, `tests/fixtures/make_fixture.py` produces both files,
so ingest is exercised on the same code path the real capture side will feed.
