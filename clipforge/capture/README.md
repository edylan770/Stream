# Layer A — Capture (spec §4)

**Phase 0. Built.** These run on the streaming PC, during the stream, and are
independent of the rest of this application (§2.1): if the input logger dies
mid-stream, one signal is lost, the stream is not.

| Script | Purpose | Spec |
|---|---|---|
| `obs_anchor.py` | record-start wall-clock epoch ms → `anchor.json` beside the recording | §4.1 |
| `marker_daemon.py` | F1/F2 hotkeys → `markers-<date>.jsonl` | §4.3 |
| `input_logger.py` | keyboard/mouse activity at 10 Hz aggregate → `input-<date>.jsonl` | §4.4 |

Three rules they obey:

- **Epoch milliseconds everywhere** (A8). Never local time, never formatted
  strings. Conversion to VOD-relative seconds happens at ingest, using the
  single `record_start_epoch_ms` anchor.
- **Never able to interrupt OBS** (§4.5). Hooks do not suppress keys, and no
  failure path raises out of a callback.
- **Never record which key was pressed.** See below.

---

## Deploying

`clipforge/capture/` depends on nothing else in this package — `contract.py`
imports only the standard library, and a test enforces that. So the whole folder
can be copied to the streaming PC and run against a bare Python:

```bash
pip install pynput obsws-python
python -m clipforge.capture.obs_anchor
```

Or, if clipforge is installed there anyway:

```bash
pip install "clipforge[capture]"
```

In OBS: **Tools → WebSocket Server Settings → Enable**. Note the port (4455)
and password.

### Start these before you start streaming

```bash
python -m clipforge.capture.obs_anchor    --password <obs-ws-password>
python -m clipforge.capture.marker_daemon --dir D:/capture
python -m clipforge.capture.input_logger  --dir D:/capture
```

Three separate processes, deliberately (§4.5). One crashing costs one signal.

### Then, afterwards

```bash
clipforge register --master D:/recordings/2026-08-14.mkv --markers D:/capture/markers-2026-08-14.jsonl
```

`anchor.json` needs no flag — `obs_anchor` writes it *beside the recording*, and
`register` looks there.

---

## Where each file goes, and why they differ

**`anchor.json` goes beside the recording.** §4.1 specifies the mechanism and
never says where the file lives. OBS's `RecordStateChanged` event carries
`outputPath`, so the anchor is written into that folder — which means
`clipforge register --master <recording>` works with no flags, and stopping and
restarting OBS mid-session produces one correct anchor per recording instead of
one file silently overwriting another.

**Markers and input go to a capture directory, one file per day.** Those daemons
have no idea where OBS is writing, and §4.5 forbids making them depend on OBS to
find out. Presses outside any recording convert to out-of-range VOD times and
are dropped at ingest, so a whole day in one file costs nothing.

---

## The privacy rule

`input_logger` installs a global keyboard hook. That hook sees **everything
typed into every window on the machine**, including passwords typed into a
browser while OBS sits idle.

§4.4's own JSONL example is rates only, so aggregate-only is what the spec asks
for. In this implementation it is structural rather than a matter of care:
`Aggregator.key()` **takes no arguments**. The pynput adapter is a one-line
lambda that discards the key object before calling it, so no function in the
module can see a key identity even if a later change wanted one to. A test
asserts the signature.

If you ever need to change that file, keep that property.

---

## File contracts

**Defined once, in [`contract.py`](contract.py).** The daemons write through it,
`tests/fixtures/make_fixture.py` fakes through it, and
`clipforge/ingest/register.py` validates against it. It used to be prose here
plus two independent implementations, which is drift waiting to happen in the
one place drift cannot be detected — a wrong anchor shifts every marker in a
stream by a constant and nothing downstream can tell. There is no checksum for
"these timestamps are forty seconds late".

### `anchor.json`

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
| `record_start_epoch_ms` | **The** sync anchor (§4.1). |
| `source` | `obs_websocket` (preferred), `hotkey_script` (§4.1's fallback), `synthetic` (fixtures). |
| `written_at_epoch_ms` | When the file was written. For the hotkey fallback the gap from the anchor is that fallback's error bar. |

`vod_time_s = (event_epoch_ms - record_start_epoch_ms) / 1000.0`. One number per
stream, no drift, no manual alignment — the entire sync solution, which is why
it is worth this much care.

**The WebSocket timestamp is taken when the event arrives**, not from anything
inside it: OBS does not report when it started. Delivery latency is tens of
milliseconds against §4.3's 5–15 second reaction window, so it does not matter —
but that is why `source` is recorded rather than assumed.

### `markers-<date>.jsonl`

```json
{"epoch_ms":1755123474789,"kind":"marker_maybe"}
{"epoch_ms":1755123491789,"kind":"marker_definite"}
```

Only those two keys. `kind` is `marker_maybe` (F1) or `marker_definite` (F2).
No VOD times, no local timestamps, no metadata — the daemon stays dumb enough
that it cannot fail in an interesting way.

§4.3's `t − 20s` retro offset is **not** applied here. It is a §17 tunable, so
the file records the observation and scoring decides what it implies.

Appended and flushed per line: a crash mid-stream costs the presses that never
happened, not the ones already made.

### `input-<date>.jsonl`

```json
{"epoch_ms":1755123456700,"keys_per_s":4.2,"mouse_vel_px_s":1840.5,"clicks_per_s":2.0}
```

10 Hz aggregate, not per-event. Mouse travel is **distance**, not displacement:
sampling position at 10 Hz would miss the fast part of a flick, which is the
entire point of the signal.

Consumed by the `input_signals` stage — Phase 3, not yet built.

---

## Hotkeys

F1/F2 per §4.3, rebindable because F1 is help or ping in most games including
Marvel Rivals:

```bash
python -m clipforge.capture.marker_daemon --bind f9=marker_definite --bind f10=marker_maybe
```

Keys are never suppressed. The game still sees them.
