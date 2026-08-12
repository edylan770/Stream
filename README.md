# ClipForge

Turns a stream recording into a reviewed shortlist of clip-worthy moments and an
FCPXML timeline you open in Resolve.

Built to [`spec/CLIPFORGE-SPEC.md`](spec/CLIPFORGE-SPEC.md). See
[`HANDOFF.md`](HANDOFF.md) for build state and every place the implementation
deliberately departs from the spec — several of those fix silent-failure bugs
and should not be reverted without reading why.

**Status: Phase 1 and the capture layer are built and working. Phases 2–7 are
not.** The [roadmap](#roadmap--what-later-phases-will-cost-you) says what each
later phase will need you to install; nothing there works yet.

---

## Quick start

```powershell
winget install --id Gyan.FFmpeg
py install 3.12
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\clipforge doctor
```

`doctor` tells you what is missing. When it is clean, double-click
**`ClipForge.cmd`** — the app opens in your browser.

---

## Prerequisites

| What | Why | Notes |
|---|---|---|
| **Python 3.12 or 3.13** | | **Not 3.14.** See below. |
| **ffmpeg + ffprobe** | probe, proxy, audio split | `winget install --id Gyan.FFmpeg`, then reopen your terminal |
| **OBS Studio** | recording, and the record-start anchor | Only on the machine you stream from |
| A GPU | Phase 2's transcription | Not needed for anything built today |

### Why not Python 3.14

WhisperX — the transcription engine §2.2 selects, and §5.7 is written entirely
around its API — declares `python <3.14` in every release from 3.7.0 onward. On
3.14 pip silently falls back to whisperx 3.2.0, which pins an old CTranslate2
with no wheel for that interpreter, and the failure surfaces as a C++ build
error for a dependency you never asked for. `pyproject.toml` pins
`>=3.11,<3.14` so you get one clear message instead.

Everything built today runs fine on 3.14. The ceiling is there for Phase 2.

---

## Installing

The project installs differently on each machine it runs on, because it does
different work on each.

```powershell
# This machine — developing and reviewing
.venv\Scripts\python -m pip install -e ".[dev]"

# The streaming PC — capture only (§2.1: Layer A must run on its own)
pip install "clipforge[capture]"

# Wherever transcription runs — Phase 2, not yet built
.venv\Scripts\python -m pip install -e ".[asr]"
```

| Extra | Pulls | Size |
|---|---|---|
| `dev` | pytest, httpx | small |
| `capture` | pynput, obsws-python | small |
| `fixtures` | piper-tts, onnxruntime | ~50 MB + voice models |
| `asr` | whisperx, torch, faster-whisper | **~2 GB of wheels**, plus models on first run |

### Machine-local settings

Copy `clipforge/config/local.yaml.example` → `clipforge/config/local.yaml`
(gitignored) for anything naming a drive, a port, or an install location.
Everything else in `clipforge/config/` is the shared tuning surface (§17) and
*is* tracked — that is where weights, thresholds and window lengths live, and
none of them are hardcoded anywhere else.

```powershell
.venv\Scripts\clipforge config show     # merged result, and which layer set each value
```

---

## Launching

| How | What it does |
|---|---|
| Double-click **`ClipForge.cmd`** | Starts the app and opens your browser |
| `.\clipforge.ps1 <command>` | Any command, without activating the venv |
| `.venv\Scripts\clipforge <command>` | The same thing, spelled out |

Both scripts find the venv themselves and tell you what to do if setup is
incomplete. To put ClipForge on your desktop: right-click `ClipForge.cmd` →
*Show more options* → *Send to* → *Desktop (create shortcut)*.

---

## Setting up capture (do this once)

Everything downstream depends on this, and §4.2 is blunt that getting the audio
tracks wrong is **unrecoverable after the fact** — mic RMS is meaningless if
game audio is mixed into it, and no later version can separate them.

### 1. OBS audio tracks — non-negotiable

*Settings → Output → Recording*, tick tracks 1–4, then in the Audio Mixer's
Advanced Audio Properties assign:

| Track | Source |
|---|---|
| 1 | Mixed (everything) |
| 2 | Mic only |
| 3 | Game audio only |
| 4 | Discord / party only |

Recording format **MKV** (crash-safe; MP4 is unrecoverable if OBS dies
mid-recording — A7).

### 2. OBS WebSocket

*Tools → WebSocket Server Settings → Enable*. Note the password.

### 3. Start the capture daemons before you stream

```powershell
python -m clipforge.capture.obs_anchor    --password <obs-ws-password>
python -m clipforge.capture.marker_daemon --dir D:\capture
python -m clipforge.capture.input_logger  --dir D:\capture
```

Three separate processes on purpose (§4.5) — if one dies you lose one signal,
not the stream. See [`clipforge/capture/README.md`](clipforge/capture/README.md)
for what each writes and where.

**While streaming:** tap **F1** when something *might* have been good, **F2**
when it definitely was. Press late and press often — §4.3 assumes a 5–15 second
reaction delay and scoring compensates. F1 and F2 are not swallowed; the game
still sees them.

> The input logger records **how much** you typed, never **what**. That is
> structural, not a policy — see its README section.

---

## Daily workflow

```powershell
.\clipforge.ps1 review
```

Then, in the app:

1. **Add a recording** — browse to the `.mkv`. It stays where it is; only a path
   is stored. Masters are 40–55 GB and nothing copies them.
2. **Run** — probe, proxy, audio split, RMS extraction, marker parsing, scoring.
   20–40 minutes for a 4-hour stream, mostly the proxy encode. Leave it.
3. **Review** — `j`/`k` to move, `1`/`2`/`3` to rate, `space` to play the window,
   `?` for why it scored. Target is 4 seconds a candidate.
4. **Export** — `.\clipforge.ps1 export <stream_id>` writes an FCPXML.
   In Resolve: *File → Import → Timeline*.

Everything the app does is also a command:

```powershell
.\clipforge.ps1 register --master D:\recordings\stream.mkv
.\clipforge.ps1 run <stream_id>
.\clipforge.ps1 score <stream_id> --list     # re-score, free and repeatable
.\clipforge.ps1 status <stream_id>
.\clipforge.ps1 metrics <stream_id>          # is review actually fast enough?
.\clipforge.ps1 export <stream_id> --min-rating 1
```

`clipforge --help` lists them all. Every command takes
`--set key.path=value` to override config for one invocation.

### Re-scoring is free

Extraction runs once; scoring is a pure function over what it stored (§6.1). Bad
weights cost nothing — change them and re-score the whole back catalogue.
**Your ratings survive it.** They are carried onto the new candidates, and the
FCPXML export reads them across every scoring generation, so a re-score can
never drop a moment you already approved.

---

## Troubleshooting

**`doctor` says ffmpeg is missing but I installed it** — reopen the terminal;
winget adds to PATH only for new shells. Or set `paths.ffmpeg` in `local.yaml`.

**The proxy encode is very slow** — `doctor` reports the encoder it picked. On a
machine with no hardware encoder it falls back to libx264, which is roughly
2.4× slower. Nothing is wrong; it just takes longer.

**"has no candidates yet"** — the pipeline ran but scoring found nothing above
threshold, or it has not run. `clipforge status <id>` shows every stage and why
it will or will not run.

**Markers are all in the wrong place** — `anchor.json` was missing when you
registered, so marker times were read as VOD seconds instead of being converted
(§4.1). `clipforge status <id>` shows the time base. Re-register with
`--anchor`.

**Review feels slow** — `clipforge metrics <id>`. §7.1 sets a hard target of 4
seconds per candidate and says that if review exceeds it, the UI gets fixed
before any other feature anywhere in the system.

**Footage moved to another drive** — `clipforge db relink --from <old> --to <new>`.

---

## Roadmap — what later phases will cost you

**None of this is built.** It is here so you know what a phase costs before
starting it, not to suggest it exists.

| Phase | Adds | You will need |
|---|---|---|
| **2** Transcript | Word-level transcript, speaker-labelled; speech rate; phrase triggers | `[asr]` (~2 GB), Whisper `large-v3` (~3 GB on first run), a CUDA GPU to be usable |
| **3** Full signals | Pitch, laughter, silence, input signals, dual profiles, preview assets | Nothing new to download; ~25 MB of preview assets per stream |
| **4** Auto-finish | Burned-in captions, vertical reframe, loudness normalisation, export presets | Nothing new; re-encode time per clip |
| **5** Digests | Per-stream structured summaries, video ideation, cross-stream compilations | An API key for a frontier model (~$0.10–0.30 per stream) |
| **6** Trends | Recurring-bit detection, clustering, an idea dashboard | Ollama for embeddings; **60+ streams before it finds anything real** |
| **7** Vision | Kill feed, multikill, clutch detection | OpenCV, plus per-game UI templates you capture yourself |

§15's advice, worth repeating: ship Phase 1, stream ten times, then pick the
next phase from what actually caused friction. The named risk is *"month three,
half the system built, zero videos published, and having become a person who
builds video tooling rather than a person who makes videos."*

---

## Backups

**There are none yet, and the database is the irreplaceable tier.** §13.2 is
explicit: signals can be re-extracted from footage, your ratings and judgment
calls cannot. Until the nightly backup is built, copy it yourself:

```powershell
.venv\Scripts\python -c "import sqlite3; sqlite3.connect(r'data\clipforge.db').execute(\"VACUUM INTO 'data\backup.db'\")"
```

Safe to run while the app is open.

---

## Development

```powershell
.venv\Scripts\python -m pytest -q
```

715 tests. The test fixtures are **synthetic** — ffmpeg `testsrc2` video with
numerically authored audio, colour bars and static, deliberately. Real footage
cannot validate a detector: nobody can say what the correct mic RMS at t=412 of
a real recording is. Each fixture's `manifest.json` carries ground truth by
construction, and every numeric test asserts against that rather than a
hardcoded number.

```powershell
.venv\Scripts\python tests\fixtures\make_fixture.py --duration 600 --name long
```

Layout follows spec §2.3. Directories for unbuilt phases (`digest/`, `ideate/`,
`trends/`) exist as a visible roadmap and contain no code.
