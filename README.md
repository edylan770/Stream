# ClipForge

Turns a stream recording into a reviewed shortlist of clip-worthy moments and an
FCPXML timeline you open in Resolve.

Built to [`spec/CLIPFORGE-SPEC.md`](spec/CLIPFORGE-SPEC.md). See
[`HANDOFF.md`](HANDOFF.md) for build state and every place the implementation
deliberately departs from the spec — several of those fix silent-failure bugs
and should not be reverted without reading why.

**Status: the capture layer and Phases 1, 2, 3 and 4 are built and working.**
That means you can take a recording all the way to a finished vertical clip with
burned-in captions. Phase 5 (digests), Phase 6 (trends) and Phase 7 (vision) are
not started — the [roadmap](#roadmap--what-later-phases-will-cost-you) says what
each will need you to install.

One piece carries a warning: **`scene_events` parses OBS's log file and has
never seen a real one**, so its patterns may not match your OBS version — and
when they do not, it produces nothing rather than an error. Next time you are at
the streaming PC, `clipforge scene-events --check "<any OBS log>"` says whether
they work, and prints a report you can act on without moving the log anywhere.
Nothing else depends on it: it feeds chapter boundaries in Phase 5 and §16
explicitly rejects it as a scoring signal.

Phase 2's transcription ships **off** (`extract.whisperx.enabled`), because it
costs a multi-GB download to feed signals nothing weights yet. Captions and the
review screen's transcript panel need it on.

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
| `fixtures` | piper-tts, onnxruntime | ~50 MB, plus ~120 MB of voice models |
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
   `[`/`]` move the window's **start** earlier/later by half a second, `{`/`}`
   its **end**. A window you adjust is the one that gets exported and rendered,
   not the detector's — the readout turns green and says how far you moved it.
   Nudging is also how §17's window-length settings get tuned: `clipforge
   metrics <id>` reports which way you tend to move boundaries and how often the
   window you were given had been clamped, so **nudge freely rather than putting
   up with a boundary you dislike.**
4. **Export** — `.\clipforge.ps1 export <stream_id>` writes an FCPXML.
   In Resolve: *File → Import → Timeline*.
5. **Render** — `.\clipforge.ps1 render <stream_id>` writes a finished vertical
   clip per approved moment, captions burned in and audio normalised to
   −14 LUFS, into the stream's `exports/`. `--preset shorts|tiktok|reels`
   picks the encode settings; they are nearly identical today and differ only
   by a duration cap, which warns and never truncates.

The two are different jobs: an FCPXML is a timeline for an editor to conform
against, a render is the postable file. §10.5 says never to extract clip files
for YouTube assembly, and to extract them only for short-form — which is what
`render` is for.

**Hook text (§8.5).** `clipforge hook <stream_id>` writes one prompt covering
every rendered clip; paste it into a frontier model, save the reply, and
`--apply` it. Nothing is chosen for you — it prints the options and a `--pick`
line. Every reply is validated: ids that do not exist are dropped and counted,
and each entry must quote its own clip verbatim, so a model answering about a
clip it never read is caught rather than believed.

Two optional edits, **both off by default**:
`--set render.mute.enabled=true` silences profanity from the word list in
`phrases.yaml` (§8.6), and `--set render.filler.enabled=true` cuts filler words
out of the mic track (§8.2). Filler removal cuts the **video** too, so on
gameplay it is a visible jump — try it with `--dry-run` first, which prints
what it would cut without encoding. `--dual` writes a muted and an unmuted copy
of each clip.

**Before your first render, check the crop.** Every coordinate in
`clipforge/config/crop_templates.yaml` is a placeholder copied from the spec's
example, and the spec's example was never measured against a real OBS layout.
The default template, `full_frame`, is a centre crop of any 16:9 source — never
wrong, but it throws away the sides of the frame. To fit your own layout:

```powershell
.\clipforge.ps1 render <stream_id> --stills      # one PNG per clip, seconds each
.\clipforge.ps1 render <stream_id> --dry-run     # the geometry and filter graph
```

Edit the numbers, look at the stills, repeat. A still costs about a second; a
full encode costs a minute, which is the wrong loop to iterate in.

Everything the app does is also a command:

```powershell
.\clipforge.ps1 register --master D:\recordings\stream.mkv
.\clipforge.ps1 run <stream_id>
.\clipforge.ps1 score <stream_id> --list     # re-score, free and repeatable
.\clipforge.ps1 status <stream_id>
.\clipforge.ps1 metrics <stream_id>          # is review actually fast enough?
.\clipforge.ps1 export <stream_id> --min-rating 1
.\clipforge.ps1 render <stream_id> --limit 3
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

**The windows are consistently too long or too short** — do not put up with it,
nudge them: `[`/`]`/`{`/`}` during review. After a handful of streams
`clipforge metrics <id>` will tell you which way you keep moving boundaries and
how many of the windows you adjusted had been clamped to
`score.window.min_window_s` / `max_window_s` — which is exactly how §17 says to
tune those two, and it needs your nudges to say anything.

**Footage moved to another drive** — `clipforge db relink --from <old> --to <new>`.

---

## Roadmap — what later phases will cost you

**None of this is built.** It is here so you know what a phase costs before
starting it, not to suggest it exists.

| Phase | Adds | You will need |
|---|---|---|
| **2** Transcript | Word-level transcript, speaker-labelled; speech rate; phrase triggers | `[asr]` (~2 GB), Whisper `large-v3` (~3 GB on first run), a CUDA GPU to be usable |

**Phase 2 is partly built.** The `whisperx` stage works — transcript with
word-level timestamps, VAD, and vocabulary seeding — but it is **off by
default**, because until speaker assignment and phrase detection land it
produces rows nothing reads. To try it:

```powershell
.venv\Scripts\python -m pip install -e ".[asr]"
.\clipforge.ps1 doctor          # reports whether CUDA is visible
```

then set `extract.whisperx.enabled: true` in `local.yaml` (with the CPU block
from `local.yaml.example` if you have no NVIDIA GPU). Measured on the test
fixture with the smallest model on CPU: 2.9% word error rate, and vocabulary
seeding took hero-name accuracy from 7/11 to 11/11.
| ~~**3** Full signals~~ | **Built** — pitch, laughter, silence, overlap, input signals, dual profiles + combined score, gated negatives, preview assets, scene events | Nothing to download. Previews cost ~40 MB and ~3 min per stream, measured on real 720p footage. `scene_events` wants one `--check` run against a real OBS log |
| **4** Auto-finish | Burned-in captions, vertical reframe, loudness normalisation, export presets | Nothing new; re-encode time per clip |
| **5** Digests | Per-stream structured summaries, video ideation, cross-stream compilations. **§11.6's semantic search is already built** (`clipforge search`, and a Search screen in the app) | An API key would be ~$0.10–0.30 per stream, but the paste round trip `clipforge hook` already uses needs none. Search needs no API key — just Ollama and a transcript |
| **6** Trends | Recurring-bit detection, clustering, an idea dashboard | Ollama for embeddings; **60+ streams before it finds anything real** |
| **7** Vision | Kill feed, multikill, clutch detection | OpenCV, plus per-game UI templates you capture yourself |

§15's advice, worth repeating: ship Phase 1, stream ten times, then pick the
next phase from what actually caused friction. The named risk is *"month three,
half the system built, zero videos published, and having become a person who
builds video tooling rather than a person who makes videos."*

---

## Backups

**The database is the irreplaceable tier.** §13.2 is explicit: signals can be
re-extracted from footage, your ratings and judgment calls cannot. Take one now:

```powershell
.\clipforge.ps1 backup
```

Safe to run while the app is open — it opens the database read-only and takes a
compacted copy, so nothing is locked and nothing is modified. On this database
that is 408 KiB down to 88 KiB in about 30 ms.

Set the nightly job up once. This prints the command; it does not run it,
because a scheduled task is a permanent change to your machine:

```powershell
.\clipforge.ps1 backup --schedule
```

Then check what you have, and prove one of them actually restores:

```powershell
.\clipforge.ps1 backup --list
.\clipforge.ps1 backup --verify
```

`--verify` is §13.3 — *"an untested backup is not a backup"*. It decompresses
the newest backup into a scratch directory, opens it with ClipForge itself, and
checks every table against the row counts recorded when the copy was taken. Run
it once after setting up the schedule, and again any time you are about to rely
on it.

**Backups go to `data/backups/`, which is on the same disk as the database.**
That covers a bad migration, a mistaken delete, and a corrupted file. It does
not cover the disk dying. Point `backup.mirror_dir` at a second drive — or at a
folder OneDrive or Dropbox already syncs — and every backup is copied there too:

```yaml
# clipforge/config/local.yaml
backup:
  mirror_dir: 'D:/ClipForgeBackups'
```

An unplugged drive warns and never fails the backup. Retention is §13.2's 30
daily plus 12 monthly, applied automatically; the newest backup is never
deleted, whatever the settings say.

---

## Development

```powershell
.venv\Scripts\python -m pytest -q
```

1654 tests, plus 3 that load a real Whisper model and need `--asr`. The test
fixtures are **synthetic** — ffmpeg `testsrc2` video with
numerically authored audio, colour bars and static, deliberately. Real footage
cannot validate a detector: nobody can say what the correct mic RMS at t=412 of
a real recording is. Each fixture's `manifest.json` carries ground truth by
construction, and every numeric test asserts against that rather than a
hardcoded number.

```powershell
.venv\Scripts\python tests\fixtures\make_fixture.py --duration 600 --name long
```

The one exception is the **speech** fixture, which Phase 2 needs because you
cannot test a transcriber on static. It is Piper TTS dialogue with authored text
and placement, so the ground truth is still known by construction. It needs the
voice models once, and tests skip cleanly without them:

```powershell
.venv\Scripts\python tests\fixtures\make_fixture.py --download-voices
.venv\Scripts\python tests\fixtures\make_fixture.py --kind speech --name speech
```

The **laughter** fixture is the third kind, for §5.5. It is band-limited noise
with a sine written onto its envelope — modulation inside 4–7 Hz and outside it
on both sides, every region at the same level so nothing can be told apart by
loudness. Needs no voice models, and is generated automatically by the tests:

```powershell
.venv\Scripts\python tests\fixtures\make_fixture.py --kind laughter --name laughter
```

It validates that the detector responds to envelope periodicity in the band and
not outside it. It cannot show that real laughter has that signature — that needs
a hand-labelled clip, and `spec/GUESSES.md` records it as the open falsifier.

Layout follows spec §2.3. Directories for unbuilt phases (`digest/`, `ideate/`,
`trends/`) exist as a visible roadmap and contain no code.
