# ClipForge — project state

Read this, then [`spec/CLIPFORGE-SPEC.md`](spec/CLIPFORGE-SPEC.md). The spec is the
requirement; this file records where the build is and which of its instructions have
been deliberately departed from.

**`git log` is the real archive.** Every commit message explains what was built and
why, including the reasoning behind each deviation. `git log --format='%s%n%n%b'` is
worth skimming before changing anything in `score/`, `render/` or `pipeline/`.

---

## Status

**Phase 1 and Phase 0 of §15 are complete. Phase 2 is in progress** — the transcript and
speaker labelling are built; phrase detection and embeddings are not.

Phase 2's stages ship **off by default** (`extract.whisperx.enabled`). §15 says to ship
Phase 1 first, and until the rest of Phase 2 lands a transcript feeds nothing while
costing a multi-GB download.

**The venv is Python 3.12.** WhisperX declares `python <3.14` in every release from
3.7.0 on, and on 3.14 pip silently resolves back to whisperx 3.2.0, which pins a
CTranslate2 with no wheel for that interpreter — so the failure arrives as a C++ build
error for a transitive dependency. `pyproject.toml` pins `>=3.11,<3.14` to make that one
legible error instead. Everything in Phase 1 runs fine on 3.14; the ceiling is for
Phase 2.

| # | Commit | What |
|---|---|---|
| 1 | `0bb502a` | skeleton, packaging, `doctor` |
| 2 | `9c55eb3` `dcfacd6` | layered config; §3.2 schema + migrations |
| 3 | `9a892b1` | synthetic fixture with ground truth by construction |
| 4 | `240a72f` | pipeline stage runner (DAG, staleness, atomic writes) |
| 5–6 | `f8454d9` `42cb994` | `register` command; `probe` stage |
| 7 | `b5e0aa8` | `proxy` stage |
| 8–9 | `b51ff82` `922fd9b` | `audio_split`; proxy encoder auto-detection |
| 10 | `4d872f1` | `audio_features` (RMS at 10 Hz) |
| 11 | `dc9b4c7` | `marker_events` + `synth-markers` |
| 12 | `89ca0d4` | scoring engine (§6.2–§6.7) |
| 13 | `2ca5046` | review screen (§7.3 subset) |
| 14a | `4c7ecd1` | pipeline jobs; `register` split into a core + a CLI |
| 14b | `94a4c19` | the shell's HTTP surface, behind a request guard |
| 14c | `f68a811` | the shell's UI — library, add-a-recording, run |
| 15 | `86ee9e9` | FCPXML export (§10.5) |
| 16 | `f13d7ac` | Phase 0 capture layer; the file contract, defined once |
| 17 | `58ba6c5` | setup docs, launchers |
| 18 | `312783d` | speech fixture (Piper TTS), for Phase 2 |
| 19 | `aaaf651` | `whisperx` stage — transcript + word timestamps (§5.7) |
| 20 | `a552654` | `speaker_assign` — §5.8, in the linear domain |
| 21 | *this* | `phrase_detect`, speech_rate, swear_density, word snapping |

861 tests pass. `.venv\Scripts\python.exe -m pytest -q`, plus 3 that need `--asr`.

**What is not built, by design:** no transcript (Phase 2), no dual profiles or preview
assets (Phase 3), no renderer (Phase 4), no digests (Phase 5), no vision (Phase 7). §15
says to ship this, use it for ten streams, and choose what comes next from observed
friction.

**Untested and needs a real machine:** `obs_anchor`'s WebSocket path. The hotkey
fallback is verified end to end here, but nothing has ever received a real
`RecordStateChanged` event. On the streaming PC: start it, hit record, confirm
`anchor.json` appears next to the `.mkv`, then `clipforge register --master <that file>`
with no `--anchor` flag.

**The one thing missing that Phase 1 does not ask for:** §13.2's nightly
`VACUUM INTO` backup. The spec calls the database the irreplaceable tier and prescribes
it; nothing implements it, there is no `data/backups/`, and the app shell is now the
first component that gets left running for hours. This is the first thing to build if
anything goes wrong before Phase 2 starts.

---

## Deviations from the spec, and why

These were each agreed explicitly. Do not revert one without reading its reason —
several fix silent-failure bugs where the spec's literal text produces code that
looks correct and is not.

**Data model**

- **Candidates are append-only generations.** §6.1 promises re-scoring is free and
  infinitely repeatable; §3.2 gives candidates no stable identity and cascades ratings
  off them, so a re-score would delete the operator's judgment calls — the one thing
  §13.2 calls irreplaceable. A new generation is created only when `config_version`
  differs; identical config replaces in place. Operator ratings carry forward by time
  overlap, tagged `rating_source='inherited'` so §14's tuning never counts one twice.
- **`config_version` is `<profile>@<sha256[:8]>`** of the canonicalized scoring config,
  not a bare profile name — two different weight sets under one name would otherwise be
  indistinguishable.
- **Artifact paths are stored relative to `data_root`**, in POSIX form, so the database
  survives moving to the streaming PC. `master_path` is absolute; `clipforge db relink`
  repairs it.
- **`record_start_epoch_ms` is nullable**, with `marker_time_base`, so footage with no
  OBS anchor can be registered.
- **`exports` gained `export_items`** — §3.2's single `candidate_id` cannot describe an
  FCPXML timeline referencing forty clips.

**Extraction**

- **Absolute dB is stored, not baselined.** §5.4.1's "delta vs. rolling baseline"
  describes how the signal is *consumed* (§6.2 step 3). Baselining at extract time
  would freeze `rolling_baseline_window_s` — a §17 tunable — into every stream ever
  processed.
- **`events.t` is the raw marker press time**, not `t − 20`. Same reason: the retro
  offset is tunable, so the row records the observation and scoring decides what it
  implies.
- **VFR detection fits PTS against a constant rate.** Measured: a CFR 30 fps MKV on a
  1 ms timebase stores alternating 33/34 ms frames, so comparing PTS deltas flags
  *every OBS recording* as variable-rate; `duration_time` reports a nominal value and
  cannot detect real VFR at all.
- **Standard frame rates are never rewritten.** Snapping picks the *nearest* rate;
  "first within tolerance" silently turned exact 60 fps into 59.94.
- **Track roles: metadata tags → §4.2 position → config override.** Two or three audio
  tracks are **refused**, not guessed: mistaking game audio for the mic poisons every
  signal and §4.2 calls it unrecoverable.

**Scoring**

- **Event kernels are unit-*peak*, not unit-area.** A unit-area Gaussian at σ=10 s
  peaks near 0.004, so §6.5's `marker_definite: 3.0` would contribute ~0.01 against
  z-scores swinging ±3 — the highest-weighted signal, silently ignored.
- **The marker kernel is an asymmetric plateau**, not a point estimate at `t−20`. §4.3
  says the reaction delay is a *range* (5–15 s). Overlapping markers combine with
  `max`, not sum — two presses eight seconds apart describe one moment.
- **Window thresholds are measured from each peak's base**, not from zero. §6.3's
  `exit = 0.35 × v_peak` assumes the composite returns to zero; on a marker plateau the
  pedestal is 3.0 and the threshold lands below it, so expansion never terminates.
  Measured: every window came out clamped at 54 s until this was fixed.
- **Peaks require a minimum value (default 0).** A local maximum in negative territory
  is a local maximum of *quietness*, and `0.35 × v_peak` is above `v_peak` when
  negative.
- **`hysteresis_enter` finally does something** — §6.3 defines it and never reads it.
  It decides whether two nearby peaks are two moments or one bump with a wobble.
- **Rolling variance is computed on centered data**, and σ has a floor
  (`score.zscore_std_floor`). Without the floor, a digitally silent stretch has σ=0 and
  every dither becomes z=50.
- **§6.6's spacing penalty is single-pass** (`accepted_only`): as written it penalises a
  third candidate twice and freezes its iteration order against scores it then mutates.
- **Auto-calibration counts candidates, not raw peaks**, and floors its target — §6.7's
  per-hour rate asks a 60 s fixture for 0.45 candidates.
- **§6.2 step 8 is a documented no-op** in Phase 1: one profile, nothing to merge across.

**Media**

- **`-ss` before `-i` is enforced in code**, not by comment — `ffmpeg.run` refuses a
  command where `-ss` has no `-i` after it (A1).
- **Proxy forces CFR** and adds `-pix_fmt yuv420p` (§5.2 omits it; a 10-bit master
  yields a proxy no browser can decode). Target height is computed in Python so small
  sources are never upscaled.
- **The proxy verifies itself after encoding** — duration, frame count, and *keyframe
  spacing*. A2's fixed GOP is the entire basis for reviewing by seeking the proxy; if a
  future ffmpeg breaks `-sc_threshold`, scrubbing degrades silently without this check.
- **Encoder detection is a test-encode**, not an `-encoders` listing: `h264_nvenc` is
  compiled into most builds and fails at open without an NVIDIA driver. Measured on the
  build machine: libx264 333 s vs h264_qsv 139.5 s for the same file.

**Review UI**

- **No autoplay.** §7.3 assumes §7.2's pre-rendered 2 s clips, which are Phase 3.
  Seek-and-hold on focus; `space` plays the window.
- **Range serving is written out**, not delegated — it is the one behaviour the whole
  review experience rests on.
- **`metrics` reports the median**, not the mean: one candidate left on screen over
  lunch would swamp an average and make §7.1's 4 s target unmeasurable.

**The app shell (§2.2 says "local web app" and nothing else)**

- **The loopback bind is not a security boundary.** Every page open in the operator's
  browser can reach this server. Harmless while the routes were read-only; a file
  browser and a run trigger are not. `review/guard.py` checks `Host` on every request
  (the only thing that catches DNS rebinding, which makes an attacker's page genuinely
  same-origin), checks `Origin` when present, and requires an `X-ClipForge` header on
  anything that changes something — a cross-origin *simple* POST needs no preflight, and
  `request.json()` parses a `text/plain` body regardless of its declared type.
- **Adding a recording is a server-side file browser, not a drop target.** A browser
  hands a page a `File` and deliberately never a path, so drag-and-drop would mean
  *uploading* a 40–55 GB master to localhost to discover where it already is.
- **`register` was split into a library core and a CLI**, so the shell can call
  `preflight()` — what registration *would* do, before anything is written. That matters
  for the anchor: without `anchor.json` a stream registers as `marker_time_base='vod'`
  and §4.3's epoch presses are read as VOD seconds, which nothing downstream can detect.
- **One writer at a time is now enforced, not assumed.** `reclaim_crashed` marks *any*
  `running` row dead, so a browser run started beside a terminal `clipforge run` would
  declare that stage dead and re-run it on top of the file it was writing.
  `JobRegistry.start` refuses when this process is already running a stream, and when
  the database shows a `running` stage with a fresh heartbeat from another process.
  Keyed on heartbeat age, so a crash's leftover row still gets reclaimed normally.
- **`clipforge review` no longer requires a reviewable stream.** It used to refuse to
  start unless something had candidates and a proxy — the exact state the shell exists
  to get you out of.

**Capture (Phase 0, §4)**

- **The file contract is defined once**, in `capture/contract.py`. It used to be prose
  in a README, a writer in the fixture generator and a validator in `register`, with the
  README itself saying "change one and you must change all three" — drift waiting to
  happen in the one place drift is undetectable, since a wrong anchor shifts every marker
  in a stream by a constant and nothing downstream can tell.
- **`capture/` imports nothing from the rest of clipforge**, enforced by an AST test.
  §2.1 requires Layer A to run without the application installed, so the folder can be
  copied to the streaming PC and run with stdlib plus `pynput`.
- **`anchor.json` is written beside the recording.** §4.1 never says where it goes.
  `RecordStateChanged` carries `outputPath` and `register.find_capture_file` already
  looks there, so `clipforge register --master <rec>` needs no flags, and stopping and
  restarting OBS mid-session yields one correct anchor per recording. Markers and input
  go to a daily file in a capture directory instead — those daemons cannot know where OBS
  is writing, and §4.5 forbids making them depend on OBS to find out.
- **The input logger cannot record which key was pressed.** `Aggregator.key()` takes no
  arguments and the pynput adapter discards the key object before calling it, so no
  function in the module can see a key identity. A global hook sees every password typed
  into every window; §4.4's own example is rates-only, but that had to be structural
  rather than a habit.
- **Hooks never suppress**, and hotkeys are rebindable. F1 is help or ping in most games
  including Marvel Rivals, and §4.5 says capture must never interrupt OBS.
- **Wall clock for timestamps, monotonic for cadence.** A8 wants epoch ms in the files;
  the 10 Hz aggregation window must be monotonic or an NTP step mid-stream stretches a
  bucket and puts a spurious spike in the one signal meant to detect spikes.

**Transcript (Phase 2, §5.7)**

- **Stages can now be *unavailable* as well as unbuilt.** `StageSpec.available(ctx)`
  returns a reason and the runner defers rather than fails. Marking `whisperx`
  implemented without this put it in the default plan, so `clipforge run` began trying
  to pull a 3 GB CUDA model on every machine — and since `execute` stops at the first
  failure, `score` never ran and streams produced no candidates. Phase 3's Ollama,
  Phase 5's API key and Phase 7's OpenCV will each want the same hook.
- **`extract.whisperx.enabled` is false by default.** §15 says to ship Phase 1 first,
  and until `speaker_assign`, `speech_rate` and `phrase_detect` exist the transcript
  feeds nothing.
- **`compute_type: auto`.** §5.7 hardcodes `float16`, which CTranslate2 *rejects* on CPU
  with an error naming neither the setting nor the cause. `auto` picks per device.
- **`language: en` by default.** §5.7 sets none, so Whisper detects per chunk — and a
  chunk of near-silence is exactly where it decides the language is Welsh.
- **`vocabulary_mode: hotwords`, not §5.7's both.** MEASURED: Whisper has positional
  encodings for 448 prompt tokens; faster-whisper truncates hotwords to 223 *and* the
  initial prompt to 223, and the framing pushes the total to 451. Seeding both raised
  `RuntimeError: No position encodings are defined for positions >= 448` from inside
  CTranslate2, with nothing pointing at the vocabulary. Terms are now trimmed to a token
  budget *before* the model sees them, and the trim is logged.
- **Seeding works, and by a lot.** MEASURED on `tiny`/CPU against the speech fixture:
  unseeded 16.3% WER and 7 of 11 hero-name occurrences ("How could I" for Hawkeye,
  "Numbers'" for "Namor's"); with hotwords 2.9% and 11 of 11. `both` came out at 5.8%
  and 10 of 11 — worse than hotwords alone, as the budget arithmetic predicts.
- **VAD is silero, not §5.7's pyannote.** whisperx's pyannote path fetches
  `pyannote/segmentation-3.0`, which needs a HuggingFace account and licence acceptance —
  contradicting §5.8's own reason for avoiding pyannote. On this machine pyannote's audio
  IO also fails outright: torchcodec cannot load against torch 2.8+cpu. §5.7's
  `vad_onset`/`vad_offset` apply to silero too (verified).
- **A4 verified, not trusted.** With VAD on, the fixture's 19.4 s silence produces zero
  segments.
- **Overlapping segments from two tracks are both kept.** §5.8's "merge by timestamp" is
  undefined when both tracks have speech at once; two people talking over each other are
  two segments, not one. `segments.seq` is assigned after the merge because §12.1 makes
  it the LLM-facing identifier.
- **Unaligned words keep their text and lose their timestamps.** Dropping them corrupts
  what §5.6 matches against; inventing times corrupts §6.3's snapping.
- **`params_hash` excludes `device` and `compute_type`**, so moving between machines does
  not invalidate a forty-minute transcription — the same reasoning as `master_identity`.

**Speaker assignment (§5.8)**

- **Energies are compared as linear power, not as the stored dB.** §5.8's pseudocode does
  `mic_e > party_e * 1.5` on values that commit 10 stores in dBFS — logarithms.
  Multiplying a negative dB by 1.5 moves the threshold *down*, so the test gets easier
  the quieter the other track is. MEASURED on the speech fixture's overlapped line: mic
  −22.72 dBFS, party −21.15 dBFS, so the party track is the louder — and the literal rule
  returns `operator`. Converting to power first returns `both`, which is what an overlap
  is. Every other line in the fixture matches its authored speaker exactly.
- **The mean is taken in the linear domain too.** Averaging dB is the geometric mean of
  the powers, which understates any window containing a peak — exactly the windows that
  matter.
- **`both` is stored**, though §3.2 comments the column as
  `'operator','party','unknown'`. §5.8's algorithm returns `both` and §8.3 depends on it
  ("alternate per word by source track"); `unknown` means there was no energy to judge by.
  No CHECK constraint, so the comment is illustrative rather than binding.
- **The 1.5 ratio is in config** (`extract.speaker.dominance_ratio`). §17's table forgot
  it, and it decides every speaker label and therefore every caption colour in §8.3.
- **No bleed handling, deliberately.** Transcribing mic and party separately would
  duplicate a line if the party's voice reached the mic — but the operator uses
  headphones, so there is no acoustic path, and C5 says not to build ahead of the data.
  **If the audio setup ever changes to speakers, duplicated captions are the symptom**,
  and `speaker_assign` is where the fix belongs since it holds both text and energies.

**Phrases and transcript signals (§5.6, §5.4.1, §6.3)**

- **A9 was being violated the moment an unweighted signal existed.** A9 requires feature
  vectors "logged in full, always, even for signals not currently weighted", but
  `score.runner.build_tracks` only loaded what the profile named. Invisible while Phase 1
  had three signals and the naive profile weighted all three — `game_rms` and `party_rms`
  were already being dropped on any multi-track recording, and `speech_rate` and
  `swear_density` would have joined them. Unweighted signals now load at weight 0: they
  contribute `weight × value` = nothing, so no score moves, but the vector fills.
- **…and are excluded from `contributing_signals`.** The vector answers "what was
  observed"; the breakdown answers "what moved the score". A row of `0.000` in the review
  UI's `?` panel is noise in the one place that explains the number beside it.
- **Phrase matching is word-boundary, not substring.** §5.6 says only "match against the
  transcript"; a substring search fires "no way" inside "I know ways around it", and each
  one is a candidate the operator rejects by hand. Apostrophes are optional in the
  pattern, so "let's" and "lets" are one config entry.
- **Only the longest repeated phrase fires.** MEASURED: a repeated four-word line fired
  every n-gram inside it — `lets go`, `go hawkeye` *and* `lets go hawkeye` at the same
  instant. Non-marker kernels combine with `sum` (§6.2 step 4), so a longer catchphrase
  would have scored higher purely for containing more n-grams. Length is not a signal.
- **`phrase_repeat` fires on the third occurrence, not the first.** Firing on the first
  would mean reaching backwards to score a moment that had not happened yet.
- **The tic filter is a stand-in for §11.2's `is_baseline_tic`**, which is "computed from
  the first ~10 streams" and cannot exist yet. `phrases.yaml` carries a hand-written
  stopword and tic list instead. **Phase 6 should replace it with the measurement** — a
  written list cannot know which phrases *this* operator says constantly; ten streams of
  transcript can.
- **Phrase events carry the speaker**, because §5.4.2 weights a party reaction above the
  operator's own and Phase 3 should be able to split that weight without re-extracting.
  `phrase_detect` therefore requires `speaker_assign`, not `whisperx`.
- **`speech_rate` counts aligned words only**, and reports the excluded share. §5.4.1
  calls it a detector of "both excitement and dead air", so a rate depressed by failed
  alignment would read as silence.
- **Word snapping (§6.3) has a leash and a direction.** §6.3 says only "snap to nearest
  word boundary". Nearest would drag a window that ends in silence to a word twenty
  seconds away, so there is a maximum distance (`score.window.snap_max_distance_s`,
  0.5 s). And nearest can clip a syllable, which is the exact thing §8.2 says snapping is
  *for* — so the start takes a word start at or before it and the end a word end at or
  after it, growing onto the boundary rather than shrinking off it. Same choice commit 15
  made for frame boundaries. A snap that would break the window bounds is skipped rather
  than re-clamped, since re-clamping would silently undo it.

**Export (§10.5)**

- **Ratings are read across generations, never `is_current`.** Inheritance only carries
  a rating forward when the windows overlap by `score.rating_inherit_min_overlap`, so a
  re-score that moves a window further than that strands the approval on a superseded
  generation where an `is_current` read cannot see it.
- **…but the most recent opinion still wins.** A plain union across generations would
  re-export a moment you rated *clip it*, then re-scored, then rated *skip*. Overlapping
  operator-rated windows are clustered into one moment and the latest `rated_at` in the
  cluster is the verdict. `rating_source='inherited'` rows never count — they are copies.
- **Times are written unreduced, over the format's timescale.** `Fraction` would reduce
  30 frames at 29.97 to `1001/1000s` — the same instant, and exactly what a naive parser
  mishandles. Final Cut emits `3603600/30000s`; so does this.
- **Starts floor and ends ceil, never round-to-nearest** (C2: a false negative costs a
  clip). With one measured exception: a time within `BOUNDARY_EPSILON` of a frame is
  *on* it. `candidates.t_start` is SQLite REAL, and `float(250 × 1001/30000)` is a
  double's width above the true rational, so `ceil` was adding a frame at each end of
  every already-aligned window.
- **A VFR master is refused.** It has no constant `frameDuration`, so a constant grid
  over it imports cleanly and is quietly misaligned. `--set export.source=proxy` (CFR by
  construction) or `--allow-vfr`.
- **Export is a command, not a §5.1 stage.** A stage re-runs unattended when its inputs
  change; an export is a decision taken after reviewing.

---

## Running it

Setup, the daily workflow and troubleshooting are in [`README.md`](README.md), which is
written for someone who has not seen this before. In short:

```bash
ClipForge.cmd                                            # double-click; opens the app
.\clipforge.ps1 doctor                                   # environment + chosen encoder
.\clipforge.ps1 export <stream_id>                       # FCPXML for Resolve
```

Launchers are scripts, not a frozen executable: once Phase 2 lands a PyInstaller build
would have to bundle torch — 3-5 GB, quarantined by antivirus, rebuilt on every
dependency change — and would still need ffmpeg and the Whisper models fetched
separately. It adds a build step without removing a setup step.

Everything the app does is also a command — `register`, `run`, `status`, `score`,
`signals`, `metrics`, `db`, `config`, `synth-markers`. All take `--set key.path=value`
to override config for one invocation.

**The fixtures are synthetic test data, not footage.** `fixture_short`, `fixture_long`
and `ntsc` are ffmpeg `testsrc2` video with numerically authored audio — colour bars and
static, deliberately. They exist because real footage cannot validate a detector: nobody
can say what the correct mic RMS at t=412 of a real recording is. Their `manifest.json`
carries the ground truth every numeric test asserts against. Regenerate with
`python tests\fixtures\make_fixture.py [--duration 600 --name long]`.

**`ntsc` exists for one reason.** Every other fixture, and `Testvid.mp4`, runs at an
integer frame rate where every boundary lands on a clean multiple and a rounding bug in
the rational path passes silently. 30000/1001 is the one that does not. Generate it with
`--fps 30000/1001`.

**`speech` is the exception to "synthetic".** Phase 2 needs something to transcribe, and
band-limited noise is exactly what makes a level detector testable and a transcriber
untestable. It is Piper TTS dialogue — speaker A on mic, B on party (§4.2) — with the
text and placement authored, so the manifest still carries ground truth by construction.
Needs the voice models:

```bash
python tests\fixtures\make_fixture.py --download-voices    # ~60 MB each, once
python tests\fixtures\make_fixture.py --kind speech --name speech
```

Tests skip cleanly when they are absent, so a fresh clone with no network still passes.
Piper's noise scales are pinned to zero: VITS samples noise during inference, and a
measuring instrument has to be identical on every machine that regenerates it.

---

## How this build has been run

Worth continuing, because it has caught real bugs:

1. **Plan before code, and disagree first.** Each session started by reading the
   relevant spec sections and reporting what was ambiguous, contradictory, or wrong
   before writing anything. Roughly a dozen genuine spec bugs were found this way.
2. **Measure, don't assume.** The VFR classifier, the encoder detection, the fixture's
   clipping, the pedestal bug and the frame-boundary epsilon were all found by checking
   rather than reasoning.
3. **Never assert against hardcoded numbers.** Every numeric test reads
   `manifest.json`, the config object, or the stage registry. A tolerance wide enough to
   pass a broken fixture is not a tolerance — 0.5 dB was hiding a 0.45 dB clipping error
   until it was tightened to 0.25.
4. **Small commits, each with a full explanation**, and stop after each for the
   operator to test.
5. **Never write to `data/clipforge.db` while verifying.** §13.2 calls it the
   irreplaceable tier and there are no backups. Copy it to a scratch directory and point
   `--set paths.db=...` at the copy. This rule exists because it was broken once, and a
   cleanup that was supposed to remove one fabricated rating also deleted every
   `review_session_duration_s` row in the database.

---

## Starting a fresh session

Phase 1 and the capture layer are done, so the pipeline can now take a real recording
with real markers in it end to end. Phase 2 (the transcript layer) is in progress.

§15 is worth re-reading before adding anything, because it is explicit:

> Ship it, use it for ten streams, then decide what to add based on what actually
> caused friction.

and names the real risk:

> month three, half the system built, zero videos published, and having become a person
> who builds video tooling rather than a person who makes videos.

**Ten real streams is the thing that has not happened.** Every weight, threshold and
window length in `clipforge/config/` is an educated guess (C5), and §17's tuning
procedure needs `signal_firing_rate_by_rating` from actual ratings on actual footage.
Nothing in Phase 2 or beyond is worth as much as that data.
