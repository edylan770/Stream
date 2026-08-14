# ClipForge — project state

Read [`CLAUDE.md`](CLAUDE.md) first — the standing rules. Then this, then
[`spec/CLIPFORGE-SPEC.md`](spec/CLIPFORGE-SPEC.md). The spec is the requirement; this
file records where the build is and which of its instructions have been deliberately
departed from.

[`spec/GUESSES.md`](spec/GUESSES.md) lists every unvalidated parameter with what would
show it is wrong. Most of this project's numbers are guesses (C5); that file is what
keeps them from looking like measurements.

**`git log` is the real archive.** Every commit message explains what was built and
why, including the reasoning behind each deviation. `git log --format='%s%n%n%b'` is
worth skimming before changing anything in `score/`, `render/` or `pipeline/`.

---

## Status

**Phases 0, 1 and 2 of §15 are complete. Phase 4 is in progress.** Phase 3 is
skipped for now by choice — §8's renderer is what turns an approved moment into
something postable, and Phase 3 adds signals that only change *which* moments
surface. Phases 3, 5, 6 and 7 are not started.

Phase 2 ships **off by default** (`extract.whisperx.enabled: false`). §15 says to ship
Phase 1 and stream ten times before adding anything, and a transcript costs a multi-GB
download to feed signals that no profile weights yet. Turn it on with that key plus the
CPU block in `local.yaml.example` if there is no NVIDIA GPU.

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
| 21 | `e37273c` | `phrase_detect`, speech_rate, swear_density, word snapping |
| 22 | `a109847` | `embeddings` (§5.10, Ollama) — Phase 2 complete |
| 23 | `968fdd0` | ASS captions (§8.3): word timeline, per-track colouring, `render:` config |
| 24 | `0a95ba6` | crop templates + `clipforge render` — the first postable file |
| 25a | `806d7af` | UI: design system, one shell bar, space for what is not built |
| 25b | `e843722` | §7.3's transcript beside the window; Render from the browser |
| 26 | *this* | loudness normalisation (two-pass, measured) and export presets |

1120 tests pass. `.venv\Scripts\python.exe -m pytest -q`, plus 3 that need `--asr`.

**What is not built, by design:** no dual profiles, laughter, pitch, input signals or
preview assets (Phase 3); no digests (Phase 5); no trends (Phase 6); no vision
(Phase 7). Phase 4 is partly built — captions, crop and `render` are in; loudness
normalization, export presets, filler removal, profanity muting and hook text are not.

---

## THE NEXT THING TO DO IS NOT CODE

**Zero streams exist.** Every weight, threshold and window length in
`clipforge/config/` is an educated guess — `spec/GUESSES.md` lists all of them and what
would show each one wrong. §17's tuning procedure needs
`signal_firing_rate_by_rating` from `tool_metrics`, which needs ratings, which needs
footage. §15 is blunt about the failure mode:

> month three, half the system built, zero videos published, and having become a person
> who builds video tooling rather than a person who makes videos.

The pipeline can take a real recording end to end today. **Record one, run it, review
it, export it.** Ten streams of that will say more about what to build next than any
amount of reading the spec.

**Three things a first real stream would settle immediately**, each currently a guess:

1. Whether `obs_anchor`'s WebSocket path works at all (see below).
2. Whether `score.peak.target_candidates_per_hour` produces a reviewable number.
3. Whether §7.1's 4-seconds-per-candidate target is met — `clipforge metrics` reports it.

### Untested, and only a real machine can test it

- **`obs_anchor`'s WebSocket path.** The hotkey fallback is verified end to end; nothing
  has ever received a real `RecordStateChanged`. Start it, hit record in OBS, confirm
  `anchor.json` lands next to the `.mkv`, then `clipforge register --master <that file>`
  with no `--anchor` flag.
- **`marker_daemon` and `input_logger` against real hotkeys.** Both are tested through
  fake sources; neither has seen `pynput`.
- **WhisperX on CUDA with `large-v3`.** Only `tiny` on CPU has run. `clipforge doctor`
  reports whether the GPU is visible before a forty-minute surprise.
- **The FCPXML export in Resolve.** Two files were generated and handed over; whether
  Resolve accepts them is unknown.

### Still missing, and not asked for by any phase

**§13.2's nightly `VACUUM INTO` backup.** The spec calls the database the irreplaceable
tier — "signals can be re-extracted from footage; the operator's judgment calls cannot" —
and prescribes a nightly job. Nothing implements it and there is no `data/backups/`.
README carries a one-liner in the meantime. This is the first thing to build the moment
there are real ratings to lose.

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

**The UI's second pass (commit 25a)**

- **One `#topbar` with a block per view, not a global bar above the views.** A
  bar stacked on top would have cost the review screen ~44px of video height,
  on the one screen where vertical space is the constraint (C4). Swapping the
  bar's contents gives the four screens the same chrome for nothing — and keeps
  every element id where the view modules already expect it.
- **`.soon` is a treatment, not a TODO comment.** Everything §7 asks for that
  no phase has built sits in its real position, greyed, with `title` naming the
  phase: §7.3's `[`/`]`/`{`/`}`/`t`/`n`/`e` keys, §7.2's three preview assets,
  Phase 4's render button, Phase 5's digest. The layout therefore does not jump
  the day one lands. The nudge keys matter most: their absence is a
  **measurement gap**, not a missing convenience — §17 tunes
  `min_window_s`/`max_window_s` against "how often the operator nudges
  boundaries during review", and with no keys that number cannot be collected
  at all (GUESSES gap 1). Greyed in the footer, it is visible instead of buried.
- **§7.4's four sections are a sentence, not four headers.** Phase 1 has one
  profile, so the list genuinely is flat; rendering "Combined / Entertainment /
  Gameplay / Marker-anchored" over one ranking would be a lie about what the
  numbers mean. The rail says what the sections need instead.
- **`review_metrics` now returns `target_ms`.** The summary screen carried its
  own hardcoded `4000` while `clipforge metrics` read
  `review.target_ms_per_candidate` from config — change the config and the two
  graded against different numbers, silently. The browser no longer knows the
  target; it is told.
- **`.stage` meant two different things** — a run-view row (a grid) and the
  review-view video column (a flex column), kept apart only by selector
  specificity. The review one is `.stage-pane` and the run one `.stage-row`.
- **A failed `enter()` shows an error view.** Three of the four views awaited a
  fetch with no `catch`, so a server that had stopped left a blank pane and an
  unhandled rejection — the operator's only signal. `router.show` catches, and
  the error view offers Try again / Back to streams.
- **A test reads the element ids out of the JS and asserts the page has them.**
  The views address the DOM as `$("some-id")`; a missing element is `null` and
  fails at the moment a key is pressed, never in a way the JSON route tests
  would see. Ids are extracted rather than listed, so ones added later are
  covered without anyone remembering.
- **`static/` stays flat.** `pyproject.toml` ships `review/static/*` — a
  **non-recursive** glob, so a `static/css/` would not be packaged in a
  non-editable install.

**The transcript panel and Render in the browser (commit 25b)**

- **§7.3's transcript is built, and the column is conditional.** "Transcript
  text for the window displayed alongside" had **zero** references to
  `transcript` or `segments` anywhere in `review/` — Phase 2 put the data in
  the database and nothing read it. But Phase 2 also ships
  `extract.whisperx.enabled: false`, so a stream processed with defaults has no
  segments at all; a permanently-visible third column would have been empty on
  every stream that exists today, costing ~320px of video width to show
  nothing. `stream_detail.has_transcript` decides, and the detail strip says
  what turning it on costs when there is nothing to show.
- **Lines, not words.** The operator is reading to decide whether the moment is
  worth clipping. Word-level timing would multiply the payload for a highlight
  nobody asked for — and `render/words.py` already exists if that changes.
- **A line is included when it OVERLAPS the window, not when it is inside it.**
  A segment starting before the window still puts speech in the clip, and
  hiding it would misreport what the clip contains.
- **Roles come from `render/words.py`'s `track_roles`/`role_for`** — the same
  provenance rule the captions use — and the colours come from
  `render.captions.styles`, sent with the payload rather than duplicated in
  CSS. If review and the export disagreed about who spoke, the operator would
  rate a moment believing one thing and watch the clip say another, and nothing
  would report it.
- **One segment query per stream, not one per candidate.** `load_candidates`
  exists to put everything in one response because a round trip per `j` press
  would spend most of C4's four seconds on latency; loading segments per window
  would have reintroduced that cost server-side. Asserted with sqlite's own
  trace hook, which also catches a query issued further down the call tree.
- **Render is a job kind, not a second mechanism.** `Job.kind` is `run` or
  `render`; `JobRegistry.start` dispatches through `_RUNNERS`, so an unknown
  kind fails before a thread starts rather than starting one that does nothing.
  Still **one job per stream**, not one per kind: a stream is being worked on
  or it is not, which is the same single-writer rule `live_elsewhere` enforces
  against other processes, and two live jobs would mean the run view following
  two logs on one screen.
- **The review summary's Render button navigates to the run view and starts it
  there.** That screen already polls the job and paints the log; a second copy
  of that machinery inside the review screen would be a second place to keep
  correct. When the hand-off cannot start — something already running, or
  nothing approved yet — it says so in the log rather than landing the operator
  on a screen that looks like it ignored the button.
- **Render is offered only once `score` is done**, for the same reason the
  review link is: rendering needs approved moments, and nothing can be approved
  before there are candidates. Otherwise the operator's first render is an
  error message.

**Loudness and export presets (§8.2/§8.3)**

- **Two passes, not §8.3's one — and the numbers are in the module.** MEASURED
  by encoding six windows of the fixture both ways and reading each file on its
  own: one-pass 1.45 LU mean error, two-pass 0.42. Over nine jittered windows
  the worst case is 3.00 against 1.70. Better on both statistics, which is why
  it ships.
- **But it is not reliably better on any ONE clip.** Shifting a window by 0.2 s
  moved a two-pass result 1.9 LU, because pass 1's gated measurement over eight
  seconds is sensitive to which blocks clear the gate. So the finished file is
  measured with `ebur128` and warned about rather than assumed, and
  `verify_tolerance_lu` is set *outside* that measured spread (2.0) instead of
  at a round 1.0 that would fire on ordinary clips.
- **The silence guard reads `ebur128`, not `loudnorm`'s own number.** THE
  finding of this commit, and it was a failing test that produced it. Two-pass
  over the fixture's authored silence lands at −32.0 LUFS against a −14 target,
  eighteen LU off, because pass 1's measurements are meaningless and pass 2
  believes them. The obvious guard does not work: `loudnorm` reports that
  window at **−17.75 LUFS**, indistinguishable from quiet speech, where a
  standalone `ebur128` reports −36.2. Its gate sits above the noise floor, so
  it averages the loudest fragments and reports a level the content lacks.
- **The guard is a gain limit, not a floor.** `max_gain_db`, because gain is
  what does the damage — lifting a noise floor twenty dB is what makes hiss
  audible — and because a limit expressed that way keeps its meaning if
  `target_lufs` moves, where a fixed floor would silently become stricter.
- **`ebur128` is run in its own pass, never chained with `loudnorm`.**
  MEASURED: the same window reads −36.2 alone and −17.8 with `ebur128` chained
  ahead of `loudnorm` in one graph, regardless of how the seek is done. Two
  decodes cost about two seconds; a guard reading the wrong number costs a clip.
- **`loudnorm` outputs 192 kHz.** MEASURED: without an `aresample` the encoder
  is handed `192000 Hz` and AAC-LC tops out at 96. Pinned in the same chain.
- **`linear=true` is requested and, on this material, never granted.** Every
  run came back `normalization_type: dynamic` — linear is refused when the
  required gain would breach the true-peak ceiling, and here it always does.
  The flag stays for quieter sources; nothing claims the result is transparent.
- **`clips.audio_filters()` is a seam, empty today.** §8.6's muting and §8.2's
  filler removal go there, and pass 1 applies the same chain — measuring
  un-muted audio and then normalising muted audio would be wrong by however
  much the muting removed, and nothing downstream could tell.
- **A preset carries encode settings, not a resolution.** §8.4 owns `output`
  and the template's `dst` rectangles are expressed in that space; two owners
  for one number is how they drift. `presets.load` refuses a preset that sets
  one.
- **The three presets are near-identical and the config says so.** They differ
  only by `max_duration_s` today. Those caps are guesses that were not verified
  from here, are marked **arbitrary** in GUESSES, and **warn without ever
  truncating** — silently cutting an approved clip loses the moment it was
  approved for (C2).

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

**Embeddings (§5.10)**

- **§5.10's first-choice model does not exist in Ollama.** It recommends
  `bge-small-en-v1.5` (384-dim); Ollama's library ships nothing under that name. The
  default is §5.10's own second choice, `nomic-embed-text` (768-dim) — ~600 MB per 100
  streams instead of 300 MB, still trivial beside a 4 GB proxy.
- **Vectors are stored L2-normalised.** §5.10's "~50 ms over 200k vectors" is a single
  matmul, which holds only over unit vectors; normalising at query time instead allocates
  a second 600 MB array per search. MEASURED: Ollama's `/api/embed` already returns unit
  vectors for this model, so it is usually a no-op — but the legacy endpoint and other
  models promise nothing, and a search that assumes unit length has to be right.
- **Nomic models use paired task prefixes.** Documents are embedded with
  `search_document: ` (config) and **a query must use `search_query: `**, which
  `extract/embeddings.py` holds as `QUERY_PREFIX` for §11.6's future search. Omitting
  them degrades retrieval *without erroring*.
- **`model` and `dim` are recorded per row and must never be compared across.** Two
  models' vectors occupy unrelated spaces; a cosine between them is finite, ordered and
  meaningless. A search filters by model rather than averaging two geometries.
- **Empty segments are skipped.** An embedding of `""` is a valid vector pointing
  somewhere arbitrary, and it would rank among real results with no text.
- **No search command was built.** §11.6's pull search is twenty lines and Phase 6 —
  with zero streams there is nothing to search, and one built now would be tuned against
  a fixture (C5).

**Captions (Phase 4, §8.3)**

- **A caption line runs to the NEXT word's start, not to its own end.** §8.3's
  pseudocode is `start, end = active_word.start, active_word.end`. A word's
  `end` is the end of its *audio* and the next word starts later, so between
  every pair of words the Dialogue line expires and the screen is blank — the
  caption strobes several times a second. Lines are emitted on the *boundaries*
  between words instead, which keeps the group continuously on screen with only
  the highlight moving. That is what §8.3 describes in prose; the code beside it
  does something else.
- **Colour comes from `segments.track`, never `segments.speaker`.** §8.3's third
  rule — "`both` → alternate per word by source track" — cannot be done from
  `speaker`: `speakers.classify` compares mic and party energy across a span and
  never reads `track`, so a segment labelled `both` holds words from exactly one
  track and has nothing inside it to alternate between. What an overlap really
  produces is *two* segments, one per track, both labelled `both`. Worse,
  `speaker` can be wrong about a segment's own author — a party-track segment
  during a mic-dominant moment is labelled `operator`, and colouring by that
  paints the party's words in the operator's colour. Track is provenance and
  cannot be wrong. One rule then implements all three of §8.3's cases. `speaker`
  is the fallback only when `track` is NULL or §4.2 found a single track.
- **Groups are built per role, and each role has its own `MarginV`.** Two people
  talking at once are two simultaneous word streams; interleaving them into one
  line reads as gibberish ("Oh my / no way / god"). Each role stacks on its own
  row. With one speaker the output is identical.
- **Line boundaries are quantised to centiseconds once, as a shared array.**
  Rounding each line's start and end independently makes adjacent lines overlap
  by up to 10 ms — two copies of the same text drawn on top of each other, which
  reads as a bold flicker — or leaves a hole. Sharing the boundary makes both
  impossible.
- **`PlayResX`/`PlayResY` are the OUTPUT resolution.** libass scales every font
  and margin by them, and §8.4's whole point is that output and source are
  different shapes. Related: `ass` must come *after* `scale` in the chain, which
  is the order §8.3's burn-in command already uses.
- **Unaligned words are interpolated, not dropped.** §5.7 deliberately stores
  null timestamps rather than inventing them. A caption cannot show a hole, so
  the gap between aligned neighbours is divided evenly and every word it touched
  is flagged and counted — a transcript that is mostly interpolated is a caption
  track that is mostly guesswork, and that should be visible.
- **The ASS file is named to the filter graph by bare filename, with ffmpeg run
  from its directory.** MEASURED, and the reason `ffmpeg.run` gained `cwd`:
  inside a filter description ffmpeg re-parses the string, so `-vf ass=C:/x.ass`
  splits at the drive letter and fails with "Unable to parse original_size" —
  naming neither the file nor the colon. Quoting plus `\:` fixes that, but
  **nothing** survives an apostrophe in a parent directory (`C:\Users\O'Brien`
  is an ordinary home directory): not `\'`, not shell-style `'\''`. Running from
  the file's own directory leaves nothing to escape. Inputs and outputs are argv
  elements, not filter values, and stay absolute.
- **`render:` sits outside `VERSIONED_SUBTREES`.** Only `extract` and `score`
  feed `config_version`. Changing a caption colour must not invalidate a
  candidate or force a re-score — nothing under `render` can change which
  moments were detected, only how an approved one is finished.

**Vertical reframe and the render command (Phase 4, §8.4)**

- **`fit` is per-region config, and the default is not `stretch`.** §8.4's own
  example gives gameplay `src` 960×800 (aspect 1.200) and `dst` 1080×1110
  (0.973) — a literal crop-then-scale stretches it vertically by 1.233×. `fill`
  covers the destination and centre-crops the overflow; `contain` fits and pads;
  `stretch` reproduces §8.4 exactly for anyone who wants it.
- **A template declares the resolution it was measured on, and a different
  aspect is refused.** Scaling x and y by different factors moves every region
  somewhere the operator did not put it, and the result is a plausible-looking
  frame that is quietly wrong. Same-shape sources scale (the 640×360 fixture
  against a 1920×1080 template).
- **Overlapping `dst` regions are refused; gaps are reported.** They are not the
  same mistake. An overlap hides part of what the template says to show, and
  there is no layout where that is intended. A gap is a band of `pad_color` —
  a legitimate letterbox, and also exactly what a typo looks like, so it is
  logged with its share of the frame rather than decided.
- **Not `vstack`.** §8.4 says "crop + scale + vstack", which only works for
  full-width regions stacked in source order. `dst` gives arbitrary rectangles,
  so the first region is `pad`ded to the full output at its own offset and the
  rest are `overlay`ed at theirs. General, and it needs no `color` source —
  which would be an infinite stream requiring a duration bound.
- **The upscale warning has a threshold, because some upscaling is inherent.**
  A 16:9 master reframed to 9:16 always enlarges by ~1.78×. Warning on any
  enlargement fires on every clip and therefore means nothing;
  `render.crop.upscale_warn_factor` (2.0) is above that floor.
- **libx264, deliberately not `encoders.select`.** That detector exists to push
  a three-hour proxy through a throughput bottleneck, and was measured doing so
  (libx264 333 s vs h264_qsv 139.5 s). A 45-second deliverable has the opposite
  priority — encoded once, watched many times — and a hardware encoder at equal
  bitrate is visibly worse. §8.3 asks for x264 by name.
- **Which audio track becomes the clip is explicit.** §8.3's burn-in has a bare
  `-af`, so ffmpeg takes stream 0, which is the mix only by luck of the OBS
  layout. `render.audio.source: auto` reuses `proxy.audio_map_index` — §5.2
  needed the same rule for the same reason. A short carrying game audio and no
  voice is a failure that survives all the way to upload.
- **`--stills` and `--dry-run` exist because the coordinates are placeholders.**
  The loop "edit numbers → wait for an encode → look" is the slow way to measure
  a layout. `--dry-run` prints the resolved geometry and the filter graph;
  `--stills` writes one PNG per moment through the identical graph. Stills are
  deliberately **un-captioned**: they exist to show where the rectangles are,
  and seeking to a representative frame would put the output clock out of step
  with the caption times.
- **`render.source` is separate from `export.source`.** An FCPXML references the
  master for an editor to conform against and may legitimately be pointed at a
  proxy; a burned-in short *is* the deliverable and wants the quality source.
- **Every render module raises from one `RenderError` base** (`render/__init__`).
  Without it the first failure a new operator meets — a mistyped template name —
  arrived as a traceback, since the command only caught its own exception type.
- **`--set key=#RRGGBB` used to become `None`.** `#` starts a YAML comment, so
  `parse_override` reduced every hex colour to an empty document and the caption
  rendered in the fallback colour with no error. A value the parser reduced to
  `None` that was not *written* as a null now keeps its raw text; `--set k=` and
  `--set k=null` still mean null. Found by a test asserting that two different
  caption colours produce two different params hashes.

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

Read [`CLAUDE.md`](CLAUDE.md) for the standing rules, this file for state, and
[`spec/GUESSES.md`](spec/GUESSES.md) before touching any number. Then open with
something like:

> Read CLAUDE.md, HANDOFF.md and spec/CLIPFORGE-SPEC.md §<the sections you are
> touching>. Phases 0–2 are done. <What you want built.> Same as before: tell me
> anything you'd do differently first, test against the fixture manifest rather than
> hardcoded numbers, small commits, stop after each.

The remaining phases, in §15's order and with what each will cost:

| Phase | Adds | Needs |
|---|---|---|
| **3** Full signals | Pitch, laughter, silence, overlap, input signals, dual profiles + combined score, gated negatives, spacing, preview assets | Nothing new to download. Phase 0's `input_logger` must have run on a real stream for four of these |
| **4** Auto-finish | ASS captions with speaker colouring, vertical reframe, loudnorm, export presets | Real crop coordinates, measured off a real OBS layout |
| **5** Digests | Structured per-stream summaries, theme/assembly passes | An API key for a frontier model (~$0.10–0.30/stream) |
| **6** Trends | N-gram bits, HDBSCAN clustering, idea dashboard, §11.6 pull search | **60+ streams before clustering finds anything real.** The n-gram half and the search are cheap and work sooner |
| **7** Vision | Kill feed, multikill, clutch, MVP | OpenCV plus per-game templates captured by hand; §5.9 says build it last and cut it if hard |

Two things worth doing before any of them, both small:

- **§13.2's backup job.** The database is the irreplaceable tier and there is none.
- **Phase 6's n-gram baseline** (`is_baseline_tic`) once ten streams exist, replacing the
  hand-written stopword list in `phrases.yaml` — the one place a guess is currently
  standing in for a measurement the corpus would provide.

But the honest answer is at the top of this file: **the next thing to do is record a
stream.**
