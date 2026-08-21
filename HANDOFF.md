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

**Phases 0, 1, 2, 3 and 4 of §15 are complete.** Phase 3 was originally skipped
by choice — §8's renderer is what turns an approved moment into something
postable, where Phase 3 only changes *which* moments surface — and was then
built out across commits 31–39. **Phase 5 is in progress** — §11.6's search
(40), §9.3's chapters (41) and §12's shared LLM plumbing (42) are in; §9.4's
digest and §10's three passes are not. Phases 6 and 7 are not started.

**Commit 43 is not a feature.** It fixes a bug that silently destroyed operator
ratings on a re-score — see the first two bullets of "Deviations" below. Nothing
else in the build depends on it; it is first because §13.2 calls ratings the
irreplaceable tier and two documented workflows walked straight into it.

One caveat on that "complete": **`scene_events` is built but its OBS log parser
has never seen a real log** and fails silently when wrong. See "Still missing"
below for the one command that settles it.

**A recording can now go all the way**: register → run → review → render →
hook, ending in a 1080×1920 MP4 with burned-in captions, normalised audio and
a hook line chosen by the operator.

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
| 26 | `7dfcc87` | loudness normalisation (two-pass, measured) and export presets |
| 27 | `2427dd5` | §8.6 profanity muting, §8.2 filler removal — both toggles, both off |
| 28 | `9a6d852` | §8.5 hook text via a paste round trip — **Phase 4 complete** |
| 29 | `572a885` | §13.2's backup + §13.3's restore test — the irreplaceable tier, covered |
| 29a | `3b0d1ab` | the backup manifest describes the copy, not the source (a race in 29) |
| 30 | `1198f4b` | §7.3's nudge keys — **GUESSES gap 1 closed**; §17 can tune the window bounds |
| 31 | `21cca7b` | pitch (§5.4.1), and what the first signal with gaps does downstream |
| 32 | `5b7aed2` | §5.4.1/§5.4.3's derived signals, and two definitions that had to change |
| 33 | `6400c72` | §5.5's laughter detector, and the three ways it was wrong |
| 34 | `f0d894a` | drop a recording on the add screen, and see its tracks first |
| 35 | `fbb039a` | §4.4's input log, and why a gap in it is not a zero |
| 36 | `8c37ac7` | §6.4's gated negatives, and a penalty that removes nothing |
| 37 | `4a5fca1` | §6.5's two profiles, and a combined ranking that agrees exactly |
| 38 | `3188777` | §7.2's preview assets — **§7.3's autoplay is real**; §7.2's own preset is 9.5× too slow |
| 39 | `4a6f5a3` | `scene_events` — **Phase 3 complete**, and the one unvalidated parser in the project |
| 40 | `3243b85` | §11.6's pull search — and the config value that was changing its results |
| 41 | `6b0e01d` | §9.3's chapters — three of four inputs dead, and a merge rule the fixture corrected |
| 42 | `aa3dfda` | `clipforge/llm/` — §12's four rules in one place, before three call sites each grew their own |
| 43 | `8d0d409` | a re-score over rated candidates no longer deletes the ratings — `config_version` covers the config, not the data |
| 44a | `48e6778` | `clipforge/moments.py` — one opinion per moment, and the two readings of "was this marked?" |
| 44b | `ba55fb3` | §14's three missing tuning metrics — ranked on a statistic that needs no threshold |
| 45a | `5d1a119` | the written rubric: storage, versioning, the `$rubric` seam — and migration `0003` |
| 45b | `8d6940e` | the rubric editor on the summary screen, and §7.3's `n` key |
| 46 | *this* | an authored three-chapter stream, and a scripted LLM source — both instruments for §9.4 |

1747 tests pass. `.venv\Scripts\python.exe -m pytest -q`, plus 3 that need `--asr`.
The suite takes 25-30 minutes; `previews` and the fixture encodes are most of it.

**What is not built, by design:** no digest and no §10 passes (the rest of Phase 5);
no trends (Phase 6); no vision (Phase 7). Of Phase 5, §11.6's pull search (40),
§9.3's chapters (41) and §12's validation and transports (42) are in — the three
items testable with no footage. **Nothing has ever called a model**: `llm.source`
is the paste round trip, and the API source reports itself unavailable.

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

**`scene_events` is built (commit 39) and UNVALIDATED.** There is no OBS on the build
machine (`%APPDATA%\obs-studio` does not exist), so its regexes are transcribed from the
documented log format and not one of them has met real text. **It is the only thing in
the project in that state**, and it fails silently when wrong — zero events, which looks
exactly like a session in which nobody switched scenes.

**What to do with it, in full, next time you are at the streaming PC:**

```bash
clipforge scene-events --check "%APPDATA%\obs-studio\logs\<any log>.txt"
```

That prints which patterns fired, the share of lines carrying a parseable timestamp, the
recording spans found, the scene timeline, and every line mentioning a scene or a
recording that no pattern claimed. **Send back that report, not the log** — OBS logs
carry machine paths, hardware details and sometimes stream URLs. Then:

1. Fix any dead pattern in `extract.scene_events.patterns` — the only place in the
   codebase with a regex for OBS's text.
2. Drop the log into `tests/fixtures/obs_logs/` (redacted if you like; only the
   timestamps, scene lines and recording banners are read). That directory is
   parametrized, so it is covered with no test changes. Delete the `*-UNVALIDATED.txt`
   files once a real one is in there.
3. Move the `scene_events` rows in `spec/GUESSES.md` from **arbitrary** to **grounded**
   and drop the banner from `obs_log.py`'s docstring.

Attach a log to a stream with `clipforge register --obs-log <file> --force`. It is low
value by the spec's own account (§16 rejects scene changes as a scorer; §9.3 makes them
tie-breaker #4 of 4), and unlike the marker and input logs **OBS keeps its logs whether
or not we ask** — so nothing was lost by building it before the format was known, and
nothing is lost by leaving it unvalidated until there is a stream to try it on.

Otherwise nothing. **§13.2's backup is built** (commits 29/29a), **§7.3's nudge keys are
built** (commit 30, closing GUESSES gap 1) and **§7.2's preview assets are built**
(commit 38) — the instrumentation exists and now wants ~10 real streams to say
anything.

Still greyed in the review footer, and still genuinely unbuilt: §7.3's `t` (tag) and
`e` (export queue). `ratings.tags` exists and nothing writes it. **`n` is built as of
commit 45b** — `ratings.note` had been accepted end to end by the server since commit 30
with no client ever sending one.

---

## Deviations from the spec, and why

These were each agreed explicitly. Do not revert one without reading its reason —
several fix silent-failure bugs where the spec's literal text produces code that
looks correct and is not.

**Data model**

- **Candidates are append-only generations.** §6.1 promises re-scoring is free and
  infinitely repeatable; §3.2 gives candidates no stable identity and cascades ratings
  off them, so a re-score would delete the operator's judgment calls — the one thing
  §13.2 calls irreplaceable. A new generation is created when `config_version` differs
  **or when the current generation carries any operator rating**; only an unrated stream
  under an unchanged config replaces in place. Operator ratings carry forward by time
  overlap, tagged `rating_source='inherited'` so §14's tuning never counts one twice.
- **…and the rating clause above was missing until commit 43, which is a bug that
  destroyed ratings.** `config_version` hashes `extract`, `score` and every profile's
  weights — it does **not** hash the signals and events those weights are applied to,
  because those are data. So an unchanged config over *changed data* took the
  replace-in-place path, which sets `prior = []` and `DELETE`s the generation, and
  `ratings.candidate_id ON DELETE CASCADE` did the rest. **Two workflows this file
  documents reach it**: `register --input-log <file> --force` and
  `register --obs-log <file> --force`, each followed by a run that cascades into `score`.
  MEASURED: the regression test failed `assert 0 == 1` before the fix — the rating was
  gone, with nothing reported. Replacing in place is an optimisation for tidy generation
  numbers on an unreviewed stream; append-only is a guarantee, and the guarantee wins
  once a human has rated anything.
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

- **Autoplay is real as of commit 38, and this is what it used to say:** "No
  autoplay. §7.3 assumes §7.2's pre-rendered 2 s clips, which are Phase 3.
  Seek-and-hold on focus; `space` plays the window." The 2 s webm now loops
  silently on focus exactly as §7.3 describes. The proxy player stays mounted
  underneath and stays seeked to the peak, so `space` starts the real window
  immediately rather than re-buffering — and the loop comes back when `space`
  stops, because the operator is still looking at the same candidate. VERIFIED
  in a browser end to end: focus loops, `j` swaps the clip, `space` hands over
  to the proxy and back.
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
  the day one lands. **It worked**: commit 30 took the nudge keys out of
  `.soon` and commit 38 took the preview assets, and neither moved anything
  around them. What is left greyed is §7.3's `t`/`e` and Phase 5's digest — commit 30 took the nudge keys, commit 38 the preview assets, and commit 45b `n`. The nudge keys matter most: their absence is a
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

**Chapter segmentation (Phase 5, §9.3)**

- **Three of §9.3's four inputs produce nothing on a default stream.** Long
  silence is live everywhere; the embedding shift needs `whisperx` (off by
  default); scene changes need an OBS log whose parser has never been validated;
  and **game changes have no producer anywhere in the system** — `streams.games`
  is an untimed JSON array and per-moment game identification is §5.9's vision,
  Phase 7. So this is a long-silence splitter today, and `clipforge chapters`
  reports which input produced each boundary and why the others did not. Same
  discipline as commit 37's live-weight table.
- **§9.3's own embedding formulation is the weaker of two, MEASURED.** It asks
  for "cosine distance between consecutive rolling-window embeddings"; against a
  maximal topic seam that form MISSES at window 2 where comparing before/after
  centroids at each gap hits at windows 2, 3 and 4. The before/after form ships.
- **…but the signal is weak either way, and that is the number to remember.**
  Even for two topics as unrelated as a hero shooter and pastry, the boundary
  peak is only **1.2–1.3× the mean** distance. `nomic-embed-text` compresses
  similarity so hard that everything is roughly half-similar to everything — the
  same property that kept `min_similarity` out of `search:`. Every boundary
  therefore carries its strength and the report prints it, so a 1.3× bump cannot
  present itself as a topic change.
- **Prominence is expressed in the distance array's OWN standard deviations.**
  MEASURED: the raw scale moves with the window — max 0.466 at window 1 against
  0.138 at window 2 — so an absolute `min_distance` would be strict at one
  setting and meaningless at another. `score/windows.py`'s existing peak finder
  does the work; only the threshold handed to it is computed per stream.
- **THE MERGE DEFERS TO THE MOST TRUSTWORTHY SOURCE, AND THE FIXTURE FORCED IT.**
  The first cut took the earliest boundary in a cluster, reasoning that starting
  a chapter at the first evidence loses no content. The speech fixture
  immediately showed why that is wrong: a **1.21-sd embedding bump at 29.4 s
  displaced a nineteen-second silence at 52.0 s**, moving the boundary 23 seconds
  early into the middle of a sentence. Priority is `silence > embedding > scene`
  — silence is the only one validated on real data and its timestamp means
  something exact, and §9.3 itself calls scene changes a tie-breaker. The
  survivor records what corroborated it.
- **Scene changes never propose a boundary alone.** §9.3 calls them "weak signal,
  tie-breaker only" and §16 rejects them as a scorer; a cluster containing
  nothing else is dropped.
- **A silence boundary is the END of the gap, not its middle** — the dead air
  belongs to the chapter that just finished, not the one about to start.
- **Chapters TILE the stream, asserted rather than assumed.** §9.2's structure is
  a partition and §9.4 chunks over it, so a gap between two chapters would
  silently drop that transcript from the digest with no error. `_assert_tiles`
  refuses to return a broken partition.
- **§9.3's 10–30 minute target is guidance.** An over-long chapter splits only at
  a boundary genuinely found inside it, never at an invented midpoint; when there
  is none, the shortfall is reported. A 95 s fixture is honestly one chapter.
- **The speech gate is §6.4's, reached through a new 3-line
  `gates.speech_activity`.** HANDOFF already records that §6.4's "ANY speech" and
  §5.4.1's "VAD on both tracks" being asked twice was a real hazard, with a test
  asserting they agree sample for sample. A third copy in `digest/` would have
  reopened it; a test asserts `chapters.py` calls the wrapper and never
  `derived.speech_gate(` directly.
- **Chapters are computed, never a table.** §3.2 declares none and §9.2 nests
  them inside the digest JSON, which is where the digest stage will put them.
- **What no test can show: that these are good chapters.** The speech fixture is
  one continuous conversation and `fixture_long` is band-limited noise. The
  mechanism is tested; the judgement needs a real transcript.

**Semantic search (Phase 5, §11.6)**

- **§5.10's "~50 ms over 200k vectors" is verified and its implied
  implementation is wrong.** MEASURED at 768 dims: the matmul is 47 ms, but
  loading the rows first costs 1004 ms and 614 MB — the read dominates the
  arithmetic 20:1, which §5.10 does not mention. Streaming in 4096-row chunks
  with a running top-K is **716 ms and 12.6 MB**: faster AND 49× lighter, so
  there was nothing to trade off.
- **`search.chunk_size` was silently changing the results.** THE finding of this
  commit, and it was a failing test that produced it. The speech fixture's three
  identical `"Let's go, Hawkeye."` lines hold identical vectors, and their
  scores came back `0.462986` at chunk=1 and `0.462985` at chunk=2. Not a
  tie-break bug: `matrix @ vector` takes a different BLAS path depending on how
  many rows it is handed, and float32 accumulation over 768 terms is not
  associative. Ranking is now computed on scores **quantised to 5 decimals**
  (`_RANK_DECIMALS`, deliberately not config) with `segment_id` breaking the
  rest — 1e-5 is far above the ~1e-6 of observed noise and far below anything
  meaningful. The unrounded score is still returned and displayed.
- **…so `test_results_are_ranked_by_descending_score` asserts descending at the
  RANKING's precision, not at full float precision.** Two hits here differ at
  the seventh decimal and come back very slightly out of raw order. Asserting
  strict descending on the raw value would be asserting that `chunk_size`
  changes results, which is the property the quantisation exists to remove.
- **There is no `min_similarity`, and that is a measurement.** MEASURED over 65
  query–document pairs: scores span **0.345–0.610** (mean 0.477, sd 0.060). A
  threshold anywhere in that band keeps roughly half of everything, related or
  not, while reading like a relevance filter. It was the obvious knob to add and
  it would have been actively harmful. A test asserts it stays absent.
- **The retrieval tests use queries sharing NO content words with their
  targets**, because that is what §11.6 claims and a "search Hawkeye, find
  Hawkeye" test would pass against a `LIKE` query. A second test asserts the
  probes share no content words, so a reworded probe cannot quietly turn the
  suite into a keyword test.
- **Asserted at recall@3, not recall@1.** MEASURED: 4 of 5 probes rank their
  target first, one ranks it third out of thirteen. Trimming the probe set until
  everything hit rank 1 would be tuning until it passes; the rank-3 case is kept
  and recorded. Thirteen lines says the mechanism works, not that retrieval is
  good — that needs a real transcript.
- **Segments are seeded from the fixture manifest, NOT transcribed.** Running
  WhisperX to test search would couple two unrelated things and let ASR errors
  decide whether retrieval passes, and would make the test depend on a multi-GB
  download. The manifest's `utterances` are authored ground truth.
- **`search:` is a top-level config subtree**, not under `extract`/`score`: a
  search parameter must never invalidate a candidate or force a re-score. Same
  reasoning as `render:` and `previews:`.
- **Search returns SEGMENTS and `candidate_id` is nullable by design.** They
  exist independently of scoring, so a memorable line is frequently nowhere near
  a detected peak. Opening a result focuses the covering candidate when there is
  one and otherwise focuses the nearest **and says so** — silently landing on an
  unrelated moment would misreport what was found. VERIFIED in a browser both
  ways.
- **`review.enter` now takes a stream id OR `{streamId, at}`**, so every existing
  caller — the library, the run view, boot's `?stream=` resume — is unchanged.
- **Three empty states, three sentences.** Nothing indexed, embedder
  unreachable, and nothing matched are different answers; showing "no results"
  for all three is how a broken search looks healthy. On shipped defaults the
  first is the COMMON case, because §5.7's transcription is off. The unreachable
  case is a 503 rather than a 500, because the operator can fix it.
- **Submit on Enter, never search-as-you-type.** A query costs an embedding round
  trip plus an index scan (~0.7 s at 200k segments), so per-keystroke querying
  would fire a dozen to answer one question. The router already refuses to
  swallow keys aimed at an `INPUT`, so the box and the library's `j`/`k` coexist
  with no second global handler.

**Scene events (Phase 3, §5.1 stage 11) — BUILT BLIND, AND SAYS SO**

- **No OBS log has ever been seen by this code**, so every regex in
  `extract.scene_events.patterns` is transcribed from the documented format and
  **none is validated**. This is a different kind of guess from everything else
  in GUESSES: those are numbers that might be badly chosen, this is a pattern
  that may not match at all — and a pattern that does not match produces zero
  events, silently, looking exactly like a session with no scene switches.
- **The whole correction path is one command.**
  `clipforge scene-events --check "<log>"` prints which patterns fired, the
  share of lines with a parseable timestamp, the recording spans found, the
  scene timeline, and every line mentioning a scene or a recording that no
  pattern claimed. It takes a FILE rather than a stream id so it runs on the
  streaming PC before anything is registered, and it prints a report so the
  *report* can be sent back rather than the log — OBS logs carry machine paths,
  hardware details and sometimes stream URLs.
- **Every regex lives in one config block and nowhere else in the codebase.**
  Fixing the parser is a YAML edit in `extract.scene_events.patterns`.
- **`tests/fixtures/obs_logs/` is parametrized.** Dropping a real log in there
  is the entire integration step — no test changes. A log with an
  `.expected.json` beside it gets exact span assertions; one without still has
  to parse. The two shipped logs are named `*-UNVALIDATED.txt` and their first
  line says they have never been near an OBS install.
- **NO WALL CLOCK IS REACHABLE FROM THE PARSER, enforced structurally.**
  `t = elapsed(scene line) − elapsed(recording start)`, both read from inside
  the same file. Reading the log's *filename* — which carries a local timestamp
  and is the obvious route — would drag in a timezone and a DST discontinuity,
  wrong by an hour twice a year with nothing downstream able to notice: §4.1's
  unrecoverable-offset class, and A8's reason for existing. A test parses
  `obs_log.py`'s AST and asserts it imports no `datetime`, `time`, `calendar`
  or `zoneinfo`, so this is a property of the module rather than a habit.
- **A log holding several recordings with nothing to tell them apart is
  REFUSED, not guessed.** One OBS session can start and stop recording
  repeatedly. The span whose logged output path ends with the master's filename
  wins; failing that, the only span wins; failing that, the stage defers with a
  message naming the fix. `patterns.recording_path` ships EMPTY for the same
  reason — guessing it would silently select the wrong recording, which is worse
  than the refusal it would replace.
- **No auto-discovery from `%APPDATA%\obs-studio\logs`**, for the same reason:
  selecting the right log needs the wall clock above. `register --obs-log <path>`
  is explicit, and the beside-the-master fallback is `obs-log.txt` **exactly** —
  a `*.txt` glob there would claim any stray notes file.
- **Scenes are stored as SPANS** (`t` .. `t_end`), the shape commit 36 settled
  for `menu_screen`. The scene already up when recording began runs from t=0:
  OBS logs a switch when it happens, not when recording starts, so dropping it
  would leave the opening of every stream with no scene at all.
- **`scene_change` carries no profile weight**, per §16 ("rejected as a
  scorer... chapter-boundary tie-breaker only"), and a test asserts it.
- **Noted and deliberately not built:** a scene named "BRB" or "Starting Soon"
  is exactly §6.4's `menu_screen` case with no Phase 7 vision needed —
  `score/gates.py` already reads `menu_screen` events and is inert only because
  nothing writes them. Wiring scene names to that gate would be scoring on a
  parser that has never seen a real log, which C5 and §16 both forbid. Revisit
  once `--check` has run against real text.

**Preview assets (Phase 3, §7.2)**

- **Assets are named by their WINDOW, never by `candidate_id`** — the change
  that makes §6.1's "re-scoring is free and infinitely repeatable" survive
  contact with this stage. §7.2's own command writes
  `previews/{candidate_id}.webm`, and candidates here are append-only
  generations: a re-score with different weights mints entirely new ids, so
  §7.2's naming turns every re-score into a full re-encode of the library. The
  clip is keyed on `t_peak` alone and the strip on the window, so a §7.3
  boundary nudge reuses the expensive asset. VERIFIED: a forced re-run and a
  forced re-score both left every file's mtime untouched.
- **§7.2's speed preset costs 13.6 minutes and was replaced. MEASURED** on real
  720p footage, five 2 s clips, SSIM against a lossless reference, extrapolated
  to §7.1's 120 candidates: §7.2 verbatim 0.9638 SSIM at 6.78 s/clip
  (**13.6 min**); `-cpu-used 8 -deadline realtime` 0.9564 at 0.71 s/clip
  (**1.4 min**). §1.3 budgets 20–40 min for *all* unattended processing and
  `extract.f0` already spends ~16 of it. 9.5× faster for 0.007 of SSIM.
- **…but VP9 itself stays, and that half is measured too.** At a matched
  ~250 KB, h264 scores 0.9416 against every VP9 variant's 0.9564+. Worth
  recording because the first comparison, at h264's default `crf 28`, looked
  four times faster *and* smaller — which is what comparing two encoders across
  different quality scales always looks like. The SSIM run is what caught it.
- **The synthetic fixture says the opposite and would have hidden all of this.**
  On `fixture_long` the same command runs at 2.04 s/clip because that source is
  640×360 — understating the real cost by **3.4×** and making §7.2's defaults
  look affordable. The numbers above are from `Testvid.mp4` for that reason.
- **§7.2's waveform PNG is deliberately NOT written.** `review/queries.py`
  already ships a downsampled envelope in the candidate payload and `review.js`
  draws it as inline SVG — vector, theme-aware, no files, no stage cost, and
  already on screen. A PNG would be a second copy of the same numbers, 120 more
  files per stream, and orphaned by every re-score. What was genuinely missing
  was §7.2's **second track**.
- **So the sparkline now draws every role, over ONE shared dB range.** It used
  to take the first of `mic_rms`/`party_rms`/`game_rms` and draw it alone —
  invisible while `mic_rms` was the only one that existed, and wrong the moment
  a stream had two people in it. The shared range is the load-bearing part:
  per-track normalisation would put a silent party mic at the same height as a
  shouting operator, so the one comparison the picture exists to support would
  be the one it could not show. Colours come from `render.captions.styles` via
  the existing `role_colours`, so the envelope, the transcript panel and the
  burned-in captions cannot disagree about who is who.
- **The role IS the kind name**, so unlike the transcript panel and the captions
  this needs no `track_roles` lookup: `extract/features.py` writes one series
  per §4.2 role (`SIGNAL_FOR_ROLE`). There is no track index to resolve and a
  kind cannot be wrong about its own role.
- **The 2 s clip SHIFTS at the ends of a recording rather than shortening.**
  §7.2's `-ss {t_peak-1}` is negative for a peak inside the first second, and a
  peak near the end has under two seconds after it. C2 says expand rather than
  contract, so the window slides; it shortens only when the recording itself is
  shorter than the clip.
- **One failed asset does not fail the stage.** A preview is a review
  convenience — losing one costs a seek, where aborting costs every asset after
  it and leaves the stage permanently un-done. The count goes to the log and to
  `preview_assets_failed` (an INVENTED name; §14 has no previews row), so a
  systematic failure is visible rather than inferred.
- **The thumb strip is one decode pass, not five seeks.** MEASURED: 0.93 s per
  strip selecting every Nth frame in a single pass, against 3.30 s for five
  separate `-ss` seeks — five process launches. The comma inside
  `select='not(mod(n,N))'` has to be escaped, because ffmpeg splits filter
  arguments on commas before the expression parser sees the string.
- **`previews` is a dependency of nothing**, and the review screen still works
  without it: null `preview_url`/`thumbstrip_url` fall back to Phase 1's
  seek-and-hold. That is what makes the stage safe to disable and safe to add to
  a library of already-processed streams.

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

**Muting and filler removal (§8.6, §8.2)**

- **Both off by default, and a default render is byte-identical to commit 26's.**
  §16 lists "profanity muting by default" among the explicitly rejected
  features. `trim_chain` returns `""` when nothing was cut rather than a trim
  spanning the whole window — an empty string makes "unchanged" true, where a
  full-span trim would only make it *probably* true.
- **Three mechanisms were measured before they were built**, because any of
  them failing would have forced a restructure:
  - `-af` accepts a branching graph (`asplit`/`atrim`/`concat` is a simple
    filtergraph), so audio cuts do **not** force audio into `-filter_complex`.
    That matters because commit 26's loudness pass 1 also uses `-af`: had it
    gone the other way, pass 1 would have measured un-cut audio while pass 2
    normalised cut audio, wrong by however much was removed and undetectable.
  - Video and audio stay in sync through cuts — MEASURED 9.000 s / 270 frames
    at 30 fps against 9.000 s of audio, on a 10 s window with a 1 s cut.
  - One `volume` filter covers several ranges, `+` being logical OR in ffmpeg's
    expression language.
- **Mute ranges are computed in source time and mapped through
  `EditPlan.map_span`**, so a mute after a cut lands where the word actually
  ended up rather than where it used to be. A word *inside* a cut is not muted
  at all — it is gone, and muting its old position would silence whatever moved
  into the gap.
- **Order is cuts → mute → loudnorm**, and it is not interchangeable: the mute
  ranges are already on the post-cut clock.
- **Cuts are planned before captions.** `render_stream` builds the cut plan,
  rebuilds the `EditPlan` with it, and only then generates the ASS — which
  drops words inside a cut automatically. The seam was built in commit 23 for
  exactly this.
- **The filler planner is mostly refusals**, and each one is a thing §8.2's
  one-line description does not mention: whole aligned words only (an
  interpolated timestamp would cut audio at a time nobody measured); only from
  `render.filler.roles`; never across another track's speech; nothing shorter
  than `min_duration_s`; cuts closer than `merge_gap_s` merged so no island of
  audio is left stuttering between them; and **the whole plan refused** if it
  would remove more than `max_share` of the clip, because that is a word-list
  problem rather than a clip full of filler.
- **Filter-graph pads are namespaced.** The trim chain uses `[k*]`/`[kt*]` and
  the crop graph's split became `[r_s*]`; two graphs sharing a pad name is a
  silent mis-wire, and they are now concatenated into one `-filter_complex`.
- **VERIFIED END TO END, not just in a filter string.** On the fixture's
  "Namor's turrets are melting me, holy shit." at 26.5–28.83 s: the mute
  interior measured **−19.0 → −43.5 dB** while 2.0–3.0 s and 4.0–4.5 s were
  untouched (−15.13 → −15.11, −27.29 → −27.50).
- **`--dual`** renders the muted and unmuted pair (§8.6), tagging the filenames
  `_clean` and `_unmuted` — but only when `--dual` is used, so turning muting on
  does not rename files the operator already has.
- **What the fixture cannot answer, and this ships anyway**: whether cutting
  filler improves a clip. Cutting a word cuts the video, and on gameplay that
  is a visible jump. The word list is general English rather than this
  operator's speech, marked **arbitrary** in GUESSES with the same falsifier as
  the profanity list — ten streams of transcript settles it.

**Hook text (§8.5, §12)**

- **No API key, so the round trip IS the product.** §2.2 puts reasoning through
  a frontier model and §12.4 prices it at cents per stream, but the operator
  does not want a paid key yet and nothing here can drive a chat website on
  their behalf. `clipforge hook` writes a prompt, the operator pastes it into
  claude.ai, and the reply is validated on the way back. That turns out to
  satisfy §12 **more** strictly than an API call would, because every check
  runs against a reply nobody controlled.
- **Real database ids in the prompt, not 1–5.** §12.2 wants a hallucinated id
  to be *detectable*, and if five clips are numbered 1 to 5 then any number a
  model invents inside that range looks valid. Export ids give a fabrication
  somewhere to fail.
- **The unit is a rendered clip.** §8.5 stores the result in
  `exports.hook_text` and that row exists only after a render. It also matches
  §8.1's loop — render the batch, watch it, pick hooks for what you will post.
- **Only the newest export per moment is offered.** `exports` is append-only,
  so re-rendering three times leaves three rows for every moment; asking a
  model about the same clip three times spends its attention and the
  operator's on nothing. Found by running it: a scratch database offered
  **fourteen** clips where three existed.
- **`--apply` stores nothing.** It validates, writes the metric, and prints the
  options with a ready-to-paste `--pick` line. §8.5 calls the hook the
  single highest-leverage decision in short-form and says it stays manual;
  storing the model's first option would be the model deciding.
- **The options are never persisted either**, which is why `--pick <id> <n>`
  needs the reply file it came from. Storing four hooks nobody chose is how
  `hook_text` stops meaning "the hook". `--pick <id> "text"` needs nothing.
- **The reply parser is tolerant** — the last balanced JSON object in the text.
  What comes out of a chat window has prose wrapped around it, and asking the
  operator to trim that by hand is the friction that makes a tool go unused.
  Same tactic `loudness.parse` needed against ffmpeg's diagnostics.
- **`llm_invalid_id_rate` is §14's own name for the metric**, checked rather
  than invented — a different name here means whatever reads it later finds
  nothing. The bad-quote drops are reported but deliberately **not** in that
  rate: §12.2 defines it over ids.
- **Drops are silent to the model, not to the operator.** §12.2 says
  non-existent ids are dropped silently; a person who cannot see that two of
  three entries were discarded has no way to tell a bad reply from clips that
  are genuinely unhookable.
- **`HookSource` has `available(cfg)`** so an API-backed source drops in beside
  `ManualHookSource` without changing anything else — the `StageSpec.available`
  and Ollama pattern. `ManualHookSource` is always available: it needs a person
  and a browser, not a credential. That source did drop in, in commit 42.

**§12's rules, in one place (Phase 5, §12) — `clipforge/llm/`**

- **The move happened before the third copy, not after.** §12's four rules were
  written once inside `render/hooks.py`, against `exports.id`. §9.4's digest
  map, §10.3's ground pass and §11.1's cluster labels need the identical four
  against different ids, and the cheapest moment to have one validator is
  before the second caller exists.
- **`validate_selections` differs per caller in exactly two arguments**: the id
  field, and a `text_for` that says which text a quote about that id has to
  appear in. Hooks pass `export_id` and the clip's transcript; the digest will
  pass §12.1's `segments.seq` and the segment's own text.
- **The refactor's evidence is that `tests/test_hooks.py` passes unmodified**,
  and separately that the prompt is **byte-identical** to what commit 41
  produced — checked by rebuilding the old string from that commit's literals
  and diffing, not by substring assertions.
- **Prompt text moved to `config/prompts.yaml`.** §17 wants tunables in config,
  and prompt quality is the one thing here that cannot be measured at all
  without footage — so it is the part most certain to be rewritten, and
  rewriting it must not mean editing Python. `llm:` is top-level and **outside
  `VERSIONED_SUBTREES`**: a prompt edit must never invalidate a candidate.
- **The JSON example stays in code**, because it has to carry a real
  `export_id`. §12.2's argument for real handles is that an invented id has
  somewhere to fail, and an example numbered 1 gives a hallucination in that
  range nowhere to fail.
- **Substitution is `$name`, not `{name}`.** Prompts contain JSON examples and
  JSON is made of braces. An unknown `$name` raises rather than being left in
  text a model then reads.
- **`prompts.digest_of` exists for §9.1.** Digests are kept forever and never
  regenerated, so two digests of one stream made by two prompts have to be
  distinguishable in origin. The digest stage stores the hash beside the output.
- **NOTHING HAS CALLED A MODEL.** `AnthropicSource` is written and reports
  itself unavailable for two reasons separately — the package is not installed
  and there is no key — because those need different fixes and one generic
  "not configured" would hide which. `clipforge doctor` shows the row.
- **An unavailable source is refused, not downgraded.** Configuring the API
  path and silently getting a paste prompt would look like the key working.
- **The one thing about the request that could be checked blind** is that it
  carries no `temperature`/`top_p`/`top_k` and no `budget_tokens`: those are a
  documented rejection on the configured model rather than a matter of taste,
  and a test asserts their absence. Everything else about it is a guess until
  a key exists, so every value is config.

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

**Backup (§13.2, §13.3)**

- **No `sqlite3` binary, no cron, no `gzip`.** §13.2's recipe is four lines of bash.
  Nothing else in this project depends on a `sqlite3` CLI, and requiring one for the job
  that must never be skipped is a way to have it skipped. MEASURED: the venv is Python
  3.12.10 with SQLite 3.49.1, past `VACUUM INTO`'s 3.27 floor — which is still checked,
  so an older interpreter fails with a sentence rather than a syntax error.
- **The source is opened READ ONLY**, over a `file:...?mode=ro` URI, deliberately not
  `db.connect`. MEASURED: `db.connect`'s `PRAGMA journal_mode=WAL` rewrites the file
  header of a database not already in WAL — a write to the one file this module exists
  to protect. MEASURED that `VACUUM INTO` works fine over a read-only connection, and
  that the source's SHA-256 is unchanged across a copy. A test asserts the connection
  *refuses* an INSERT, so the guarantee is structural rather than a habit.
- **It is safe against a live database, and that was measured rather than trusted.**
  A5 puts the database in WAL so the review UI writes while the pipeline runs, so a
  04:00 backup will meet an open connection. MEASURED: with 626 KB of committed,
  deliberately un-checkpointed WAL outstanding, every one of those rows was in the copy.
  Had it not been, this would produce a file that looks like a backup and is missing the
  newest ratings — worse than none.
- **§13.2's own script fails on its second run of a day.** Its filename is
  `clipforge_$(date +%F).db`, and MEASURED, `VACUUM INTO` raises `output file already
  exists` on a non-empty target. So the spec's snippet breaks at the moment someone runs
  it by hand to be safe. Here that is an explicit refusal naming `--force`. MEASURED and
  worth knowing: the refusal covers only a *non-empty* target — a zero-byte file is
  silently accepted and overwritten, which is why every write goes through
  `atomic_output` (A10), which unlinks first.
- **Retention is 30 daily + 12 monthly, and the RULE is not in the spec.** Keep the
  newest `keep_daily`; then the newest file in each of the `keep_monthly` most recent
  months; delete the rest. **The newest backup is never deletable** whatever the config
  says, so a `keep_daily: 0` typo cannot empty the directory. **Prune only ever deletes
  files it can name** — anything not matching `clipforge_YYYY-MM-DD.db[.gz]` exactly is
  invisible to it, including a date-shaped name that is not a date.
- **No B2/S3 upload** (C5: no account, zero streams). But a `.gz` beside its source is
  not protection against the disk dying, and the command **says so on every run** rather
  than letting the word "backup" imply otherwise. `backup.mirror_dir` is the version
  available with no credentials: a second plain directory, an external drive or a synced
  folder. A failed mirror WARNS and never fails the backup — an unplugged drive must not
  mean no backup at all.
- **§13.3 is not a one-off.** The spec says to test the restore path "once, early. Then
  trust it"; trusting it for five months is the failure being prevented. `--verify`
  decompresses to a scratch directory, runs SQLite's `integrity_check`, opens the copy
  with the application's own `db.open_db(migrate_to_latest=False)` — which already
  refuses an unexpected schema version — and compares every table against the manifest
  written beside the backup.
- **Verification compares against a manifest, not the live database and not literals.**
  A month-old backup is legitimately behind the live database, so comparing the two would
  fail for an honest reason and train the operator to ignore the check; hardcoded counts
  would pass a backup that had lost rows. The counts are recorded at copy time and
  checked against themselves — the same rule every numeric test here follows.
- **…and the manifest is read from the COPY, not from the source** (fixed in 29a; the
  first cut read it from the source and shipped a race). A5 puts the database in WAL so
  the review UI writes while other things run, so anything committed between a
  source-side count and the `VACUUM INTO` is in the backup and not in the manifest —
  and `--verify` then reports a problem on a backup that is perfectly good. A nightly
  check that cries wolf stops being read, which is the failure §13.3 exists to prevent.
  It cannot be closed with a transaction: SQLite refuses to VACUUM from inside one.
  Reading the copy makes the manifest true by construction. The test drives a writer
  straight into the gap and fails without the fix.
- **The gzip header is stripped of its name and clock** (`filename=''`, `mtime=0`).
  Otherwise every nightly differs from the last even when the database did not, which
  defeats any mirror that deduplicates and makes comparing two backups impossible.
- **The scheduled task is printed, never created**, and it goes through `clipforge.ps1`
  rather than `clipforge.exe`: that launcher `Push-Location`s to the project root, and
  `config.find_project_root` walks up from the working directory — a task run from
  `System32` would otherwise resolve `./data` somewhere else and quietly back up nothing.
- **`backup_duration_s` is an INVENTED metric name.** §14's table has no backup row.
  Recorded anyway (C6), after the copy — so it describes a backup the backup does not
  contain — and never fatally: a database that cannot be written to is a reason to keep
  the backup, not to throw it away.
- **`clipforge doctor` reports the newest backup's age.** A nightly job that silently
  stopped looks exactly like one that is working, right up until it matters.

**§7.3's nudge keys (commit 30)**

- **Nothing new in the schema, and nothing new downstream.** `ratings.adjusted_start`/
  `adjusted_end` already existed, `render/selection.py` already preferred them, and
  `score/runner._inherit_ratings` already carried them across a re-score — so the keys
  reach the FCPXML and the renderer without either knowing they exist. `candidates` is
  never touched: §3.2's `CHECK (t_peak BETWEEN t_start AND t_end)` makes a nudge past the
  peak an IntegrityError, and candidates are the detector's output that a re-score is
  free to replace.
- **A nudge is NOT clamped to `min_window_s`/`max_window_s`, and that is the point.**
  Those are the two values §17 tunes *against these nudges*, so refusing a window outside
  their range would make the measurement circular — the operator could never record "I
  wanted this shorter than 8 seconds". Only arithmetic clamps apply: a window may not
  invert or leave the recording. VERIFIED in a browser: nudged down to 0.6 s against a
  `min_window_s` of 8.
- **A count cannot tune anything, so the row carries direction.** §17 asks "how often the
  operator nudges", which does not say which way to move a bound. `window_nudge_s`
  (an **invented** name — §14 has no such metric) records start/end deltas, keypresses,
  whether the window still contains its peak, and **whether the detector's window had
  been sitting exactly on a clamp**. That last field is the one that moves a number:
  repeatedly extending windows that arrived at exactly 60 s *is* `max_window_s` being too
  low.
- **The nudge and the rating are different events with different routes.**
  `ratings.rating` is NOT NULL, so storing an unrated nudge would mean inventing a
  rating — and a fabricated rating corrupts §14's `signal_firing_rate_by_rating`, which
  §17 calls the primary weight-tuning input. So the *window* rides along with the rating
  (§7.3's actual flow: watch, adjust, rate) and the *observation* posts to its own route
  when the operator leaves the candidate. A nudge followed by walking away is still
  counted. VERIFIED: the nudge route left the ratings table untouched.
- **An absent adjustment means "no opinion", not "revert".** `save_rating` COALESCEs, so
  re-rating a moment trimmed in an earlier session keeps the trim. The payload also
  returns the stored adjustment, or reopening a stream would show the detector's window
  and silently hide the operator's own.
- **An adjusted window suppresses unadjusted ones in its cluster** —
  `render/selection.py`. FOUND BY A FAILING TEST. `approved_moments` unions every rated
  window in a cluster so a re-score that trimmed one cannot shorten a moment approved at
  its original length (C2); but with a nudge in the mix, an overlapping older generation
  put the trimmed seconds straight back. The operator cuts dead air off the front and the
  export restores it, with nothing reporting that it did. A hand-set boundary is an
  explicit statement where a rated candidate window is not, so it wins. Two nudged rows
  in one cluster still union — both are the operator speaking.
- **The review modules are now checked for duplicate declarations in one scope.**
  MEASURED THE HARD WAY: a second `const span` in `drawSpark` is a SyntaxError, an ES
  module that does not parse does not load *at all*, and the entire review screen went
  blank while every Python test passed — none of them execute the JS. The check is in
  Python rather than shelling out to node, because there is no node on this machine and a
  test that always skips is not a test.

**One opinion per moment, in one module (`clipforge/moments.py`)**

- **The move happened as the second caller arrived, not after.** `Moment`,
  `rated_candidates`, `cluster` and `verdict` were written inside
  `render/selection.py` against what goes on a timeline. §14's
  `signal_firing_rate_by_rating` needs the identical rule — one opinion per
  moment, across generations, latest wins — against feature vectors, and §14's
  stated hazard is counting one judgment twice, which is the exact thing the
  rule prevents. Same argument commit 42 made for `clipforge/llm/`.
- **`render/selection.py` re-exports them and keeps `approved_moments`.**
  Deciding what goes on a timeline IS a render concern, and its two extra rules
  — union the cluster's windows, unless a hand-set boundary suppresses the
  unadjusted ones — are about frames rather than about ratings.
- **The evidence is `tests/test_selection.py` passing UNMODIFIED**, plus a test
  asserting `selection.verdict is moments.verdict` rather than `==`. A copied
  function passes every behavioural test today and drifts the first time one of
  the two is edited; `is` is the only assertion a second implementation cannot
  satisfy.
- **`marker_anchored` was one expression serving two incompatible questions.**
  It lived inline in `review/queries.py` as `press inside the window OR a
  non-zero marker contribution`. §7.4's safety net wants that union — §4.3's
  plateau runs 25 s either side of a press, so a window merely *near* one should
  not be buried by the weights. §14's `marker_precision` must not: it reads
  `contributing_signals`, which `features.breakdown` builds from **weighted**
  tracks only, so **the loose answer changes when a weight changes** — and a
  weight-tuning input that moves when the weights move cannot tune them.
  `moments.marker_anchoring` now returns both facts named separately
  (`press_inside`, `contributed`) and each caller states which it means. One
  derivation, two readings, no drift — the hazard HANDOFF already records for
  `gates.speech_activity`, one layer up. The review UI's behaviour is unchanged.
- **`marker_times` selects by `source`, not by `kind`.** §3.2's rationale for the
  `events` shape is "new sensor = new `source` value"; a third marker hotkey
  would arrive as a new kind under the same source, and a kind filter would
  ignore it silently.

**§14's weight-tuning metrics (`clipforge/tuning.py`, `metrics --tuning`)**

- **§14 names four metrics that bear on tuning and three of them did not exist.**
  `approval_rate` did. `signal_firing_rate_by_rating` — which §14 calls the
  primary weight-tuning input and §17 builds its whole procedure on — did not,
  nor did `marker_precision` or `marker_recall_proxy`. **Nothing in the codebase
  had ever read `candidates.feature_vector`**; A9 has been filling that column
  since Phase 1 for exactly this.
- **THE RANKING COLUMN IS THRESHOLD-FREE, and that is the load-bearing choice.**
  §14 says "which signals fired", but `feature_vector` holds three different
  kinds of number: a z-score for `continuous`, a kernel level 0..1 for `events`
  and `composite`, and §6.4's gate ramp 0..1 for `afk`/`menu_screen`. One firing
  threshold across those cannot mean one thing. Signals are ranked on
  **separation** instead — P(a *clip it* moment outscores a *skip* one), the
  normalised Mann-Whitney U over `combined.ranks`. Rank-based, so it reads the
  same on all three scales; and **below 0.5 means a signal discriminates the
  wrong way**, which a firing rate cannot show as such.
- **§14's literal firing rate is reported beside it**, because §14 names it and
  §17 says to pull that name out of `tool_metrics`. Events, composites and gates
  fire at `> 0` — grounded, since that is what a zero kernel level means — so
  `tuning.firing_threshold_z` is the ONE arbitrary number here rather than three,
  and a test asserts it cannot change the ranking.
- **It reads through `clipforge/moments.py`, and the obvious query would have
  returned zero.** `review_metrics`' `is_current = 1 AND rating_source =
  'operator'` finds nothing on a re-scored stream: after commit 43 the operator's
  row is on the superseded generation and the current one carries an
  `'inherited'` copy the filter excludes. The primary tuning input would have
  read zero on exactly the corpus it exists to measure.
- **It REFUSES rather than printing a table that looks like evidence.** Below
  `tuning.min_rated_moments` there is no ranking at all, and below
  `tuning.min_moments_per_class` an individual signal prints its reason instead
  of a number. VERIFIED against a scratch copy of the real database: 6 streams, 3
  ratings, and it declines — which is the correct answer.
- **Only the schema's DECLARED keys are scored.** MEASURED: every candidate in
  the real database predates `feature_schema` version 2, and its vectors carry
  context keys the current writer no longer emits. Iterating the stored JSON
  rather than `feature_schema.keys` would have ranked `mic_rms_db` — an absolute
  dB level — as a signal.
- **`marker_precision` takes `press_inside`, never the looser reading.** See the
  `moments.py` section above: the loose half reads `contributing_signals`, which
  is built from weighted tracks, so it moves when a weight moves. A test zeroes
  the marker weight and asserts the precision is unmoved.
- **`--record` is opt-in, not every run.** §17 says to pull the metric from
  `tool_metrics`, so it must be writable — but this is a pure function of ratings
  and feature vectors, both kept forever, so nothing is lost by not recording and
  C6 does not compel it. A report command that wrote on every invocation would
  fill the table with duplicates of a number it can always recompute.
- **`metrics --json` keeps its shape unless `--tuning` is passed.** Nesting the
  old flat `{stream_id: metrics}` unconditionally would break anything already
  parsing it, for the convenience of a section that caller did not ask for.

**§14's `approval_rate` was wrong on any re-scored stream (fixed in 44b)**

- `review_metrics`' `by_rating` filtered `rating_source = 'operator'` while
  `load_candidates` LEFT JOINs ratings with **no such filter**. So the review
  screen showed an inherited rating as rated, and the summary beside it said
  `rated 0` and `approval rate 0.0%`: **a summary disagreeing with the screen it
  summarises.** The filter is gone; `is_current = 1` is one generation and a
  candidate has at most one `ratings` row, so nothing can be counted twice at
  this scope. §14's count-one-judgment-once hazard belongs to `tuning.py`, which
  spans generations and reads through `moments` for that reason.
- **`by_source` is now in the payload** and `clipforge metrics` prints how many
  ratings were carried over by a re-score rather than made about this generation.
  Counted, because the screen counts them — but never silently.
- **`score/combined.py`'s `_ranks` is now `ranks`.** §14's separation needs the
  identical ties-averaged helper, and reaching into another module's underscore
  name is how two copies of a helper come to exist.

**The written rubric (`clipforge/rubric.py`) — the learning layer's other half**

- **This has NO § REFERENCE, and that omission is the point.** §17 tunes weights
  from `signal_firing_rate_by_rating` and §14 calls that the primary tuning
  input — but a firing rate cannot carry "the bit only works when she doesn't
  see it coming", and no feature vector ever will. The rubric is a versioned
  plain-language document the operator writes after a review batch, fed into the
  LLM ranking and ideation prompts. It works at n=1 where fitted weights need
  dozens of streams, and it is still interpretable six months later.
- **IT MUST NEVER REACH SCORING, and that is enforced structurally.** Scoring is
  deterministic (C1) and §6.1 promises re-scoring is free and infinitely
  repeatable; a rubric on that path would make every re-score depend on prose
  `config_version` cannot even describe, since the text lives in a table rather
  than in config. A test AST-walks every module under `clipforge/score/` and
  asserts none imports it — the shape `test_capture.py` uses for §2.1.
- **Append-only. Never UPDATE.** §9.1 keeps digests forever and never
  regenerates them, so a digest made under v3 is a different artifact from one
  made under v4 and v3 has to stay readable. `digests.rubric_version` records
  which was in force. A test asserts the module has no `update`/`delete`/
  `replace` at all: the absence of the verb is the guarantee.
- **No `rubric_of()` hash**, deliberately, despite `prompts.digest_of()` sitting
  right beside it. That one exists because a prompt template lives in a config
  file editable in place, so its identity is its content. A rubric row is
  immutable by construction, so the version integer IS its identity.
- **One free-text column, not named sections.** `what_worked`/`what_didnt`/
  `watch_for` would decide the shape of the operator's thinking before a single
  rubric had been written. Markdown headings inside `text` cost nothing later.
- **Oversized is WARNED about, never truncated** (`rubric.warn_chars`, 6000,
  arbitrary). Every downstream prompt carries the whole text; silently dropping
  the end of the operator's judgement is the exact failure this prevents.
- **`ABSENT` is a sentence, not an empty string.** `prompts.render` uses
  `Template.substitute`, which raises on anything unsupplied — so `$rubric`
  always needs a value, and a blank section under a heading reads as though
  something went missing and invites a model to invent guidance that is not
  there.
- **Nothing consumes it yet.** The seam is proved against a temp prompts file,
  and a test asserts no shipped prompt contains `$rubric` — adding one to the
  hook prompt would break `render/hooks.py`, whose caller supplies exactly four
  named values.

**Migration 0003, and three gaps closed while the tables were still empty**

- **RUN `clipforge db init` BEFORE THE REVIEW SERVER.** `review/app.py`'s
  `connect()` opens with `migrate_to_latest=False`, so every route fails loudly
  until the schema is migrated. That is correct — a server reading a schema it
  does not understand is worse — but this is the first migration since `0002`.
- **`open_loops.segment_id`** (for §9.4's digest). §9.2's JSON gives every loop
  one and the table had nowhere to put it, so §12.3's verbatim-quote check ran
  at ingest and then discarded its own evidence.
- **`digests.prompt_hash` and `digests.rubric_version`** (for §9.4 and the
  rubric). `llm.prompts.digest_of` has existed since commit 42 with nowhere to
  store its result.
- **`performance` re-keyed to `(export_id, platform)`.** §3.2 keyed it on
  `export_id` alone, so one export could carry one performance row — but the
  unit of an export here is a rendered clip and the three presets differ only by
  `max_duration_s`, so the same file posted to Shorts and TikTok is one export
  with two performances and the schema refused the second. A rebuild, since
  SQLite cannot alter a primary key; safe because the table is empty and nothing
  references it. `platform` became NOT NULL, because SQLite permits NULLs in a
  non-INTEGER primary key column and two null-platform rows would have defeated
  the change.
- **All four in ONE migration**, because every affected table is empty today and
  a column is far cheaper to add before that stops being true — `0001_init.sql`
  already argues this for whole tables ("an empty table costs nothing;
  retrofitting one costs a migration and a backfill").

**The rubric editor and §7.3's `n` key (the review UI)**

- **The editor is a panel inside `#summary`, not a new view.** `#summary` is a
  sibling of `#review-main` inside `<main id="view-review">`, hand-toggled by
  `review.js`; the router only ever toggles the view. So this needed no
  `router.register`, no `BARS` entry and no new JS module — markup plus handlers
  in the module that already owns that screen.
- **It is on the summary rather than the candidate screen** because C4 budgets
  four seconds a candidate and says to fix the UI before adding anything
  anywhere if that slips. A textarea does not belong in that loop, and the
  summary is the moment the opinion has just formed.
- **The textarea prefills with the current version**, so the operator amends
  rather than retypes — and `POST /api/rubric` therefore returns
  `unchanged: true` WITHOUT writing when the text matches. Without that, opening
  the summary and clicking Save would mint a version saying nothing new, and
  §9.1's reasoning makes every version permanent.
- **§7.3's `n` rides with the rating, and that is the same wall commit 30 hit.**
  `ratings.rating` is NOT NULL, so a note on an unrated candidate could only be
  stored by inventing a rating — and a fabricated rating corrupts §14's
  `signal_firing_rate_by_rating`, which commit 44b just built. So: already rated
  posts immediately against the existing rating; not yet rated holds the note
  and `rate()` sends it with the verdict. `renderVerdict` shows a held note as
  *pending*, because "it will be saved when you rate this" is a promise the
  operator would otherwise have to take on trust.
- **Moving off a candidate closes an open note box and discards it.** Carrying
  it over would let `j` file a note against the wrong moment; pressing `j`
  mid-sentence means abandoning the sentence.
- **`api_rate` caught only `ValueError` where `api_nudge` beside it caught
  `KeyError` too**, so a rating body without `rating` surfaced as an unhandled
  500 rather than a 400. Unreachable while the only client always sent one.
  Fixed, with a test.
- **`search.js` was in NEITHER hardcoded module list** in `tests/test_review_api.py`
  and in no test file at all, so commit 40's module had been covered by neither
  the duplicate-declaration check nor the element-id check — and the first of
  those exists because a stray `const` once blanked the whole review screen past
  a green suite. Added to both, and the second list now reuses `MODULES` instead
  of repeating it.
- **The rubric route tests get their OWN database.** `test_review_api.py`'s
  `processed` fixture is session-scoped and mutable — which is why
  `test_metrics_include_the_approval_rate` asserts a range rather than a value.
  "Nothing has been written yet" is not a range.
- **VERIFIED IN A BROWSER, and it found something.** `saveRubric` set its status
  line and then awaited `loadRubric()`, which clears that line — so the operator
  clicked Save and watched the confirmation vanish. Reordered. No Python test
  executes this JS, which is the whole reason the browser pass exists. Also
  verified: `n` on an unrated candidate shows *note pending*, rating it lands
  the note in `ratings` with `rating_source='operator'`, and a rubric written in
  the browser reads back through `clipforge rubric --list` and `--diff`.

**The authored three-chapter fixture, and a scripted LLM source (commit 46)**

- **§9.4's digest is a map-reduce over CHAPTERS, and nothing here had more than
  one.** The speech fixture is 95 s and thirteen utterances — this file already
  called it "honestly one chapter" — and `fixture_long` is band-limited noise
  with no transcript. The digest could have been built with its central
  structure untested.
- **MEDIA-FREE, and that is the design decision.** `chapters.silence_boundaries`
  reads `signal_series` through §6.4's gate and never touches audio, so
  `tests/fixtures/transcript.py` writes the arrays directly. No ffmpeg, no
  Ollama, no network: **17 tests in 6.4 seconds** on a fresh clone.
  `test_chapters.py`'s own fixture still covers §9.3 against real measured audio
  at 95 s — the two are complementary, and both should stay.
- **The signal is bursts and pauses, not a flat level, and that is FORCED.**
  `derived.speech_gate` is `rms > rolling_mean + vad.margin_db`, so a constant
  "speech" level can never exceed its own rolling baseline: a flat stretch reads
  as silence everywhere and the whole stream comes out as one chapter. Real
  speech alternates, the baseline settles between the two, and the gate
  separates them. A test asserts the authored levels differ by more than the
  margin, so the constraint is recorded rather than rediscovered.
- **MEASURED, and it corrected the fixture rather than the gate:** the gate marks
  exactly `vad.hangover_s` × `score_grid_hz` = **4 samples at the head of each
  gap and none after**. That is `speech_gate` holding the flag past the last
  burst, deliberately, so one sentence does not become nine utterances — the
  same overrun GUESSES records on the speech fixture. The first version of the
  precondition asserted zero speech anywhere in the gap and failed; the
  assertion was wrong, not the levels. It now reads the hangover from config and
  checks past it, and separately checks the gap still clears `min_silence_s`
  with the hangover removed.
- **The authored chapters come out exactly right**: 3 chapters at 1790.1 s and
  3560.1 s against authored boundaries of 1790.0 and 3560.0 — one grid sample,
  which is the tolerance the test derives from `score.score_grid_hz` rather than
  writing down. 29.8 / 29.5 / 29.7 minutes, so this is **the first fixture where
  `target_min_s` / `target_max_s` mean anything**; a 95-second stream cannot
  exercise a 10–30 minute target.
- **Phrases are run through the real detector, not planted as rows.** The
  fixture seeds segments and the test calls `phrases.run(ctx)`, so the planted
  repeat proves the code path instead of restating it.
- **`tests/fakes.py`'s `ScriptedSource` is INJECTED, never registered.**
  `sources.SOURCES` is untouched and `source_for` still refuses unknown names; a
  stage takes `source=None` and a test passes a fake, the pattern
  `transcript.run(ctx, transcriber=None)` set. A fake reachable from config is
  one that can be selected by accident.
- **It records the PROMPTS, not just the replies.** What a digest asked for is
  as much a part of §12 as what it did with the answer — §12.1 says the model
  never sees a timestamp, and the only way to check that is to read the prompt
  that was sent. Retrofitting that later would mean rewriting the fake.
- **Calling more often than scripted raises `ExhaustedError` rather than
  returning a default.** §12.4 budgets a whole stream's reasoning; a map that
  quietly ran twice per chapter would otherwise be invisible.
- **The four reply shapes are all exercised**: accepted; hallucinated id dropped
  **and counted** into `llm_invalid_id_rate`; a real quote from the WRONG
  segment dropped as `bad-quote` with the id rate left clean; and two
  unparseable shapes returning None rather than raising — `malformed` (no JSON
  at all) and `truncated` (an object that never closes, which is what a
  `max_tokens` cutoff produces and what a naive "find the first `{`" parser
  would take).

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
.\clipforge.ps1 backup                                   # §13.2; --schedule for nightly
```

Launchers are scripts, not a frozen executable: once Phase 2 lands a PyInstaller build
would have to bundle torch — 3-5 GB, quarantined by antivirus, rebuilt on every
dependency change — and would still need ffmpeg and the Whisper models fetched
separately. It adds a build step without removing a setup step.

Everything the app does is also a command — `register`, `run`, `status`, `score`,
`signals`, `metrics`, `backup`, `db`, `config`, `synth-markers`. All take
`--set key.path=value` to override config for one invocation.

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

[`NEXT-SESSION.md`](NEXT-SESSION.md) is the short version — state, what a first
real stream would settle, and what is worth building if you are not streaming
yet. Read that, then [`CLAUDE.md`](CLAUDE.md) for the standing rules, this file
for the deviations, and [`spec/GUESSES.md`](spec/GUESSES.md) before touching any
number. Then open with something like:

> Read CLAUDE.md, HANDOFF.md and spec/CLIPFORGE-SPEC.md §<the sections you are
> touching>. Phases 0–2 are done. <What you want built.> Same as before: tell me
> anything you'd do differently first, test against the fixture manifest rather than
> hardcoded numbers, small commits, stop after each.

The remaining phases, in §15's order and with what each will cost:

| Phase | Adds | Needs |
|---|---|---|
| ~~**3** Full signals~~ | **DONE, commits 31–38**, except `scene_events` — see above. Pitch, laughter, silence, overlap, input signals, dual profiles + combined score, gated negatives, preview assets | Phase 0's `input_logger` must have run on a real stream before four of these mean anything |
| **4** Auto-finish | ASS captions with speaker colouring, vertical reframe, loudnorm, export presets | Real crop coordinates, measured off a real OBS layout |
| **5** Digests | Structured per-stream summaries, theme/assembly passes, §11.6's pull search | An API key would cost ~$0.10–0.30/stream, but the manual paste round trip `clipforge hook` already uses works with none. The search half needs no model at all |
| **6** Trends | N-gram bits, HDBSCAN clustering, idea dashboard, §11.6 pull search | **60+ streams before clustering finds anything real.** The n-gram half and the search are cheap and work sooner |
| **7** Vision | Kill feed, multikill, clutch, MVP | OpenCV plus per-game templates captured by hand; §5.9 says build it last and cut it if hard |

One thing worth doing before any of them:

- **Phase 6's n-gram baseline** (`is_baseline_tic`) once ten streams exist, replacing the
  hand-written stopword list in `phrases.yaml` — the one place a guess is currently
  standing in for a measurement the corpus would provide.

But the honest answer is at the top of this file: **the next thing to do is record a
stream.**
